"""
pitch_aggregates.py — Per-player aggregates computed from Statcast pitch data.

Fills gaps that BRef/Savant endpoints don't expose:
  batters  — gb_pct, fps_pct, pitches_per_pa
  pitchers — gb_pct, zone_pct, csw_pct, pitches_per_bf

All metrics are raw rates (0–1 scale); normalization happens downstream
in team_portrait._normalize_batting / _normalize_pitching.

Uses the cached Statcast parquet (pull_statcast_season) — no extra API calls.
In-process cache (_CACHE) prevents double-computation when batting and pitching
merge functions both call compute_pitch_aggregates in the same seed run.
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event-type sets
# ---------------------------------------------------------------------------

_SWUNG: frozenset[str] = frozenset({
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "foul_bunt", "missed_bunt",
})

_CSW: frozenset[str] = frozenset({
    "called_strike", "swinging_strike", "swinging_strike_blocked",
})

_SWING_MISS: frozenset[str] = frozenset({
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
})

# ---------------------------------------------------------------------------
# In-process cache keyed by season
# ---------------------------------------------------------------------------

_CACHE: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batter_aggregates(sc: pd.DataFrame, min_pa: int = 50) -> pd.DataFrame:
    """
    Per-batter aggregates from pitch-level Statcast.

    Returns DataFrame with columns:
        key_mlbam, gb_pct, fps_pct, pitches_per_pa, zone_swing_pct, contact_pct
    """
    df = sc[["batter", "game_pk", "at_bat_number", "pitch_number",
             "description", "bb_type", "zone"]].copy()

    # Ground ball flag
    df["_is_gb"]   = df["bb_type"] == "ground_ball"
    df["_has_bip"] = df["bb_type"].notna()

    # First-pitch swing flag
    df["_is_fp"]    = df["pitch_number"] == 1
    df["_fp_swung"] = df["_is_fp"] & df["description"].isin(_SWUNG)

    # Zone swing% — pitches in Statcast zones 1–9 (strike zone)
    df["_in_zone"]    = pd.to_numeric(df["zone"], errors="coerce").between(1, 9)
    df["_zone_swing"] = df["_in_zone"] & df["description"].isin(_SWUNG)

    # Contact rate — swings that don't miss
    df["_is_swing"]      = df["description"].isin(_SWUNG)
    df["_is_swing_miss"] = df["description"].isin(_SWING_MISS)

    # Chase (O-Swing) — swings at pitches outside Statcast zones 1–9.
    # Computed here because the Savant chase_pct column is only ~34% populated;
    # this fills the gap for plate-discipline traits downstream.
    df["_has_zone"] = pd.to_numeric(df["zone"], errors="coerce").notna()
    df["_oz"]       = df["_has_zone"] & ~df["_in_zone"]
    df["_oz_swing"] = df["_oz"] & df["_is_swing"]

    # Aggregate counting stats per batter
    agg = df.groupby("batter", sort=False).agg(
        gb_n          = ("_is_gb",        "sum"),
        bip_n         = ("_has_bip",      "sum"),
        fp_swung_n    = ("_fp_swung",     "sum"),
        fp_n          = ("_is_fp",        "sum"),
        total_pitches = ("pitch_number",  "count"),
        zone_swing_n  = ("_zone_swing",   "sum"),
        zone_pitch_n  = ("_in_zone",      "sum"),
        swing_n       = ("_is_swing",     "sum"),
        swing_miss_n  = ("_is_swing_miss","sum"),
        oz_n          = ("_oz",           "sum"),
        oz_swing_n    = ("_oz_swing",     "sum"),
    )

    # PA count = distinct (game_pk, at_bat_number) per batter
    pa_counts = (
        df.drop_duplicates(["batter", "game_pk", "at_bat_number"])
        .groupby("batter", sort=False)
        .size()
        .rename("pa_n")
    )
    agg = agg.join(pa_counts).fillna({"pa_n": 0})

    agg = agg[agg["pa_n"] >= min_pa].copy()

    agg["gb_pct"]         = np.where(agg["bip_n"] > 0,       agg["gb_n"] / agg["bip_n"],                              np.nan)
    agg["fps_pct"]        = np.where(agg["fp_n"] > 0,        agg["fp_swung_n"] / agg["fp_n"],                         np.nan)
    agg["pitches_per_pa"] = np.where(agg["pa_n"] > 0,        agg["total_pitches"] / agg["pa_n"],                      np.nan)
    agg["zone_swing_pct"] = np.where(agg["zone_pitch_n"] > 0, agg["zone_swing_n"] / agg["zone_pitch_n"],              np.nan)
    agg["contact_pct"]    = np.where(agg["swing_n"] > 0,     (agg["swing_n"] - agg["swing_miss_n"]) / agg["swing_n"], np.nan)
    agg["chase_sc_pct"]   = np.where(agg["oz_n"] > 0,        agg["oz_swing_n"] / agg["oz_n"],                         np.nan)

    result = agg[["gb_pct", "fps_pct", "pitches_per_pa", "zone_swing_pct",
                  "contact_pct", "chase_sc_pct"]].reset_index()
    result = result.rename(columns={"batter": "key_mlbam"})
    log.info("Batter pitch aggregates — %d players (min_pa=%d)", len(result), min_pa)
    return result


def _arsenal_entropy(sc: pd.DataFrame) -> pd.Series:
    """Shannon entropy of pitch_type distribution per pitcher (arsenal diversity)."""
    valid = sc[sc["pitch_type"].notna() & (sc["pitch_type"] != "")]
    pt = valid.groupby(["pitcher", "pitch_type"]).size()
    totals = pt.groupby("pitcher").sum()
    probs  = pt / totals
    entropy = (-probs * np.log(probs + 1e-9)).groupby("pitcher").sum()
    return entropy.rename("arsenal_diversity")


def _pitcher_aggregates(sc: pd.DataFrame, min_bf: int = 30) -> pd.DataFrame:
    """
    Per-pitcher aggregates from pitch-level Statcast.

    Returns DataFrame with columns:
        key_mlbam, gb_pct, zone_pct, csw_pct, pitches_per_bf,
        avg_spin_rate, arsenal_diversity
    """
    df = sc[["pitcher", "game_pk", "at_bat_number", "batter",
             "pitch_number", "description", "bb_type",
             "plate_x", "plate_z", "sz_top", "sz_bot",
             "release_spin_rate", "pitch_type"]].copy()

    # Ground ball flag
    df["_is_gb"]   = df["bb_type"] == "ground_ball"
    df["_has_bip"] = df["bb_type"].notna()

    # Zone flag — use per-pitch sz_top/sz_bot, fall back to ABS-standard league averages
    # ABS averages: top=3.38 ft, bot=1.59 ft, width=±0.833 ft (17" plate + ½ ball radius)
    _ABS_TOP   = 3.38
    _ABS_BOT   = 1.59
    _ABS_WIDTH = 0.833
    plate_x = pd.to_numeric(df["plate_x"], errors="coerce")
    plate_z = pd.to_numeric(df["plate_z"], errors="coerce")
    sz_top  = pd.to_numeric(df["sz_top"], errors="coerce").fillna(_ABS_TOP)
    sz_bot  = pd.to_numeric(df["sz_bot"], errors="coerce").fillna(_ABS_BOT)
    df["_in_zone"]      = (plate_x.abs() <= _ABS_WIDTH) & (plate_z >= sz_bot) & (plate_z <= sz_top)
    df["_has_location"] = plate_x.notna() & plate_z.notna()

    # CSW flag
    df["_is_csw"] = df["description"].isin(_CSW)

    # Spin rate
    df["_spin"]     = pd.to_numeric(df["release_spin_rate"], errors="coerce")
    df["_has_spin"] = df["_spin"].notna()

    agg = df.groupby("pitcher", sort=False).agg(
        gb_n          = ("_is_gb",        "sum"),
        bip_n         = ("_has_bip",      "sum"),
        zone_n        = ("_in_zone",      "sum"),
        location_n    = ("_has_location", "sum"),
        csw_n         = ("_is_csw",       "sum"),
        total_pitches = ("pitch_number",  "count"),
        spin_sum      = ("_spin",         "sum"),
        spin_n        = ("_has_spin",     "sum"),
    )

    # BF count = distinct (game_pk, at_bat_number, batter) per pitcher
    bf_counts = (
        df.drop_duplicates(["pitcher", "game_pk", "at_bat_number", "batter"])
        .groupby("pitcher", sort=False)
        .size()
        .rename("bf_n")
    )
    agg = agg.join(bf_counts).fillna({"bf_n": 0})

    agg = agg[agg["bf_n"] >= min_bf].copy()

    agg["gb_pct"]         = np.where(agg["bip_n"]      > 0, agg["gb_n"]    / agg["bip_n"],                   np.nan)
    agg["zone_pct"]       = np.where(agg["location_n"] > 0, agg["zone_n"]  / agg["location_n"],               np.nan)
    agg["csw_pct"]        = np.where(agg["total_pitches"] > 0, agg["csw_n"] / agg["total_pitches"],           np.nan)
    agg["pitches_per_bf"] = np.where(agg["bf_n"]        > 0, agg["total_pitches"] / agg["bf_n"],              np.nan)
    agg["avg_spin_rate"]  = np.where(agg["spin_n"]      > 0, agg["spin_sum"] / agg["spin_n"],                 np.nan)

    # Arsenal diversity — Shannon entropy of pitch mix
    entropy = _arsenal_entropy(df)
    agg = agg.join(entropy, how="left")

    result = agg[["gb_pct", "zone_pct", "csw_pct", "pitches_per_bf",
                  "avg_spin_rate", "arsenal_diversity"]].reset_index()
    result = result.rename(columns={"pitcher": "key_mlbam"})
    log.info("Pitcher pitch aggregates — %d pitchers (min_bf=%d)", len(result), min_bf)
    return result


# ---------------------------------------------------------------------------
# Opener usage (all 30 teams from full Statcast)
# ---------------------------------------------------------------------------

def compute_platoon_optimization(statcast: pd.DataFrame) -> pd.Series:
    """
    For each pitching team, fraction of PAs where pitcher and batter share handedness
    (same-hand matchups: RHP vs RHH, LHP vs LHH).

    Higher = team creates more same-hand matchups for its pitchers.
    Returns Series indexed by team abbreviation, values in [0, 1].
    """
    sc = statcast[["pitcher", "batter", "p_throws", "stand",
                   "inning_topbot", "home_team", "away_team",
                   "game_pk", "at_bat_number"]].copy()

    # Only count one row per plate appearance (avoid per-pitch duplication)
    sc = sc.drop_duplicates(["game_pk", "at_bat_number"])

    sc["pitching_team"] = np.where(
        sc["inning_topbot"] == "Top",
        sc["home_team"],
        sc["away_team"],
    )

    sc = sc.dropna(subset=["p_throws", "stand", "pitching_team"])
    sc["same_hand"] = sc["p_throws"] == sc["stand"]

    stats = sc.groupby("pitching_team").agg(
        same_hand_pa=("same_hand", "sum"),
        total_pa=("same_hand", "count"),
    )
    result = (stats["same_hand_pa"] / stats["total_pa"]).rename("platoon_pct")
    log.info("Platoon optimization computed — %d teams", len(result))
    return result


def compute_batter_spray_stats(
    statcast: pd.DataFrame,
    min_bip: int = 20,
) -> pd.DataFrame:
    """
    Per-batter spray chart statistics from Statcast hit coordinates.

    Uses hc_x / hc_y to compute spray angle, then classifies each batted
    ball as pull, gap, center, or oppo.  Gaps (LC and RC) are the ±20–50°
    band from the field center line.

    Args:
        statcast: full-season Statcast DataFrame (must have hc_x, hc_y,
                  bb_type, events, batter columns)
        min_bip:  minimum balls-in-play required to include a batter

    Returns DataFrame with columns:
        key_mlbam  — MLBAM batter ID
        bip        — total balls in play
        xb_pct     — (2B + 3B) / bip   (extra-base contact rate)
        gap_pct    — BIP with |spray_angle| in [20, 50]° / bip
        pull_pct   — BIP with |spray_angle| > 50° to pull side / bip
        oppo_pct   — BIP with |spray_angle| > 50° to oppo side / bip
        hr_per_bip — HR / bip
        hr_fb      — HR / fly balls (pure power purity signal; None if no fly balls)
    """
    # Keep only pitches with valid hit coordinates (batted balls)
    sc = statcast[
        statcast["hc_x"].notna() &
        statcast["hc_y"].notna() &
        statcast["bb_type"].notna()
    ].copy()

    if sc.empty:
        return pd.DataFrame(columns=[
            "key_mlbam", "bip", "xb_pct", "gap_pct",
            "pull_pct", "oppo_pct", "hr_per_bip", "hr_fb",
        ])

    # ── Spray angle ──────────────────────────────────────────────────────────
    # Home plate anchor in Statcast SVG coordinate space.
    # Positive angle → right-field side; negative → left-field side.
    HP_X, HP_Y = 126.0, 203.0
    sc["_angle"] = np.degrees(np.arctan2(
        sc["hc_x"].astype(float) - HP_X,
        HP_Y - sc["hc_y"].astype(float),   # y-axis flipped
    ))

    # Pull side for RHH is RF (positive angle); for LHH it's LF (negative).
    # Gap zones are field-location-based (±20–50°) regardless of handedness —
    # both LC and RC gaps sit in this band.
    sc["_abs_angle"]  = sc["_angle"].abs()
    sc["_in_gap"]     = sc["_abs_angle"].between(20, 50)
    sc["_is_pull"]    = sc["_abs_angle"] > 50   # extreme pull or oppo

    # Label pull vs oppo by handedness for the pull_pct / oppo_pct split
    rh_mask = (sc.get("stand", "R") == "R") | (sc.get("p_throws", "R") == "R")
    # For RHH: positive angle = pull (RF); for LHH: negative angle = pull (LF)
    sc["_is_pull_side"] = np.where(
        sc.get("stand", pd.Series("R", index=sc.index)) == "R",
        sc["_angle"] > 50,    # RHH pull = RF (positive)
        sc["_angle"] < -50,   # LHH pull = LF (negative)
    )
    sc["_is_oppo_side"] = sc["_is_pull"] & ~sc["_is_pull_side"]

    # Event flags
    sc["_is_double"] = sc["events"] == "double"
    sc["_is_triple"] = sc["events"] == "triple"

    # Fly ball flag — needed for HR/FB denominator
    sc["_is_flyball"] = sc["bb_type"] == "fly_ball"

    agg = sc.groupby("batter").agg(
        bip           = ("hc_x",          "count"),
        doubles       = ("_is_double",     "sum"),
        triples       = ("_is_triple",     "sum"),
        fly_balls     = ("_is_flyball",    "sum"),
        gap_bip       = ("_in_gap",        "sum"),
        pull_bip      = ("_is_pull_side",  "sum"),
        oppo_bip      = ("_is_oppo_side",  "sum"),
    ).reset_index()

    # HR count from the FULL statcast dataframe (not spray-filtered rows) so that
    # HRs with missing hc_x coords (wall-scrapers, tracking gaps ~1% of HRs) are
    # included in the numerator. The fly ball denominator still comes from the
    # spray-filtered rows where hc coords are valid.
    all_hr = (
        statcast[statcast["events"] == "home_run"]
        .groupby("batter").size()
        .rename("hr")
        .reset_index()
    )
    agg = agg.merge(all_hr, on="batter", how="left")
    agg["hr"] = agg["hr"].fillna(0)

    agg = agg[agg["bip"] >= min_bip].copy()
    bip = agg["bip"].clip(lower=1)

    agg["xb_pct"]    = (agg["doubles"] + agg["triples"]) / bip
    agg["gap_pct"]   = agg["gap_bip"]  / bip
    agg["pull_pct"]  = agg["pull_bip"] / bip
    agg["oppo_pct"]  = agg["oppo_bip"] / bip
    agg["hr_per_bip"]= agg["hr"]       / bip
    # HR/FB: home runs per fly ball — pure power purity signal.
    # Requires at least 15 fly balls for a reliable estimate; std dev drops from
    # ~0.22 at 0-5 FB to ~0.08 at 15+ FB (empirically validated on 2023 Statcast).
    # NaN for extreme GB hitters and low-sample players.
    MIN_FLY_BALLS = 15
    agg["hr_fb"] = np.where(
        agg["fly_balls"] >= MIN_FLY_BALLS,
        agg["hr"] / agg["fly_balls"],
        np.nan,
    )

    log.info("Batter spray stats — %d batters (min_bip=%d)", len(agg), min_bip)
    return agg.rename(columns={"batter": "key_mlbam"})[
        ["key_mlbam", "bip", "xb_pct", "gap_pct", "pull_pct", "oppo_pct", "hr_per_bip", "hr_fb"]
    ]


def compute_opener_usage(statcast: pd.DataFrame) -> pd.Series:
    """
    For each team, fraction of games where the first pitcher faced ≤ 9 batters (opener).
    Returns Series indexed by team abbreviation, values in [0, 1].
    """
    sc = statcast[["game_pk", "pitcher", "batter", "at_bat_number",
                   "inning_topbot", "home_team", "away_team", "pitch_number"]].copy()

    # First pitcher per (game_pk, side) = pitcher who threw pitch number 1
    first = (
        sc[sc["pitch_number"] == 1]
        .drop_duplicates(["game_pk", "inning_topbot"])
        [["game_pk", "inning_topbot", "pitcher", "home_team", "away_team"]]
    )

    # Count distinct batters faced per (game_pk, pitcher) to detect short outings
    bf_per_game = (
        sc.drop_duplicates(["game_pk", "pitcher", "at_bat_number", "batter"])
        .groupby(["game_pk", "pitcher"])
        .size()
        .reset_index(name="game_bf")
    )

    first = first.merge(bf_per_game, on=["game_pk", "pitcher"], how="left")
    first["is_opener"] = first["game_bf"].fillna(0) <= 9  # ≤ 9 BF ≈ 3 innings

    # Pitching team: when inning_topbot == 'Top', away team bats → home team pitches
    first["pitching_team"] = np.where(
        first["inning_topbot"] == "Top",
        first["home_team"],
        first["away_team"],
    )

    stats = first.groupby("pitching_team").agg(
        opener_games=("is_opener", "sum"),
        total_games=("is_opener", "count"),
    )
    result = (stats["opener_games"] / stats["total_games"]).rename("opener_pct")
    log.info("Opener usage computed — %d teams", len(result))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pitch_aggregates(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (batter_agg, pitcher_agg) for the given season.

    Both DataFrames have key_mlbam as the join key.
    Results are cached in-process so a seed run that calls this for both
    batting and pitching only reads the Statcast parquet once.
    """
    if season in _CACHE:
        return _CACHE[season]

    from ingest import pull_statcast_season
    log.info("Computing pitch aggregates for season %d", season)
    sc = pull_statcast_season(season)

    batter_agg  = _batter_aggregates(sc)
    pitcher_agg = _pitcher_aggregates(sc)

    _CACHE[season] = (batter_agg, pitcher_agg)
    return batter_agg, pitcher_agg


# ---------------------------------------------------------------------------
# Pitcher leverage exposure — |ΔWE| per plate appearance
# ---------------------------------------------------------------------------

_LEVERAGE_CACHE: dict[int, pd.DataFrame] = {}

def compute_pitcher_leverage(statcast: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Empirical leverage proxy per pitcher: mean absolute home-win-expectancy
    swing per plate appearance faced. A reliever used in tight late spots
    lives in high-|ΔWE| PAs; a mop-up arm's PAs barely move the needle.

    Computed entirely from statcast delta_home_win_exp (100% populated) —
    no external leverage-index table needed.

    Returns DataFrame: pitcher, n_pa, mean_abs_dwe
    """
    if season in _LEVERAGE_CACHE:
        return _LEVERAGE_CACHE[season]

    sc = statcast[["pitcher", "game_pk", "at_bat_number",
                   "delta_home_win_exp"]].copy()
    sc["_dwe"] = pd.to_numeric(sc["delta_home_win_exp"], errors="coerce").abs()
    # One row per PA: the PA's total |ΔWE| is on its final pitch, but summing
    # per-pitch |ΔWE| within the PA captures mid-PA swings too — use PA sum.
    pa = (sc.groupby(["pitcher", "game_pk", "at_bat_number"])["_dwe"]
            .sum().reset_index())
    agg = pa.groupby("pitcher").agg(
        n_pa=("_dwe", "count"),
        mean_abs_dwe=("_dwe", "mean"),
    ).reset_index()

    _LEVERAGE_CACHE[season] = agg
    log.info("Pitcher leverage computed — %d pitchers", len(agg))
    return agg


# ---------------------------------------------------------------------------
# Catcher framing — shadow-zone called-strike rate above league expectation
# ---------------------------------------------------------------------------

_FRAMING_CACHE: dict[int, pd.DataFrame] = {}

# ABS-standard zone bounds (same convention as _pitcher_aggregates)
_FR_TOP, _FR_BOT, _FR_W = 3.38, 1.59, 0.833
_SHADOW_BAND = 0.25   # ft each side of the zone edge = the frameable region

def compute_catcher_framing(statcast: pd.DataFrame, season: int,
                            min_taken: int = 500) -> pd.DataFrame:
    """
    Framing proxy per catcher, computed from raw pitches (the Savant framing
    endpoint is broken in pybaseball; this is the same idea from first
    principles): on TAKEN pitches in the shadow band around the zone edge,
    how far above/below the league called-strike rate does this catcher sit?

    Returns DataFrame:
        catcher (MLBAM), shadow_taken, cs_rate, league_cs_rate,
        framing_above (cs_rate − league, in percentage points),
        framing_pct (percentile among catchers with ≥ min_taken)
    """
    if season in _FRAMING_CACHE:
        return _FRAMING_CACHE[season]

    sc = statcast[["fielder_2", "description", "plate_x", "plate_z",
                   "sz_top", "sz_bot"]].copy()
    sc = sc[sc["description"].isin(["called_strike", "ball"])]

    px = pd.to_numeric(sc["plate_x"], errors="coerce")
    pz = pd.to_numeric(sc["plate_z"], errors="coerce")
    top = pd.to_numeric(sc["sz_top"], errors="coerce").fillna(_FR_TOP)
    bot = pd.to_numeric(sc["sz_bot"], errors="coerce").fillna(_FR_BOT)

    # Distance outside the zone on each axis (0 inside); shadow = within the
    # band of an edge, either side.
    dx = (px.abs() - _FR_W).clip(lower=None)
    dz_hi = pz - top
    dz_lo = bot - pz
    # Signed "distance from in-zone region": positive = outside
    outside = pd.concat([dx, dz_hi, dz_lo], axis=1).max(axis=1)
    shadow = outside.abs() <= _SHADOW_BAND
    sc = sc[shadow & px.notna() & pz.notna()]

    sc["_cs"] = sc["description"] == "called_strike"
    league_rate = float(sc["_cs"].mean())

    agg = sc.groupby("fielder_2").agg(
        shadow_taken=("_cs", "count"),
        cs_rate=("_cs", "mean"),
    ).reset_index().rename(columns={"fielder_2": "catcher"})
    agg["league_cs_rate"] = league_rate
    agg["framing_above"] = (agg["cs_rate"] - league_rate) * 100.0

    qual = agg[agg["shadow_taken"] >= min_taken].copy()
    qual["framing_pct"] = qual["framing_above"].rank(pct=True) * 100.0
    agg = agg.merge(qual[["catcher", "framing_pct"]], on="catcher", how="left")

    _FRAMING_CACHE[season] = agg
    log.info("Catcher framing computed — %d catchers (%d qualified), "
             "league shadow CS rate %.1f%%",
             len(agg), len(qual), league_rate * 100)
    return agg


# ---------------------------------------------------------------------------
# Catcher pop time (Savant leaderboard via pybaseball; cached to CSV)
# ---------------------------------------------------------------------------

def load_catcher_poptime(season: int) -> pd.DataFrame:
    """
    Pop time to 2B per catcher, cached to processed/poptime_{season}.csv.
    Adds pop_pct — percentile among listed catchers (LOWER time = better,
    so a low percentile = fast exchange).
    Returns empty frame if the endpoint is unavailable (tags simply skip).
    """
    cache = _HERE / "processed" / f"poptime_{season}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    try:
        import pybaseball
        df = pybaseball.statcast_catcher_poptime(season)
        df = df.rename(columns={"entity_id": "catcher"})
        df["catcher"] = pd.to_numeric(df["catcher"], errors="coerce")
        df["pop_2b"] = pd.to_numeric(df.get("pop_2b_sba"), errors="coerce")
        df = df[df["catcher"].notna() & df["pop_2b"].notna()]
        df["pop_pct"] = df["pop_2b"].rank(pct=True) * 100.0
        out = df[["catcher", "pop_2b", "pop_pct"]].copy()
        out.to_csv(cache, index=False)
        log.info("Catcher poptime — %d catchers cached", len(out))
        return out
    except Exception as exc:
        log.warning("Catcher poptime unavailable: %s", exc)
        return pd.DataFrame(columns=["catcher", "pop_2b", "pop_pct"])


# ---------------------------------------------------------------------------
# Baserunning advancement — extra bases taken, from base-state transitions
# ---------------------------------------------------------------------------

_ADVANCE_CACHE: dict[int, "pd.DataFrame"] = {}

def compute_batter_advancement(statcast: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Extra-bases-taken per RUNNER, from base-state transitions between
    consecutive PAs (des narration is only ~26% populated — never use it).

    Opportunities pooled across the three standard XBT situations:
      • runner on 1B, batter singles  → did he reach 3B?
      • runner on 2B, batter singles  → did he score?
      • runner on 1B, batter doubles  → did he score?

    A runner "scored" if he appears on no base in the next PA and the
    half-inning continued with more runs than outs added... ambiguity is
    expensive, so instead: advancement is credited from where the runner
    IS next PA; runners absent from the next PA's bases in a continuing
    half-inning with a run scored on the play are counted as scored
    (delta home/away score check). PAs that end the half-inning are
    dropped — conservative denominators beat guessed numerators.

    Returns per-runner DataFrame:
      key_mlbam, opps, advances, xbt_rate,
      first_to_third_opps, first_to_third_n
    """
    if season in _ADVANCE_CACHE:
        return _ADVANCE_CACHE[season]

    cols = ["game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot",
            "events", "on_1b", "on_2b", "on_3b", "post_bat_score", "bat_score"]
    have = [c for c in cols if c in statcast.columns]
    sc = statcast[have].copy()

    # Final pitch of every PA (events non-null), plus each PA's FIRST pitch
    # base state (the state the previous PA's runners ended in).
    sc = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    finals = sc[sc["events"].notna()].drop_duplicates(
        ["game_pk", "at_bat_number"], keep="last")
    firsts = sc.drop_duplicates(["game_pk", "at_bat_number"], keep="first")

    first_state = firsts.set_index(["game_pk", "at_bat_number"])[
        ["on_1b", "on_2b", "on_3b", "inning", "inning_topbot"]]

    rows: dict[int, dict] = {}

    def _bump(rid, kind, advanced):
        r = rows.setdefault(int(rid), {"opps": 0, "advances": 0,
                                       "ft_opps": 0, "ft_n": 0})
        r["opps"] += 1
        r["advances"] += int(advanced)
        if kind == "1b_single":
            r["ft_opps"] += 1
            r["ft_n"] += int(advanced)

    scored_ok = {"post_bat_score", "bat_score"} <= set(have)

    for f in finals.itertuples():
        ev = f.events
        if ev not in ("single", "double"):
            continue
        nxt_key = (f.game_pk, f.at_bat_number + 1)
        if nxt_key not in first_state.index:
            continue  # end of half-inning (or data edge) — drop, don't guess
        nxt = first_state.loc[nxt_key]
        if nxt["inning"] != f.inning or nxt["inning_topbot"] != f.inning_topbot:
            continue  # half-inning ended on the play — drop

        nxt_bases = {int(x) for x in (nxt["on_1b"], nxt["on_2b"], nxt["on_3b"])
                     if pd.notna(x)}
        runs_scored = (int(f.post_bat_score) - int(f.bat_score)) if scored_ok else 0

        def _resolve(rid, target_base_val, station_base_val):
            """advanced if on target base next PA; scored counts as advanced;
            on station base = held; otherwise ambiguous → drop."""
            rid = int(rid)
            if pd.notna(target_base_val) and int(target_base_val) == rid:
                return True
            if pd.notna(station_base_val) and int(station_base_val) == rid:
                return False
            if rid not in nxt_bases and runs_scored > 0:
                return True   # off the bases in a continuing inning + run(s) home
            return None       # forced out / pinch-runner / ambiguity — drop

        if ev == "single":
            if pd.notna(f.on_1b):
                adv = _resolve(f.on_1b, nxt["on_3b"], nxt["on_2b"])
                if adv is not None:
                    _bump(f.on_1b, "1b_single", adv)
            if pd.notna(f.on_2b):
                adv = _resolve(f.on_2b, None, nxt["on_3b"])
                # target for 2B runner on a single = home (absent + run scored)
                rid = int(f.on_2b)
                if pd.notna(nxt["on_3b"]) and int(nxt["on_3b"]) == rid:
                    _bump(rid, "2b_single", False)
                elif rid not in nxt_bases and runs_scored > 0:
                    _bump(rid, "2b_single", True)
        elif ev == "double":
            if pd.notna(f.on_1b):
                rid = int(f.on_1b)
                if pd.notna(nxt["on_3b"]) and int(nxt["on_3b"]) == rid:
                    _bump(rid, "1b_double", False)
                elif rid not in nxt_bases and runs_scored > 0:
                    _bump(rid, "1b_double", True)

    out = pd.DataFrame([
        {"key_mlbam": rid, "opps": r["opps"], "advances": r["advances"],
         "xbt_rate": r["advances"] / r["opps"] if r["opps"] else float("nan"),
         "first_to_third_opps": r["ft_opps"], "first_to_third_n": r["ft_n"]}
        for rid, r in rows.items()
    ])
    _ADVANCE_CACHE[season] = out
    log.info("Batter advancement — %d runners, %d opportunities",
             len(out), int(out["opps"].sum()) if not out.empty else 0)
    return out


# ---------------------------------------------------------------------------
# Position appearances — home position + versatility, from fielder_2..9
# ---------------------------------------------------------------------------

_POSITION_CACHE: dict[int, "pd.DataFrame"] = {}

_POS_NAMES = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS",
              7: "LF", 8: "CF", 9: "RF"}
MIN_DEFENSIVE_PITCHES = 300   # below → home position "DH"
POS_MIN_PITCHES       = 150   # a position "counts" at ≥ this OR ≥5% share

def compute_position_appearances(statcast: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Per player: defensive pitch counts by position (fielder_2..9 alignment
    columns — verified 100% populated), home_position, and positions_played.

    Returns: key_mlbam, home_position, positions_played, total_def_pitches,
             pos_detail (dict pos → share)
    """
    if season in _POSITION_CACHE:
        return _POSITION_CACHE[season]

    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for n in range(2, 10):
        col = f"fielder_{n}"
        if col not in statcast.columns:
            continue
        vc = pd.to_numeric(statcast[col], errors="coerce").dropna().astype(int).value_counts()
        for pid, c in vc.items():
            counts[pid][_POS_NAMES[n]] += int(c)

    rows = []
    for pid, ctr in counts.items():
        total = sum(ctr.values())
        if total < MIN_DEFENSIVE_PITCHES:
            home = "DH"
            played = []
        else:
            home = ctr.most_common(1)[0][0]
            played = [p for p, c in ctr.items()
                      if c >= POS_MIN_PITCHES or c / total >= 0.05]
        rows.append({"key_mlbam": pid, "home_position": home,
                     "positions_played": len(played),
                     "total_def_pitches": total,
                     "pos_detail": {p: round(c / total, 3)
                                    for p, c in ctr.most_common()}})
    out = pd.DataFrame(rows)
    _POSITION_CACHE[season] = out
    log.info("Position appearances — %d players", len(out))
    return out


# ---------------------------------------------------------------------------
# Runner control — mid-PA advances allowed per pitcher (mechanism-agnostic)
# ---------------------------------------------------------------------------

_RUNNER_CTRL_CACHE: dict[int, "pd.DataFrame"] = {}

def compute_runner_control(statcast: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Per pitcher: how often runners moved up a base BETWEEN pitches within a
    PA (stolen bases, wild pitches, passed balls and balks collectively —
    the base-state data cannot attribute mechanism, and the tag's evidence
    says so). Opportunities = PAs with 1B occupied at any point.

    Returns: pitcher, opps, advances_allowed, advance_rate
    """
    if season in _RUNNER_CTRL_CACHE:
        return _RUNNER_CTRL_CACHE[season]

    cols = ["pitcher", "game_pk", "at_bat_number", "pitch_number",
            "on_1b", "on_2b", "on_3b"]
    sc = statcast[[c for c in cols if c in statcast.columns]].copy()
    sc = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"])

    for b in ("on_1b", "on_2b", "on_3b"):
        sc[b] = pd.to_numeric(sc[b], errors="coerce")

    grp_key = ["game_pk", "at_bat_number"]
    # Advance detected when a runner id appears on a HIGHER base than the
    # previous pitch of the same PA.
    prev_1b = sc.groupby(grp_key)["on_1b"].shift()
    same_pa = sc.groupby(grp_key)["pitch_number"].shift().notna()
    moved_2b = same_pa & prev_1b.notna() & (sc["on_2b"] == prev_1b)
    moved_3b = same_pa & prev_1b.notna() & (sc["on_3b"] == prev_1b)
    prev_2b = sc.groupby(grp_key)["on_2b"].shift()
    moved_23 = same_pa & prev_2b.notna() & (sc["on_3b"] == prev_2b)
    sc["_adv"] = (moved_2b | moved_3b | moved_23)

    pa = sc.groupby(["pitcher"] + grp_key).agg(
        had_1b=("on_1b", lambda x: x.notna().any()),
        advs=("_adv", "sum"),
    ).reset_index()
    qual = pa[pa["had_1b"].astype(bool)]
    out = qual.groupby("pitcher").agg(
        opps=("had_1b", "count"),
        advances_allowed=("advs", "sum"),
    ).reset_index()
    out["advance_rate"] = out["advances_allowed"] / out["opps"]

    _RUNNER_CTRL_CACHE[season] = out
    log.info("Runner control — %d pitchers", len(out))
    return out


# ---------------------------------------------------------------------------
# Savant leaderboard CSVs — catcher blocking + pitch tempo (cached)
# ---------------------------------------------------------------------------

def load_catcher_blocking(season: int) -> pd.DataFrame:
    """
    Savant catcher-blocking leaderboard, cached to processed/blocking_{season}.csv.
    blocking_pct = percentile of blocks_above_average among listed catchers.
    Empty frame on fetch failure (tags simply skip).
    """
    cache = _HERE / "processed" / f"blocking_{season}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    try:
        import requests, io
        url = ("https://baseballsavant.mlb.com/leaderboard/catcher-blocking"
               f"?game_type=Regular&season_end={season}&season_start={season}"
               "&split=no&team=&type=Cat&with_team_only=1&csv=true")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df = df.rename(columns={"player_id": "catcher"})
        df["catcher"] = pd.to_numeric(df["catcher"], errors="coerce")
        df["blocks_above_average"] = pd.to_numeric(
            df["blocks_above_average"], errors="coerce")
        df = df[df["catcher"].notna() & df["blocks_above_average"].notna()]
        df["blocking_pct"] = df["blocks_above_average"].rank(pct=True) * 100.0
        out = df[["catcher", "blocks_above_average", "n_pbwp", "blocking_pct"]].copy()
        out.to_csv(cache, index=False)
        log.info("Catcher blocking — %d catchers cached", len(out))
        return out
    except Exception as exc:
        log.warning("Catcher blocking unavailable: %s", exc)
        return pd.DataFrame(columns=["catcher", "blocks_above_average",
                                     "n_pbwp", "blocking_pct"])


def load_pitch_tempo(season: int, min_pitches: int = 300) -> pd.DataFrame:
    """
    Savant pitch-tempo leaderboard (empty-bases median seconds between
    pitches), cached to processed/tempo_{season}.csv.

    The raw CSV repeats column NAMES (total_pitches / median_seconds_empty
    each appear twice) — columns are taken positionally.
    tempo_pct = percentile of empty-bases tempo (higher = slower).
    """
    cache = _HERE / "processed" / f"tempo_{season}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    try:
        import requests, io
        url = ("https://baseballsavant.mlb.com/leaderboard/pitch-tempo"
               f"?type=Pit&min_pitches=100&season_end={season}"
               f"&season_start={season}&split_pitches=no&team="
               "&with_team_only=1&csv=true")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # Positional: 0 entity_id, 4 total_pitches, 6 total_pitches_empty,
        # 7 median_seconds_empty (first occurrence)
        df.columns = [f"c{i}" for i in range(len(df.columns))]
        out = pd.DataFrame({
            "pitcher":        pd.to_numeric(df["c0"], errors="coerce"),
            "pitches_empty":  pd.to_numeric(df["c6"], errors="coerce"),
            "tempo_empty_s":  pd.to_numeric(df["c7"], errors="coerce"),
        })
        out = out[out["pitcher"].notna() & out["tempo_empty_s"].notna()
                  & (out["pitches_empty"] >= min_pitches)]
        out["tempo_pct"] = out["tempo_empty_s"].rank(pct=True) * 100.0
        out.to_csv(cache, index=False)
        log.info("Pitch tempo — %d pitchers cached", len(out))
        return out
    except Exception as exc:
        log.warning("Pitch tempo unavailable: %s", exc)
        return pd.DataFrame(columns=["pitcher", "pitches_empty",
                                     "tempo_empty_s", "tempo_pct"])


# ---------------------------------------------------------------------------
# Batter L/R split stats
# ---------------------------------------------------------------------------

#: wOBA weights for computing from raw events (2023 CBA values)
_WOBA_WEIGHTS = {
    "walk":       0.690,
    "intent_walk":0.690,
    "hit_by_pitch":0.720,
    "single":     0.880,
    "double":     1.242,
    "triple":     1.569,
    "home_run":   2.004,
}
_WOBA_DENOM_EVENTS = {
    "strikeout", "field_out", "force_out", "grounded_into_double_play",
    "fielders_choice", "double_play", "fielders_choice_out",
    "strikeout_double_play", "field_error", "truncated_pa",
}

def _split_stats(group: pd.DataFrame) -> dict:
    """Compute AVG / OBP / SLG / OPS / wOBA for a subset of plate appearance rows."""
    ev = group["events"].dropna()
    if ev.empty:
        return {}

    singles  = (ev == "single").sum()
    doubles  = (ev == "double").sum()
    triples  = (ev == "triple").sum()
    hrs      = (ev == "home_run").sum()
    bb       = ev.isin(["walk", "intent_walk"]).sum()
    hbp      = (ev == "hit_by_pitch").sum()
    sf       = (ev == "sac_fly").sum()
    hits     = singles + doubles + triples + hrs
    ab_events = {"single","double","triple","home_run","strikeout","field_out",
                 "force_out","grounded_into_double_play","fielders_choice",
                 "double_play","fielders_choice_out","strikeout_double_play",
                 "field_error","truncated_pa","fielders_choice_out"}
    ab = ev.isin(ab_events).sum()
    pa = ab + bb + hbp + sf

    avg = hits / ab              if ab  > 0 else float("nan")
    obp = (hits + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) > 0 else float("nan")
    tb  = singles + 2*doubles + 3*triples + 4*hrs
    slg = tb / ab                if ab  > 0 else float("nan")
    ops = (obp + slg)            if (not np.isnan(obp) and not np.isnan(slg)) else float("nan")

    # wOBA
    woba_num = sum(_WOBA_WEIGHTS.get(e, 0) for e in ev)
    woba_den = ab + bb + hbp + sf
    woba = woba_num / woba_den   if woba_den > 0 else float("nan")

    return {
        "pa": int(pa), "ab": int(ab),
        "avg": round(avg, 3), "obp": round(obp, 3),
        "slg": round(slg, 3), "ops": round(ops, 3),
        "woba": round(woba, 3),
    }


def compute_batter_splits(
    statcast: pd.DataFrame,
    hitter_ids: set[int],
    min_pa: int = 20,
) -> dict[int, dict]:
    """
    Compute AVG / OBP / SLG / OPS / wOBA split by pitcher handedness (LHP vs RHP)
    for each batter in hitter_ids, plus the batter's side profile.

    Returns {batter_id: {"vs_lhp": {...}, "vs_rhp": {...},
                          "bats": "L"|"R"|"S", "stand_l_share": float}}
    Only includes splits where the player faced >= min_pa of that handedness.
    bats = "S" (switch) when the batter took ≥ 15% of PAs from each side.
    """
    needed = ["batter", "p_throws", "events", "stand",
              "game_pk", "at_bat_number"]
    sc = statcast[
        statcast["batter"].isin(hitter_ids) & statcast["events"].notna()
    ][[c for c in needed if c in statcast.columns]].copy()

    if sc.empty:
        return {}

    results: dict[int, dict] = {}
    for batter_id, grp in sc.groupby("batter"):
        entry: dict = {}
        for hand, label in [("L", "vs_lhp"), ("R", "vs_rhp")]:
            subset = grp[grp["p_throws"] == hand]
            stats  = _split_stats(subset)
            if stats and stats.get("pa", 0) >= min_pa:
                entry[label] = stats

        # Batter side profile — per-PA stand distribution (switch detection)
        if "stand" in grp.columns:
            pa_rows = grp.drop_duplicates(["game_pk", "at_bat_number"]) \
                if {"game_pk", "at_bat_number"}.issubset(grp.columns) else grp
            stands = pa_rows["stand"].dropna()
            if len(stands) > 0:
                l_share = float((stands == "L").mean())
                entry["stand_l_share"] = round(l_share, 3)
                if 0.15 <= l_share <= 0.85:
                    entry["bats"] = "S"
                else:
                    entry["bats"] = "L" if l_share > 0.85 else "R"

        if entry:
            results[int(batter_id)] = entry

    log.info("Batter splits computed — %d batters", len(results))
    return results


def compute_pitcher_splits(
    statcast: pd.DataFrame,
    pitcher_ids: set[int],
    min_bf: int = 40,
) -> dict[int, dict]:
    """
    Pitcher-side platoon splits: wOBA allowed vs LHH and vs RHH, plus how
    the pitcher was DEPLOYED (share of batters faced who shared his hand —
    high share on a reliever = matchup weapon usage).

    Returns {pitcher_id: {"vs_lhh": {...}, "vs_rhh": {...},
                           "p_throws": "L"|"R",
                           "same_hand_bf_share": float}}
    Splits included only when >= min_bf of that side was faced.
    """
    needed = ["pitcher", "stand", "p_throws", "events",
              "game_pk", "at_bat_number", "batter"]
    sc = statcast[
        statcast["pitcher"].isin(pitcher_ids) & statcast["events"].notna()
    ][[c for c in needed if c in statcast.columns]].copy()

    if sc.empty:
        return {}

    results: dict[int, dict] = {}
    for pitcher_id, grp in sc.groupby("pitcher"):
        entry: dict = {}
        throws = grp["p_throws"].mode()
        throws = str(throws.iloc[0]) if not throws.empty else None
        entry["p_throws"] = throws

        for hand, label in [("L", "vs_lhh"), ("R", "vs_rhh")]:
            subset = grp[grp["stand"] == hand]
            stats  = _split_stats(subset)
            if stats and stats.get("pa", 0) >= min_bf:
                entry[label] = stats

        # Deployment: same-hand share of BF (per-PA, not per-pitch)
        pa_rows = grp.drop_duplicates(["game_pk", "at_bat_number"]) \
            if {"game_pk", "at_bat_number"}.issubset(grp.columns) else grp
        stands = pa_rows["stand"].dropna()
        if throws and len(stands) > 0:
            entry["same_hand_bf_share"] = round(float((stands == throws).mean()), 3)

        if entry.get("vs_lhh") or entry.get("vs_rhh"):
            results[int(pitcher_id)] = entry

    log.info("Pitcher splits computed — %d pitchers", len(results))
    return results


# ---------------------------------------------------------------------------
# Team pitcher arsenal — per pitch type × pitcher trajectories
# ---------------------------------------------------------------------------

_PHYSICS_COLS = ["vx0", "vy0", "vz0", "ax", "ay", "az",
                 "release_pos_x", "release_pos_y", "release_pos_z"]
_HOME_PLATE_Y = 1.4167   # feet from back of home plate

def compute_team_arsenal_trajectories(
    statcast: pd.DataFrame,
    pitcher_ids: set[int],
    player_info: dict[int, dict],
    min_pitches: int = 30,
) -> dict:
    """
    For each pitch type thrown by pitchers in pitcher_ids, compute per-pitcher
    average physics parameters sufficient to reconstruct the 3D flight path.

    Returns:
        {
          "pitch_types": [sorted list of available pitch types],
          "by_pitch_type": {
              "4-Seam Fastball": [
                  {
                    "name": "Justin Verlander",
                    "player_id": 434378,
                    "p_throws": "R",
                    "pitch_count": 450,
                    "usage_pct": 0.45,
                    "velo": 93.5,
                    "whiff_rate": 0.28,
                    "run_value_per100": -1.2,
                    "vx0": ..., "vy0": ..., "vz0": ...,
                    "ax": ..., "ay": ..., "az": ...,
                    "release_x": ..., "release_y": ..., "release_z": ...,
                  }, ...
              ],
              ...
          }
        }

    Physics params allow trajectory reconstruction via:
        pos(t) = release + v0·t + ½·a·t²
    where t runs from 0 to time of flight (~0.4s).
    ax/ay/az already incorporate Magnus force and aerodynamic drag.
    """
    needed = ["pitcher", "pitch_name", "p_throws", "release_speed",
              "description", "delta_pitcher_run_exp"] + _PHYSICS_COLS
    available = [c for c in needed if c in statcast.columns]

    sc = statcast[
        statcast["pitcher"].isin(pitcher_ids) & statcast["pitch_name"].notna()
    ][available].copy()

    if sc.empty:
        return {"pitch_types": [], "by_pitch_type": {}}

    sc["_whiff"] = sc["description"].isin(
        ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
    )
    sc["_rv"] = pd.to_numeric(sc.get("delta_pitcher_run_exp", pd.Series(dtype=float)),
                              errors="coerce")

    # Total pitches per pitcher (for usage %)
    pitcher_totals = sc.groupby("pitcher").size().to_dict()

    by_pitch: dict[str, list] = {}

    for (pitcher_id, pitch_type), grp in sc.groupby(["pitcher", "pitch_name"]):
        pitcher_id = int(pitcher_id)
        if len(grp) < min_pitches:
            continue

        # Drop rows missing physics params
        phys_grp = grp.dropna(subset=[c for c in _PHYSICS_COLS if c in grp.columns])
        if len(phys_grp) < min_pitches // 2:
            continue

        info     = player_info.get(pitcher_id, {})
        name     = info.get("name") or f"ID {pitcher_id}"
        p_throws = grp["p_throws"].mode().iloc[0] if not grp["p_throws"].empty else "R"
        total    = pitcher_totals.get(pitcher_id, 1)
        usage    = len(grp) / total

        velo      = float(phys_grp["release_speed"].mean())
        whiff_rate= float(grp["_whiff"].mean())
        rv_raw    = grp["_rv"].dropna()
        rv_per100 = float(rv_raw.mean() * 100) if not rv_raw.empty else float("nan")

        entry = {
            "name":              name,
            "player_id":         pitcher_id,
            "p_throws":          str(p_throws),
            "pitch_count":       len(grp),
            "usage_pct":         round(usage, 3),
            "velo":              round(velo, 1),
            "whiff_rate":        round(whiff_rate, 3),
            "run_value_per100":  round(rv_per100, 2) if not np.isnan(rv_per100) else None,
            # Mean physics params for trajectory reconstruction
            "vx0":  round(float(phys_grp["vx0"].mean()), 4),
            "vy0":  round(float(phys_grp["vy0"].mean()), 4),
            "vz0":  round(float(phys_grp["vz0"].mean()), 4),
            "ax":   round(float(phys_grp["ax"].mean()),  4),
            "ay":   round(float(phys_grp["ay"].mean()),  4),
            "az":   round(float(phys_grp["az"].mean()),  4),
            "release_x": round(float(phys_grp["release_pos_x"].mean()), 3),
            "release_y": round(float(phys_grp["release_pos_y"].mean()), 3),
            "release_z": round(float(phys_grp["release_pos_z"].mean()), 3),
        }
        by_pitch.setdefault(pitch_type, []).append(entry)

    # Sort pitchers within each type by usage descending
    for pt in by_pitch:
        by_pitch[pt].sort(key=lambda e: -e["usage_pct"])

    pitch_types = sorted(by_pitch.keys())
    log.info("Arsenal trajectories: %d pitch types, %d pitcher×type combos",
             len(pitch_types), sum(len(v) for v in by_pitch.values()))
    return {"pitch_types": pitch_types, "by_pitch_type": by_pitch}
