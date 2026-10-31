"""
team_portrait.py — Integration layer: build a unified team portrait.

Combines all prior modules into one callable that returns a full analytical
snapshot of a team for a given season.

Team-player affiliation from Statcast:
  inning_topbot == 'Top'  → away team is batting  → pitcher is home_team
  inning_topbot == 'Bot'  → home team is batting   → pitcher is away_team

Portrait structure:
  {
    team, season,
    players: {
      hitters: [{player_id, name, metrics, archetype, temporal}],
      starters: [...],
      bullpen:  [...],
    },
    team_metrics: {aggregated PA-weighted metrics},
    park:         {park factor result from park_effects},
    philosophy:   {scores A1–C4, confidences per dimension},
    temporal:     {mode, pa, games},
    spin:         {pitcher_level spin efficiency summary},
    tunneling:    {pitcher_level tunneling summary},
    data_coverage: {fraction of expected metrics present},
  }
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Bump whenever the shape of the portrait dict returned by build_team_portrait
# changes (new top-level keys, renamed fields, etc.) — callers use this to
# detect and discard stale cached/stored portraits built against an older shape.
# 20 does not change the SHAPE — it invalidates content. The schema-19
# portraits were built against normalization pools (data/processed/
# *_history_*.csv, *_prior_*.parquet) cached before the Savant percentile
# repair, so a pitcher's true 93 mph was ranked against a pool still holding
# percentiles and came back as the 93rd percentile. `elite velo` fired on
# 89.8% of pitcher-seasons against a 90th-percentile gate, and `soft tosser`
# and `contact suppressor` on none at all. Bumping the version is what
# discards those portraits everywhere, Supabase Storage included.
# 21 adds the per-player `gate_inputs` block — everything build_*_traits was
# called with, so tags can be recomputed by scripts/rederive_traits.py without
# a rebuild. This is the last rebuild a pure gate change should ever require.
# 22: spray_data zones are all batter-relative (pull_line … oppo_line); the
# old pull/rc_gap/center/lc_gap/oppo keys had pull and oppo reversed. And
# players carry their TEAM stint (pa/bf/g/gs/war) beside *_total season
# fields and season_teams; players under the minimum are in
# players.also_appeared.
# 23: the per-swing whiff rate is Whiff_pct (was SwStr_pct, a different rate's
# name), and an arsenal's out_pitch carries swstr_pct (was whiff_pct). Whiffs
# include foul tips, as Savant counts them.
# 24: players carry tag_states — per tag family, whether the tag is present,
# measured and not distinctive (absent), or unmeasured and why — and team
# identity units carry population_coverage and family_states beside
# trait_density. Catchers are three populations (framing, blocking, pop time)
# instead of one. Framing is location- and count-adjusted; extra base taker
# ranks attempts.
# 25: one name per percentile (metric_contracts). Hitter metrics no longer
# carry "_pct"-stripped aliases or raw duplicates (`ISO` → `ISO_pct`,
# `BB_pct` → `BB_pct_pct`, `K` dropped); season_metrics use the same names
# as the era percentiles (`K_pct_raw` → `K_pct_raw_pct`, `OAA` → `OAA_pct`,
# `SprintSpeed` → `sprint_speed_pct`). Luck and the within-team spin and
# tunnel percentiles are gone. The "Contact" card label is "Avoids Ks".
# 26: pitcher metrics drop the pitcher_archetypes aliases (`K_pct_pct` for
# `K_pct_pitch_pct`, likewise Whiff, BB, GB) and the raw Zone_pct / CSW_pct
# fractions; only declared percentiles remain. Pitcher card and Scout keys
# follow (`K_pct_pitch_pct` …).
# 27: philosophy_metrics read the percentiles by their own names: no
# K_inv_pct / BB_pitch_inv_pct (both inverted an already-inverted rank), no
# TeamK / AvgVelo / GB aliases, and the staff's whiff rate is the staff's.
# Team aggregates average declared percentiles only. Philosophy names
# describe the roster: "Control and Contact Management", "Arsenal Variety and
# Matchup Usage", "Defense and Park Support", "Young, Team-Controlled Roster".
PORTRAIT_SCHEMA_VERSION = 27

from ingest import (
    pull_statcast_range,
    pull_statcast_season,
    pull_fg_batting,
    pull_fg_pitching,
    pull_batting_history,
    pull_pitching_history,
    pull_bwar,
    load_chadwick,
    compute_reliability,
    apply_reliability_weight,
    normalize_percentile,
    era_pool_seasons,
    PROCESSED_DIR,
)
from reliability import (
    is_limited_sample,
    LIMITED_SAMPLE_PA,
    LIMITED_SAMPLE_BF,
)
from database import (
    query_batting, query_pitching,
    season_batting_seeded, query_debut_seasons,
)
from spin_efficiency import build_pitcher_spin_profile
from tunneling import build_tunnel_profile
from park_effects import build_park_profile
from philosophy import build_philosophy_summary
from pitcher_archetypes import build_starter_profile, build_bullpen_profile
from temporal import determine_mode, process_metrics

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Player info cache (name + age from Chadwick)
# ---------------------------------------------------------------------------

_PLAYER_INFO_CACHE: Optional[dict[int, dict]] = None


def _get_player_info_map() -> dict[int, dict]:
    """
    Build a {mlbam_id: {name, age_as_of_today}} lookup from Chadwick.
    Cached in-process after first call.
    """
    global _PLAYER_INFO_CACHE
    if _PLAYER_INFO_CACHE is not None:
        return _PLAYER_INFO_CACHE

    import datetime
    today = datetime.date.today()

    cols = ["key_mlbam", "name_first", "name_last", "birth_year", "birth_month", "birth_day"]
    try:
        cw = load_chadwick()
        # Load birth columns separately (not pulled by default load_chadwick)
        from ingest import REF_DIR
        raw = pd.read_csv(
            REF_DIR / "chadwick_crosswalk.csv",
            usecols=["key_mlbam", "name_first", "name_last",
                     "birth_year", "birth_month", "birth_day"],
            low_memory=False,
        )
        raw["key_mlbam"] = pd.to_numeric(raw["key_mlbam"], errors="coerce")
        raw = raw[raw["key_mlbam"].notna()].drop_duplicates("key_mlbam")

        result: dict[int, dict] = {}
        for _, row in raw.iterrows():
            mlbam = int(row["key_mlbam"])
            first = str(row.get("name_first") or "").strip()
            last  = str(row.get("name_last")  or "").strip()
            name  = f"{first} {last}".strip() or None

            age: Optional[int] = None
            try:
                by = int(row["birth_year"])
                bm = int(row["birth_month"])
                bd = int(row["birth_day"])
                born = datetime.date(by, bm, bd)
                age = today.year - born.year - (
                    (today.month, today.day) < (born.month, born.day)
                )
            except Exception:
                pass

            result[mlbam] = {"name": name, "age": age}

        _PLAYER_INFO_CACHE = result
        log.info("Player info map built — %d entries", len(result))
    except Exception as exc:
        log.warning("Could not build player info map: %s", exc)
        _PLAYER_INFO_CACHE = {}

    return _PLAYER_INFO_CACHE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum PA / BF WITH THIS TEAM for a player to carry tags and count toward
# the team's weighted metrics. Below it the player is still on the roster,
# listed under players.also_appeared.
MIN_HITTER_PA    = 30
MIN_PITCHER_BF   = 20
# Minimum BF to count as a starter candidate
STARTER_MIN_BF   = 50
# Innings threshold to separate starters from bullpen (rough proxy: BF/3)
STARTER_IP_PROXY = 20   # ~60 BF for starters

# Savant column name map → our internal names
BATTING_COL_MAP: dict[str, str] = {
    # Savant column names
    "k_percent":        "K_pct",
    "bb_percent":       "BB_pct",
    "hard_hit_percent": "HardHit_pct",
    "sprint_speed":     "sprint_speed",
    "whiff_percent":    "Whiff_pct",
    "chase_percent":    "OSwing_pct",
    "xwoba":            "xwOBA",
    "brl_percent":      "Barrel_pct",
    "xba":              "xBA",
    "xslg":             "xSLG",
    "xiso":             "ISO",
    "xobp":             "OBP",
    "ba":               "AVG",
    "pa":               "PA",
    "exit_velocity":    "EV_avg",
    # BRef-computed raw rates
    "K_rate":           "K_pct",
    "BB_rate":          "BB_pct",
    # BRef direct columns (identity maps)
    "ISO":              "ISO",
    "OBP":              "OBP",
    "AVG":              "AVG",
    "PA":               "PA",
    "SB":               "SB",
    "CS":               "CS",
    # Supabase DB column names (lowercase snake_case)
    "k_rate":           "K_pct",
    "bb_rate":          "BB_pct",
    "hard_hit_pct":     "HardHit_pct",
    "barrel_pct":       "Barrel_pct",
    "whiff_pct":        "Whiff_pct",
    "chase_pct":        "OSwing_pct",
    "avg":              "AVG",
    "obp":              "OBP",
    "slg":              "SLG",
    "iso":              "ISO",
    "sb":               "SB",
    "cs":               "CS",
    # New statcast-derived columns
    "gb_pct":           "BatGB_pct",
    "fps_pct":          "FPS_pct",
    "pitches_per_pa":   "PitchesPerPA",
    "war":              "WAR_bat",
    "zone_swing_pct":   "ZSwing_pct",
    "contact_pct":      "Contact_pct",
    # Spray chart metrics (computed from Statcast hc_x/hc_y)
    "xb_pct":    "XB_pct",       # (2B + 3B) / BIP — gap contact rate
    "gap_pct":   "GapTend_pct",  # fraction of BIP 15–30° from centre (the alleys)
    "pull_pct":  "Pull_pct",     # fraction of BIP to pull side
    "oppo_pct":  "Oppo_pct",     # fraction of BIP to opposite field
    "hr_fb":     "HR_FB_pct",    # HR / fly balls — per-player power purity
}

PITCHING_COL_MAP: dict[str, str] = {
    # Savant column names
    "k_percent":        "K_pct_pitch",
    "bb_percent":       "BB_pct_pitch",
    "hard_hit_percent": "HardHit_allowed",
    "xwoba":            "xwOBA_allowed",
    "pa":               "BF",
    "whiff_percent":    "Whiff_pct_pitch",
    "brl_percent":      "Barrel_allowed",
    "fb_velocity":      "avg_velo",
    # BRef-computed raw rates
    "K_rate_pitch":     "K_pct_pitch",
    "BB_rate_pitch":    "BB_pct_pitch",
    # BRef direct columns
    "BF":               "BF",
    # Supabase DB column names (lowercase snake_case)
    "k_rate":           "K_pct_pitch",
    "bb_rate":          "BB_pct_pitch",
    "hard_hit_pct":     "HardHit_allowed",
    "barrel_pct":       "Barrel_allowed",
    "whiff_pct":        "Whiff_pct_pitch",
    "xwoba_allowed":    "xwOBA_allowed",
    "fb_velocity":      "avg_velo",
    "gs":               "GS",
    "bf":               "BF",
    # New statcast-derived + BRef columns
    "gb_pct":           "GB_pct_pitch",
    "zone_pct":         "Zone_pct",
    "csw_pct":          "CSW_pct",
    "pitches_per_bf":   "PitchesPerBF",
    "war":              "WAR_pitch",
    "fip":              "FIP",
    "avg_spin_rate":    "AvgSpinRate",
    "arsenal_diversity":"ArsenalDiversity",
}

# Rank direction of every metric the two maps above produce: True means the
# percentile is inverted (100 - rank), so a LOW raw value earns a HIGH pct.
# Declared for every target, not defaulted, because a default is how hard-hit
# and barrel rate allowed came to be ranked backwards: they were left out on
# the belief that Savant had already inverted them, then ingest replaced the
# Savant percentiles with raw rates allowed, and `contact suppressor` went to
# the pitchers allowing the MOST hard contact (Graceffo 2025: 53.6% hard-hit,
# ranked 98.6). validation.validate_metric_directions fails on any target
# missing here.
#
# Hitters rank raw except K%: OSwing_pct_pct is "chases more", the gates read
# it that way. Pitchers rank so that a high pct is good for the pitcher on
# every outcome (K, BB, contact quality, FIP); shape and usage metrics (GB%,
# zone%, velocity, spin) rank raw.
# Both tables are DERIVED from metric_contracts, where each percentile also
# states its cohort and what a high rank means; validation V27 checks every
# stored percentile against those contracts.
from metric_contracts import rank_invert as _rank_invert  # noqa: E402
BATTING_RANK_INVERT: dict[str, bool] = _rank_invert("H.metrics")
PITCHING_RANK_INVERT: dict[str, bool] = _rank_invert("P.metrics")

# Curated percentile keys shown on pitcher cards (starters AND relievers —
# the two carry an identical schema)
from metric_contracts import PITCHER_DISPLAY as _PITCHER_DISPLAY  # noqa: E402
_PITCHER_DISPLAY_KEYS = list(_PITCHER_DISPLAY)



# ---------------------------------------------------------------------------
# Step 1 — Extract team players from Statcast
# ---------------------------------------------------------------------------

def _extract_team_players(
    statcast: pd.DataFrame,
    team: str,
) -> tuple[set[int], set[int]]:
    """
    From pitch-level Statcast data, return (batter_mlbam_ids, pitcher_mlbam_ids)
    for the given team.

    Vectorised: pre-filter top/bottom halves once, then use boolean mask
    on the categorical home/away column rather than two separate string-filters.
    """
    bot = statcast["inning_topbot"] == "Bot"
    top = ~bot

    home_team_mask = statcast["home_team"] == team
    away_team_mask = statcast["away_team"] == team

    # Hitters: home team bats in bottom half; away team bats in top half
    hitter_ids: set[int] = set(
        pd.concat([
            statcast.loc[bot & home_team_mask, "batter"],
            statcast.loc[top & away_team_mask, "batter"],
        ]).dropna().astype(int).unique()
    )
    # Pitchers: home team pitches in top half; away team pitches in bottom half
    pitcher_ids: set[int] = set(
        pd.concat([
            statcast.loc[top & home_team_mask, "pitcher"],
            statcast.loc[bot & away_team_mask, "pitcher"],
        ]).dropna().astype(int).unique()
    )
    log.info("Team %s — found %d hitters, %d pitchers in Statcast slice",
             team, len(hitter_ids), len(pitcher_ids))
    return hitter_ids, pitcher_ids


# ---------------------------------------------------------------------------
# Step 1b — All-team hitter mapping + cross-team spread metrics
# ---------------------------------------------------------------------------

def _all_team_hitter_ids(statcast: pd.DataFrame) -> dict[str, set[int]]:
    """
    Return {team_abbr: set_of_batter_mlbam_ids} for every team.

    Vectorised: two groupby operations instead of 30× boolean-filter over
    the full DataFrame. ~30x faster than the naive per-team loop.
    """
    # Bottom half (home team bats) — group by home_team to get home batters
    bot = statcast[statcast["inning_topbot"] == "Bot"][["home_team", "batter"]].dropna()
    home_batters = (
        bot.astype({"batter": "int64"}, errors="ignore")
        .groupby("home_team")["batter"]
        .apply(set)
    )
    # Top half (away team bats) — group by away_team to get away batters
    top = statcast[statcast["inning_topbot"] == "Top"][["away_team", "batter"]].dropna()
    away_batters = (
        top.astype({"batter": "int64"}, errors="ignore")
        .groupby("away_team")["batter"]
        .apply(set)
    )
    all_teams = set(home_batters.index) | set(away_batters.index)
    return {
        team: home_batters.get(team, set()) | away_batters.get(team, set())
        for team in all_teams
    }


def _league_team_spin_pct(statcast: pd.DataFrame, league_spin: pd.DataFrame) -> pd.Series:
    """
    Percentile (0–100) of each pitching team's spin efficiency across the league.

    Team value = mean of the league pool's spin_efficiency_pct (already ranked
    within pitch type, then across every pitcher in the season), weighted by
    pitches. A pitcher who changed teams carries his total_pitches into each
    club in proportion to the Statcast pitches he threw for it.
    """
    sc = statcast[["pitcher", "inning_topbot", "home_team", "away_team"]].copy()
    sc["pitching_team"] = np.where(
        sc["inning_topbot"] == "Top", sc["home_team"], sc["away_team"],
    )
    counts = (
        sc.dropna(subset=["pitcher", "pitching_team"])
        .groupby(["pitcher", "pitching_team"]).size()
        .rename("n").reset_index()
    )
    counts["share"] = counts["n"] / counts.groupby("pitcher")["n"].transform("sum")

    pool = league_spin[["pitcher", "spin_efficiency_pct", "total_pitches"]].dropna()
    m = counts.merge(pool, on="pitcher", how="inner")
    m["w"] = m["total_pitches"] * m["share"]
    m["wx"] = m["w"] * m["spin_efficiency_pct"]
    g = m.groupby("pitching_team")[["wx", "w"]].sum()
    team_mean = g["wx"] / g["w"].where(g["w"] > 0)
    return normalize_percentile(team_mean)


def cross_team_path(season: int) -> Path:
    """
    The per-season cross-team rows file. `_v2` because the unversioned files
    summed season-wide PA and WAR into every team a player joined and counted
    a traded hitter's batted balls for both clubs; `_v3` because the tenure
    columns now come from MLB service time (service_time.py), not a calendar
    estimate. Reading an older file into a pool would mix two definitions.
    """
    return PROCESSED_DIR / f"cross_team_metrics_v3_{season}.csv"


def cross_team_rows(
    statcast: pd.DataFrame,
    batting_full: pd.DataFrame,
    season: int,
    bat_stints: pd.DataFrame,
    bwar_bat: pd.DataFrame,
    debut_seasons: dict[int, int],
    est_service_time: dict[int, float],
) -> pd.DataFrame:
    """
    One row per team: spread/concentration, tenure and lineup-power metrics,
    from each team's OWN stints — its players' PA and WAR with the team, its
    own batted balls. Season rates (ISO, xwOBA, SLG, barrel%) are the
    player's, weighted by what the team got. Written to cross_team_path().
    """
    all_bat = batting_full.rename(columns=BATTING_COL_MAP)
    _league_iso = pd.to_numeric(all_bat.get("ISO", pd.Series(dtype=float)), errors="coerce")
    league_median_iso = float(_league_iso.median()) if _league_iso.notna().sum() > 0 else 0.15

    bat_team = np.where(statcast["inning_topbot"] == "Top",
                        statcast["away_team"], statcast["home_team"])
    rows = []
    for team in sorted(bat_stints["team"].dropna().unique()):
        ids = set(bat_stints.loc[bat_stints["team"] == team, "batter"].astype(int))
        tb, _ = _team_stints_batting(batting_full[batting_full["key_mlbam"].isin(ids)],
                                     bat_stints, bwar_bat, team)
        tb = tb.rename(columns=BATTING_COL_MAP)

        iso     = pd.to_numeric(tb.get("ISO",       pd.Series(dtype=float)), errors="coerce")
        xwoba   = pd.to_numeric(tb.get("xwOBA",     pd.Series(dtype=float)), errors="coerce")
        slg     = pd.to_numeric(tb.get("SLG",       pd.Series(dtype=float)), errors="coerce")
        barrel  = pd.to_numeric(tb.get("Barrel_pct",pd.Series(dtype=float)), errors="coerce")
        pa      = pd.to_numeric(tb.get("PA",        pd.Series(dtype=float)), errors="coerce").fillna(1)
        war     = pd.to_numeric(tb.get("WAR_bat",   pd.Series(dtype=float)), errors="coerce")
        total_pa = float(pa.sum())

        # WAR concentration: top-3 WAR share; fall back to ISO×PA proxy when WAR unavailable
        if war.notna().sum() >= 3:
            war_clipped = war.fillna(0).clip(lower=0)
            total_war = float(war_clipped.sum())
            war_concentration = float(war_clipped.nlargest(3).sum() / total_war) if total_war > 0 else np.nan
            roster_floor = float((war > 0).mean()) if war.notna().sum() > 0 else np.nan
        else:
            prod = (iso.fillna(0) * pa).clip(lower=0)
            total_prod = float(prod.sum())
            war_concentration = float(prod.nlargest(3).sum() / total_prod) if total_prod > 0 else np.nan
            roster_floor = float((iso.fillna(0) > 0.050).mean()) if iso.notna().sum() > 0 else np.nan

        war_variance = float(war.std()) if war.notna().sum() > 1 else np.nan

        # MLB service time per player (service_time.py) — years.days float
        # (5.143 = 5 years 143 days), from the official lists and the roster
        # ledger. Falls back to calendar-year tenure only without a source.
        # CBA thresholds: <3.0 pre-arb · 3.0-5.999 arb-eligible · ≥6.0 free agent.
        service_vals, pa_vals = [], []
        for pid, player_pa in zip(tb["key_mlbam"].astype(int), pa):
            est = est_service_time.get(pid)
            if est is None or (isinstance(est, float) and np.isnan(est)):
                debut = debut_seasons.get(pid)
                if debut is None:
                    continue
                est = float(int(season) - int(debut))
            service_vals.append(est)
            pa_vals.append(float(player_pa))

        if service_vals:
            svc_arr = np.array(service_vals, dtype=float)
            pa_arr  = np.array(pa_vals,      dtype=float)
            avg_tenure       = float(np.average(svc_arr, weights=pa_arr))   # kept as avg_tenure for METRIC_MAP compat
            new_player_share = float((svc_arr < 3.0).sum()  / len(svc_arr))  # pre-arb (< 3.000 EST)
            veteran_share    = float((svc_arr >= 6.0).sum() / len(svc_arr))  # FA eligible (≥ 6.000 EST)
            arb_share        = float(((svc_arr >= 3.0) & (svc_arr < 6.0)).sum() / len(svc_arr))
        else:
            avg_tenure = new_player_share = veteran_share = arb_share = np.nan

        # HR/FB, XBH and gap share from THIS team's batted balls
        team_sc    = statcast[(bat_team == team) & statcast["batter"].isin(ids)]
        hr_count   = int((team_sc["events"] == "home_run").sum())
        double_count = int((team_sc["events"] == "double").sum())
        triple_count = int((team_sc["events"] == "triple").sum())
        fb_count   = int((team_sc["bb_type"] == "fly_ball").sum())
        bip_count  = int(team_sc["bb_type"].notna().sum())

        # A2 — Gap contact metrics (doubles+triples / BIP)
        xb_count = double_count + triple_count
        team_xb_rate = float(xb_count / bip_count) if bip_count > 0 else np.nan
        # power_source_ratio: HR / total XBH (1.0 = all HRs, 0.0 = all gap hits)
        total_xbh = hr_count + xb_count
        power_source_ratio = float(hr_count / total_xbh) if total_xbh > 0 else np.nan

        team_sc_bip = team_sc[team_sc["hc_x"].notna() & team_sc["hc_y"].notna()]
        if not team_sc_bip.empty:
            from pitch_aggregates import LINE_ANGLE_DEG, PULL_ANGLE_DEG, spray_angle
            angles = spray_angle(team_sc_bip["hc_x"], team_sc_bip["hc_y"])
            team_gap_pct = float(angles.abs().between(
                PULL_ANGLE_DEG, LINE_ANGLE_DEG, inclusive="right").sum() / len(angles))
        else:
            team_gap_pct = np.nan

        # A4 — Lineup Power metrics: PA-weighted SLG and Barrel%
        team_slg    = float((slg.fillna(0) * pa).sum() / total_pa)    if total_pa > 0 and slg.notna().any()    else np.nan
        team_barrel = float((barrel.fillna(0) * pa).sum() / total_pa) if total_pa > 0 and barrel.notna().any() else np.nan
        # Fraction of PA from batters with ISO >= league median (power breadth)
        above_median_pa = float(pa[iso >= league_median_iso].sum()) if iso.notna().any() else np.nan
        power_contributors = float(above_median_pa / total_pa) if total_pa > 0 and not np.isnan(above_median_pa) else np.nan

        rows.append({
            "team":               team,
            "iso_spread":         float(iso.std())                            if iso.notna().sum() > 1   else np.nan,
            "xwoba_spread":       float(xwoba.std())                         if xwoba.notna().sum() > 1 else np.nan,
            "pa_concentration":   float(pa.nlargest(2).sum() / total_pa)     if total_pa > 0            else np.nan,
            "war_concentration":  war_concentration,
            "war_variance":       war_variance,
            "roster_floor":       roster_floor,
            "avg_tenure":         avg_tenure,        # PA-weighted avg EST (years)
            "new_player_share":   new_player_share,  # fraction pre-arb (EST < 3.0)
            "veteran_share":      veteran_share,     # fraction FA eligible (EST >= 6.0)
            "arb_share":          arb_share,         # fraction arb-eligible (3.0 <= EST < 6.0)
            "hr_fb":              float(hr_count / fb_count)                 if fb_count > 0            else np.nan,
            "team_hr":            float(hr_count),
            "team_slg":           team_slg,
            "power_contributors": power_contributors,
            "team_barrel":        team_barrel,
            "team_xb_rate":       team_xb_rate,
            "team_gap_pct":       team_gap_pct,
            "power_source_ratio": power_source_ratio,   # display metadata, not scored
        })

    cross = pd.DataFrame(rows)
    try:
        cross.to_csv(cross_team_path(season), index=False)
    except Exception as exc:
        log.warning("Could not save cross-team rows for %d: %s", season, exc)
    return cross


def ensure_cross_team_rows(season: int, force: bool = False) -> bool:
    """
    Build one season's cross-team rows if absent (or forced). Every season's
    rows must exist BEFORE any portrait ranks against them: they used to be
    written as a side effect of each season's first portrait, so in a full
    rebuild a season ranked against whichever seasons happened to have been
    built before it. refresh.py calls this for every season ahead of the
    portrait stage. Returns True if it wrote the file.
    """
    if cross_team_path(season).exists() and not force:
        return False
    from pitch_aggregates import compute_stints
    statcast = pull_statcast_season(season, top_up=False)
    batting_full, _, bwar_bat, _ = _load_season_players(season)
    bat_stints, _ = compute_stints(statcast, season)
    hitter_ids = [int(p) for p in bat_stints["batter"].unique()]
    from service_time import service_years_days
    cross_team_rows(statcast, batting_full, season, bat_stints, bwar_bat,
                    query_debut_seasons(),
                    service_years_days(hitter_ids, season))
    return True


_CROSS_TEAM_METRIC_MAP = [
    ("iso_spread",        "ISO_spread_pct",        False),
    ("xwoba_spread",      "wRCplus_spread_pct",    False),
    ("pa_concentration",  "PA_concentration_pct",  False),
    ("war_concentration", "WAR_concentration_pct", False),
    ("war_variance",      "WAR_variance_inv_pct",  True),
    ("roster_floor",      "RosterFloor_pct",       False),
    ("avg_tenure",        "AvgTenure_inv_pct",     True),
    ("avg_tenure",        "AvgTenure_pct",         False),
    ("new_player_share",  "PreArbShare_pct",       False),  # pre-arb: EST < 3.0
    ("veteran_share",     "FAShare_pct",           False),  # FA eligible: EST >= 6.0
    ("arb_share",         "ArbShare_pct",          False),  # arb-eligible: 3.0 <= EST < 6.0
    ("hr_fb",             "HR_FB_pct",             False),
    ("team_hr",           "TeamHR_pct",            False),
    ("team_slg",          "TeamSLG_pct",           False),
    ("power_contributors","PowerContributors_pct", False),
    ("team_barrel",       "TeamBarrel_pct",        False),
    ("team_xb_rate",      "TeamXB_pct",            False),  # A2 — gap contact
    ("team_gap_pct",      "TeamGap_pct",           False),
]


def _compute_cross_team_metrics(
    statcast: pd.DataFrame,
    batting_full: pd.DataFrame,
    target_team: str,
    season: int,
    debut_seasons: dict[int, int],
    est_service_time: dict[int, float],
    bat_stints: pd.DataFrame,
    bwar_bat: pd.DataFrame,
) -> tuple[dict[str, float], Optional[pd.Series]]:
    """
    The target team's percentile in each cross-team metric, ranked against the
    team-seasons of its ERA POOL (ingest.era_pool_seasons — finished seasons
    plus its own), the same pool its player percentiles use. It used to rank
    against every cached season, future ones included, so what 2019 was
    compared with depended on what had been built on the machine.

    Returns (metrics, the target team's raw row or None).
    """
    from ingest import era_pool_seasons
    path = cross_team_path(season)
    if path.exists():
        cross = pd.read_csv(path)
    else:
        cross = cross_team_rows(statcast, batting_full, season, bat_stints, bwar_bat,
                                debut_seasons, est_service_time)

    frames = [cross.assign(pool_season=season)]
    for other in era_pool_seasons(season):
        p = cross_team_path(other)
        if other != season and p.exists():
            frames.append(pd.read_csv(p).assign(pool_season=other))
    pool = pd.concat(frames, ignore_index=True)

    cross = cross.set_index("team")
    out: dict[str, float] = {}
    if target_team not in cross.index:
        log.warning("Cross-team: target team %s not in current season cross frame", target_team)
        return out, None
    for col, key, invert in _CROSS_TEAM_METRIC_MAP:
        if col not in pool.columns or pool[col].notna().sum() <= 1:
            continue
        target_val = cross.loc[target_team, col]
        if target_val is None or (isinstance(target_val, float) and np.isnan(target_val)):
            continue
        pool_vals = pool[col].dropna()
        if invert:
            pct = float((pool_vals < target_val).sum() / len(pool_vals) * 100)
        else:
            pct = float((pool_vals <= target_val).sum() / len(pool_vals) * 100)
        out[key] = pct

    log.info("Cross-team metrics — %d populated for %s (pool: %d team-seasons, %d seasons)",
             len(out), target_team, len(pool), pool["pool_season"].nunique())
    return out, cross.loc[target_team]


# ---------------------------------------------------------------------------
# Step 1c — Team stints: what this club got from each player
# ---------------------------------------------------------------------------
# The roster is everyone who appeared for the team. Each player's row keeps
# the SEASON rates (one player, whichever uniform) but the workload, WAR and
# deployment become this team's stint — the season totals
# move to *_total fields. Soto 2022: 436 PA / 3.73 WAR on WSH and 228 / 1.74
# on SD, "664 total" on both, instead of 664 PA and 5.47 WAR credited twice.
#
# Below MIN_HITTER_PA / MIN_PITCHER_BF with this team a player is listed
# under `also_appeared` — on the roster, without tags or weight.

def _col(frame: pd.DataFrame, *names: str) -> Optional[str]:
    return next((c for c in names if c in frame.columns), None)


def _season_teams(stints: pd.DataFrame, id_col: str, work: str) -> dict[int, list[str]]:
    """{player: teams played for, most work first} — for players with 2+."""
    multi = stints[stints.groupby(id_col)["team"].transform("nunique") > 1]
    return {int(pid): grp.sort_values(work, ascending=False)["team"].tolist()
            for pid, grp in multi.groupby(id_col)}


def _team_stints_batting(rows: pd.DataFrame, stints: pd.DataFrame,
                         war_stints: pd.DataFrame, team: str
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(qualified rows, also-appeared rows). PA/WAR become this team's."""
    rows = rows.copy()
    pa_col = _col(rows, "pa", "PA") or "pa"
    war_col = _col(rows, "war", "WAR") or "war"
    ids = rows["key_mlbam"].astype(int)
    mine = stints[stints["team"] == team].set_index("batter")
    rows["pa_total"] = pd.to_numeric(rows.get(pa_col), errors="coerce")
    rows[pa_col] = ids.map(mine["pa"]).fillna(0).astype(int).values
    rows["games_team"] = ids.map(mine["g"]).fillna(0).astype(int).values
    rows["war_total"] = pd.to_numeric(rows.get(war_col), errors="coerce")
    w = war_stints[war_stints["team"] == team].set_index("key_mlbam")["war"]
    rows[war_col] = ids.map(w).values
    teams = _season_teams(stints, "batter", "pa")
    rows["season_teams"] = [teams.get(int(i)) for i in ids]
    keep = rows[pa_col] >= MIN_HITTER_PA
    return rows[keep], rows[~keep]


def _team_stints_pitching(rows: pd.DataFrame, stints: pd.DataFrame,
                          war_stints: pd.DataFrame, team: str
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(qualified rows, also-appeared rows). BF/G/GS/WAR and the deployment
    inputs (same-hand share, leverage) become this team's."""
    rows = rows.copy()
    bf_col = _col(rows, "bf", "BF") or "bf"
    g_col = _col(rows, "g", "G") or "g"
    gs_col = _col(rows, "gs", "GS") or "gs"
    war_col = _col(rows, "war", "WAR") or "war"
    ids = rows["key_mlbam"].astype(int)
    mine = stints[stints["team"] == team].set_index("pitcher")
    for col, stint in ((bf_col, "bf"), (g_col, "g"), (gs_col, "gs")):
        rows[f"{stint}_total"] = pd.to_numeric(rows.get(col), errors="coerce")
        rows[col] = ids.map(mine[stint]).fillna(0).astype(int).values
    rows["same_hand_bf_share_team"] = ids.map(mine["same_hand_bf_share"]).values
    if "mean_abs_dwe" in mine.columns:
        rows["leverage_team"] = ids.map(mine["mean_abs_dwe"]).values
        rows["leverage_n_team"] = ids.map(mine["lev_n_pa"]).values
    rows["war_total"] = pd.to_numeric(rows.get(war_col), errors="coerce")
    w = war_stints[war_stints["team"] == team].set_index("key_mlbam")["war"]
    rows[war_col] = ids.map(w).values
    teams = _season_teams(stints, "pitcher", "bf")
    rows["season_teams"] = [teams.get(int(i)) for i in ids]
    keep = rows[bf_col] >= MIN_PITCHER_BF
    return rows[keep], rows[~keep]


def _also_appeared(bat: pd.DataFrame, pit: pd.DataFrame) -> list[dict]:
    """Players under the minimum with this team: listed, never tagged or weighted."""
    info = _get_player_info_map()
    out = []
    for frame, side, work, total in ((bat, "hitter", "pa", "pa_total"),
                                     (pit, "pitcher", "bf", "bf_total")):
        if frame.empty:
            continue
        wcol = _col(frame, work, work.upper())
        for _, r in frame.iterrows():
            pid = int(r["key_mlbam"])
            n = int(r[wcol]) if wcol and not pd.isna(r[wcol]) else 0
            if n <= 0:
                continue
            war = r.get("war", r.get("WAR"))
            out.append(_jsonable({
                "player_id": pid,
                "name": info.get(pid, {}).get("name") or r.get("name"),
                "side": side,
                work: n,
                total: r.get(total),
                "war": war,
                "season_teams": r.get("season_teams"),
            }))
    return sorted(out, key=lambda d: -(d.get("pa") or d.get("bf") or 0))


# ---------------------------------------------------------------------------
# Step 2 — Load and filter player-level aggregates
# ---------------------------------------------------------------------------

def _load_batting_for_team(
    season: int,
    hitter_ids: set[int],
) -> pd.DataFrame:
    """
    Load batting aggregates and filter to team's hitters.
    Returns DataFrame with key_mlbam and normalised metric columns.
    """
    batting = pull_fg_batting(season)

    # Savant fallback: player_id is already key_mlbam
    if "key_mlbam" not in batting.columns and "player_id" in batting.columns:
        batting = batting.rename(columns={"player_id": "key_mlbam"})

    batting["key_mlbam"] = pd.to_numeric(batting.get("key_mlbam"), errors="coerce")
    team_bat = batting[batting["key_mlbam"].isin(hitter_ids)].copy()
    log.info("Batting — %d / %d hitters matched in aggregate", len(team_bat), len(hitter_ids))
    return team_bat


def _load_pitching_for_team(
    season: int,
    pitcher_ids: set[int],
) -> pd.DataFrame:
    """
    Load pitching aggregates and filter to team's pitchers.
    """
    pitching = pull_fg_pitching(season)

    if "key_mlbam" not in pitching.columns and "player_id" in pitching.columns:
        pitching = pitching.rename(columns={"player_id": "key_mlbam"})

    pitching["key_mlbam"] = pd.to_numeric(pitching.get("key_mlbam"), errors="coerce")
    team_pit = pitching[pitching["key_mlbam"].isin(pitcher_ids)].copy()
    log.info("Pitching — %d / %d pitchers matched in aggregate", len(team_pit), len(pitcher_ids))
    return team_pit


# ---------------------------------------------------------------------------
# Step 3 — Normalize individual player metrics
# ---------------------------------------------------------------------------

def _pool_rank(pool_series: pd.Series, team_series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Rank each value in team_series against the pool distribution using searchsorted.

    This avoids the merge-based approach which creates duplicate rows when
    the pool contains multiple seasons (same key_mlbam appears N times).

    Returns a Series of percentile ranks (0–100) aligned to team_series.index.
    """
    pool_vals = pd.to_numeric(pool_series, errors="coerce").dropna().values
    if len(pool_vals) == 0:
        return pd.Series(np.nan, index=team_series.index)

    sorted_pool = np.sort(pool_vals)
    n = len(sorted_pool)

    def rank_one(v):
        if pd.isna(v):
            return np.nan
        # Fraction of pool values strictly below v, averaged with fraction ≤ v
        # (equivalent to scipy percentileofscore kind='mean')
        lo = float(np.searchsorted(sorted_pool, v, side="left"))
        hi = float(np.searchsorted(sorted_pool, v, side="right"))
        pct = ((lo + hi) / 2.0 / n) * 100.0
        return (100.0 - pct) if invert else pct

    team_numeric = pd.to_numeric(team_series, errors="coerce")
    return team_numeric.map(rank_one)


def _normalize_batting(batting_pool: pd.DataFrame, team_bat: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-rank batting metrics within the historical pool (multi-season),
    then attach those ranks to the current-team subset.

    batting_pool — multi-season DataFrame (all seeded seasons × all 30 teams).
                   "70" means 70th pct of the Statcast era, not just this year.
    team_bat     — current-season team subset (filtered by key_mlbam).

    Returns team_bat with _pct columns appended.  No merge — each player's
    current-season value is ranked against the pool via searchsorted, so the
    row count is always equal to len(team_bat).
    """
    if batting_pool.empty or team_bat.empty:
        return team_bat

    # Rename Savant columns to internal names
    full = batting_pool.rename(columns=BATTING_COL_MAP)
    team = team_bat.rename(columns=BATTING_COL_MAP).copy()

    # Derive OBP–SLG gap per player (patience/contact signal vs. raw power)
    if "OBP" in full.columns and "SLG" in full.columns:
        full["OBP_SLG_gap"] = (
            pd.to_numeric(full["OBP"], errors="coerce")
            - pd.to_numeric(full["SLG"], errors="coerce")
        )
    if "OBP" in team.columns and "SLG" in team.columns:
        team["OBP_SLG_gap"] = (
            pd.to_numeric(team["OBP"], errors="coerce")
            - pd.to_numeric(team["SLG"], errors="coerce")
        )

    # Metrics to rank: all BATTING_COL_MAP targets + derived OBP_SLG_gap
    metrics_to_rank = list(dict.fromkeys(BATTING_COL_MAP.values()))
    if "OBP_SLG_gap" in full.columns:
        metrics_to_rank.append("OBP_SLG_gap")

    for metric in metrics_to_rank:
        if metric not in full.columns:
            continue
        if metric not in team.columns:
            continue
        col_name = f"{metric}_pct"
        team[col_name] = _pool_rank(full[metric], team[metric],
                                    invert=BATTING_RANK_INVERT[metric])

    # K_pct_raw_pct: non-inverted K% percentile for TTO classifier
    # (K_pct_pct is inverted; TTO classifier needs raw strikeout rank)
    if "K_pct" in full.columns and "K_pct" in team.columns:
        team["K_pct_raw_pct"] = _pool_rank(full["K_pct"], team["K_pct"], invert=False)

    return team


def _normalize_pitching(pitching_pool: pd.DataFrame, team_pit: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-rank pitching metrics within the historical pool (multi-season).
    Same pool-rank approach as _normalize_batting — no merge, no duplicate rows.
    """
    if pitching_pool.empty or team_pit.empty:
        return team_pit

    full = pitching_pool.rename(columns=PITCHING_COL_MAP)
    team = team_pit.rename(columns=PITCHING_COL_MAP).copy()

    for metric in dict.fromkeys(PITCHING_COL_MAP.values()):
        if metric not in full.columns or metric not in team.columns:
            continue
        col_name = f"{metric}_pct"
        team[col_name] = _pool_rank(full[metric], team[metric],
                                    invert=PITCHING_RANK_INVERT[metric])

    return team


# ---------------------------------------------------------------------------
# Step 4 — Team-level aggregate metrics (PA-weighted)
# ---------------------------------------------------------------------------

def _aggregate_team_batting(team_bat: pd.DataFrame) -> dict[str, Optional[float]]:
    """
    PA-weighted mean of each percentile metric across the team's hitters.
    Also computes derived spread metrics (wRC+ spread, ISO spread).
    """
    if team_bat.empty:
        return {}

    # Detect PA column
    pa_col = next((c for c in ["pa", "PA", "plate_appearances"] if c in team_bat.columns), None)
    if pa_col is None:
        log.warning("No PA column found in batting aggregate — using equal weights")
        weights = pd.Series(np.ones(len(team_bat)), index=team_bat.index)
    else:
        weights = pd.to_numeric(team_bat[pa_col], errors="coerce").fillna(1.0)

    result: dict[str, Optional[float]] = {}
    total_w = weights.sum()
    if total_w == 0:
        return result

    # Declared percentiles only: raw rates end in "_pct" too (HR_FB_pct is a
    # rate), and averaged in here they were read downstream as percentiles.
    from metric_contracts import lookup as _contract
    pct_cols = [c for c in team_bat.columns
                if c.endswith("_pct") and _contract("H.metrics", c) is not None]
    for col in pct_cols:
        vals = pd.to_numeric(team_bat[col], errors="coerce")
        valid_mask = vals.notna()
        if valid_mask.sum() == 0:
            result[col] = None
        else:
            w = weights[valid_mask]
            v = vals[valid_mask]
            result[col] = float((v * w).sum() / w.sum())

    # Spread metrics — std dev of underlying raw values
    for raw, spread_key in [("wRC_plus", "wRCplus_spread"), ("ISO", "ISO_spread")]:
        col = raw if raw in team_bat.columns else None
        if col and team_bat[col].notna().sum() > 1:
            result[f"{spread_key}_raw"] = float(team_bat[col].std())

    # PA concentration: top-2 PA / total PA
    if pa_col and weights.notna().sum() >= 2:
        top2 = weights.nlargest(2).sum()
        result["PA_concentration_raw"] = float(top2 / total_w)

    return result


def _aggregate_team_pitching(team_pit: pd.DataFrame) -> dict[str, Optional[float]]:
    """
    BF-weighted mean of pitching percentile metrics across the team's pitchers.
    """
    if team_pit.empty:
        return {}

    bf_col = next(
        (c for c in ["BF", "bf", "batters_faced", "n", "TBF"] if c in team_pit.columns),
        None,
    )
    if bf_col is None:
        log.warning("No BF column found in pitching aggregate — using equal weights")
        weights = pd.Series(np.ones(len(team_pit)), index=team_pit.index)
    else:
        weights = pd.to_numeric(team_pit[bf_col], errors="coerce").fillna(1.0)

    result: dict[str, Optional[float]] = {}
    total_w = weights.sum()
    if total_w == 0:
        return result

    from metric_contracts import lookup as _contract
    pct_cols = [c for c in team_pit.columns
                if c.endswith("_pct") and _contract("P.metrics", c) is not None]
    for col in pct_cols:
        vals = pd.to_numeric(team_pit[col], errors="coerce")
        valid_mask = vals.notna()
        if valid_mask.sum() == 0:
            result[col] = None
        else:
            w = weights[valid_mask]
            v = vals[valid_mask]
            result[col] = float((v * w).sum() / w.sum())

    return result


# ---------------------------------------------------------------------------
# Step 5 — Hitter archetype classification
# ---------------------------------------------------------------------------

# Steal attempts are ranked ONCE per season, league-wide, over every hitter
# with this many times on base — zero-attempt players included, so `low steal
# attempts` reaches the players who never run. It was ranked within each
# team's hitters, among those with 2+ attempts: identical seasons got
# different ranks on different clubs (Zobrist 2015: 33.3 on ATH, 20.0 on KC),
# and a player who never ran could not be "low".
STEAL_RANK_MIN_ON_BASE = 50


def _league_attempt_pcts(batting_full: pd.DataFrame) -> pd.Series:
    """
    Attempt-rate percentile per player for the season, indexed by key_mlbam:
    (SB + CS) / times on base (OBP x PA), ranked among hitters with
    STEAL_RANK_MIN_ON_BASE times on base. Ties share their average rank, so
    the zero-attempt group sits together near the bottom.
    """
    full = batting_full.rename(columns=BATTING_COL_MAP)
    full = full.drop_duplicates("key_mlbam")
    num = lambda c: pd.to_numeric(full.get(c), errors="coerce")
    tob = num("OBP").fillna(0) * num("PA").fillna(0)
    rate = (num("SB").fillna(0) + num("CS").fillna(0)) / tob.where(tob > 0)
    elig = tob >= STEAL_RANK_MIN_ON_BASE
    pct = rate[elig].rank(pct=True, method="average") * 100.0
    return pd.Series(pct.values, index=pd.to_numeric(full.loc[elig, "key_mlbam"]).astype(int))


def _classify_hitters(
    team_bat: pd.DataFrame,
    split_map: Optional[dict[int, dict]] = None,
    catcher_map: Optional[dict[int, dict]] = None,
    advancement_map: Optional[dict[int, dict]] = None,
    positions_map: Optional[dict[int, dict]] = None,
    attempt_pcts: Optional[pd.Series] = None,
) -> list[dict]:
    """
    Build hitter profile for each player in team_bat.
    Requires percentile columns (_pct suffix) to be present.
    split_map — {batter_id: L/R splits} feeding the platoon trait family.
    Returns list of hitter profile dicts (traits + spectrum + metrics_pct).
    """
    from ingest import normalize_percentile

    def _nz(v, default=0.0):
        """NaN-safe coercion: NaN is truthy, so `NaN or 0` never falls back."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # Season attempt-rate rank, league-wide (_league_attempt_pcts), the same
    # number on every team the player appeared for.
    def _att_pct(pid) -> float:
        if attempt_pcts is None or pd.isna(pid):
            return np.nan
        return float(attempt_pcts.get(int(pid), np.nan))

    profiles = []
    for idx, row in team_bat.iterrows():
        player_id = row.get("key_mlbam")
        if pd.isna(player_id):
            continue

        # Era percentiles, one key each: the ranked column's name, and only
        # columns metric_contracts declares as percentiles — a "_pct" suffix
        # is not enough, since raw rates carry one too (BB_pct is the walk
        # rate, BB_pct_pct its rank). This used to store every "_pct" column
        # and its stripped twin, so `BB_pct` was the raw rate or the
        # percentile depending on column order, `K` was the raw K rate, and
        # `ISO`, `AVG`, `SB` were percentiles under raw names.
        from metric_contracts import lookup as _contract
        metrics: dict[str, float] = {}
        for col in team_bat.columns:
            if col.endswith("_pct") and _contract("H.metrics", col) is not None:
                val = row.get(col)
                if val is not None and not pd.isna(val):
                    metrics[col] = float(val)

        # Raw sprint speed (ft/sec) for Speed modifier hard threshold.
        # sprint_speed is ft/s from Savant's sprint leaderboard, DB or CSV.
        sprint_raw = row.get("sprint_speed") or row.get("Sprint Speed")
        sprint_raw = float(sprint_raw) if sprint_raw is not None and not pd.isna(sprint_raw) else None

        # Raw K rate (fraction, e.g. 0.259) for Complete Hitter K% ceiling.
        # Stored under "K_pct" after BATTING_COL_MAP rename; fall back to "k_rate".
        k_raw = row.get("K_pct") or row.get("k_rate") or row.get("K_rate")
        if k_raw is not None and not pd.isna(k_raw):
            metrics["k_rate_raw"] = float(k_raw)

        # Attempt rate percentile (cross-player, pre-computed above)
        att_pct = _att_pct(player_id)
        att_pct = None if np.isnan(att_pct) else att_pct

        # SB/CS — available from BRef merge; Savant-only falls back to 0
        obp_val = _nz(row.get("OBP")) or _nz(row.get("obp"))
        # Two workloads. pa_team: what THIS club got — the displayed PA, the
        # team weight, the deployment gates. pa_total: the season, all teams —
        # the sample the season-wide rates (and so every non-deployment tag's
        # reliability gate) rest on.
        pa_team  = int(_nz(row.get("PA")) or _nz(row.get("pa")))
        pa_total = int(_nz(row.get("pa_total")) or pa_team)
        pa_val   = float(pa_total or 1)
        profile: dict = {"player_id": int(player_id)}
        info = _get_player_info_map().get(int(player_id), {})
        profile["name"] = info.get("name")
        # Prefer DB age (BRef season-specific) over Chadwick today-computed age
        db_age = row.get("age") or row.get("Age")
        profile["age"]  = (int(db_age) if db_age is not None and not pd.isna(db_age) else None) or info.get("age")
        profile["pa"]   = pa_team
        profile["pa_total"] = pa_total
        profile["home_position"] = ((positions_map or {}).get(int(player_id))
                                    or {}).get("home_position")
        war_raw = row.get("WAR_bat")
        profile["war"]  = float(war_raw) if war_raw is not None and not pd.isna(war_raw) else None
        war_tot = row.get("war_total")
        profile["war_total"] = float(war_tot) if war_tot is not None and not pd.isna(war_tot) else None
        # Teams of a traded player, most PA first — the card says the tags'
        # evidence is the full season across them.
        teams = row.get("season_teams")
        if isinstance(teams, list) and len(teams) > 1:
            profile["season_teams"] = teams

        # ── Trait tags (archetype-replacement layer) ──────────────────────────
        # Season-pool gate overrides: _szn columns → hitter_traits gate keys
        season_metrics: dict[str, float] = {}
        for col in team_bat.columns:
            scol = str(col)
            if scol.endswith("_szn"):
                val = row.get(col)
                if val is not None and not pd.isna(val):
                    season_metrics[scol[:-4]] = float(val)
        try:
            from hitter_traits import build_hitter_traits
            profile["traits"], profile["spectrum"] = build_hitter_traits(
                metrics=metrics,
                sprint_speed_raw=sprint_raw,
                sb=int(_nz(row.get("SB", 0))),
                cs=int(_nz(row.get("CS", 0))),
                opportunities=int(obp_val * pa_val),
                attempt_rate_pct=att_pct,
                season_metrics=season_metrics,
                splits=(split_map or {}).get(int(player_id)),
                catching=(catcher_map or {}).get(int(player_id)),
                advancement=(advancement_map or {}).get(int(player_id)),
                positions=(positions_map or {}).get(int(player_id)),
                pa=pa_total,
                pa_team=pa_team,
            )
        except Exception as _ht_exc:
            log.warning("hitter traits failed for %s: %s", player_id, _ht_exc)
            profile["traits"], profile["spectrum"] = [], None

        # Everything build_hitter_traits was called with, kept so the tags can
        # be recomputed WITHOUT rebuilding the portrait. Traits are a pure
        # function of these; the portrait used to throw them away, so changing
        # a gate meant an hour-long rebuild that recomputed metrics which had
        # not moved. See scripts/rederive_traits.py.
        profile["gate_inputs"] = _jsonable({
            "metrics":          metrics,
            "season_metrics":   season_metrics,
            "sprint_speed_raw": sprint_raw,
            "sb":               int(_nz(row.get("SB", 0))),
            "cs":               int(_nz(row.get("CS", 0))),
            "opportunities":    int(obp_val * pa_val),
            "attempt_rate_pct": att_pct,
            "splits":           (split_map or {}).get(int(player_id)),
            "catching":         (catcher_map or {}).get(int(player_id)),
            "advancement":      (advancement_map or {}).get(int(player_id)),
            "positions":        (positions_map or {}).get(int(player_id)),
            "pa":               pa_total,
            "pa_team":          pa_team,
        })

        # Eligible-population membership — the denominator each tag is scored
        # against in _trait_density (see TAG_POPULATION).
        profile["populations"] = _hitter_populations(
            sprint_raw=sprint_raw,
            season_metrics=season_metrics,
            catching=(catcher_map or {}).get(int(player_id)),
            advancement=(advancement_map or {}).get(int(player_id)),
            splits=(split_map or {}).get(int(player_id)),
            positions=(positions_map or {}).get(int(player_id)),
            pa=pa_total,
            metrics=metrics,
            attempt_rate_pct=att_pct,
            attempts=int(_nz(row.get("SB", 0))) + int(_nz(row.get("CS", 0))),
        )

        # One player-level marker rather than a per-metric reliability score:
        # a reader needs to know "don't lean on this profile", not to audit
        # thirteen separate denominators. The per-tag gate has already run
        # inside build_hitter_traits.
        profile["limited_sample"] = is_limited_sample(pa_total, LIMITED_SAMPLE_PA)

        # Store curated percentile metrics for the interactive player chart.
        # Each raw_key is the exact key that _classify_hitters puts in `metrics`
        # (the col.removesuffix("_pct") form OR the full col name — both present).
        # The labels and the keys behind them live in metric_contracts, so
        # the dashboard can say what each one ranks.
        from metric_contracts import HITTER_DISPLAY as _HD
        _DISPLAY_KEYS = [(key, label) for label, key in _HD.items()]
        profile["metrics_pct"] = {
            label: float(metrics[raw_key])
            for raw_key, label in _DISPLAY_KEYS
            if raw_key in metrics
            and not (isinstance(metrics[raw_key], float) and np.isnan(metrics[raw_key]))
        }
        profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
# Step 6 — Pitcher archetype classification
# ---------------------------------------------------------------------------

def _classify_pitchers(
    team:      str,
    season:    int,
    team_pit:  pd.DataFrame,
    spin_pitcher: Optional[pd.DataFrame],
    tunneling_pitcher: Optional[pd.DataFrame],
    pitch_mix_df=None,
    league_tunnel: Optional[pd.DataFrame] = None,
    league_spin:   Optional[pd.DataFrame] = None,
    league_discipline: Optional[pd.DataFrame] = None,
    league_mechanics:  Optional[pd.DataFrame] = None,
    pitching_full: Optional[pd.DataFrame] = None,
    pitcher_splits: Optional[dict[int, dict]] = None,
    leverage_df: Optional[pd.DataFrame] = None,
    tempo_map: Optional[dict[int, dict]] = None,
    runner_ctrl_map: Optional[dict[int, dict]] = None,
    league_sequencing: Optional[pd.DataFrame] = None,
    pitch_stints: Optional[pd.DataFrame] = None,
) -> tuple[list[dict], dict]:
    """
    Classify starters individually (build_starter_profile per pitcher) and
    build one collective bullpen profile (build_bullpen_profile for the team).

    Returns (starters_list, bullpen_profile_dict).
    """
    starters: list[dict] = []
    bullpen_arms: list[dict] = []

    if team_pit.empty:
        return starters, {}, bullpen_arms

    # Detect BF column for starter vs. bullpen split
    bf_col = next(
        (c for c in ["BF", "bf", "batters_faced", "n", "TBF"] if c in team_pit.columns),
        None,
    )

    bullpen_metrics_acc: list[dict[str, float]] = []
    bullpen_weights: list[float] = []

    # ── Role pool thresholds — league starter/reliever usage quantiles ──────
    # Computed once so "workhorse" means top-of-THE-LEAGUE workload, not top
    # of this team's five starters — and over player-TEAM stints, because the
    # role tags are deployment: what one club did with the pitcher. Scherzer
    # 2021 was 19 starts for WSH and 11 for LAD; the season's 30 made him a
    # workhorse on both before.
    role_pool: dict[str, float] = {}
    try:
        if pitch_stints is not None and not pitch_stints.empty:
            lp = pitch_stints[pitch_stints["g"].fillna(0) > 0].copy()
            lp["_is_starter"] = (lp["gs"].fillna(0) / lp["g"]) > 0.5
            sp_pool = lp[lp["_is_starter"]]
            rp_pool = lp[~lp["_is_starter"]]
            if len(sp_pool) > 5:
                role_pool["starter_bf_hi"] = float(sp_pool["bf"].quantile(0.85))
                qual = sp_pool[sp_pool["gs"] >= 10]
                if len(qual) > 5:
                    role_pool["starter_bfgs_lo"] = float(
                        (qual["bf"] / qual["gs"]).quantile(0.15))
            if len(rp_pool) > 5:
                role_pool["reliever_g_hi"] = float(rp_pool["g"].quantile(0.85))
                qual = rp_pool[rp_pool["g"] >= 15]
                if len(qual) > 5:
                    role_pool["reliever_bfg_hi"] = float(
                        (qual["bf"] / qual["g"]).quantile(0.85))
                # Leverage-exposure thresholds among reliever stints (min 80 PA)
                if "mean_abs_dwe" in rp_pool.columns:
                    rl = rp_pool[rp_pool["lev_n_pa"].fillna(0) >= 80]
                    if len(rl) > 10:
                        role_pool["reliever_lev_hi"] = float(
                            rl["mean_abs_dwe"].quantile(0.85))
                        role_pool["reliever_lev_lo"] = float(
                            rl["mean_abs_dwe"].quantile(0.15))
    except Exception as _rp_exc:
        log.warning("Role pool thresholds failed: %s", _rp_exc)

    for _, row in team_pit.iterrows():
        player_id = row.get("key_mlbam")
        if pd.isna(player_id):
            continue
        player_id = int(player_id)

        # bf/g/gs: this team's stint (deployment, display, weight).
        # bf_total: the season, all teams (the reliability gate's sample).
        bf = int(row.get(bf_col, 0) or 0) if bf_col else 0
        if bf < MIN_PITCHER_BF:
            continue
        _bft = row.get("bf_total")
        bf_total = int(_bft) if _bft is not None and not pd.isna(_bft) else bf
        _wt = row.get("war_total")
        war_total = float(_wt) if _wt is not None and not pd.isna(_wt) else None
        _teams = row.get("season_teams")
        season_teams = _teams if isinstance(_teams, list) and len(_teams) > 1 else None
        # Leverage and same-hand share with THIS team (deployment).
        _lev = row.get("leverage_team")
        leverage = (float(_lev) if _lev is not None and not pd.isna(_lev)
                    and (row.get("leverage_n_team") or 0) >= 80 else None)
        splits = (pitcher_splits or {}).get(player_id)
        _share = row.get("same_hand_bf_share_team")
        if splits is not None and _share is not None and not pd.isna(_share):
            splits = {**splits, "same_hand_bf_share": float(_share)}

        # Era percentiles under their one name, declared ones only — raw
        # columns end in "_pct" too (Zone_pct, CSW_pct are fractions) and
        # used to land here beside the percentiles (metric_contracts).
        from metric_contracts import lookup as _contract
        metrics: dict[str, float] = {}
        for col in team_pit.columns:
            if col.endswith("_pct") and _contract("P.metrics", col) is not None:
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    metrics[col] = float(val)   # keep full _pct name for archetypes

        # Attach tunneling / plate discipline per pitcher.
        # League pools rank against ALL pitchers in the season — required for
        # the classifier keys (TunnelScore_pct, SwStr_per_velo_pct, …), which
        # live in league-percentile space. The per-team frames only rank
        # within this team's ~30 arms and are kept as display-level fallback.
        def _attach_pct(df, src_col: str, dest_key: str):
            if df is None or "pitcher" not in df.columns:
                return
            row_ = df[df["pitcher"] == player_id]
            if row_.empty:
                return
            val = row_.iloc[0].get(src_col)
            if val is not None and not np.isnan(float(val or np.nan)):
                metrics[dest_key] = float(val)

        _attach_pct(league_tunnel, "tunnel_score_pct",    "TunnelScore_pct")
        _attach_pct(league_discipline, "swstr_per_velo_pct",     "SwStr_per_velo_pct")
        _attach_pct(league_discipline, "first_pitch_strike_pct", "FirstPitchStrike_pct")

        # HardHit_allowed_pct / Barrel_allowed_pct are already "higher =
        # better" — ranked inverted in _normalize_pitching (PITCHING_RANK_INVERT).
        # K_inv_pct: P3 scoring uses low-K as a signal (ground-ball style, not strikeout).
        if "K_pct_pitch_pct" in metrics:
            metrics["K_inv_pct"] = 100.0 - metrics["K_pct_pitch_pct"]

        # Starter vs reliever: GS / G fraction only — more starts than relief appearances.
        # G (total appearances) is stored in Supabase after the schema update.
        gs = int(row.get("GS", 0) or 0)
        g  = row.get("G") or row.get("g")
        g  = int(g) if g is not None and not pd.isna(g) else None
        if g is not None and g > 0:
            is_starter = gs / g > 0.5
        else:
            # G not available (pre-schema-update data) — skip rather than guess
            is_starter = False

        if is_starter:
            profile = build_starter_profile(
                player_id, metrics,
                season=season, pitch_mix_df=pitch_mix_df,
            )
            info = _get_player_info_map().get(player_id, {})
            # Prefer Chadwick name; fall back to DB name column if available
            db_name = row.get("name")
            profile["name"] = info.get("name") or (str(db_name).strip() if db_name else None)
            # Prefer DB age (BRef season-specific) over Chadwick today-computed age
            _db_age = row.get("age")
            profile["age"]  = (int(_db_age) if _db_age is not None and not pd.isna(_db_age) else None) or info.get("age")
            profile["bf"]   = bf
            profile["bf_total"] = bf_total
            profile["g"], profile["gs"] = g, gs
            war_raw = row.get("WAR_pitch")
            profile["war"]  = float(war_raw) if war_raw is not None and not pd.isna(war_raw) else None
            profile["war_total"] = war_total
            if season_teams:
                profile["season_teams"] = season_teams
            try:
                from pitcher_traits import build_pitcher_traits
                profile["traits"] = build_pitcher_traits(
                    player_id, league_mechanics, league_sequencing,
                    league_discipline, league_tunnel, league_spin,
                    arsenal_profile=profile.get("arsenal_profile"),
                    metrics=metrics,
                    bf=bf, g=g, gs=gs, is_starter=True, role_pool=role_pool,
                    leverage=leverage,
                    splits=splits,
                    tempo=(tempo_map or {}).get(player_id),
                    runner_control=(runner_ctrl_map or {}).get(player_id),
                    bf_total=bf_total,
                )
            except Exception as _tr_exc:
                log.warning("traits failed for %s: %s", player_id, _tr_exc)
                profile["traits"] = []
            profile["populations"] = _pitcher_populations(
                player_id, True, league_mechanics,
                tempo=(tempo_map or {}).get(player_id),
                splits=splits,
                runner_control=(runner_ctrl_map or {}).get(player_id),
            )
            profile["limited_sample"] = is_limited_sample(bf_total, LIMITED_SAMPLE_BF)
            # See the hitter equivalent. The five league frames are stored as
            # this pitcher's row from each (pool_rows), not whole: the rows
            # already carry the league percentiles, and storing them makes a
            # replay independent of modules/processed on the building machine.
            profile["gate_inputs"] = _jsonable({
                "metrics":        metrics,
                "bf": bf, "g": g, "gs": gs, "is_starter": True,
                "bf_total":       bf_total,
                "leverage":       leverage,
                "splits":         splits,
                "tempo":          (tempo_map or {}).get(player_id),
                "runner_control": (runner_ctrl_map or {}).get(player_id),
                "pool_rows":      _pool_rows(
                    player_id,
                    league_mechanics=league_mechanics,
                    league_sequencing=league_sequencing,
                    league_discipline=league_discipline,
                    league_tunnel=league_tunnel,
                    league_spin=league_spin),
            })
            # Archetype boxes retired — traits are the identity layer now.
            # modifiers (platoon/command/arsenal/ace) is fully redundant with
            # the trait families and is dropped from the portrait.
            profile.pop("primary", None)
            profile.pop("modifiers", None)
            # Aligned pitcher schema: handedness from statcast splits (the
            # build_starter_profile default was a hard-coded "R"), plus the
            # same curated metrics_pct the reliever cards use.
            profile["handedness"] = (
                ((pitcher_splits or {}).get(player_id) or {}).get("p_throws")
                or profile.get("handedness")
            )
            profile["metrics_pct"] = {
                k: metrics[k] for k in _PITCHER_DISPLAY_KEYS if k in metrics
            }
            starters.append(profile)
        else:
            bullpen_metrics_acc.append(metrics)
            bullpen_weights.append(float(bf))
            info = _get_player_info_map().get(player_id, {})
            war_raw = row.get("WAR_pitch")
            db_name_rel = row.get("name")
            bullpen_arms.append({
                "player_id": player_id,
                "name":      info.get("name") or (str(db_name_rel).strip() if db_name_rel else None),
                "age":       (int(row.get("age")) if row.get("age") is not None and not pd.isna(row.get("age")) else None) or info.get("age"),
                "handedness": ((pitcher_splits or {}).get(player_id) or {}).get("p_throws"),
                "bf":        bf,
                "bf_total":  bf_total,
                "g":         g,
                "gs":        gs,
                "war":       float(war_raw) if war_raw is not None and not pd.isna(war_raw) else None,
                "war_total": war_total,
                **({"season_teams": season_teams} if season_teams else {}),
                "metrics_pct": {
                    k: metrics[k] for k in _PITCHER_DISPLAY_KEYS if k in metrics
                },
            })
            # Arsenal profile for relievers too — pitch_mix covers them, and
            # without it the arsenal family (two-pitch, out-pitch, fastball
            # family) was starters-only.
            _bp_arsenal = None
            try:
                from pitch_mix import build_arsenal_profile
                if pitch_mix_df is not None:
                    _bp_arsenal = build_arsenal_profile(
                        player_id, season, pitch_mix_df,
                        tunnel_pct=metrics.get("TunnelScore_pct"),
                    )
                    if _bp_arsenal is not None:
                        _bp_arsenal = {k: v for k, v in _bp_arsenal.items()
                                       if k != "pitch_rows"}
                        bullpen_arms[-1]["arsenal_profile"] = _bp_arsenal
            except Exception as _ap_exc:
                log.warning("bullpen arsenal failed for %s: %s", player_id, _ap_exc)
            try:
                from pitcher_traits import build_pitcher_traits
                bullpen_arms[-1]["traits"] = build_pitcher_traits(
                    player_id, league_mechanics, league_sequencing,
                    league_discipline, league_tunnel, league_spin,
                    arsenal_profile=_bp_arsenal,
                    metrics=metrics,
                    bf=bf, g=g, gs=gs, is_starter=False, role_pool=role_pool,
                    leverage=leverage,
                    splits=splits,
                    tempo=(tempo_map or {}).get(player_id),
                    runner_control=(runner_ctrl_map or {}).get(player_id),
                    bf_total=bf_total,
                )
            except Exception as _tr_exc:
                log.warning("traits failed for %s: %s", player_id, _tr_exc)
                bullpen_arms[-1]["traits"] = []
            bullpen_arms[-1]["populations"] = _pitcher_populations(
                player_id, False, league_mechanics,
                tempo=(tempo_map or {}).get(player_id),
                splits=splits,
                runner_control=(runner_ctrl_map or {}).get(player_id),
            )
            bullpen_arms[-1]["limited_sample"] = is_limited_sample(bf_total, LIMITED_SAMPLE_BF)
            bullpen_arms[-1]["gate_inputs"] = _jsonable({
                "metrics":        metrics,
                "bf": bf, "g": g, "gs": gs, "is_starter": False,
                "bf_total":       bf_total,
                "leverage":       leverage,
                "splits":         splits,
                "tempo":          (tempo_map or {}).get(player_id),
                "runner_control": (runner_ctrl_map or {}).get(player_id),
                "pool_rows":      _pool_rows(
                    player_id,
                    league_mechanics=league_mechanics,
                    league_sequencing=league_sequencing,
                    league_discipline=league_discipline,
                    league_tunnel=league_tunnel,
                    league_spin=league_spin),
            })

    # Build collective bullpen profile from BF-weighted mean metrics
    bullpen_profile: dict = {}
    if bullpen_metrics_acc:
        total_w = sum(bullpen_weights)
        if total_w > 0:
            all_keys: set[str] = set()
            for m in bullpen_metrics_acc:
                all_keys.update(m.keys())
            bp_metrics: dict[str, float] = {}
            for k in all_keys:
                vals = [
                    (bullpen_metrics_acc[i].get(k, np.nan), bullpen_weights[i])
                    for i in range(len(bullpen_metrics_acc))
                ]
                valid = [(v, w) for v, w in vals if not np.isnan(v)]
                if valid:
                    bp_metrics[k] = float(
                        sum(v * w for v, w in valid) / sum(w for _, w in valid)
                    )
            try:
                bullpen_profile = build_bullpen_profile(team, season, bp_metrics)
            except Exception as exc:
                log.warning("build_bullpen_profile failed: %s", exc)
                bullpen_profile = {"team": team, "season": season, "metrics": bp_metrics}

    # role_pool is four league quantiles shared by every pitcher on the staff,
    # so it rides on the bullpen profile rather than being copied per player.
    bullpen_profile = bullpen_profile or {}
    bullpen_profile["role_pool"] = _jsonable(role_pool)

    return starters, bullpen_profile, bullpen_arms


# ---------------------------------------------------------------------------
# Step 7 — Philosophy metric assembly
# ---------------------------------------------------------------------------

def _build_philosophy_metrics(
    team_batting_agg: dict[str, Optional[float]],
    team_pitching_agg: dict[str, Optional[float]],
    park: Optional[dict],
    spin_team: Optional[dict],
    tunneling_team: Optional[dict],
    cross_team_metrics: Optional[dict] = None,
    oaa_val: Optional[float] = None,
    def_runs_val: Optional[float] = None,
    opener_val: Optional[float] = None,
    platoon_val: Optional[float] = None,
    turnover_val: Optional[float] = None,
) -> dict[str, float]:
    """
    Merge aggregated team metrics into the flat dict expected by philosophy.py.

    All values should already be 0–100 percentile ranks from the aggregation step.
    Derived/inverted keys are computed here.
    """
    metrics: dict[str, Optional[float]] = {}

    # ── Offensive metrics ─────────────────────────────────────────────────
    # Savant pct columns come in as e.g. K_pct_pct (K% percentile rank)
    # philosophy.py expects keys like ISO_pct, BB_pct_pct, etc.
    for key, val in team_batting_agg.items():
        metrics[key] = val

    # Derive inverted keys where needed
    def _inv(key: str) -> Optional[float]:
        v = metrics.get(key)
        return (100.0 - v) if v is not None else None

    # Caller-inverted keys: each inverts a PLAIN percentile (metric_contracts),
    # never one that is already inverted. K_inv_pct and BB_pitch_inv_pct did
    # exactly that and are gone — philosophy reads K_pct_pct / K_pct_raw_pct
    # and BB_pct_pitch_pct directly.
    metrics["Sprint_inv_pct"]        = _inv("sprint_speed_pct")
    metrics["PitchesPerPA_inv_pct"]  = _inv("PitchesPerPA_pct")
    metrics["BB_inv_pct"]            = _inv("BB_pct_pct")

    # ── Pitching metrics ──────────────────────────────────────────────────
    # Pitcher percentiles have their own names (K_pct_pitch_pct, not K_pct_pct),
    # so nothing here can be shadowed by the batting average of the same name
    # — which is how "staff whiff rate" used to read the hitters' whiff rate.
    for key, val in team_pitching_agg.items():
        metrics.setdefault(key, val)

    # ── Park factor ───────────────────────────────────────────────────────
    if park and "pitcher_friendly" in park:
        metrics["ParkPitcherFriendly_pct"] = park["pitcher_friendly"]

    # ── Tunneling quality (B3) ────────────────────────────────────────────
    if tunneling_team and "tunnel_score_team_pct" in tunneling_team:
        metrics["TunnelScore_pct"] = tunneling_team["tunnel_score_team_pct"]

    # ── CSW% — prefer Statcast-derived; fall back to the staff's whiff rate ──
    if "CSW_pct_pct" not in metrics and "Whiff_pct_pitch_pct" in metrics:
        metrics["CSW_pct_pct"] = metrics["Whiff_pct_pitch_pct"]

    # ── Cross-team spread/concentration metrics ───────────────────────────
    if cross_team_metrics:
        for k, v in cross_team_metrics.items():
            metrics.setdefault(k, v)

    # ── OAA + defensive runs ─────────────────────────────────────────────
    if oaa_val is not None:
        metrics["OAA_pct"] = oaa_val
        metrics.setdefault("TeamDefense_pct", oaa_val)  # B2 proxy
    if def_runs_val is not None:
        metrics["DRS_pct"] = def_runs_val               # BRef defensive runs (replaces placeholder)
        metrics.setdefault("TeamDefense_pct", def_runs_val)  # B2 proxy when OAA absent
    if oaa_val is not None and def_runs_val is not None:
        # Both available — use average for B2 team defense
        metrics["TeamDefense_pct"] = (oaa_val + def_runs_val) / 2.0

    if opener_val is not None:
        metrics["OpenerUsage_pct"] = opener_val

    if platoon_val is not None:
        metrics["PlatoonOptimization_pct"] = platoon_val

    if turnover_val is not None:
        metrics["Turnover_pct"] = turnover_val
        # CoreRetention is the inverse of Turnover (100 - turnover percentile)
        metrics.setdefault("CoreRetention_pct", 100.0 - turnover_val)

    # Remove None values; keep only floats
    clean: dict[str, float] = {
        k: float(v) for k, v in metrics.items()
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    }
    return clean


# ---------------------------------------------------------------------------
# Step 8 — Team-level temporal mode
# ---------------------------------------------------------------------------

def _team_temporal(
    statcast: pd.DataFrame,
    team: str,
    season: int,
) -> dict:
    """
    Determine temporal mode for the team using team-level PA and games played.
    """
    # Home games
    home = statcast[statcast["home_team"] == team]
    away = statcast[statcast["away_team"] == team]

    games = set(home["game_pk"].dropna().unique()) | set(away["game_pk"].dropna().unique())
    games_played = len(games)

    # PA proxy: count complete plate appearances (events not null)
    team_pa_home = home[(home["inning_topbot"] == "Bot") & home["events"].notna()]
    team_pa_away = away[(away["inning_topbot"] == "Top") & away["events"].notna()]
    team_pa = len(team_pa_home) + len(team_pa_away)

    mode = determine_mode(team_pa, games_played)
    return {"mode": mode, "pa": team_pa, "games": games_played}


# ---------------------------------------------------------------------------
# Step 8b — Roster turnover (season-over-season player set delta)
# ---------------------------------------------------------------------------

def _compute_turnover(statcast: pd.DataFrame, season: int) -> dict[str, float]:
    """
    PA-weighted batter continuity + BF-weighted pitcher continuity.

    For each team, asks: what fraction of this season's production (PA for
    batters, BF for pitchers) came from players who were also on this team
    last season?

    Keeping your 3 stars who account for 70% of PA = high continuity.
    Keeping 10 bench players who account for 15% of PA = low continuity.

    Returns {team: continuity_rate} where 1.0 = full core retained.
    Cached to data/processed/team_turnover_{season}.csv.
    """
    from ingest import PROCESSED_DIR
    cache = PROCESSED_DIR / f"team_turnover_{season}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        log.info("Loading team_turnover_%d from cache", season)
        return dict(zip(df["team"], df["turnover_rate"]))

    raw_dir = PROCESSED_DIR.parent / "raw"
    prev_parquet = raw_dir / f"statcast_{season - 1}.parquet"
    if not prev_parquet.exists():
        log.info("No prior-season Statcast cache for %d — skipping turnover", season - 1)
        return {}

    # Load only the columns we need from prior season
    prev_sc = pd.read_parquet(
        str(prev_parquet),
        columns=["batter", "pitcher", "home_team", "away_team", "inning_topbot", "events"],
    )

    def _team_col(df: pd.DataFrame) -> pd.Series:
        """Batting team for each row."""
        return np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])

    def _pitching_team_col(df: pd.DataFrame) -> pd.Series:
        """Pitching team for each row."""
        return np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])

    # ── Current season: PA per batter per team, BF per pitcher per team ──────
    # PA = rows where events is not null (end of plate appearance)
    pa_rows = statcast[statcast["events"].notna()].copy()
    pa_rows["bat_team"] = _team_col(pa_rows)
    pa_rows["pit_team"] = _pitching_team_col(pa_rows)

    cur_bat_pa = pa_rows.groupby(["bat_team", "batter"]).size().reset_index(name="pa")
    cur_pit_bf = pa_rows.groupby(["pit_team", "pitcher"]).size().reset_index(name="bf")

    # ── Prior season: which players appeared for each team ───────────────────
    prev_pa_rows = prev_sc[prev_sc["events"].notna()].copy()
    prev_pa_rows["bat_team"] = _team_col(prev_pa_rows)
    prev_pa_rows["pit_team"] = _pitching_team_col(prev_pa_rows)

    prev_bat_teams: dict[str, set] = (
        prev_pa_rows.groupby("bat_team")["batter"]
        .apply(set).to_dict()
    )
    prev_pit_teams: dict[str, set] = (
        prev_pa_rows.groupby("pit_team")["pitcher"]
        .apply(set).to_dict()
    )

    # ── Compute continuity per team ───────────────────────────────────────────
    all_teams = set(cur_bat_pa["bat_team"].unique()) | set(cur_pit_bf["pit_team"].unique())
    rates: dict[str, float] = {}

    for team in all_teams:
        # Batter continuity — PA-weighted
        team_bat = cur_bat_pa[cur_bat_pa["bat_team"] == team]
        total_pa = team_bat["pa"].sum()
        if total_pa > 0:
            prior_batters = prev_bat_teams.get(team, set())
            retained_pa = team_bat[team_bat["batter"].isin(prior_batters)]["pa"].sum()
            bat_continuity = retained_pa / total_pa
        else:
            bat_continuity = np.nan

        # Pitcher continuity — BF-weighted
        team_pit = cur_pit_bf[cur_pit_bf["pit_team"] == team]
        total_bf = team_pit["bf"].sum()
        if total_bf > 0:
            prior_pitchers = prev_pit_teams.get(team, set())
            retained_bf = team_pit[team_pit["pitcher"].isin(prior_pitchers)]["bf"].sum()
            pit_continuity = retained_bf / total_bf
        else:
            pit_continuity = np.nan

        # Combined: average of both sides (equal weight)
        components = [v for v in [bat_continuity, pit_continuity] if not np.isnan(v)]
        if components:
            rates[team] = float(np.mean(components))

    pd.DataFrame({"team": list(rates), "turnover_rate": list(rates.values())}).to_csv(
        cache, index=False
    )
    log.info(
        "Turnover computed (PA+BF weighted) — %d teams (season %d vs %d)",
        len(rates), season, season - 1,
    )
    return rates


# ---------------------------------------------------------------------------
# Step 9 — Data coverage diagnostic
# ---------------------------------------------------------------------------

def _data_coverage(philosophy_metrics: dict[str, float]) -> float:
    """
    Return fraction of philosophy metric slots that have data.
    Denominator = total unique metric keys across all 11 philosophies.
    """
    from philosophy import PHILOSOPHY_DEFS
    all_keys: set[str] = set()
    for defn in PHILOSOPHY_DEFS.values():
        all_keys.update(defn["weights"].keys())

    present = sum(1 for k in all_keys if k in philosophy_metrics)
    return round(present / len(all_keys), 3) if all_keys else 0.0


# ---------------------------------------------------------------------------
# Construction vs Results — projected profile from prior-year stats
# ---------------------------------------------------------------------------

def _compute_projected_profile(
    season:     int,
    hitter_ids: set[int],
    team_bat:   pd.DataFrame,
) -> dict:
    """
    Build a projected team profile from prior-year player performance.

    For each player who appeared for the team in `season`, looks up their
    (season-1) stats from player_batting, classifies them, and weights their
    contribution by actual PA accumulated with this team in `season`.

    Returns a projection dict or {} if prior-season data is unavailable
    (e.g. season=2015 has no 2014 data, or season=2016 with 2015 pool).

    Notes:
    - Rookies / players with no prior-year data are excluded and tracked
      as `excluded_pa_pct`.
    - Trade-deadline acquisitions are naturally down-weighted because they
      have fewer PA with the team — no special handling needed.
    - Prior season 2020 (⚡ shortened) is flagged but included as-is.
    """
    from hitter_archetypes import COMPLETE_THRESHOLDS
    from collections import Counter

    prior = season - 1
    if prior < 2015:
        return {}

    # Cache prior-year batting to avoid a repeated DB round-trip
    # (prior-year data is immutable for historical seasons)
    _prior_cache = PROCESSED_DIR / f"batting_prior_{prior}.parquet"
    try:
        if _prior_cache.exists():
            prior_bat = pd.read_parquet(_prior_cache)
        else:
            prior_bat = query_batting(prior)
            if prior_bat is not None and not prior_bat.empty:
                try:
                    prior_bat.to_parquet(_prior_cache, index=False)
                except Exception:
                    pass
    except Exception as exc:
        log.warning("Projected profile: could not load %d batting: %s", prior, exc)
        return {}

    if prior_bat is None or prior_bat.empty:
        return {}

    # Normalize prior-year pool within the prior season (not current season)
    NORM_SPECS = [
        ("iso",        "ISO_pct",        False),
        ("obp",        "OBP_pct",        False),
        ("bb_rate",    "BB_pct_pct",     False),
        ("k_rate",     "K_pct_pct",      True),    # high = fewer Ks (Contact)
        ("barrel_pct", "Barrel_pct_pct", False),
        ("avg",        "AVG_pct",        False),
        ("contact_pct","Contact_pct_pct",False),
        ("xwoba",      "xwOBA_pct",      False),
    ]
    prior_bat = prior_bat.copy()
    for raw_col, pct_col, invert in NORM_SPECS:
        if raw_col in prior_bat.columns:
            prior_bat[pct_col] = normalize_percentile(
                pd.to_numeric(prior_bat[raw_col], errors="coerce"), invert=invert)

    if "k_rate" in prior_bat.columns:
        prior_bat["K_pct_raw_pct_noninv"] = normalize_percentile(
            pd.to_numeric(prior_bat["k_rate"], errors="coerce"), invert=False)
    if "obp" in prior_bat.columns and "slg" in prior_bat.columns:
        prior_bat["OBP_SLG_gap"] = (pd.to_numeric(prior_bat["obp"], errors="coerce") -
                                     pd.to_numeric(prior_bat["slg"], errors="coerce"))
        prior_bat["OBP_SLG_gap_pct"] = normalize_percentile(prior_bat["OBP_SLG_gap"])

    prior_map = {int(r["key_mlbam"]): r
                 for _, r in prior_bat.iterrows()
                 if pd.notna(r.get("key_mlbam"))}

    # PA weights from current season team_bat
    pa_col = "PA" if "PA" in team_bat.columns else "pa"
    id_col = "key_mlbam"
    pa_weights: dict[int, float] = {}
    for _, row in team_bat.iterrows():
        pid = int(row.get(id_col, 0) or 0)
        pa  = float(row.get(pa_col, 0) or 0)
        if pid and pa > 0:
            pa_weights[pid] = pa

    total_pa    = sum(pa_weights.values())
    covered_pa  = 0.0
    trait_pa: dict[str, float] = Counter()
    proj_metrics: dict[str, list[float]] = {}  # metric → [(value, weight), ...]

    pct_cols_prior = [c for c in prior_bat.columns
                      if c.endswith("_pct") and not c.startswith("_")]

    for pid in hitter_ids:
        pa = pa_weights.get(pid, 0)
        if pa == 0 or pid not in prior_map:
            continue

        row = prior_map[pid]
        metrics: dict[str, float] = {}
        from metric_contracts import lookup as _contract
        for col in pct_cols_prior:
            val = row.get(col)
            if val is not None and not pd.isna(val) and _contract("H.metrics", col) is not None:
                metrics[col] = float(val)

        k_ni = row.get("K_pct_raw_pct_noninv")
        if k_ni is not None and not pd.isna(k_ni):
            metrics["K_pct_raw_pct"] = float(k_ni)
        if "BB_pct_pct" in metrics:
            metrics["BB_inv_pct"] = 100.0 - metrics["BB_pct_pct"]

        sprint  = row.get("sprint_speed")
        k_raw   = row.get("k_rate")
        if k_raw is not None and not pd.isna(k_raw):
            metrics["k_rate_raw"] = float(k_raw)

        from hitter_traits import build_hitter_traits
        traits, _spec = build_hitter_traits(
            metrics=metrics,
            sprint_speed_raw=(float(sprint) if sprint is not None
                              and not pd.isna(sprint) else None),
        )
        for t in traits:
            trait_pa[t["tag"]] += pa
        covered_pa += pa

        # Collect weighted philosophy-relevant metrics for projected scores
        for key, val in metrics.items():
            if key.endswith("_pct") or key in COMPLETE_THRESHOLDS:
                proj_metrics.setdefault(key, []).append((val, pa))

    if total_pa == 0 or covered_pa == 0:
        return {}

    # PA-weighted projected trait density (share of covered PA per tag)
    trait_density = {k: v / covered_pa
                     for k, v in sorted(trait_pa.items(), key=lambda kv: -kv[1])}

    # PA-weighted philosophy metric averages (for projected radar)
    proj_philosophy: dict[str, float] = {}
    for key, pairs in proj_metrics.items():
        total_w = sum(w for _, w in pairs)
        if total_w > 0:
            proj_philosophy[key] = sum(v * w for v, w in pairs) / total_w

    return {
        "trait_density":      trait_density,
        "philosophy_metrics": proj_philosophy,
        "coverage":           covered_pa / total_pa,
        "excluded_pa_pct":    1.0 - (covered_pa / total_pa),
        "prior_season":       prior,
        "prior_season_flag":  (prior == 2020),
    }


def _compute_pitching_projection(
    season:      int,
    pitcher_ids: set[int],
    team_pit:    pd.DataFrame,
) -> dict:
    """
    BF-weighted projection of pitching-side philosophy metrics (B1/B2) from
    each current-roster pitcher's prior-year individual performance.

    Mirrors `_compute_projected_profile` but for pitchers, weighting each
    pitcher's prior-year percentile rank by the batters-faced (BF) they
    actually accumulated with this team in `season`.

    Coverage is intentionally partial — B1 ("Stuff Dominant") is fully
    player-level (velo, K%, whiff%, spin rate) and projects cleanly, but B2
    ("Command + Contact Mgmt") includes `TeamDefense_pct`, a team-aggregate
    fielding metric that has nothing to do with any individual pitcher's
    prior performance — so it's deliberately left out and B2 will project
    at ~80% coverage (3 of its 4 weighted inputs). B3 ("Pitch Design") and
    B4 ("Defensive Infrastructure") are *not* projected at all: they lean on
    team-level constructs — spin-efficiency/tunneling pipelines, arsenal
    diversity, platoon optimization, opener usage, OAA/DRS, park factors —
    that can't be meaningfully reconstructed from a single player's prior
    raw stats without re-running the full team pipeline on a hypothetical
    roster. Projecting them from one player-level proxy (e.g. CSW% alone for
    B3) would be more misleading than just leaving them blank.
    """
    from collections import Counter  # noqa: F401  (parity with hitter fn; unused here)

    prior = season - 1
    if prior < 2015:
        return {}

    _prior_cache = PROCESSED_DIR / f"pitching_prior_{prior}.parquet"
    try:
        if _prior_cache.exists():
            prior_pit = pd.read_parquet(_prior_cache)
        else:
            prior_pit = query_pitching(prior)
            if prior_pit is not None and not prior_pit.empty:
                try:
                    prior_pit.to_parquet(_prior_cache, index=False)
                except Exception:
                    pass
    except Exception as exc:
        log.warning("Pitching projection: could not load %d pitching: %s", prior, exc)
        return {}

    if prior_pit is None or prior_pit.empty:
        return {}

    # Normalize prior-year pool within the prior season — only the metrics
    # that feed B1 (fully) and B2 (partially; minus TeamDefense_pct).
    NORM_SPECS = [
        ("k_rate",        "K_pct_pitch_pct",     False),
        ("fb_velocity",   "avg_velo_pct",        False),
        ("whiff_pct",     "Whiff_pct_pitch_pct", False),
        ("avg_spin_rate", "AvgSpinRate_pct",     False),
        ("bb_rate",       "BB_pct_pitch_pct",    True),   # high = fewer walks
        ("zone_pct",      "Zone_pct_pct",        False),
        ("gb_pct",        "GB_pct_pitch_pct",    False),
    ]
    prior_pit = prior_pit.copy()
    for raw_col, pct_col, invert in NORM_SPECS:
        if raw_col in prior_pit.columns:
            prior_pit[pct_col] = normalize_percentile(
                pd.to_numeric(prior_pit[raw_col], errors="coerce"), invert=invert)

    prior_map = {int(r["key_mlbam"]): r
                 for _, r in prior_pit.iterrows()
                 if pd.notna(r.get("key_mlbam"))}

    # BF weights from current-season team_pit (mirrors PA-weighting for hitters)
    bf_col = next((c for c in ["BF", "bf", "batters_faced", "TBF"] if c in team_pit.columns), None)
    id_col = "key_mlbam"
    bf_weights: dict[int, float] = {}
    for _, row in team_pit.iterrows():
        pid = int(row.get(id_col, 0) or 0)
        bf  = float(row.get(bf_col, 0) or 0) if bf_col else 0.0
        if pid and bf > 0:
            bf_weights[pid] = bf

    total_bf   = sum(bf_weights.values())
    covered_bf = 0.0
    proj_metrics: dict[str, list[tuple[float, float]]] = {}

    PCT_KEYS = ["K_pct_pitch_pct", "avg_velo_pct", "Whiff_pct_pitch_pct", "AvgSpinRate_pct",
                "Zone_pct_pct", "GB_pct_pitch_pct", "BB_pct_pitch_pct"]

    for pid in pitcher_ids:
        bf = bf_weights.get(pid, 0)
        if bf == 0 or pid not in prior_map:
            continue
        row = prior_map[pid]
        covered_bf += bf

        for key in PCT_KEYS:
            val = row.get(key)
            if val is not None and not pd.isna(val):
                proj_metrics.setdefault(key, []).append((float(val), bf))


    if total_bf == 0 or covered_bf == 0:
        return {}

    proj_philosophy: dict[str, float] = {}
    for key, pairs in proj_metrics.items():
        total_w = sum(w for _, w in pairs)
        if total_w > 0:
            proj_philosophy[key] = sum(v * w for v, w in pairs) / total_w

    return {
        "philosophy_metrics": proj_philosophy,
        "coverage":           covered_bf / total_bf,
    }


# ---------------------------------------------------------------------------
# Step 19 — Team identity synthesis
# ---------------------------------------------------------------------------

# Offense shares are computed from hitter trait tags (hitter_traits.py):
#   power   — PA share carrying `power bat` or `plus power`
#   contact — PA share carrying `contact bat`
#   complete— PA share carrying `complete` (the collapse badge; counts toward
#             neither side so power/contact stay pure signals)
_OFFENSE_POWER_TAGS   = {"power bat", "plus power"}
# _OFFENSE_CONTACT_TAGS / _OFFENSE_COMPLETE_TAGS retired with `contact bat`
# and `complete`; the shares they fed were only used by the label layer.

# Rotation out-mechanism now reads outcome-tag densities (pitcher_traits.py):
# `bat-misser` share vs `ground-baller` share, with a margin before either
# side claims the label — mirrors the bullpen's ±15 band.
_MECHANISM_MARGIN = 0.15
# Tunnel-dependent rotation label: BF share carried by `tunneler` starters
_TUNNEL_ROTATION_SHARE = 0.50


# ---------------------------------------------------------------------------
# Tag populations — the eligible denominator for each tag
# ---------------------------------------------------------------------------
# Most tags can fire for anyone, so their denominator is the whole unit. But
# population-gated tags can only fire for a subset (catchers, players with an
# OAA reading, pitchers with a tempo sample...). Dividing those by the whole
# unit's PA/BF understates them by roughly the subset's share — an elite
# framer on a team whose catchers take 12% of the PA reads as 0.12 density no
# matter how good he is, which is why catcher and fielding tags could never
# surface in a team fingerprint. Each tag is divided by the weight of the
# players who could actually have earned it.
POP_ALL = "all"

TAG_POPULATION: dict[str, str] = {
    # catcher craft — only catchers with a reading for THAT skill. One shared
    # `catchers` population let pop time alone qualify a catcher for the
    # framing denominator (2025: 89 rows in it, 77 with a framing reading).
    "elite framer": "catchers_framing", "poor framer": "catchers_framing",
    "good blocker": "catchers_blocking", "bad blocker": "catchers_blocking",
    "quick pop":    "catchers_pop",
    # fielding — needs an OAA / arm-strength reading (DHs and most catchers
    # have neither)
    "elite defender":      "fielders_oaa",
    "plus defender":       "fielders_oaa",
    "defensive liability": "fielders_oaa",
    "cannon arm":          "fielders_arm",
    # speed tags need a sprint-speed reading (nulled in unseeded seasons)
    "elite speed": "speed", "fast": "speed", "station-to-station": "speed",
    # baserunning advancement needs XBT opportunities
    "extra base taker": "xbt",
    # platoon — needs statcast splits; the gap tags additionally need a
    # qualifying sample from both sides
    "switch hitter":      "splits",
    "platoon liability":  "splits_qual",
    "reverse split":      "splits_qual",
    "platoon-vulnerable": "splits_qual",
    "platoon specialist": "splits_qual",
    # handedness is known only where splits are
    "left-handed hitter":  "splits", "right-handed hitter":  "splits",
    "left-handed pitcher": "splits_qual", "right-handed pitcher": "splits_qual",
    # zone-judgment tags need pitch-level chase data (~80% of hitters)
    "patient": "chase", "free swinger": "chase", "aggressive": "chase",
    # gap hitter needs spray data (hc_x/hc_y)
    "gap hitter": "spray",
    # running game — the decision to run is scored against runners with
    # opportunities, the result of running against those who actually went
    "high steal attempts": "steal_opps", "low steal attempts": "steal_opps",
    "high steal rate":     "steal_attempts", "low steal rate": "steal_attempts",
    # small-sample floors already enforced in the trait gates
    "rarely strikes out": "pa200",
    "super-utility":      "positions",
    # pitcher mechanics need ≥150 pitches; tempo needs a tempo sample
    "submarine": "mechanics", "sidearm": "mechanics", "over-the-top": "mechanics",
    "wide release": "mechanics", "deep extension": "mechanics",
    "short extension": "mechanics",
    "quick pitcher": "tempo", "slow pitcher": "tempo",
    "controls runners": "runner_control",
    # role tags are scoped to the pool they were ranked within
    "two-pitch": "starters", "workhorse": "starters",
    "short-outing starter": "starters",
    "heavy usage": "relievers", "mop-up duty": "relievers",
    "high-leverage arm": "relievers", "multi-inning reliever": "relievers",
}


# ---------------------------------------------------------------------------
# Tag kind — who controls the thing the tag describes
# ---------------------------------------------------------------------------
# The second of the v3 axes to actually reach code (population was the first).
# `kind` orders tags by CONTROL, from what nobody chose to what the front
# office decided:
#
#   attribute  — nobody chose it (handedness, arm slot)
#   tool       — physical capacity (sprint speed, velocity, arm strength)
#   behavior   — a choice the player makes (chase, steal attempts, pitch mix)
#   result     — what came out (strikeout rate, barrel rate, framing runs)
#   deployment — what the organisation did with them (leverage, usage, role)
#   noise      — luck
#
# Without this axis every tag competed for the same identity slots, so a team
# fingerprint could lead with "right-handed hitter" — true, unchosen, and
# useless as a description of how they play. Measured across 360 team-seasons,
# handedness held 6.8% of all fingerprint slots and 42% of team-seasons
# carried at least one. Kind does not DELETE those tags; it files them, so an
# attribute is reported as an attribute instead of outranking a skill.
#
# Keyed by tag name. Kind is a property of the VOCABULARY, not of a
# player-season — `free swinger` is a behavior whoever carries it — so this is
# a lookup map rather than a field stamped on every trait instance. That also
# means adding it needs no portrait rebuild.
KIND_ATTRIBUTE  = "attribute"
KIND_TOOL       = "tool"
KIND_BEHAVIOR   = "behavior"
KIND_RESULT     = "result"
KIND_DEPLOYMENT = "deployment"

# Display order, least controllable → most.
TAG_KINDS = [KIND_ATTRIBUTE, KIND_TOOL, KIND_BEHAVIOR, KIND_RESULT,
             KIND_DEPLOYMENT]

TAG_KIND: dict[str, str] = {}


def _assign_kind(kind: str, *tags: str) -> None:
    for t in tags:
        TAG_KIND[t] = kind


_assign_kind(
    KIND_ATTRIBUTE,
    # handedness
    "left-handed hitter", "right-handed hitter", "switch hitter",
    "left-handed pitcher", "right-handed pitcher",
    # delivery geometry — a pitcher's build and slot, not a season's choice
    "over-the-top", "sidearm", "submarine", "wide release",
    "deep extension", "short extension",
)

_assign_kind(
    KIND_TOOL,
    "elite speed", "fast", "station-to-station",
    # Velocity is one axis with three ends. `soft tosser` was missing here
    # only because the broken normalization pool had starved it to zero
    # incidence — it did not exist in the portraits this map was built from.
    "elite velo", "plus velo", "soft tosser",
    "cannon arm", "quick pop",
    "high spin efficiency", "ride four-seam",
)

_assign_kind(
    KIND_BEHAVIOR,
    # swing decisions
    "patient", "free swinger", "aggressive", "zone hunter",
    # what they do with the bat
    "pull-heavy", "oppo bat", "air-ball bat", "ground-ball bat",
    # the decision to run (the RESULT of running is a result tag)
    "high steal attempts", "low steal attempts", "extra base taker",
    # how a pitcher works
    "count-shifter", "first-pitch attacker", "patterned", "steady mix",
    "unpredictable", "quick pitcher", "slow pitcher", "tunneler",
    # what he chooses to throw
    "deep arsenal", "two-pitch", "cutter-primary", "sinker-baller",
    "knuckleballer",
)

_assign_kind(
    KIND_RESULT,
    "power bat", "plus power", "gap hitter", "weak contact",
    "high-K", "rarely strikes out", "walk machine",
    "high steal rate", "low steal rate",
    "elite defender", "plus defender", "defensive liability",
    "elite framer", "poor framer", "good blocker", "bad blocker",
    "bat-misser", "ground-baller", "fly-ball prone", "pitch-to-contact",
    "elite control", "plus control", "walk prone", "whiffs above velo",
    "contact suppressor", "controls runners",
    "platoon liability", "reverse split", "platoon-vulnerable",
)

_assign_kind(
    KIND_DEPLOYMENT,
    "everyday player", "heavy usage", "high-leverage arm", "mop-up duty",
    "multi-inning reliever", "short-outing starter", "swingman", "workhorse",
    "platoon specialist", "super-utility",
)



# Tags that carve up ONE underlying axis. At team level several ends carry
# density at once, so a fingerprint row could spend both its slots stating a
# single fact: "left-handed hitter -22pts · right-handed hitter +26pts" is one
# thing said twice, once as a presence and once as an absence. Measured across
# 360 team-seasons, 8.2% of two-tag rows were such a pair and 76 of 81 were
# handedness. Only one member of a group is shown.
#
# These are GROUPS, not pairs, because batting handedness is three-way. CLE
# 2016 read `right-handed hitter -31` beside `left-handed hitter -1` — both
# negative, which is impossible for two ends of one axis and obvious once you
# remember switch hitters exist. A team deep in switch hitters is down in BOTH
# L and R, and pairing L against R would have reported that as "not
# right-handed" rather than as what it is.
TAG_EXCLUSIVE_GROUPS: list[frozenset[str]] = [
    frozenset({"left-handed hitter", "right-handed hitter", "switch hitter"}),
    frozenset({"left-handed pitcher", "right-handed pitcher"}),
    frozenset({"high steal attempts", "low steal attempts"}),
    frozenset({"high steal rate", "low steal rate"}),
    frozenset({"elite framer", "poor framer"}),
    frozenset({"good blocker", "bad blocker"}),
    frozenset({"air-ball bat", "ground-ball bat"}),
    frozenset({"pull-heavy", "oppo bat"}),
    frozenset({"elite defender", "plus defender", "defensive liability"}),
    frozenset({"elite control", "plus control", "walk prone"}),
    frozenset({"bat-misser", "pitch-to-contact"}),
    frozenset({"quick pitcher", "slow pitcher"}),
    frozenset({"deep arsenal", "two-pitch"}),
    frozenset({"patient", "free swinger"}),
    frozenset({"high-K", "rarely strikes out"}),
    frozenset({"elite speed", "fast", "station-to-station"}),
    frozenset({"elite velo", "plus velo", "soft tosser"}),
    frozenset({"power bat", "plus power", "weak contact"}),
]

# tag → the other members of its group
TAG_EXCLUSIVE: dict[str, frozenset[str]] = {
    tag: (grp - {tag}) for grp in TAG_EXCLUSIVE_GROUPS for tag in grp
}


def _ordinal(n: float) -> str:
    """1st / 2nd / 3rd / 4th … — the fit notes read as prose, and a hard-coded
    'th' produced '3th percentile'. 11-13 take 'th' regardless of last digit."""
    i = int(round(n))
    if 11 <= (i % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}"


# ---------------------------------------------------------------------------
# Build fingerprint — staleness a portrait can detect about itself
# ---------------------------------------------------------------------------
# PORTRAIT_SCHEMA_VERSION answers "was this built against a different SHAPE".
# It cannot answer "was this built against different DATA or different GATES",
# and that is the failure this project kept hitting — four times in one day:
#
#   * the metric-unit repair fixed the database and left three layers of
#     derived cache holding percentiles, so the rebuild read stale inputs
#   * 31 portraits carried the current schema but pre-repair units, and
#     build_league_identity skipped them as cache hits
#   * HOU_2025 was written by a dev server against a half-regenerated pool,
#     twelve minutes before the rebuild reached that season, and was skipped
#     for the same reason
#   * two tag-logic changes (the reliability floor, `zone hunter`) forced full
#     rebuilds even though no metric had moved
#
# Every one of those is the same shape: a cache that does not know its inputs
# changed. The fingerprint hashes what a portrait actually depends on —
# the normalization pools it was ranked against, and every gate constant and
# vocabulary map the tags were cut with. A mismatch means rebuild, decided by
# the data rather than by remembering to bump a number.
#
# PER SEASON. It used to hash every pool file of every season, so one night of
# 2027 games would have marked all 360 portraits stale, 2015 included. A
# portrait now hashes only what it reads: the root files of the seasons in its
# era pool (finished seasons + its own — ingest.era_pool_seasons) and its own
# season's tunnel pool. Finished seasons share a pool that stops moving until
# the next season finishes, so in season only the current one goes stale.
_FINGERPRINT_CACHE: dict[int, str] = {}
_GATE_FINGERPRINT_CACHE: Optional[str] = None

# Root files hashed for every season in the portrait's era pool. Content-hashed
# rather than mtime-checked: regenerating a cache byte-identically should NOT
# invalidate anything. The derived history/prior pools are not hashed: history
# is a pure function of these roots, and both are created lazily mid-build,
# which is what used to move the fingerprint under a running rebuild and force
# a second pass.
# The cross-team rows of every era-pool season are what the team-construction
# percentiles rank against, so they are pool files exactly like the fg roots.
_FINGERPRINT_POOL_FILES = ("fg_batting_{s}.csv", "fg_pitching_{s}.csv",
                           "cross_team_metrics_v3_{s}.csv")

# Pools that live outside PROCESSED_DIR, hashed for the portrait's own season.
# tunneling.py writes tunnel_league_{s}.parquet into modules/processed; left
# out, a change to how tunnel_score is computed could not move the fingerprint
# and every portrait kept a `tunneler` tag cut by the old maths.
#
# Pitch mix, plate discipline, mechanics and sequencing were missing for the
# same reason, and had no rebuild either: a portrait cut from a 93-day-old
# pitch mix still matched the fingerprint.
# refresh.stage_league rebuilds all six when Statcast moves.
_LEAGUE_DIR = Path(__file__).resolve().parent / "processed"
_FINGERPRINT_EXTRA_POOLS = tuple((_LEAGUE_DIR, pat) for pat in (
    "tunnel_league_{s}.parquet",
    "pitch_mix_{s}.parquet",
    "spin_league_{s}.parquet",   # `high spin efficiency`, team spin percentile
    "plate_discipline_league_{s}.parquet",
    "mechanics_league_{s}.parquet",
    "sequencing_league_{s}.parquet",
))


def _jsonable(obj):
    """
    Coerce numpy/pandas scalars and NaN to plain JSON types.

    The gate inputs come straight out of pandas rows, so they arrive as
    np.float64 / np.int64 / NaT and would either blow up json.dump or
    round-trip as the string "nan" — which reads as a value rather than as
    missing, and would silently change what a gate sees on re-derive.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def _pool_rows(pitcher_id: int, **frames: Optional[pd.DataFrame]) -> dict:
    """
    This pitcher's own row from each per-season league frame, keyed by the
    build_pitcher_traits argument name, so the stored gate inputs are enough
    to replay his tags on their own.

    The frames are already ranked league-wide, so one row carries the
    percentile the gate compares against — the rest of the frame is only how
    that percentile was reached. Without these rows a replay had to reload the
    frames from modules/processed, which exists on the machine that built the
    portrait and nowhere else.
    """
    from pitcher_traits import _pool_row
    out = {}
    for key, frame in frames.items():
        row = _pool_row(frame, pitcher_id)
        out[key] = _jsonable(row.to_dict()) if row is not None else None
    return out


def _stable_repr(v) -> str:
    """Order-independent repr, so dict/set iteration order cannot churn the hash."""
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_stable_repr(v[k])}" for k in sorted(map(str, v))) + "}"
    if isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(_stable_repr(x) for x in v)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_stable_repr(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def _pool_digest(season: int) -> str:
    """Content hash of the pool files one season's portraits read."""
    from ingest import era_pool_seasons
    paths = [PROCESSED_DIR / pat.format(s=s)
             for s in era_pool_seasons(season) for pat in _FINGERPRINT_POOL_FILES]
    paths += [directory / pat.format(s=season)
              for directory, pat in _FINGERPRINT_EXTRA_POOLS]
    # Service time (roster control): the frozen per-player totals it rests on
    # — identical on the build machine and the nightly runner, which has
    # neither the MLS lists nor the full roster ledger.
    from service_time import fingerprint_inputs
    paths += fingerprint_inputs(season)
    parts = []
    for path in paths:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            # Absent is part of the state: a portrait built without a file
            # must not match one built with it.
            parts.append(f"{path.name}:missing")
            continue
        parts.append(f"{path.name}:{h.hexdigest()[:16]}")
    return "|".join(parts)


def _gate_digest() -> str:
    """Every tag gate constant and vocabulary map, from the modules that cut tags."""
    import hitter_traits
    import pitcher_traits
    import pitch_aggregates
    import reliability

    # pitch_aggregates was missing, and it is not a bystander: it owns the
    # strike-zone geometry behind zone%, chase%, CSW% and framing, all of
    # which feed tags. A change there moved tag output while leaving the
    # fingerprint identical — a portrait built before it and one built after
    # were indistinguishable, which is the exact failure this digest exists to
    # catch.
    items = []
    for mod in (hitter_traits, pitcher_traits, pitch_aggregates, reliability):
        for name in sorted(dir(mod)):
            # Module-level CONSTANTS are the gates: thresholds, floors, maps.
            if not name.isupper() or name.startswith("_"):
                continue
            val = getattr(mod, name)
            if isinstance(val, (int, float, str, bool, dict, list, tuple, set, frozenset)):
                items.append(f"{mod.__name__}.{name}={_stable_repr(val)}")
    # Vocabulary axes and rank directions defined in this module. A direction
    # flip changes every tag cut from that metric's percentile.
    for name in ("TAG_POPULATION", "TAG_KIND", "TAG_KINDS", "TAG_EXCLUSIVE_GROUPS",
                 "BATTING_RANK_INVERT", "PITCHING_RANK_INVERT",
                 "MIN_HITTER_PA", "MIN_PITCHER_BF"):
        items.append(f"team_portrait.{name}={_stable_repr(globals()[name])}")
    return "|".join(items)


# The pool files were never the whole input. Two rebuilds in October 2026
# reported every portrait current when none was: the seed had rewritten the
# season totals portraits read from the database, and a code change (paging
# past the API's row cap) had changed what they read — neither is a file here.
# So the fingerprint also hashes the database rows and the code.
#
# Code is split the way rederive_traits needs it: the TAG modules turn stored
# gate inputs into tags, so a change there can be replayed; every other module
# computes the inputs themselves, so a change there means rebuild.
_MODULES_DIR = Path(__file__).resolve().parent
_GATE_CODE_MODULES = ("hitter_traits", "pitcher_traits", "reliability")
_DB_DIGEST_CACHE: dict[int, str] = {}
_DATA_FINGERPRINT_CACHE: dict[int, str] = {}


def _build_code_files() -> list[Path]:
    files = [p for p in sorted(_MODULES_DIR.glob("*.py"))
             if p.stem not in _GATE_CODE_MODULES and p.stem != "validation"]
    return files + [_MODULES_DIR.parent / "scripts" / "build_league_identity.py"]


def _code_digest(paths: list[Path]) -> str:
    """
    Hash of these files' code: comments, blank lines and docstrings dropped,
    so rewording a comment does not mark 360 portraits stale. Tokens rather
    than ast.dump, whose output changes between Python versions.
    """
    import io
    import tokenize
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING)
    stmt_start = (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
    parts = []
    for path in paths:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
                if t.type not in skip]
        kept, prev = [], tokenize.NEWLINE
        for i, t in enumerate(toks):
            nxt = toks[i + 1].type if i + 1 < len(toks) else tokenize.ENDMARKER
            # A string that is a whole statement is a docstring (or inert).
            if t.type == tokenize.STRING and prev in stmt_start and nxt == tokenize.NEWLINE:
                continue
            kept.append(f"{t.type}:{t.string}")
            prev = t.type
        parts.append(f"{path.name}:{hashlib.sha256(chr(31).join(kept).encode()).hexdigest()[:16]}")
    return "|".join(parts)


def _db_digest(season: int) -> str:
    """
    Content hash of the database rows a season's portraits read: its own
    season totals and the prior season's (the projection). An unseeded season
    reads the fg_* CSV instead, which _pool_digest already covers.
    """
    from database import query_batting, query_pitching
    parts = []
    for s in (season - 1, season):
        if s not in _DB_DIGEST_CACHE:
            sub = []
            for name, query in (("batting", query_batting), ("pitching", query_pitching)):
                df = query(s)
                if df.empty:
                    sub.append(f"{name}:none")
                    continue
                df = df.sort_values("key_mlbam")[sorted(df.columns)]
                sub.append(f"{name}:{len(df)}:"
                           f"{hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]}")
            _DB_DIGEST_CACHE[s] = ",".join(sub)
        parts.append(f"db{s}:{_DB_DIGEST_CACHE[s]}")
    return "|".join(parts)


def data_fingerprint(season: int, refresh: bool = False) -> str:
    """
    Hash of what a season's portrait FEATURES were computed from: its pool
    files, its database rows, and the code that computes them — everything
    but the tag logic. Stored on each portrait, so rederive_traits can refuse
    to replay tags over features that have since changed.
    """
    season = int(season)
    if refresh or season not in _DATA_FINGERPRINT_CACHE:
        blob = "||".join((_pool_digest(season), _db_digest(season),
                          _code_digest(_build_code_files())))
        _DATA_FINGERPRINT_CACHE[season] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return _DATA_FINGERPRINT_CACHE[season]


def build_fingerprint(season: int, refresh: bool = False) -> str:
    """
    Short hash of everything one season's portraits depend on beyond their own
    team-season: data_fingerprint (pools, database rows, feature code) plus
    the gate constants, vocabulary and tag code the tags were cut with.
    Memoised per process and season — inputs do not change mid-run; pass
    refresh=True after changing them.
    """
    season = int(season)
    if refresh or season not in _FINGERPRINT_CACHE:
        gate_code = _code_digest([_MODULES_DIR / f"{m}.py" for m in _GATE_CODE_MODULES])
        blob = "||".join((data_fingerprint(season, refresh), _gate_digest(), gate_code))
        _FINGERPRINT_CACHE[season] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return _FINGERPRINT_CACHE[season]


def clear_fingerprint_cache() -> None:
    """Forget every memoised fingerprint (after pools or the database changed)."""
    _FINGERPRINT_CACHE.clear()
    _DATA_FINGERPRINT_CACHE.clear()
    _DB_DIGEST_CACHE.clear()


def gate_fingerprint(refresh: bool = False) -> str:
    """
    The code half of build_fingerprint: gate constants and vocabulary only.

    build_fingerprint also hashes the pool files, and those exist only where
    portraits are BUILT. This half is computable anywhere the code is, so the
    dashboard uses it to refuse a published portrait cut by different tag
    logic than the code it is running — new gates deployed over old portraits.
    """
    global _GATE_FINGERPRINT_CACHE
    if _GATE_FINGERPRINT_CACHE is None or refresh:
        _GATE_FINGERPRINT_CACHE = hashlib.sha256(
            _gate_digest().encode()).hexdigest()[:16]
    return _GATE_FINGERPRINT_CACHE


def portrait_is_current(portrait: Optional[dict]) -> bool:
    """
    Is this stored portrait current, for the machine that builds them?

    Schema must match, and so must its season's build fingerprint (pools,
    database rows, code, gates). A portrait with no fingerprint is not
    current: it cannot be checked, and every schema-22 build carries one.

    The dashboard never calls this: it cannot see the pools. It filters
    published portraits on schema_version and gate_fingerprint instead
    (dashboard/store.py).
    """
    if not portrait:
        return False
    if portrait.get("schema_version") != PORTRAIT_SCHEMA_VERSION:
        return False
    stamped = portrait.get("build_fingerprint")
    if stamped is None:
        return False
    season = portrait.get("season")
    return season is not None and stamped == build_fingerprint(int(season))


def kind_of(tag: str) -> Optional[str]:
    """Kind for a tag, or None if the tag is outside the vocabulary.

    The `<pitch> out pitch` tags are generated dynamically per pitch type, so
    they are matched by suffix rather than enumerated — choosing which pitch
    to put hitters away with is a behavior.
    """
    if tag in TAG_KIND:
        return TAG_KIND[tag]
    if tag.endswith("out pitch"):
        return KIND_BEHAVIOR
    return None


def _hitter_populations(
    sprint_raw:   Optional[float],
    season_metrics: dict,
    catching:     Optional[dict],
    advancement:  Optional[dict],
    splits:       Optional[dict],
    positions:    Optional[dict],
    pa:           int,
    metrics:      Optional[dict] = None,
    attempt_rate_pct: Optional[float] = None,
    attempts:     int = 0,
) -> list[str]:
    """Which tag populations this hitter belongs to — mirrors the gates in
    hitter_traits.build_hitter_traits."""
    from hitter_traits import (PLATOON_MIN_PA, RARELY_K_MIN_PA, XBT_MIN_OPPS,
                               STEAL_RATE_MIN_ATTEMPTS)

    from metric_contracts import lookup as _contract
    pops = [POP_ALL]
    sm = season_metrics or {}

    # Both lookups refuse a name metric_contracts does not declare. A stale
    # name does not fail on its own — it reads as "no value" and quietly
    # empties a population; schema 25's rename did exactly that to chase and
    # spray until this check existed.
    def _has(key: str) -> bool:
        if _contract("H.season", key) is None:
            raise KeyError(f"undeclared season percentile {key!r}")
        v = sm.get(key)
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    c = catching or {}
    for key, pop in (("framing_pct", "catchers_framing"),
                     ("blocking_pct", "catchers_blocking"),
                     ("pop_pct", "catchers_pop")):
        if c.get(key) is not None and not (isinstance(c[key], float) and np.isnan(c[key])):
            pops.append(pop)
    if _has("OAA_pct"):
        pops.append("fielders_oaa")
    if _has("ArmStrength_pct"):
        pops.append("fielders_arm")
    if sprint_raw is not None:
        pops.append("speed")
    if (advancement or {}).get("opps", 0) >= XBT_MIN_OPPS:
        pops.append("xbt")
    if positions:
        pops.append("positions")
    if pa >= RARELY_K_MIN_PA:
        pops.append("pa200")

    m = metrics or {}

    def _hasm(key: str) -> bool:
        if _contract("H.metrics", key) is None:
            raise KeyError(f"undeclared era percentile {key!r}")
        v = m.get(key)
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    if _hasm("OSwing_pct_pct"):
        pops.append("chase")
    if _hasm("XB_pct_pct") and _hasm("GapTend_pct_pct"):
        pops.append("spray")
    if attempt_rate_pct is not None and not (
            isinstance(attempt_rate_pct, float) and np.isnan(attempt_rate_pct)):
        pops.append("steal_opps")
    if attempts >= STEAL_RATE_MIN_ATTEMPTS:
        pops.append("steal_attempts")

    sp = splits or {}
    if sp.get("bats"):
        pops.append("splits")
        bats = sp.get("bats")
        if bats == "S":
            pops.append("splits_qual")   # switch hitters are tagged, not gapped
        elif bats in ("L", "R"):
            same = sp.get("vs_lhp" if bats == "L" else "vs_rhp") or {}
            opp  = sp.get("vs_rhp" if bats == "L" else "vs_lhp") or {}
            if (same.get("pa", 0) >= PLATOON_MIN_PA
                    and opp.get("pa", 0) >= PLATOON_MIN_PA):
                pops.append("splits_qual")
    return pops


def _pitcher_populations(
    player_id:      int,
    is_starter:     bool,
    league_mechanics,
    tempo:          Optional[dict],
    splits:         Optional[dict],
    runner_control: Optional[dict],
) -> list[str]:
    """Which tag populations this pitcher belongs to — mirrors the gates in
    pitcher_traits.build_pitcher_traits."""
    pops = [POP_ALL, "starters" if is_starter else "relievers"]
    try:
        if (league_mechanics is not None and len(league_mechanics)
                and int(player_id) in set(league_mechanics["pitcher"])):
            pops.append("mechanics")
    except Exception:
        pass
    if tempo:
        pops.append("tempo")
    if splits:
        pops.append("splits_qual")
    if runner_control:
        pops.append("runner_control")
    return pops


# ---------------------------------------------------------------------------
# Tag states — what a missing tag means
# ---------------------------------------------------------------------------
# Tags are positive assertions, so an untagged player used to read as an
# ordinary one whatever the reason: measured and unremarkable, too few pitches
# to say, no reading at all, or a skill he has no occasion to show. Each
# player now carries one state per tag FAMILY (the tags that read one
# measurement — elite/poor framer is framing):
#
#   present              a tag in the family fired
#   absent               every applicable tag in the family could have fired
#                        and none did: measured, not distinctive
#   insufficient_sample  there is a reading, but below a gate's sample floor
#                        or a pool's qualifying minimum
#   missing_data         no reading (includes Savant leaderboards that list
#                        qualified players only — sprint speed, OAA, arm,
#                        blocking, pop time, tempo — whose cutoffs we can't see)
#   not_applicable       no tag in the family can apply to him (catcher craft
#                        for a non-catcher, starter workload for a reliever,
#                        steal success for a player who never ran)
#
# A family is `absent` only when ALL its applicable tags were evaluable:
# a 100-PA hitter can be checked for `high-K` but not `rarely strikes out`, so
# his strikeout family is insufficient_sample, not "measured and typical".
# States are computed from the stored traits and gate_inputs only, so
# rederive_traits recomputes them exactly as a build would.
TAG_STATES = ("present", "absent", "insufficient_sample", "missing_data",
              "not_applicable")

HITTER_TAG_FAMILIES: dict[str, tuple[str, ...]] = {
    "handedness":     ("left-handed hitter", "right-handed hitter", "switch hitter"),
    "power":          ("power bat", "plus power", "weak contact"),
    "gap power":      ("gap hitter",),
    "strikeouts":     ("high-K", "rarely strikes out"),
    "walks":          ("walk machine",),
    "chase":          ("patient", "free swinger"),
    "zone hunting":   ("zone hunter",),
    "first-pitch swings": ("aggressive",),
    "direction":      ("pull-heavy", "oppo bat"),
    "trajectory":     ("air-ball bat", "ground-ball bat"),
    "speed":          ("elite speed", "fast", "station-to-station"),
    "steal attempts": ("high steal attempts", "low steal attempts"),
    "steal success":  ("high steal rate", "low steal rate"),
    "extra bases":    ("extra base taker",),
    "fielding":       ("elite defender", "plus defender", "defensive liability"),
    "arm":            ("cannon arm",),
    "framing":        ("elite framer", "poor framer"),
    "blocking":       ("good blocker", "bad blocker"),
    "pop time":       ("quick pop",),
    "platoon":        ("platoon liability", "reverse split"),
    "playing time":   ("everyday player",),
    "versatility":    ("super-utility",),
}

PITCHER_TAG_FAMILIES: dict[str, tuple[str, ...]] = {
    "handedness":      ("left-handed pitcher", "right-handed pitcher"),
    "arm slot":        ("over-the-top", "sidearm", "submarine"),
    "release width":   ("wide release",),
    "extension":       ("deep extension", "short extension"),
    "velocity":        ("elite velo", "plus velo", "soft tosser"),
    "fastball":        ("ride four-seam", "sinker-baller", "cutter-primary"),
    "arsenal depth":   ("deep arsenal", "two-pitch"),
    "knuckleball":     ("knuckleballer",),
    "out pitch":       (),          # "<pitch> out pitch", matched by suffix
    "spin efficiency": ("high spin efficiency",),
    "tunneling":       ("tunneler",),
    "whiffs vs velo":  ("whiffs above velo",),
    "pitch sequencing": ("unpredictable", "patterned"),
    "count shifting":  ("count-shifter", "steady mix"),
    "first-pitch strikes": ("first-pitch attacker",),
    "tempo":           ("quick pitcher", "slow pitcher"),
    "strikeouts":      ("bat-misser", "pitch-to-contact"),
    "ground balls":    ("ground-baller", "fly-ball prone"),
    "contact quality": ("contact suppressor",),
    "control":         ("elite control", "plus control", "walk prone"),
    "runner control":  ("controls runners",),
    "platoon":         ("platoon-vulnerable", "reverse split", "platoon specialist"),
    "starter workload": ("workhorse", "short-outing starter"),
    "reliever usage":  ("heavy usage", "multi-inning reliever"),
    "leverage":        ("high-leverage arm", "mop-up duty"),
    "swing role":      ("swingman",),
}

# Pitcher tags read from a league pool the pitcher must qualify for
# (gate_inputs.pool_rows); a missing row means too few pitches.
_PITCHER_TAG_POOL: dict[str, str] = {
    **dict.fromkeys(["unpredictable", "patterned", "count-shifter", "steady mix"],
                    "league_sequencing"),
    "first-pitch attacker": "league_discipline",
    "whiffs above velo":    "league_discipline",
    "tunneler":             "league_tunnel",
    "high spin efficiency": "league_spin",
}
_ARSENAL_TAGS = {"ride four-seam", "sinker-baller", "cutter-primary", "deep arsenal",
                 "two-pitch", "knuckleballer"}
_ROLE_POPS = {"starters", "relievers"}


def _is_num(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def _hitter_populations_from_inputs(gi: dict) -> list[str]:
    """_hitter_populations from a profile's stored gate_inputs."""
    return _hitter_populations(
        sprint_raw=gi.get("sprint_speed_raw"),
        season_metrics=gi.get("season_metrics") or {},
        catching=gi.get("catching"),
        advancement=gi.get("advancement"),
        splits=gi.get("splits"),
        positions=gi.get("positions"),
        pa=int(gi.get("pa") or 0),
        metrics=gi.get("metrics") or {},
        attempt_rate_pct=gi.get("attempt_rate_pct"),
        attempts=int(gi.get("sb") or 0) + int(gi.get("cs") or 0),
    )


def _pitcher_populations_from_inputs(gi: dict, is_starter: bool) -> list[str]:
    """_pitcher_populations from stored gate_inputs (mechanics membership is
    the pitcher's mechanics pool row)."""
    pops = [POP_ALL, "starters" if is_starter else "relievers"]
    if (gi.get("pool_rows") or {}).get("league_mechanics") is not None:
        pops.append("mechanics")
    if gi.get("tempo"):
        pops.append("tempo")
    if gi.get("splits"):
        pops.append("splits_qual")
    if gi.get("runner_control"):
        pops.append("runner_control")
    return pops


def _family_state(tags: tuple, fired: set, evaluate) -> str:
    """Fold per-tag states into one family state. `evaluate(tag)` returns
    "ok", "not_applicable", "insufficient_sample" or "missing_data"."""
    if fired & set(tags):
        return "present"
    states = [evaluate(t) for t in tags]
    states = [st for st in states if st != "not_applicable"]
    if not states:
        return "not_applicable"
    if all(st == "ok" for st in states):
        return "absent"
    return ("insufficient_sample" if "insufficient_sample" in states
            else "missing_data")


def hitter_tag_states(traits: list[dict], gi: dict) -> dict[str, str]:
    """State of every hitter tag family for one profile (see TAG_STATES)."""
    from reliability import tag_supported
    gi = gi or {}
    fired = {t.get("tag") for t in traits or []}
    pops = set(_hitter_populations_from_inputs(gi))
    pa = gi.get("pa")
    catching = gi.get("catching") or {}
    detail = (gi.get("positions") or {}).get("pos_detail") or {}
    caught = bool(catching) or "C" in detail
    attempts = int(gi.get("sb") or 0) + int(gi.get("cs") or 0)

    def evaluate(tag: str) -> str:
        pop = TAG_POPULATION.get(tag, POP_ALL)
        if pop in pops:
            return "ok" if tag_supported(tag, pa) else "insufficient_sample"
        if pop.startswith("catchers_") and not caught:
            return "not_applicable"
        if pop == "steal_attempts" and attempts == 0:
            return "not_applicable"
        if pop == "catchers_framing":
            return ("insufficient_sample" if catching.get("framing_taken")
                    else "missing_data")
        if pop in ("pa200", "steal_opps", "steal_attempts", "xbt", "splits_qual"):
            return "insufficient_sample"
        return "missing_data"

    return {fam: _family_state(tags, fired, evaluate)
            for fam, tags in HITTER_TAG_FAMILIES.items()}


def pitcher_tag_states(traits: list[dict], gi: dict, is_starter: bool,
                       arsenal_profile: Optional[dict]) -> dict[str, str]:
    """State of every pitcher tag family for one profile (see TAG_STATES)."""
    from reliability import tag_supported
    gi = gi or {}
    fired = {t.get("tag") for t in traits or []}
    pops = set(_pitcher_populations_from_inputs(gi, is_starter))
    n = gi.get("bf_total") if gi.get("bf_total") is not None else gi.get("bf")
    pool_rows = gi.get("pool_rows") or {}
    metrics = gi.get("metrics") or {}

    def evaluate(tag: str) -> str:
        pop = TAG_POPULATION.get(tag, POP_ALL)
        if pop in _ROLE_POPS and pop not in pops:
            return "not_applicable"
        if pop not in pops:
            return "missing_data" if pop == "tempo" else "insufficient_sample"
        pool = _PITCHER_TAG_POOL.get(tag)
        if pool and pool_rows.get(pool) is None:
            return "insufficient_sample"
        if tag in ("over-the-top", "sidearm", "submarine") and not _is_num(
                (pool_rows.get("league_mechanics") or {}).get("arm_angle")):
            return "missing_data"     # Statcast records arm angle from 2020
        if tag in _ARSENAL_TAGS and not arsenal_profile:
            return "insufficient_sample"
        if tag in ("elite velo", "plus velo", "soft tosser") \
                and not _is_num(metrics.get("avg_velo_pct")):
            return "missing_data"
        return "ok" if tag_supported(tag, n) else "insufficient_sample"

    out = {}
    for fam, tags in PITCHER_TAG_FAMILIES.items():
        if fam == "out pitch":
            if any(t and t.endswith(" out pitch") for t in fired):
                out[fam] = "present"
            elif not arsenal_profile:
                out[fam] = "insufficient_sample"
            else:
                out[fam] = ("absent" if tag_supported("any out pitch", n)
                            else "insufficient_sample")
            continue
        out[fam] = _family_state(tags, fired, evaluate)
    return out


def _population_coverage(players: list[dict], weight_key: str) -> dict[str, float]:
    """Share of a unit's workload in each tag population — the coverage that
    goes beside a conditional trait_density. A tag's density is its share of
    the eligible workload; this says how much of the unit that eligible
    workload is."""
    pop_w: dict[str, float] = {}
    total = 0.0
    for pl in players:
        w = float(pl.get(weight_key) or 0)
        if w <= 0:
            continue
        total += w
        for pop in (pl.get("populations") or [POP_ALL]):
            pop_w[pop] = pop_w.get(pop, 0.0) + w
    if total <= 0:
        return {}
    return {pop: round(w / total, 4) for pop, w in sorted(pop_w.items())}


def _family_state_shares(players: list[dict], weight_key: str) -> dict[str, dict[str, float]]:
    """Workload share of a unit in each state, per tag family. Where `absent`
    is thin, a missing tag says little about the unit — the obstacle was
    coverage, not typicality."""
    acc: dict[str, dict[str, float]] = {}
    total = 0.0
    for pl in players:
        w = float(pl.get(weight_key) or 0)
        states = pl.get("tag_states")
        if w <= 0 or not states:
            continue
        total += w
        for fam, st in states.items():
            d = acc.setdefault(fam, {})
            d[st] = d.get(st, 0.0) + w
    if total <= 0:
        return {}
    return {fam: {st: round(v / total, 4) for st, v in d.items()}
            for fam, d in acc.items()}


def attach_tag_states(hitters: list[dict], starters: list[dict],
                      bullpen_arms: list[dict]) -> None:
    """Set `populations` and `tag_states` on every profile from its traits and
    gate_inputs. Shared by the build and rederive_traits, so a replay of the
    gates moves the states with the tags."""
    for h in hitters or []:
        gi = h.get("gate_inputs")
        if gi:
            h["populations"] = _hitter_populations_from_inputs(gi)
            h["tag_states"] = hitter_tag_states(h.get("traits") or [], gi)
    for group, is_starter in ((starters, True), (bullpen_arms, False)):
        for p in group or []:
            gi = p.get("gate_inputs")
            if gi:
                st = gi.get("is_starter", is_starter)
                p["populations"] = _pitcher_populations_from_inputs(gi, st)
                p["tag_states"] = pitcher_tag_states(
                    p.get("traits") or [], gi, st, p.get("arsenal_profile"))


def _trait_density(arms: list[dict], weight_key: str = "bf") -> dict[str, float]:
    """
    Weighted share of a unit's workload carried by players holding each trait
    tag. This is the trait-based replacement for archetype counting: the
    unit's shape IS this distribution — no winner-take-all label.

    Each tag is divided by the weight of the players ELIGIBLE for it (see
    TAG_POPULATION), not the whole unit. Players carry their eligibility in
    `populations`; profiles built before that field existed fall back to the
    whole-unit denominator, which is the pre-fix behaviour.
    """
    tag_w:  dict[str, float] = {}
    pop_w:  dict[str, float] = {}
    total_w = 0.0
    for arm in arms:
        w = float(arm.get(weight_key) or 0)
        if w <= 0:
            continue
        total_w += w
        for pop in (arm.get("populations") or [POP_ALL]):
            pop_w[pop] = pop_w.get(pop, 0.0) + w
        for tr in arm.get("traits") or []:
            tag = tr.get("tag")
            if tag:
                tag_w[tag] = tag_w.get(tag, 0.0) + w
    if total_w <= 0:
        return {}

    out: dict[str, float] = {}
    for tag, w in tag_w.items():
        denom = pop_w.get(TAG_POPULATION.get(tag, POP_ALL), 0.0) or total_w
        out[tag] = round(min(w / denom, 1.0), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _build_team_identity(
    hitter_profiles:   list[dict],
    starters:          list[dict],
    bullpen_profile:   dict,
    philosophy_metrics: dict,
    bullpen_arms:      Optional[list[dict]] = None,
) -> dict:
    """
    Synthesize a team-level "identity" summary from the player-level
    archetype classifications already computed for the portrait:

      offense — PA-weighted hitter archetype distribution + headline label
      rotation — BF-weighted starter archetype + arsenal-approach distribution,
                 ace count, and a headline label
      bullpen — out-mechanism / leverage-structure passthrough
      fit — qualitative read on rotation-vs-defense alignment and
            whether the bullpen's out-mechanism complements the rotation's

    This is a synthesis pass only — it derives new summary fields from
    existing per-player profiles, it does not recompute any underlying
    metrics.
    """
    identity: dict = {"offense": {}, "rotation": {}, "bullpen": {}, "fit": {}}

    # ── Offense: PA-weighted trait shares ──────────────────────────────────
    offense_density = _trait_density(hitter_profiles, weight_key="pa")
    total_pa = sum(float(hp.get("pa") or 0) for hp in hitter_profiles)

    if total_pa > 0 and offense_density:
        # Headline labels retired. They were the least emergent thing in the
        # system — human-chosen cutoffs ("power_share >= 0.40") assigning a
        # team to a bucket. A team's offensive identity is now read from where
        # it actually sits in tag-density space (see build_league_identity:
        # fingerprint deviations + nearest neighbours), which needs no
        # thresholds and follows the roster rather than a taxonomy.
        identity["offense"] = {
            "trait_density": offense_density,
            "power_share":   sum(v for k, v in offense_density.items()
                                 if k in _OFFENSE_POWER_TAGS),
        }

    # ── Rotation: BF-weighted trait densities + approach distribution ──────
    approach_bf: dict[str, float] = {}
    total_bf = 0.0
    for sp in starters:
        bf = float(sp.get("bf") or 0)
        if bf <= 0:
            continue
        arsenal = sp.get("arsenal_profile") or {}
        approach = arsenal.get("approach")
        if approach:
            approach_bf[approach] = approach_bf.get(approach, 0.0) + bf
        total_bf += bf

    rotation_density = _trait_density(starters)

    if total_bf > 0:
        approach_dist = {k: v / total_bf for k, v in approach_bf.items()}
        dominant_approach = max(approach_dist, key=approach_dist.get) if approach_dist else None

        # Out-mechanism from outcome-tag densities, with a margin (±15 pts,
        # mirroring the bullpen's K-vs-GB band) before either side claims it.
        k_share  = rotation_density.get("bat-misser", 0.0)
        gb_share = rotation_density.get("ground-baller", 0.0)
        if k_share >= gb_share + _MECHANISM_MARGIN:
            rotation_mechanism = "Strikeout-driven"
        elif gb_share >= k_share + _MECHANISM_MARGIN:
            rotation_mechanism = "Ground-ball-driven"
        elif k_share or gb_share:
            rotation_mechanism = "Mixed"
        else:
            rotation_mechanism = None

        # Rotation headline label retired for the same reason as the offense
        # one. `out_mechanism` survives: it is not a bucket assignment but a
        # side-by-side comparison of two measured densities with an explicit
        # margin, and the fit notes below read it.
        identity["rotation"] = {
            "trait_density":     rotation_density,
            "approach_dist":     approach_dist,
            "dominant_approach": dominant_approach,
            "tunneler_share":    rotation_density.get("tunneler", 0.0),
            "out_mechanism":     rotation_mechanism,
        }

    # ── Bullpen: passthrough of collective profile ─────────────────────────
    if bullpen_profile:
        identity["bullpen"] = {
            "out_mechanism":      bullpen_profile.get("out_mechanism"),
            "leverage_structure": bullpen_profile.get("leverage_structure"),
        }
    if bullpen_arms:
        density = _trait_density(bullpen_arms)
        if density:
            identity.setdefault("bullpen", {})["trait_density"] = density

    # Coverage beside every density: how much of the unit each population
    # is, and per tag family how much of the workload was measured at all.
    for unit, players, wkey in (("offense", hitter_profiles, "pa"),
                                ("rotation", starters, "bf"),
                                ("bullpen", bullpen_arms or [], "bf")):
        if identity.get(unit, {}).get("trait_density") is None:
            continue
        identity[unit]["population_coverage"] = _population_coverage(players, wkey)
        shares = _family_state_shares(players, wkey)
        if shares:
            identity[unit]["family_states"] = shares

    # ── Fit: rotation vs defense, bullpen complement ────────────────────────
    rotation_mech = identity["rotation"].get("out_mechanism")
    bullpen_mech  = identity["bullpen"].get("out_mechanism")
    team_defense_pct = philosophy_metrics.get("TeamDefense_pct")

    # Descriptive facts only — state the two figures side by side and let the
    # reader connect them. No evaluation of whether a pairing is good/bad.
    if rotation_mech and team_defense_pct is not None:
        band = ("top-tier" if team_defense_pct >= 60
                else "bottom-tier" if team_defense_pct < 40
                else "middle-tier")
        identity["fit"]["rotation_vs_defense"] = (
            f"{rotation_mech} rotation; team defense ranks {_ordinal(team_defense_pct)} "
            f"percentile ({band})."
        )

    if rotation_mech and bullpen_mech:
        if rotation_mech == bullpen_mech:
            identity["fit"]["bullpen_complement"] = (
                f"Rotation and bullpen are both {bullpen_mech.lower()}."
            )
        else:
            identity["fit"]["bullpen_complement"] = (
                f"Rotation is {rotation_mech.lower()}; "
                f"bullpen is {bullpen_mech.lower()}."
            )

    return identity


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _load_season_players(season: int):
    """
    (batting_full, pitching_full, bwar_bat, bwar_pitch) for one season.

    batting_full / pitching_full hold SEASON TOTALS, one row per player — what
    the player did, all teams together; their `war` is the season sum. The
    bwar frames keep one row per team stint (ingest.pull_bwar) for what each
    team got.
    """
    if season_batting_seeded(season):
        log.info("Loading season %d from Supabase", season)
        batting_full  = query_batting(season)
        pitching_full = query_pitching(season)
    else:
        log.info("Season %d not in Supabase — pulling from BRef/Savant", season)
        batting_full  = pull_fg_batting(season)
        pitching_full = pull_fg_pitching(season)

    # Ensure key_mlbam column exists (Savant fallback uses player_id)
    for df_obj in [batting_full, pitching_full]:
        if "key_mlbam" not in df_obj.columns and "player_id" in df_obj.columns:
            df_obj.rename(columns={"player_id": "key_mlbam"}, inplace=True)

    batting_full["key_mlbam"]  = pd.to_numeric(batting_full.get("key_mlbam"),  errors="coerce")
    pitching_full["key_mlbam"] = pd.to_numeric(pitching_full.get("key_mlbam"), errors="coerce")

    # Merge bWAR season totals into batting_full / pitching_full — overrides
    # the empty DB war column. The per-team stints are returned alongside.
    bwar_bat = pd.DataFrame(columns=["key_mlbam", "team", "war"])
    bwar_pitch = pd.DataFrame(columns=["key_mlbam", "team", "war"])
    try:
        bwar_bat, bwar_pitch = pull_bwar(season)
        for frame, side in ((bwar_bat, "batting"), (bwar_pitch, "pitching")):
            if frame.empty:
                continue
            frame["key_mlbam"] = pd.to_numeric(frame["key_mlbam"], errors="coerce")
            total = frame.groupby("key_mlbam", as_index=False)["war"].sum()
            if side == "batting":
                batting_full = batting_full.drop(columns=["war"], errors="ignore").merge(
                    total, on="key_mlbam", how="left")
            else:
                pitching_full = pitching_full.drop(columns=["war"], errors="ignore").merge(
                    total, on="key_mlbam", how="left")
    except Exception as exc:
        log.warning("bWAR merge failed: %s", exc)
    return batting_full, pitching_full, bwar_bat, bwar_pitch


def build_team_portrait(
    team: str,
    season: int,
    statcast: Optional[pd.DataFrame] = None,
    statcast_label: str = "test",
) -> dict:
    """
    Build a full analytical portrait of one team for one season.

    Args:
        team:          MLB team abbreviation (e.g. 'HOU', 'NYY')
        season:        season year (2015+)
        statcast:      pre-loaded Statcast DataFrame (optional; loaded from
                       cache/network if not provided)
        statcast_label: cache label used when statcast is None and a full
                        season pull is needed (only for non-standard date ranges)

    Returns:
        Nested portrait dict (see module docstring for structure).
    """
    log.info("Building portrait — team=%s season=%d", team, season)

    # ── 1. Statcast data ──────────────────────────────────────────────────
    if statcast is None:
        statcast = pull_statcast_season(season)

    # Filter to games involving this team
    team_sc = statcast[
        (statcast["home_team"] == team) | (statcast["away_team"] == team)
    ].copy()

    if team_sc.empty:
        log.warning("No Statcast rows found for team=%s — returning minimal portrait", team)
        return {"team": team, "season": season, "error": "no_statcast_data"}

    # ── 2. Player IDs from Statcast ───────────────────────────────────────
    hitter_ids, pitcher_ids = _extract_team_players(statcast, team)

    # ── 3. Load and normalize player aggregates ───────────────────────────
    # current-season pool: used for team filtering and cross-team metrics
    batting_full, pitching_full, bwar_bat, bwar_pitch = _load_season_players(season)

    # ── 3c. Spray chart stats from Statcast hc_x / hc_y ─────────────────────
    # Computed once for the full season and merged into batting_full so that
    # normalization in _normalize_batting ranks against the season pool.
    # Spray stats are NOT stored in the DB — computed fresh each portrait build.
    try:
        from pitch_aggregates import compute_batter_spray_stats
        spray = compute_batter_spray_stats(statcast, min_bip=20)
        if not spray.empty:
            spray["key_mlbam"] = pd.to_numeric(spray["key_mlbam"], errors="coerce")
            batting_full = batting_full.merge(
                spray[["key_mlbam", "xb_pct", "gap_pct", "pull_pct", "oppo_pct", "hr_per_bip", "hr_fb"]],
                on="key_mlbam", how="left",
            )
            log.info("Spray stats merged — %d batters with spray data",
                     batting_full["xb_pct"].notna().sum())
    except Exception as exc:
        log.warning("Spray stats merge failed: %s", exc)

    # ── 3a-ii. Chase/FPS backfill from Statcast pitch aggregates ──────────
    # Savant's chase_pct column is only ~34% populated in the DB, which
    # starved the plate-discipline traits (aggressive / elite discipline).
    # Backfill nulls with rates computed directly from the season's pitches.
    try:
        from pitch_aggregates import compute_pitch_aggregates
        _bat_agg, _ = compute_pitch_aggregates(season)
        _fill = _bat_agg[["key_mlbam", "chase_sc_pct", "fps_pct",
                          "zone_swing_pct"]].rename(
            columns={"chase_sc_pct": "_chase_fill", "fps_pct": "_fps_fill",
                     "zone_swing_pct": "_zswing_fill"})
        batting_full = batting_full.merge(_fill, on="key_mlbam", how="left")
        # UNIT MISMATCH — do NOT fillna() one into the other. The DB's
        # chase_pct/fps_pct are PERCENTILES (1–100, median ~49); the Statcast
        # columns are RATES (0–1, median ~0.28). Filling nulls mixed both
        # scales into one column, and the downstream percentile rank then
        # sorted by DATA SOURCE rather than by behaviour — every DB-sourced
        # player outranked every backfilled one. `free swinger` was really
        # tagging "has a Savant value" and `patient` "was backfilled"
        # (corr with true chase rate: +0.27).
        # The Statcast rate is computed uniformly from raw pitches for every
        # player, so it is used ALONE. Players without it stay null and simply
        # miss the zone-judgment tags — which is what the `chase` population
        # already encodes.
        for dest, src in [("chase_pct", "_chase_fill"), ("fps_pct", "_fps_fill"),
                          ("zone_swing_pct", "_zswing_fill")]:
            batting_full[dest] = pd.to_numeric(batting_full[src], errors="coerce")

        # Swing discrimination — strikes attacked minus balls chased. This is
        # the `zone hunter` axis, and it has to be differenced from the RAW
        # rates here rather than from their percentiles downstream: a
        # percentile of a difference is not the difference of two percentiles.
        #
        # Zone-swing rate alone was not worth a tag. Measured on 2,040
        # qualified batter-seasons (2021-25) it correlates +0.81 with
        # first-pitch-swing rate, which `aggressive` already gates on — above
        # the containment that retired `hard contact` (0.82) and `gyro-heavy`
        # (0.79). The difference decorrelates it to +0.26: swinging a lot and
        # swinging at the RIGHT pitches are separate skills, and only the
        # second one is news.
        batting_full["swing_discrim"] = (
            pd.to_numeric(batting_full["zone_swing_pct"], errors="coerce")
            - pd.to_numeric(batting_full["chase_pct"], errors="coerce"))
        batting_full = batting_full.drop(
            columns=["_chase_fill", "_fps_fill", "_zswing_fill"])
        log.info("Chase/FPS from Statcast rates — chase %d/%d non-null",
                 batting_full["chase_pct"].notna().sum(), len(batting_full))
    except Exception as exc:
        log.warning("Chase/FPS backfill failed: %s", exc)

    # ── 3b. Multi-season historical pool for percentile normalization ─────
    # Percentiles are computed against the Statcast era so that "70" means 70th
    # pct of the era, not just this year's 30 teams. The era is every FINISHED
    # season plus this one (ingest.era_pool_seasons): finished seasons all
    # share one pool that does not move while the next season is played, so a
    # night of new games re-ranks only the season it happened in.
    #
    # This used to be query_seeded_seasons(), which read player_batting through
    # the API's 1,000-row cap and returned whichever seasons came back first —
    # 2015-2025 without 2026 when replaced, so 2026 portraits ranked against a
    # different era than every other season.
    try:
        seeded_seasons = era_pool_seasons(season)
        batting_pool  = pull_batting_history(seeded_seasons)
        pitching_pool = pull_pitching_history(seeded_seasons)
        # Normalise key_mlbam in pool
        for pool in [batting_pool, pitching_pool]:
            if not pool.empty:
                if "key_mlbam" not in pool.columns and "player_id" in pool.columns:
                    pool.rename(columns={"player_id": "key_mlbam"}, inplace=True)
                pool["key_mlbam"] = pd.to_numeric(pool.get("key_mlbam"), errors="coerce")
        log.info("Historical pool: %d batting rows / %d pitching rows across %d seasons",
                 len(batting_pool), len(pitching_pool), len(seeded_seasons))
    except Exception as exc:
        log.warning("Historical pool failed — falling back to single-season: %s", exc)
        batting_pool  = batting_full
        pitching_pool = pitching_full

    from pitch_aggregates import compute_stints
    bat_stints, pit_stints = compute_stints(statcast, season)
    team_bat_raw, _bat_cameos = _team_stints_batting(
        batting_full[batting_full["key_mlbam"].isin(hitter_ids)], bat_stints, bwar_bat, team)
    team_pit_raw, _pit_cameos = _team_stints_pitching(
        pitching_full[pitching_full["key_mlbam"].isin(pitcher_ids)], pit_stints, bwar_pitch, team)
    also_appeared = _also_appeared(_bat_cameos, _pit_cameos)

    # Normalize against multi-season historical pool
    team_bat = _normalize_batting(batting_pool,  team_bat_raw)
    team_pit = _normalize_pitching(pitching_pool, team_pit_raw)

    # ── Spray stat normalization (current-season only — not in DB pool) ───────
    # batting_full has spray stats merged; batting_pool (multi-season DB) doesn't.
    # Rank spray metrics against the full current-season batting_full pool.
    SPRAY_COLS = [
        ("xb_pct",   False),   # higher = more doubles+triples per BIP
        ("gap_pct",  False),   # higher = more BIP in gap zones
        ("pull_pct", False),
        ("oppo_pct", False),
        ("hr_fb",    False),   # HR / fly balls — power purity; higher = more HR power
    ]
    for raw_col, invert in SPRAY_COLS:
        renamed_col = BATTING_COL_MAP.get(raw_col)
        if renamed_col and raw_col in batting_full.columns:
            pool_series = pd.to_numeric(batting_full[raw_col], errors="coerce")
            team_series = pd.to_numeric(team_bat_raw[raw_col], errors="coerce") \
                if raw_col in team_bat_raw.columns else pd.Series(dtype=float)
            if pool_series.notna().sum() > 1 and team_series.notna().sum() > 0:
                pct_col = f"{renamed_col}_pct"
                team_bat[pct_col] = _pool_rank(pool_series, team_series, invert=invert)

    # ── Season-pool gate percentiles for hitter trait tags ─────────────────
    # Hybrid normalization: display percentiles stay era-based (above), but
    # TAG GATES rank within the current season so a tag means "distinctive vs
    # this year's peers" and incidence doesn't drift with the run environment
    # (e.g. `high-K` firing on 32% of hitters in a high-K era).
    # Columns get a `_szn` suffix; _classify_hitters routes them to the gates
    # as season_metrics, under the SAME names as the era percentiles — the
    # dict says the cohort, the key says the metric (metric_contracts), and
    # hitter_traits lets a season rank override the era one key by key.
    # raw_col = pre-rename batting_full column; key = the percentile's name.
    _SEASON_TAG_METRICS = [
        # (batting_full col, gate key,   invert)
        ("k_rate",       "K_pct_raw_pct",  False),  # higher = more Ks
        ("bb_rate",      "BB_pct_pct",     False),
        ("chase_pct",    "OSwing_pct_pct", False),  # higher = chases more
        ("fps_pct",      "FPS_pct_pct",    False),
        ("iso",          "ISO_pct",        False),
        ("ba",           "AVG_pct",        False),
        ("obp",          "OBP_pct",        False),
        ("brl_percent",  "Barrel_pct_pct", False),
        ("hard_hit_percent", "HardHit_pct_pct", False),
        ("gb_pct",       "BatGB_pct_pct",  False),  # batter GB% (air/ground tags)
        ("sprint_speed", "sprint_speed_pct", False),  # station-to-station gate
        ("swing_discrim","SwingDiscrim_pct", False), # zone-hunter gate
    ]
    def _first_col(frame: pd.DataFrame, col: str) -> Optional[pd.Series]:
        """Column as Series; first occurrence if the rename created duplicates."""
        if col not in frame.columns:
            return None
        s = frame.loc[:, col]
        return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s

    _renamed_full = batting_full.rename(columns=BATTING_COL_MAP)
    _renamed_team = team_bat_raw.rename(columns=BATTING_COL_MAP)
    for raw_col, key, invert in _SEASON_TAG_METRICS:
        try:
            col = BATTING_COL_MAP.get(raw_col, raw_col)
            src_full = _first_col(_renamed_full, col)
            src_team = _first_col(_renamed_team, col)
            if src_full is None or src_team is None:
                continue
            pool_series = pd.to_numeric(src_full, errors="coerce")
            team_series = pd.to_numeric(src_team, errors="coerce")
            if pool_series.notna().sum() > 1 and team_series.notna().sum() > 0:
                team_bat[f"{key}_szn"] = _pool_rank(
                    pool_series, team_series, invert=invert).values
        except Exception as exc:
            log.warning("Season-pool rank failed for %s: %s", key, exc)

    # PAvol_szn — the `everyday player` gate. A deployment tag, so it ranks
    # what THIS team gave the player (stint PA) against every player-team
    # stint in the league, not the season total: a regular traded in July is
    # an everyday player for neither club's full season, and a club that
    # had the player for 25 games did not make an everyday player of them.
    try:
        _pa_col = _col(team_bat_raw, "pa", "PA")
        team_bat["PAvol_pct_szn"] = _pool_rank(
            bat_stints["pa"], pd.to_numeric(team_bat_raw[_pa_col], errors="coerce")).values
    except Exception as exc:
        log.warning("Stint PA rank failed: %s", exc)

    # Pool-sourced season ranks — these columns live in the history pool
    # (Savant CSV path), not in query_batting. Rank within THIS season's
    # slice (filtering the pool to it: an era map once leaked other seasons'
    # values via dict overwrite) and map to team players by MLBAM ID.
    #   OAA          — outs above average (fielding family)
    #   ArmStrength  — throwing arm (fielding family)
    # (Luck — wOBA − xwOBA — was ranked here too; its tags are retired.)
    _POOL_SEASON_METRICS = [
        ("oaa",                      "OAA_pct"),
        ("arm_strength",             "ArmStrength_pct"),
    ]
    try:
        _lp = batting_pool
        _yr_col = next((c for c in ["_history_season", "year", "season"]
                        if c in _lp.columns), None)
        if _yr_col is not None:
            szn_all = _lp[pd.to_numeric(_lp[_yr_col], errors="coerce") == season]
            from ingest import normalize_percentile as _npct
            for src_col, key in _POOL_SEASON_METRICS:
                if src_col not in szn_all.columns:
                    continue
                szn_slice = szn_all[szn_all[src_col].notna()]
                if len(szn_slice) <= 1:
                    continue
                pcts = _npct(pd.to_numeric(szn_slice[src_col], errors="coerce"))
                pmap = dict(zip(
                    pd.to_numeric(szn_slice["key_mlbam"], errors="coerce"),
                    pcts,
                ))
                team_bat[f"{key}_szn"] = team_bat["key_mlbam"].map(
                    lambda x: pmap.get(float(x)) if pd.notna(x) else None)
    except Exception as exc:
        log.warning("Pool-season ranking failed: %s", exc)

    log.info("Season-pool tag gates ranked — %d _szn columns",
             sum(1 for c in team_bat.columns if str(c).endswith("_szn")))

    # ── 4. Spin efficiency ────────────────────────────────────────────────
    pitcher_sc = team_sc[team_sc["pitcher"].isin(pitcher_ids)].copy()
    spin_pitch_level: Optional[pd.DataFrame] = None
    spin_pitcher_level: Optional[pd.DataFrame] = None
    try:
        if not pitcher_sc.empty:
            spin_pitch_level, spin_pitcher_level = build_pitcher_spin_profile(
                pitcher_sc, season=season)
            log.info("Spin efficiency computed for %d pitchers", len(spin_pitcher_level))
    except Exception as exc:
        log.warning("Spin efficiency failed: %s", exc)

    # ── 5. Tunneling ──────────────────────────────────────────────────────
    tun_pitcher_level: Optional[pd.DataFrame] = None
    try:
        if not pitcher_sc.empty:
            tun_pitcher_level = build_tunnel_profile(pitcher_sc)
            log.info("Tunneling computed — %d pitchers", len(tun_pitcher_level))
    except Exception as exc:
        log.warning("Tunneling failed: %s", exc)

    # ── 6. Park factors ───────────────────────────────────────────────────
    park_result: Optional[dict] = None
    try:
        park_df = build_park_profile(statcast, season)
        if not park_df.empty:
            team_park = park_df[park_df["team"] == team]
            park_result = team_park.to_dict("records")[0] if not team_park.empty else None
    except Exception as exc:
        log.warning("Park factors failed: %s", exc)

    # ── 7. Temporal mode ──────────────────────────────────────────────────
    temporal_info = _team_temporal(statcast, team, season)

    # ── 8. Team-level aggregated metrics ─────────────────────────────────
    team_batting_agg  = _aggregate_team_batting(team_bat)
    team_pitching_agg = _aggregate_team_pitching(team_pit)

    # ── 9. Spin efficiency team summary ──────────────────────────────────
    # League-wide spin pool — percentiles vs ALL pitchers in the season (the
    # per-team frame above only ranks within this team's arms). Also reused
    # for the per-pitcher classifier keys below.
    _league_spin = None
    try:
        from spin_efficiency import load_league_spin
        _league_spin = load_league_spin(season, statcast=statcast)
    except Exception as _ls_exc:
        log.warning("League spin pool unavailable: %s", _ls_exc)

    spin_team_summary: Optional[dict] = None
    if spin_pitcher_level is not None and not spin_pitcher_level.empty:
        se_col = "spin_efficiency_weighted"
        if se_col in spin_pitcher_level.columns:
            valid = spin_pitcher_level[se_col].dropna()
            if len(valid) > 0:
                team_mean_se = float(valid.mean())

                # Team percentile: pitch-weighted mean of each pitcher's league
                # spin_efficiency_pct, ranked across the 30 pitching teams.
                se_pct = 50.0
                if _league_spin is not None and not _league_spin.empty:
                    league_team_pct = _league_team_spin_pct(statcast, _league_spin)
                    if team in league_team_pct.index and league_team_pct.notna().sum() > 1:
                        se_pct = float(league_team_pct.loc[team])

                spin_team_summary = {
                    "mean_spin_efficiency":         team_mean_se,
                    "weighted_spin_efficiency_pct": se_pct,
                    "pitcher_count":                len(valid),
                }

    # ── 10a. Cross-team spread/concentration + tenure metrics ─────────────
    cross_team = {}
    _power_source_ratio: Optional[float] = None
    _roster_ctrl: dict = {}
    try:
        debut_seasons, est_service_time = {}, {}
        if not cross_team_path(season).exists():
            debut_seasons    = query_debut_seasons()
            all_hitter_ids   = [int(p) for p in bat_stints["batter"].unique()]
            from service_time import service_years_days
            est_service_time = service_years_days(all_hitter_ids, season)
        cross_team, _row = _compute_cross_team_metrics(
            statcast, batting_full, team, season, debut_seasons, est_service_time,
            bat_stints, bwar_bat)
        # Raw values for display (not scored): power_source_ratio and the
        # roster-control shares.
        if _row is not None:
            _v = _row.get("power_source_ratio")
            _power_source_ratio = None if _v is None or pd.isna(_v) else float(_v)
            for _rc_col, _rc_key in [
                ("new_player_share", "pre_arb"),
                ("arb_share",        "arb"),
                ("veteran_share",    "fa_eligible"),
                ("avg_tenure",       "avg_tenure"),
            ]:
                _v = _row.get(_rc_col)
                _roster_ctrl[_rc_key] = None if _v is None or pd.isna(_v) else float(_v)
    except Exception as exc:
        log.warning("Cross-team metrics failed: %s", exc)

    # ── 10b. OAA (team-level, all positions) ─────────────────────────────
    oaa_val: Optional[float] = None
    try:
        from ingest import pull_fielding_oaa
        oaa_df = pull_fielding_oaa(season)
        if not oaa_df.empty and "oaa_total" in oaa_df.columns:
            oaa_series = oaa_df.set_index("team")["oaa_total"]
            oaa_pcts   = normalize_percentile(oaa_series)
            if team in oaa_pcts.index:
                oaa_val = float(oaa_pcts.loc[team])
    except Exception as exc:
        log.warning("OAA failed: %s", exc)

    # ── 10bb. Defensive runs above average (BRef bWAR component) ─────────
    def_runs_val: Optional[float] = None
    try:
        from ingest import pull_def_runs
        def_runs_df = pull_def_runs(season)
        if not def_runs_df.empty and "def_runs_total" in def_runs_df.columns:
            def_series = def_runs_df.set_index("team")["def_runs_total"]
            def_pcts   = normalize_percentile(def_series)
            if team in def_pcts.index:
                def_runs_val = float(def_pcts.loc[team])
    except Exception as exc:
        log.warning("Defensive runs failed: %s", exc)

    # ── 10c. Opener usage (all 30 teams from Statcast) ────────────────────
    opener_val: Optional[float] = None
    try:
        from pitch_aggregates import compute_opener_usage
        opener_series = compute_opener_usage(statcast)
        opener_pcts   = normalize_percentile(opener_series)
        if team in opener_pcts.index:
            opener_val = float(opener_pcts.loc[team])
    except Exception as exc:
        log.warning("Opener usage failed: %s", exc)

    # ── 10d. Platoon optimization (all 30 teams from Statcast) ────────────
    platoon_val: Optional[float] = None
    try:
        from pitch_aggregates import compute_platoon_optimization
        platoon_series = compute_platoon_optimization(statcast)
        platoon_pcts   = normalize_percentile(platoon_series)
        if team in platoon_pcts.index:
            platoon_val = float(platoon_pcts.loc[team])
    except Exception as exc:
        log.warning("Platoon optimization failed: %s", exc)

    # ── 10e. Roster turnover vs. prior season ─────────────────────────────
    turnover_val: Optional[float] = None
    try:
        turnover_rates = _compute_turnover(statcast, season)
        if turnover_rates:
            turnover_series = pd.Series(turnover_rates)
            turnover_pcts   = normalize_percentile(turnover_series)
            if team in turnover_pcts.index:
                turnover_val = float(turnover_pcts.loc[team])
    except Exception as exc:
        log.warning("Turnover failed: %s", exc)

    # ── 9b. Tunneling team summary ────────────────────────────────────────
    tunneling_team_summary: Optional[dict] = None
    if tun_pitcher_level is not None and not tun_pitcher_level.empty:
        try:
            # BF-weight tunnel_score_pct across pitchers on this team.
            # Pitcher IDs come from tun_pitcher_level["pitcher"]; we join
            # to team_pit to get BF weights, falling back to equal weights.
            tun_df = tun_pitcher_level.copy()
            if "pitcher" in tun_df.columns and "tunnel_score_pct" in tun_df.columns:
                tun_df["pitcher"] = tun_df["pitcher"].astype(int)
                # BF weights: this team's stint BF. team_pit is keyed by
                # key_mlbam (renamed BF), never "pitcher", so this join used to
                # find nothing and every pitcher weighed 1.0.
                bf_map: dict[int, float] = {}
                _bfc = _col(team_pit, "BF", "bf")
                if _bfc:
                    for _, row in team_pit.iterrows():
                        pid, bf = row.get("key_mlbam"), row.get(_bfc)
                        if not pd.isna(pid) and bf is not None and not pd.isna(bf) and bf:
                            bf_map[int(pid)] = float(bf)

                tun_df["_weight"] = tun_df["pitcher"].map(bf_map).fillna(1.0)
                valid = tun_df["tunnel_score_pct"].notna()
                if valid.sum() > 0:
                    w   = tun_df.loc[valid, "_weight"]
                    v   = tun_df.loc[valid, "tunnel_score_pct"]
                    team_tunnel_mean = float((v * w).sum() / w.sum())

                    # Percentile rank against all 30 teams: pitch-count-weighted
                    # mean of each team's pitchers' league tunnel scores, from
                    # the cached league-wide pool.
                    tun_team_pct = 50.0
                    try:
                        from tunneling import load_league_tunnel
                        pool = load_league_tunnel(season, statcast=statcast)
                        sc_tun = statcast[["pitcher", "inning_topbot",
                                           "home_team", "away_team"]].copy()
                        sc_tun["pitching_team"] = np.where(
                            sc_tun["inning_topbot"] == "Top",
                            sc_tun["home_team"], sc_tun["away_team"],
                        )
                        counts = (
                            sc_tun.dropna(subset=["pitching_team"])
                            .groupby(["pitching_team", "pitcher"])
                            .size().rename("n_pitches").reset_index()
                        )
                        merged = counts.merge(
                            pool[["pitcher", "tunnel_score_weighted"]], on="pitcher",
                        ).dropna(subset=["tunnel_score_weighted"])
                        team_means = (
                            merged.groupby("pitching_team")
                            .apply(lambda g: np.average(g["tunnel_score_weighted"],
                                                        weights=g["n_pitches"]),
                                   include_groups=False)
                        )
                        if team in team_means.index and team_means.notna().sum() > 1:
                            tun_team_pct = float(normalize_percentile(team_means).loc[team])
                    except Exception as _tp_exc:
                        log.warning("Team tunnel percentile fallback (50.0): %s", _tp_exc)

                    tunneling_team_summary = {
                        "mean_tunnel_score":     team_tunnel_mean,
                        "tunnel_score_team_pct": tun_team_pct,
                        "pitcher_count":         int(valid.sum()),
                    }
                    log.info(
                        "Tunneling team summary — mean=%.1f pct=%.1f n=%d",
                        team_tunnel_mean, tun_team_pct, valid.sum(),
                    )
        except Exception as exc:
            log.warning("Tunneling team aggregation failed: %s", exc)

    # ── 10. Philosophy metrics assembly ───────────────────────────────────
    philosophy_metrics = _build_philosophy_metrics(
        team_batting_agg,
        team_pitching_agg,
        park_result,
        spin_team_summary,
        tunneling_team_summary,
        cross_team_metrics=cross_team,
        oaa_val=oaa_val,
        def_runs_val=def_runs_val,
        opener_val=opener_val,
        platoon_val=platoon_val,
        turnover_val=turnover_val,
    )

    # ── 11. Philosophy scoring ────────────────────────────────────────────
    philosophy = build_philosophy_summary(team, season, philosophy_metrics)

    # ── 12. Hitter classification (traits) ────────────────────────────────
    # Batter L/R splits computed up front — they feed the platoon trait
    # family during classification AND get attached as hp["splits"] below.
    _split_map: dict[int, dict] = {}
    try:
        from pitch_aggregates import compute_batter_splits
        _split_map = compute_batter_splits(statcast, hitter_ids, min_pa=20)
    except Exception as exc:
        log.warning("Batter splits failed: %s", exc)

    # Running game + versatility maps (all base-state / alignment derived)
    _advancement_map: dict[int, dict] = {}
    _positions_map: dict[int, dict] = {}
    try:
        from pitch_aggregates import (compute_batter_advancement,
                                      compute_position_appearances)
        _adv = compute_batter_advancement(statcast, season)
        if not _adv.empty:
            qual = _adv[_adv["opps"] >= 10].copy()
            # Ranked on how often he goes; whether he makes it is evidence.
            qual["xbt_pct"] = qual["attempt_rate"].rank(pct=True) * 100.0
            pctm = dict(zip(qual["key_mlbam"], qual["xbt_pct"]))
            for r in _adv.itertuples():
                _advancement_map[int(r.key_mlbam)] = {
                    "opps": int(r.opps), "attempts": int(r.attempts),
                    "advances": int(r.advances),
                    "thrown_out": int(r.thrown_out),
                    "attempt_rate": float(r.attempt_rate),
                    "xbt_rate": float(r.xbt_rate),
                    "xbt_pct": pctm.get(r.key_mlbam),
                    "first_to_third_n": int(r.first_to_third_n),
                    "first_to_third_opps": int(r.first_to_third_opps),
                }
        # Positions THIS team played the player at (`super-utility` is a
        # deployment tag; home_position is shown on this team's card).
        _pos = compute_position_appearances(statcast, season)
        _pos = _pos[_pos["team"] == team]
        for r in _pos.itertuples():
            _positions_map[int(r.key_mlbam)] = {
                "home_position": r.home_position,
                "positions_played": int(r.positions_played),
                "pos_detail": r.pos_detail,
            }
    except Exception as exc:
        log.warning("Advancement/positions maps failed: %s", exc)

    # Catcher craft map — framing (from raw pitches) + pop time (Savant)
    _catcher_map: dict[int, dict] = {}
    try:
        from pitch_aggregates import compute_catcher_framing, load_catcher_poptime
        _fr = compute_catcher_framing(statcast, season)
        for r in _fr.itertuples():
            # Every catcher's edge-pitch count, so an unranked one reads as
            # too few pitches (tag_states) rather than as no data.
            entry = {"framing_taken": int(r.shadow_taken)}
            if pd.notna(getattr(r, "framing_pct", None)):
                entry.update({"framing_pct":   float(r.framing_pct),
                              "framing_above": float(r.framing_above)})
            _catcher_map[int(r.catcher)] = entry
        _pop = load_catcher_poptime(season)
        for r in _pop.itertuples():
            _catcher_map.setdefault(int(r.catcher), {}).update(
                {"pop_pct": float(r.pop_pct), "pop_2b": float(r.pop_2b)})
        from pitch_aggregates import load_catcher_blocking
        _blk = load_catcher_blocking(season)
        for r in _blk.itertuples():
            _catcher_map.setdefault(int(r.catcher), {}).update(
                {"blocking_pct": float(r.blocking_pct),
                 "blocks_above_average": float(r.blocks_above_average),
                 "n_pbwp": float(r.n_pbwp)})
    except Exception as exc:
        log.warning("Catcher craft map failed: %s", exc)

    hitter_profiles = _classify_hitters(team_bat, split_map=_split_map,
                                        catcher_map=_catcher_map,
                                        advancement_map=_advancement_map,
                                        positions_map=_positions_map,
                                        attempt_pcts=_league_attempt_pcts(batting_full))

    # ── 13. Pitcher archetypes ────────────────────────────────────────────
    # Load pitch-mix table once here so every starter in the loop shares it
    _pitch_mix_df = None
    try:
        from pitch_mix import load_pitch_mix
        _pitch_mix_df = load_pitch_mix(season)
        log.info("Pitch mix loaded: %d rows for %d pitchers", len(_pitch_mix_df), _pitch_mix_df.pitcher.nunique())
    except Exception as _pm_exc:
        log.warning("Pitch mix unavailable for %s %s: %s", team, season, _pm_exc)

    # League-wide tunnel pool — percentiles vs ALL pitchers in the season
    # (the per-team frames above only rank within this team's arms). The
    # league spin pool was loaded in step 9.
    _league_tunnel = None
    try:
        from tunneling import load_league_tunnel
        _league_tunnel = load_league_tunnel(season, statcast=statcast)
    except Exception as _lt_exc:
        log.warning("League tunnel pool unavailable: %s", _lt_exc)
    _league_disc = None
    try:
        from pitch_mix import load_league_plate_discipline
        _league_disc = load_league_plate_discipline(season, statcast=statcast)
    except Exception as _ld_exc:
        log.warning("League plate-discipline pool unavailable: %s", _ld_exc)
    _league_mech = None
    _league_seq = None
    try:
        from pitcher_traits import load_league_mechanics, load_league_sequencing
        _league_mech = load_league_mechanics(season, statcast=statcast)
        _league_seq  = load_league_sequencing(season, statcast=statcast)
    except Exception as _lm_exc:
        log.warning("League mechanics/sequencing pools unavailable: %s", _lm_exc)

    # Pitcher L/R splits + deployment (platoon trait family).
    # Full-season statcast, NOT team_sc: a midseason acquisition's games with
    # his former club aren't in team_sc (e.g. 2023 Verlander's Mets starts).
    _pitcher_split_map: dict[int, dict] = {}
    try:
        from pitch_aggregates import compute_pitcher_splits
        _pitcher_split_map = compute_pitcher_splits(statcast, pitcher_ids, min_bf=40)
    except Exception as _ps_exc:
        log.warning("Pitcher splits failed: %s", _ps_exc)

    # Leverage exposure — |ΔWE| per PA, league-wide (role family gates)
    _leverage_df = None
    try:
        from pitch_aggregates import compute_pitcher_leverage
        _leverage_df = compute_pitcher_leverage(statcast, season)
    except Exception as _lv_exc:
        log.warning("Pitcher leverage failed: %s", _lv_exc)

    # Tempo (savant) + runner control (base-state) maps
    _tempo_map: dict[int, dict] = {}
    _runner_ctrl_map: dict[int, dict] = {}
    try:
        from pitch_aggregates import load_pitch_tempo, compute_runner_control
        for r in load_pitch_tempo(season).itertuples():
            _tempo_map[int(r.pitcher)] = {
                "tempo_pct": float(r.tempo_pct),
                "tempo_empty_s": float(r.tempo_empty_s)}
        _rc = compute_runner_control(statcast, season)
        if not _rc.empty:
            qual = _rc[_rc["opps"] >= 100].copy()
            qual["advance_pct"] = qual["advance_rate"].rank(pct=True) * 100.0
            pctm = dict(zip(qual["pitcher"], qual["advance_pct"]))
            for r in _rc.itertuples():
                _runner_ctrl_map[int(r.pitcher)] = {
                    "opps": int(r.opps),
                    "advances_allowed": int(r.advances_allowed),
                    "advance_rate": float(r.advance_rate),
                    "advance_pct": pctm.get(r.pitcher),
                }
    except Exception as exc:
        log.warning("Tempo/runner-control maps failed: %s", exc)

    starters, bullpen_profile, bullpen_arms = _classify_pitchers(
        team, season, team_pit, spin_pitcher_level, tun_pitcher_level,
        pitch_mix_df=_pitch_mix_df,
        league_tunnel=_league_tunnel, league_spin=_league_spin,
        league_discipline=_league_disc, league_mechanics=_league_mech,
        pitching_full=pitching_full, pitcher_splits=_pitcher_split_map,
        leverage_df=_leverage_df,
        tempo_map=_tempo_map, runner_ctrl_map=_runner_ctrl_map,
        league_sequencing=_league_seq,
        pitch_stints=pit_stints,
    )

    # ── 14. Data coverage ─────────────────────────────────────────────────
    coverage = _data_coverage(philosophy_metrics)

    # ── 14b. Team identity synthesis ────────────────────────────────────────
    attach_tag_states(hitter_profiles, starters, bullpen_arms)
    team_identity: dict = {}
    try:
        team_identity = _build_team_identity(
            hitter_profiles, starters, bullpen_profile, philosophy_metrics,
            bullpen_arms=bullpen_arms,
        )
    except Exception as exc:
        log.warning("Team identity synthesis failed: %s", exc)

    # ── 16. Construction vs Results — projected profile ───────────────────
    projected: dict = {}
    try:
        projected = _compute_projected_profile(season, hitter_ids, team_bat)
        if projected:
            log.info(
                "Projected profile — prior=%d  coverage=%.0f%%  excluded=%.0f%%",
                projected["prior_season"],
                projected["coverage"] * 100,
                projected["excluded_pa_pct"] * 100,
            )
            # Merge in a BF-weighted pitching-side projection (B1 fully,
            # B2 partially — see _compute_pitching_projection docstring for
            # why B3/B4 are deliberately left unprojected).
            try:
                pitch_proj = _compute_pitching_projection(season, pitcher_ids, team_pit)
                if pitch_proj.get("philosophy_metrics"):
                    projected.setdefault("philosophy_metrics", {}).update(
                        pitch_proj["philosophy_metrics"])
                    projected["pitching_coverage"] = pitch_proj["coverage"]
                    log.info("Pitching projection merged — BF coverage=%.0f%%",
                             pitch_proj["coverage"] * 100)
            except Exception as exc:
                log.warning("Pitching projection failed: %s", exc)
    except Exception as exc:
        log.warning("Projected profile failed: %s", exc)

    # ── 15. Spray chart data ──────────────────────────────────────────────
    spray_data: dict = {}
    try:
        from pitch_aggregates import pull_angle, spray_angle, spray_zone
        # This team's balls in play only: selecting by batter alone gave a
        # traded hitter's balls for another club to both teams' charts.
        bat_team = np.where(statcast["inning_topbot"] == "Top",
                            statcast["away_team"], statcast["home_team"])
        team_bip = statcast[
            (bat_team == team) & statcast["batter"].isin(hitter_ids) &
            statcast["hc_x"].notna() & statcast["hc_y"].notna()
        ].copy()

        def _event_bucket(ev: str) -> str:
            if ev == "home_run": return "hr"
            if ev == "double":   return "double"
            if ev == "triple":   return "triple"
            if ev == "single":              return "single"
            return "out"

        team_bip["_bucket"] = team_bip["events"].fillna("out").apply(_event_bucket)
        n_bip = len(team_bip)

        # Five zones, all handedness-aware (pitch_aggregates.spray_zone):
        # pull line, pull gap, center, oppo gap, oppo line. The chart used to
        # mix pull/oppo with ABSOLUTE left/right gaps, which cannot be summed
        # across a lineup of both hands, and had pull and oppo reversed.
        SPRAY_ZONES = ("pull_line", "pull_gap", "center", "oppo_gap", "oppo_line")

        def _zone_shares(frame: pd.DataFrame) -> dict[str, float]:
            z = spray_zone(pull_angle(spray_angle(frame["hc_x"], frame["hc_y"]),
                                      frame["stand"]))
            n = int(z.notna().sum())
            return {k: (float((z == k).sum() / n) if n else 0.0) for k in SPRAY_ZONES}

        team_zones = _zone_shares(team_bip)
        # League-wide directional averages (all batters this season)
        lg_zones = _zone_shares(statcast[statcast["hc_x"].notna() & statcast["hc_y"].notna()])

        hr_n     = int((team_bip["_bucket"] == "hr").sum())
        double_n = int((team_bip["_bucket"] == "double").sum())
        triple_n = int((team_bip["_bucket"] == "triple").sum())
        xbh_n    = double_n + triple_n
        s_n   = int((team_bip["_bucket"] == "single").sum())
        stand_cts   = team_bip["stand"].value_counts().to_dict() if "stand" in team_bip.columns else {}
        total_stand = sum(stand_cts.values()) or 1

        # Raw batted-ball points are event-level data, not summary — they go
        # to a sidecar file (data/processed/spray/{TEAM}_{season}.json) that
        # the spray chart loads on demand. The portrait keeps only the
        # scalar zone shares / counts (~1 KB instead of ~120 KB).
        _spray_raw = {
            "hc_x":       [float(x) for x in team_bip["hc_x"].tolist()[:5000]],
            "hc_y":       [float(y) for y in team_bip["hc_y"].tolist()[:5000]],
            "event_type": team_bip["_bucket"].tolist()[:5000],
            "stand":      (team_bip["stand"].tolist()[:5000] if "stand" in team_bip.columns else []),
        }
        try:
            import json as _json
            _spray_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "spray"
            _spray_dir.mkdir(parents=True, exist_ok=True)
            (_spray_dir / f"{team}_{season}.json").write_text(
                _json.dumps(_spray_raw))
        except Exception as _sp_exc:
            log.warning("Spray sidecar write failed: %s", _sp_exc)

        spray_data = {
            **{f"{k}_pct": v for k, v in team_zones.items()},
            **{f"lg_{k}_pct": v for k, v in lg_zones.items()},
            "hr_count":     hr_n,
            "xbh_count":    xbh_n,
            "double_count": double_n,
            "triple_count": triple_n,
            "single_count": s_n,
            "stand_pct":    {k: v/total_stand for k, v in stand_cts.items()},
        }
        log.info("Spray data — %d BIP  HR=%d 2B/3B=%d 1B=%d", n_bip, hr_n, xbh_n, s_n)
    except Exception as exc:
        log.warning("Spray data failed: %s", exc)

    # ── 17. Batter L/R split stats (computed in step 12; attach here) ───────
    for hp in hitter_profiles:
        pid = hp.get("player_id")
        if pid and int(pid) in _split_map:
            hp["splits"] = _split_map[int(pid)]

    # ── 18. Arsenal trajectories ─────────────────────────────────────────────
    arsenal_trajectories: dict = {}
    try:
        from pitch_aggregates import compute_team_arsenal_trajectories
        arsenal_trajectories = compute_team_arsenal_trajectories(
            statcast, pitcher_ids, _get_player_info_map(), min_pitches=30
        )
    except Exception as exc:
        log.warning("Arsenal trajectories failed: %s", exc)

    portrait = {
        "schema_version": PORTRAIT_SCHEMA_VERSION,
        # What this portrait was built AGAINST — pools and gates. See
        # build_fingerprint: the schema catches a change of shape, this
        # catches a change of inputs.
        "build_fingerprint": build_fingerprint(season),
        # Its feature half: what rederive_traits must find unchanged.
        "data_fingerprint":  data_fingerprint(season),
        # Its gate-constant half alone, for readers without the pools.
        "gate_fingerprint":  gate_fingerprint(),
        "team":   team,
        "season": season,
        # Everyone who appeared for this team, most work with the team first.
        # Under MIN_HITTER_PA / MIN_PITCHER_BF a player is in also_appeared:
        # on the roster, without tags or weight.
        "players": {
            "hitters":      sorted(hitter_profiles, key=lambda p: -(p.get("pa") or 0)),
            "starters":     sorted(starters, key=lambda p: -(p.get("bf") or 0)),
            "bullpen_arms": sorted(bullpen_arms, key=lambda p: -(p.get("bf") or 0)),
            "also_appeared": also_appeared,
        },
        # Team-level collective (scores/out-mechanism/leverage-structure) —
        # lives beside the other team-level rollups, not inside players.
        "bullpen_collective": bullpen_profile,
        "team_metrics": {
            "batting":            team_batting_agg,
            "pitching":           team_pitching_agg,
            "power_source_ratio": _power_source_ratio,  # raw HR/(HR+2B+3B) — display only
            "roster_control":     _roster_ctrl,          # raw service-time shares — display only
        },
        "park":        park_result,
        "philosophy":         philosophy,
        "philosophy_metrics": philosophy_metrics,
        "temporal":           temporal_info,
        "spin":        spin_team_summary,
        "tunneling":   {
            "pitcher_level": (
                tun_pitcher_level.to_dict("records")
                if tun_pitcher_level is not None
                else None
            ),
            "team_summary": tunneling_team_summary,
        },
        "data_coverage": coverage,
        "spray_data":          spray_data,
        "projected":           projected,
        "arsenal_trajectories": arsenal_trajectories,
        "team_identity":       team_identity,
    }

    log.info(
        "Portrait complete — team=%s season=%d | coverage=%.0f%% | mode=%s | "
        "hitters=%d starters=%d relievers=%d bullpen_profile=%s",
        team, season, coverage * 100, temporal_info["mode"],
        len(hitter_profiles), len(starters), len(bullpen_arms), bool(bullpen_profile),
    )
    return portrait


# ---------------------------------------------------------------------------
# Batch: list of (team, season) → flat DataFrame
# ---------------------------------------------------------------------------

def portraits_to_dataframe(portraits: list[dict]) -> pd.DataFrame:
    """
    Convert a list of team portraits to a flat summary DataFrame.

    One row per team × season. Columns:
      team, season, data_coverage, temporal_mode,
      offense_primary, offense_conf, offense_hybrid,
      pitching_primary, pitching_conf, pitching_hybrid,
      roster_primary, roster_conf, roster_hybrid,
      A1…C4 (philosophy scores),
      hitter_count, starter_count, bullpen_count,
    """
    rows = []
    for p in portraits:
        if "error" in p:
            rows.append({"team": p["team"], "season": p["season"], "error": p["error"]})
            continue

        row: dict = {
            "team":          p["team"],
            "season":        p["season"],
            "data_coverage": p.get("data_coverage"),
            "temporal_mode": p.get("temporal", {}).get("mode"),
        }

        # Philosophy scores
        phil = p.get("philosophy", {})
        for code, info in phil.get("scores", {}).items():
            row[code] = info.get("score")

        # Dimension confidences
        for dim, conf in phil.get("confidences", {}).items():
            row[f"{dim}_primary"] = conf.get("primary")
            row[f"{dim}_conf"]    = conf.get("display_confidence")
            row[f"{dim}_hybrid"]  = conf.get("hybrid")

        # Roster counts
        row["hitter_count"]  = len(p.get("players", {}).get("hitters", []))
        row["starter_count"] = len(p.get("players", {}).get("starters", []))
        row["bullpen_arms"]  = bool(p.get("bullpen_collective"))

        rows.append(row)

    return pd.DataFrame(rows)
