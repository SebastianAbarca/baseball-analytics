"""
pitch_mix.py — Per-pitcher pitch-type aggregation and arsenal profile builder.

Tier 2 data layer: aggregates Statcast pitch-by-pitch data into per-pitcher-season
movement and outcome metrics per pitch type, then classifies arsenal structure.

Two outputs:
  1. pitch_mix_{season}.parquet  — long format, one row per pitcher × pitch_type
     Columns: pitcher, season, p_throws, pitch_type, n, usage_pct,
              avg_velo, avg_ivb (pfx_z), avg_hb (pfx_x),
              avg_spin, avg_spin_axis, whiff_pct, gb_pct, csw_pct

  2. build_arsenal_profile(pitcher_id, season) → structured dict
     Used by build_starter_profile as the new primary classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
PROCESSED_DIR = _HERE / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# Minimum pitches of a type to include in profile
MIN_PITCH_TYPE_N   = 50
# Minimum usage % to consider a pitch meaningful
MIN_USAGE_PCT      = 0.05   # 5%
# Minimum total pitches for a qualifying pitcher-season
MIN_PITCHER_PITCHES = 150

# ---------------------------------------------------------------------------
# Pitch family taxonomy
# ---------------------------------------------------------------------------

# Primary family grouping
PITCH_FAMILY: dict[str, str] = {
    "FF": "fastball",      # 4-seam
    "FA": "fastball",      # generic fastball (older data)
    "FT": "fastball",      # 2-seam (legacy label)
    "SI": "fastball",      # sinker
    "FC": "fastball",      # cutter
    "SL": "breaking",      # slider
    "ST": "breaking",      # sweeper (Statcast 2023+)
    "SV": "breaking",      # slurve
    "CU": "breaking",      # curveball
    "KC": "breaking",      # knuckle-curve
    "CS": "breaking",      # slow curve
    "CH": "offspeed",      # changeup
    "FS": "offspeed",      # splitter
    "FO": "offspeed",      # forkball
    "SC": "offspeed",      # screwball
    "KN": "knuckleball",
    "EP": "eephus",
    "PO": "pitchout",
}

# Human-readable sub-type labels used in the profile
PITCH_LABEL: dict[str, str] = {
    "FF": "4-Seam",
    "FA": "4-Seam",
    "FT": "2-Seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle-Curve",
    "CS": "Slow Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",
    "SC": "Screwball",
    "KN": "Knuckleball",
    "EP": "Eephus",
}

# Which pitch codes count toward the fastball family
FASTBALL_CODES  = {"FF", "FA", "FT", "SI", "FC"}
BREAKING_CODES  = {"SL", "ST", "SV", "CU", "KC", "CS"}
OFFSPEED_CODES  = {"CH", "FS", "FO", "SC"}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(season: int) -> Path:
    return PROCESSED_DIR / f"pitch_mix_{season}.parquet"


def _load_statcast(season: int) -> pd.DataFrame:
    """Load Statcast season data via ingest module."""
    import sys
    sys.path.insert(0, str(_HERE))
    from ingest import pull_statcast_season
    return pull_statcast_season(season)


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def build_pitch_mix(season: int, force: bool = False) -> pd.DataFrame:
    """
    Aggregate Statcast pitch-by-pitch into per-pitcher × pitch_type rows.

    Cached to processed/pitch_mix_{season}.parquet.  Pass force=True to
    rebuild even when the cache exists.

    Returns DataFrame with columns:
        pitcher, season, p_throws,
        pitch_type, n, usage_pct,
        avg_velo, avg_ivb, avg_hb,
        avg_spin, avg_spin_axis,
        whiff_pct, gb_pct, csw_pct, swing_pct
    """
    cache = _cache_path(season)
    if cache.exists() and not force:
        log.info("pitch_mix %d: loading from cache", season)
        return pd.read_parquet(cache)

    log.info("pitch_mix %d: building from Statcast…", season)
    sc = _load_statcast(season)

    # Drop irrelevant pitch types (pitchouts, eephus)
    sc = sc[~sc["pitch_type"].isin({"PO", "EP", "SC"})].copy()
    sc = sc[sc["pitch_type"].notna() & (sc["pitch_type"] != "")]

    # Swing / whiff / CSW flags per pitch
    sc["is_swing"]          = sc["description"].isin({
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "hit_into_play",
    })
    sc["is_whiff"]          = sc["description"].isin({
        "swinging_strike", "swinging_strike_blocked", "foul_tip",
    })
    sc["is_called_strike"]  = sc["description"] == "called_strike"
    sc["is_csw"]            = sc["is_whiff"] | sc["is_called_strike"]
    sc["is_gb"]             = sc["bb_type"] == "ground_ball"
    sc["is_bip"]            = sc["bb_type"].isin(
        {"ground_ball", "fly_ball", "line_drive", "popup"}
    )

    grp = sc.groupby(["pitcher", "game_year", "p_throws", "pitch_type"])

    rows = []
    # Per-pitcher totals for usage %
    totals = sc.groupby(["pitcher", "game_year"])["pitch_type"].count().rename("total_n")

    agg = grp.agg(
        n               = ("pitch_type",        "count"),
        avg_velo        = ("release_speed",      "mean"),
        avg_ivb         = ("pfx_z",              "mean"),
        avg_hb          = ("pfx_x",              "mean"),
        avg_spin        = ("release_spin_rate",  "mean"),
        avg_spin_axis   = ("spin_axis",          "mean"),
        n_swing         = ("is_swing",           "sum"),
        n_whiff         = ("is_whiff",           "sum"),
        n_csw           = ("is_csw",             "sum"),
        n_gb            = ("is_gb",              "sum"),
        n_bip           = ("is_bip",             "sum"),
    ).reset_index()

    agg = agg.merge(
        totals.reset_index().rename(columns={"game_year": "game_year"}),
        on=["pitcher", "game_year"],
    )

    agg["usage_pct"]  = agg["n"] / agg["total_n"]
    agg["whiff_pct"]  = np.where(agg["n"] > 0, agg["n_whiff"] / agg["n"], np.nan)
    agg["csw_pct"]    = np.where(agg["n"] > 0, agg["n_csw"]   / agg["n"], np.nan)
    agg["swing_pct"]  = np.where(agg["n"] > 0, agg["n_swing"] / agg["n"], np.nan)
    agg["gb_pct"]     = np.where(agg["n_bip"] > 0, agg["n_gb"] / agg["n_bip"], np.nan)

    # Drop low-volume pitch types
    agg = agg[agg["n"] >= MIN_PITCH_TYPE_N].copy()
    # Drop pitchers below minimum total pitches
    keep = totals[totals >= MIN_PITCHER_PITCHES].reset_index()
    agg = agg.merge(keep[["pitcher", "game_year"]], on=["pitcher", "game_year"])

    # Rename game_year → season for consistency
    agg = agg.rename(columns={"game_year": "season"})

    keep_cols = [
        "pitcher", "season", "p_throws", "pitch_type",
        "n", "usage_pct",
        "avg_velo", "avg_ivb", "avg_hb",
        "avg_spin", "avg_spin_axis",
        "whiff_pct", "gb_pct", "csw_pct", "swing_pct",
    ]
    result = agg[keep_cols].round(
        {"usage_pct": 4, "avg_velo": 2, "avg_ivb": 3, "avg_hb": 3,
         "avg_spin": 1, "avg_spin_axis": 1,
         "whiff_pct": 4, "gb_pct": 4, "csw_pct": 4, "swing_pct": 4}
    )

    result.to_parquet(cache, index=False)
    log.info("pitch_mix %d: saved %d rows to cache", season, len(result))
    return result


def load_pitch_mix(season: int) -> pd.DataFrame:
    """Load pitch mix for a season, building from Statcast if needed."""
    return build_pitch_mix(season)


# ---------------------------------------------------------------------------
# League plate-discipline pool — SwStr-per-velo and first-pitch-strike
# ---------------------------------------------------------------------------

# First-pitch outcomes that count as a strike (called, swung at, or in play)
_FPS_STRIKE_DESCRIPTIONS = {
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "bunt_foul_tip", "missed_bunt",
    "hit_into_play",
}

# Minimum first pitches (≈ PAs) for a stable first-pitch-strike rate
_MIN_FIRST_PITCHES = 50


def load_league_plate_discipline(
    season: int,
    statcast: Optional[pd.DataFrame] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    League-wide per-pitcher plate-discipline pool, cached to parquet.

    Columns (one row per qualifying pitcher):
        swstr_rate          — swinging strikes / all pitches (raw)
        fb_velo             — mean fastball-family release speed (overall mean
                              if the pitcher throws no fastballs)
        swstr_resid         — swstr_rate minus the league linear fit on fb_velo:
                              whiffs beyond what velocity predicts (deception)
        fps_rate            — first-pitch strikes / first pitches (raw)
        swstr_per_velo_pct  — percentile rank of swstr_resid (0–100)
        first_pitch_strike_pct — percentile rank of fps_rate (0–100)

    Percentiles are ranked against all qualifying pitchers in the season, so
    they live in the same league-percentile space as the other archetype
    metrics (TunnelScore_pct, SpinEfficiency_pct, …).
    """
    cache = PROCESSED_DIR / f"plate_discipline_league_{season}.parquet"
    if cache.exists() and not force:
        log.info("plate_discipline_league %d: loading from cache", season)
        return pd.read_parquet(cache)

    if statcast is None:
        statcast = _load_statcast(season)

    log.info("plate_discipline_league %d: building from Statcast…", season)
    sc = statcast[["pitcher", "pitch_number", "description",
                   "release_speed", "pitch_type"]].copy()

    # ── SwStr% over all pitches ────────────────────────────────────────────
    sc["_is_whiff"] = sc["description"].isin(
        {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
    )
    per = sc.groupby("pitcher").agg(
        n_pitches  = ("description", "count"),
        swstr_rate = ("_is_whiff",   "mean"),
    )

    # ── Fastball velocity (overall mean fallback) ──────────────────────────
    fb = sc[sc["pitch_type"].isin(FASTBALL_CODES)]
    fb_velo  = fb.groupby("pitcher")["release_speed"].mean()
    any_velo = sc.groupby("pitcher")["release_speed"].mean()
    per["fb_velo"] = fb_velo.reindex(per.index).fillna(any_velo)

    # ── First-pitch strike rate ────────────────────────────────────────────
    first = sc[sc["pitch_number"] == 1]
    fps = first.groupby("pitcher").agg(
        n_first  = ("description", "count"),
        fps_rate = ("description", lambda s: s.isin(_FPS_STRIKE_DESCRIPTIONS).mean()),
    )
    per = per.join(fps)

    # Qualifying pool
    per = per[(per["n_pitches"] >= MIN_PITCHER_PITCHES)
              & (per["n_first"] >= _MIN_FIRST_PITCHES)
              & per["fb_velo"].notna()].copy()
    if per.empty:
        log.warning("plate_discipline_league %d: no qualifying pitchers", season)
        return per.reset_index()

    # ── Velocity-adjusted whiff: residual off the league linear fit ────────
    slope, intercept = np.polyfit(per["fb_velo"], per["swstr_rate"], 1)
    per["swstr_resid"] = per["swstr_rate"] - (slope * per["fb_velo"] + intercept)

    per["swstr_per_velo_pct"]     = per["swstr_resid"].rank(pct=True) * 100.0
    per["first_pitch_strike_pct"] = per["fps_rate"].rank(pct=True) * 100.0

    out = per.reset_index()
    out.to_parquet(cache, index=False)
    log.info("plate_discipline_league %d: saved %d pitchers "
             "(swstr~velo slope=%.4f/mph)", season, len(out), slope)
    return out


# ---------------------------------------------------------------------------
# Arsenal profile classifier
# ---------------------------------------------------------------------------

def classify_fastball_family(fb_rows: pd.DataFrame) -> dict:
    """
    Given only the fastball-family rows for one pitcher, classify FB family.

    Returns:
        family:        '4-seam' | 'sinker' | 'cutter' | 'multi-FB' | None
        primary_pitch: pitch_type code of dominant FB
        avg_velo:      velo of primary FB
        avg_ivb:       IVB (pfx_z) of primary FB — 'ride' signal
        avg_hb:        HB (pfx_x) of primary FB
        usage_pct:     usage of primary FB
    """
    if fb_rows.empty:
        return {
            "family": None, "primary_pitch": None,
            "avg_velo": None, "avg_ivb": None, "avg_hb": None, "usage_pct": 0.0,
        }

    by_usage = fb_rows.sort_values("usage_pct", ascending=False)
    top = by_usage.iloc[0]
    top_pct = top["usage_pct"]
    top_code = top["pitch_type"]

    # Total fastball usage across all FB types
    total_fb_pct = fb_rows["usage_pct"].sum()

    # Classify based on dominant pitch code + usage concentration
    if top_code in {"FF", "FA"} and top_pct >= 0.30:
        family = "4-seam"
    elif top_code in {"SI", "FT"} and top_pct >= 0.25:
        family = "sinker"
    elif top_code == "FC" and top_pct >= 0.20:
        family = "cutter"
    elif len(fb_rows) >= 2 and total_fb_pct >= 0.40 and top_pct < 0.40:
        # Multiple FB types each meaningful, no single dominant one
        family = "multi-FB"
    elif top_pct >= 0.15:
        # Something is thrown but doesn't clearly dominate — label by code
        code_map = {"FF": "4-seam", "FA": "4-seam", "FT": "sinker",
                    "SI": "sinker", "FC": "cutter"}
        family = code_map.get(top_code, "4-seam")
    else:
        family = None

    return {
        "family":        family,
        "primary_pitch": top_code,
        "avg_velo":      round(float(top["avg_velo"]), 1) if pd.notna(top["avg_velo"]) else None,
        "avg_ivb":       round(float(top["avg_ivb"]), 2) if pd.notna(top["avg_ivb"]) else None,
        "avg_hb":        round(float(top["avg_hb"]), 2) if pd.notna(top["avg_hb"]) else None,
        "usage_pct":     round(float(top_pct), 3),
    }


def classify_out_pitch(non_fb_rows: pd.DataFrame) -> dict:
    """
    Identify the primary out pitch (secondary weapon) from non-fastball pitches.

    Ranks by whiff_pct × usage_pct to reward pitches that miss bats AND are
    thrown enough to matter — avoids elevating a 5% dart with a high whiff rate.

    Returns:
        pitch_type:  code
        family:      'slider' | 'sweeper' | 'curveball' | 'changeup' |
                     'splitter' | 'knuckle-curve' | 'slurve' | None
        avg_velo, avg_ivb, avg_hb, avg_spin_axis
        usage_pct, whiff_pct
        shape:       'lateral' | 'drop' | 'drop-sweep' | 'fade' | 'rise-drop'
    """
    if non_fb_rows.empty:
        return {"pitch_type": None, "family": None, "usage_pct": 0.0,
                "whiff_pct": None, "shape": None,
                "avg_velo": None, "avg_ivb": None, "avg_hb": None, "avg_spin_axis": None}

    rows = non_fb_rows[non_fb_rows["usage_pct"] >= MIN_USAGE_PCT].copy()
    if rows.empty:
        rows = non_fb_rows.copy()

    rows["leverage"] = rows["whiff_pct"].fillna(0) * rows["usage_pct"]
    top = rows.sort_values("leverage", ascending=False).iloc[0]
    code = top["pitch_type"]

    # Sub-type label
    family_map = {
        "SL": "slider", "ST": "sweeper", "SV": "slurve",
        "CU": "curveball", "KC": "knuckle-curve", "CS": "curveball",
        "CH": "changeup", "FS": "splitter", "FO": "forkball",
    }
    family = family_map.get(code, PITCH_LABEL.get(code, code))

    # Shape based on movement quadrant
    ivb = float(top["avg_ivb"]) if pd.notna(top["avg_ivb"]) else 0.0
    hb  = float(top["avg_hb"])  if pd.notna(top["avg_hb"])  else 0.0
    if code in {"CH", "FS", "FO"}:
        shape = "fade"          # arm-side fade / drop
    elif abs(hb) >= 1.0 and abs(ivb) < 0.4:
        shape = "lateral"       # sweeper / pure horizontal break
    elif ivb < -0.6 and abs(hb) < 0.8:
        shape = "drop"          # 12-6 curve
    elif ivb < -0.4 and abs(hb) >= 0.6:
        shape = "drop-sweep"    # diagonal (slurve/knuckle-curve)
    elif abs(hb) >= 0.6 and ivb > -0.4:
        shape = "lateral"       # slider with glove-side cut
    else:
        shape = "drop"

    return {
        "pitch_type":    code,
        "family":        family,
        "usage_pct":     round(float(top["usage_pct"]), 3),
        "whiff_pct":     round(float(top["whiff_pct"]), 3) if pd.notna(top["whiff_pct"]) else None,
        "shape":         shape,
        "avg_velo":      round(float(top["avg_velo"]), 1) if pd.notna(top["avg_velo"]) else None,
        "avg_ivb":       round(float(top["avg_ivb"]), 2) if pd.notna(top["avg_ivb"]) else None,
        "avg_hb":        round(float(top["avg_hb"]), 2) if pd.notna(top["avg_hb"]) else None,
        "avg_spin_axis": round(float(top["avg_spin_axis"]), 0) if pd.notna(top["avg_spin_axis"]) else None,
    }


# tunnel-dependent gate: pitcher's league-wide tunnel score percentile.
# Top quartile = tunneling is a genuinely distinguishing trait, not just
# "throws a fastball and a breaking ball" (the old IVB-delta heuristic
# fired for ~every FF+SL/CU combo and labeled 18/30 rotations tunnel-dependent).
TUNNEL_DEPENDENT_PCT = 75.0


def _approach(
    fb_usage: float,
    all_rows: pd.DataFrame,
    fb_info: dict,
    out_pitch: dict,
    tunnel_pct: Optional[float] = None,
) -> str:
    """
    Determine approach label from usage and tunneling signal.

    tunnel-dependent — league tunnel score percentile >= 75 (Gausman/Webb
                       type: release paths converge, plate locations diverge)
    fastball-first   — FB family > 55% of all pitches
    secondary-led    — best non-FB pitch has higher leverage than all FB pitches
    balanced         — neither clearly dominant (spread across 3+ pitches)
    """
    if tunnel_pct is not None and float(tunnel_pct) >= TUNNEL_DEPENDENT_PCT:
        return "tunnel-dependent"

    if fb_usage >= 0.55:
        return "fastball-first"

    # Check if a non-FB pitch has the highest leverage
    non_fb = all_rows[~all_rows["pitch_type"].isin(FASTBALL_CODES)]
    if not non_fb.empty:
        top_non_fb_lev = (non_fb["whiff_pct"].fillna(0) * non_fb["usage_pct"]).max()
        all_fb_lev     = (
            all_rows[all_rows["pitch_type"].isin(FASTBALL_CODES)]["whiff_pct"].fillna(0)
            * all_rows[all_rows["pitch_type"].isin(FASTBALL_CODES)]["usage_pct"]
        ).max() if not all_rows[all_rows["pitch_type"].isin(FASTBALL_CODES)].empty else 0.0
        if top_non_fb_lev > all_fb_lev * 1.1:
            return "secondary-led"

    return "balanced"


def build_arsenal_profile(
    pitcher_id: int,
    season: int,
    pitch_mix_df: Optional[pd.DataFrame] = None,
    tunnel_pct: Optional[float] = None,
) -> dict:
    """
    Build full arsenal profile for one pitcher-season.

    Args:
        pitcher_id:   MLBAM pitcher ID
        season:       season year
        pitch_mix_df: pre-loaded pitch_mix DataFrame (avoids reload if already cached)
        tunnel_pct:   league-wide tunnel score percentile (0–100) — gates the
                      tunnel-dependent approach label when >= TUNNEL_DEPENDENT_PCT

    Returns structured dict:
    {
        "fastball":   { family, primary_pitch, avg_velo, avg_ivb, avg_hb, usage_pct }
        "out_pitch":  { pitch_type, family, usage_pct, whiff_pct, shape,
                        avg_velo, avg_ivb, avg_hb, avg_spin_axis }
        "supporting": ["CH", "CU"]   # pitches > MIN_USAGE_PCT, not primary FB or out_pitch
        "approach":   "fastball-first" | "secondary-led" | "balanced" | "tunnel-dependent"
        "depth":      "one-pitch" | "two-pitch" | "multi-pitch"
        "knuckleballer": bool
        "display":    "Ride-first 4S · Slider"   # short human-readable label
        "pitch_rows": DataFrame (all pitch type rows for this pitcher-season)
    }

    Returns None if pitcher has no qualifying pitch data for the season.
    """
    if pitch_mix_df is None:
        pitch_mix_df = load_pitch_mix(season)

    rows = pitch_mix_df[
        (pitch_mix_df["pitcher"] == pitcher_id) &
        (pitch_mix_df["season"]  == season)
    ].copy()

    if rows.empty:
        return None

    # Knuckleball check first — completely different profile
    kn_row = rows[rows["pitch_type"] == "KN"]
    if not kn_row.empty and kn_row.iloc[0]["usage_pct"] >= 0.30:
        return {
            "fastball":    {"family": None, "primary_pitch": None, "avg_velo": None,
                            "avg_ivb": None, "avg_hb": None, "usage_pct": 0.0},
            "out_pitch":   {"pitch_type": "KN", "family": "knuckleball",
                            "usage_pct": float(kn_row.iloc[0]["usage_pct"]),
                            "whiff_pct": float(kn_row.iloc[0]["whiff_pct"]) if pd.notna(kn_row.iloc[0]["whiff_pct"]) else None,
                            "shape": "knuckleball", "avg_velo": None,
                            "avg_ivb": None, "avg_hb": None, "avg_spin_axis": None},
            "supporting":  [],
            "approach":    "knuckleball",
            "depth":       "one-pitch",
            "knuckleballer": True,
            "display":     "Knuckleball",
            "pitch_rows":  rows,
        }

    fb_rows  = rows[rows["pitch_type"].isin(FASTBALL_CODES)]
    non_fb   = rows[~rows["pitch_type"].isin(FASTBALL_CODES) &
                    ~rows["pitch_type"].isin({"KN", "EP"})]

    fb_info    = classify_fastball_family(fb_rows)
    out_pitch  = classify_out_pitch(non_fb)
    fb_usage   = float(fb_rows["usage_pct"].sum())
    approach   = _approach(fb_usage, rows, fb_info, out_pitch, tunnel_pct=tunnel_pct)

    # Supporting pitches: anything > MIN_USAGE_PCT not already captured as
    # primary FB or out pitch
    exclude = {fb_info.get("primary_pitch"), out_pitch.get("pitch_type")}
    supporting = sorted(
        rows[
            (rows["usage_pct"] >= MIN_USAGE_PCT) &
            (~rows["pitch_type"].isin(exclude))
        ]["pitch_type"].tolist()
    )

    # Arsenal depth
    meaningful_pitches = rows[rows["usage_pct"] >= MIN_USAGE_PCT]
    n_pitches = len(meaningful_pitches)
    if n_pitches <= 1:
        depth = "one-pitch"
    elif n_pitches == 2:
        depth = "two-pitch"
    else:
        depth = "multi-pitch"

    # Human-readable display label
    fb_label  = PITCH_LABEL.get(fb_info.get("primary_pitch", ""), fb_info.get("family", ""))
    out_label = PITCH_LABEL.get(out_pitch.get("pitch_type", ""), out_pitch.get("family", ""))
    if fb_label and out_label:
        display = f"{fb_label} · {out_label}"
    elif fb_label:
        display = fb_label
    elif out_label:
        display = out_label
    else:
        display = "Unknown"

    return {
        "fastball":      fb_info,
        "out_pitch":     out_pitch,
        "supporting":    supporting,
        "approach":      approach,
        "depth":         depth,
        "knuckleballer": False,
        "display":       display,
        "pitch_rows":    rows,
    }
