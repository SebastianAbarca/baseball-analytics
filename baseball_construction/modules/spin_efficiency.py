"""
spin_efficiency.py — active spin (spin efficiency) per pitcher × pitch type.

Spin efficiency is the share of a pitch's spin that is transverse to its
flight (active spin) and so produces Magnus movement; the rest is gyro spin.

Two sources, by season:

  2020+     Baseball Savant's measured active spin (3D spin vector at
            release, from the active-spin leaderboard). Not inferred from
            movement, so seam-shifted wake does not contaminate it.
  pre-2020  A physics estimate from the 9-parameter trajectory fit:
              1. Magnus acceleration = (ax, ay, az + g) minus its component
                 along the mid-flight velocity (removes gravity and drag)
              2. C_L  = 2·m·a_M / (ρ·A·v²), ρ from the home park's elevation
              3. S_T  = inverse of the Sawicki lift curve
                        C_L = 1.5·S (S < 0.1), 0.09 + 0.6·S (S ≥ 0.1)
              4. efficiency = S_T / S, S = r·ω/v from total spin
              5. × a per-pitch-type level factor fitted to Savant 2020–2025,
                 so pre-2020 values read on the same scale as measured ones

            Against Savant measured (2024–26, pitcher × pitch type) this ranks
            at Spearman 0.92 FF, 0.93 CU, 0.83 SL, 0.59 CH — level with
            Savant's own movement-inferred numbers. CH/FS stay weak because
            movement alone cannot separate seam-shifted wake from spin.

Percentiles are ranked WITHIN pitch type: a pooled rank mostly measures how
few sliders a pitcher throws, since gyro spin is what makes a slider.

Pipeline:
  1. compute_spin_efficiency(df)           — pitch-level physics estimate
  2. aggregate_pitcher_pitch_type(df)      — group to pitcher × pitch_type
     (attach_savant_active_spin replaces the estimate with Savant where it exists)
  3. apply_reliability_weighting(agg)      — shrink toward the pitch type's mean
  4. normalize_spin_efficiency(agg)        — percentile 0–100 within pitch type
  5. build_pitcher_spin_profile(df)        — full pipeline in one call
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ingest import (
    STABILIZATION,
    apply_reliability_weight,
    compute_reliability,
    normalize_percentile,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bump when the estimator or its inputs change — cached league pools built by
# an older method are rebuilt instead of served.
SPIN_METHOD_VERSION = 2

# First season Savant publishes measured (spin-based) active spin
MEASURED_FROM_SEASON = 2020

SAVANT_ACTIVE_SPIN_URL = (
    "https://baseballsavant.mlb.com/leaderboard/active-spin"
    "?year={season}_spin-based&min={min_pitches}&hand=&csv=true"
)

# Savant leaderboard column → Statcast pitch_type codes it covers
SAVANT_PITCH_COLUMNS = {
    "active_spin_fourseam": ["FF"],
    "active_spin_sinker":   ["SI"],
    "active_spin_cutter":   ["FC"],
    "active_spin_changeup": ["CH"],
    "active_spin_splitter": ["FS"],
    "active_spin_curve":    ["CU", "KC"],
    "active_spin_slider":   ["SL"],
    "active_spin_sweeper":  ["ST"],
    "active_spin_slurve":   ["SV"],
}

# Ball and air. ρ at sea level, ~70°F; thinned by park elevation.
GRAVITY_FT_S2     = 32.174
BALL_MASS_KG      = 0.145
BALL_RADIUS_M     = 9.125 / (2 * np.pi) * 0.0254
BALL_AREA_M2      = np.pi * BALL_RADIUS_M ** 2
RHO_SEA_LEVEL     = 1.19     # kg/m³
AIR_SCALE_HEIGHT_FT = 27_000
FT_TO_M           = 0.3048
PLATE_FRONT_Y_FT  = 17 / 12

# Home-park elevation (ft). Roofs and weather are ignored; only elevation
# moves air density enough to matter (Coors ≈ 18% thinner).
PARK_ELEVATION_FT = {
    "ARI": 1060, "AZ": 1060, "ATL": 1050, "BAL": 30, "BOS": 20, "CHC": 600,
    "CWS": 600, "CIN": 490, "CLE": 650, "COL": 5200, "DET": 600, "HOU": 40,
    "KC": 750, "LAA": 160, "LAD": 500, "MIA": 10, "MIL": 600, "MIN": 840,
    "NYM": 20, "NYY": 50, "OAK": 20, "ATH": 30, "PHI": 20, "PIT": 730,
    "SD": 20, "SEA": 20, "SF": 0, "STL": 460, "TB": 20, "TEX": 550,
    "TOR": 300, "WSH": 30,
}
DEFAULT_ELEVATION_FT = 300

# Physics estimate → Savant scale, per pitch type: median(Savant measured) /
# median(physics) over pitcher × pitch_type rows (≥ 50 pitches), 2020–2025.
# Each holds within ±5% season to season (SV, 51 rows, is noisier).
# Gyro-heavy pitches need the most: the lift curve is least certain at the
# small transverse spin factors of sliders and cutters. Sinkers need < 1
# because seam-shifted wake adds movement the estimate credits to spin.
# Changes levels only; within-type ranks are unaffected.
PITCH_TYPE_LEVEL = {
    "FF": 1.062, "SI": 0.912, "FC": 1.915, "CH": 1.053, "FS": 1.152,
    "CU": 1.215, "KC": 1.244, "SL": 1.650, "ST": 1.032, "SV": 1.054,
}
PITCH_TYPE_LEVEL_FALLBACK = {"FA": "FF", "CS": "CU", "SC": "CH"}

# Pitch types to exclude from spin efficiency analysis
# (knuckleballs and eephus have near-zero intentional spin)
EXCLUDE_PITCH_TYPES = {"KN", "EP", "PO", "IN", "FO"}

# Minimum pitches to include a pitcher × pitch_type row in any output
MIN_PITCHES_TO_INCLUDE = 10

PHYSICS_COLUMNS = ["release_spin_rate", "vx0", "vy0", "vz0", "ax", "ay", "az"]

# ---------------------------------------------------------------------------
# Step 1 — Pitch-level physics estimate
# ---------------------------------------------------------------------------

def _time_to_y(df: pd.DataFrame, y_ft) -> pd.Series:
    """Time (s) from the y = 50 ft fit origin until the ball reaches y_ft."""
    a = 0.5 * df["ay"]
    b = df["vy0"]
    c = 50.0 - y_ft
    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


def _invert_lift(cl: pd.Series) -> pd.Series:
    """Transverse spin factor S_T from C_L, inverting the Sawicki lift curve."""
    return pd.Series(np.where(cl < 0.15, cl / 1.5, (cl - 0.09) / 0.6), index=cl.index)


def _level_factor(pitch_type: pd.Series) -> pd.Series:
    mapped = pitch_type.map(lambda t: PITCH_TYPE_LEVEL_FALLBACK.get(t, t))
    return mapped.map(PITCH_TYPE_LEVEL).fillna(1.0)


def compute_spin_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics estimate of active spin for every pitch row.

    Adds columns:
      magnus_accel     — Magnus acceleration, ft/s² (gravity and drag removed)
      lift_coeff       — C_L implied by that acceleration
      spin_factor      — S = r·ω/v from total spin
      spin_efficiency  — S_T / S on the Savant scale, clipped to [0, 1]

    Rows missing any trajectory or spin input receive NaN spin_efficiency.
    """
    df = df.copy()

    missing_cols = [c for c in PHYSICS_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"compute_spin_efficiency: missing columns {missing_cols}")

    # Nullable (pd.NA) dtypes break np.where; work in plain float64
    for c in PHYSICS_COLUMNS + ["release_pos_y", "release_extension"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    if "release_pos_y" in df.columns:
        release_y = df["release_pos_y"]
    elif "release_extension" in df.columns:
        release_y = 60.5 - df["release_extension"]
    else:
        release_y = pd.Series(54.0, index=df.index)
    release_y = release_y.fillna(54.0)

    # Velocity at mid-flight (release → front of plate)
    t_mid = (_time_to_y(df, PLATE_FRONT_Y_FT) + _time_to_y(df, release_y)) / 2
    vx = df["vx0"] + df["ax"] * t_mid
    vy = df["vy0"] + df["ay"] * t_mid
    vz = df["vz0"] + df["az"] * t_mid
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    # Magnus acts perpendicular to velocity: drop gravity, then the along-v part
    ax, ay, az = df["ax"], df["ay"], df["az"] + GRAVITY_FT_S2
    along = (ax * vx + ay * vy + az * vz) / speed
    df["magnus_accel"] = np.sqrt(
        (ax - along * vx / speed) ** 2
        + (ay - along * vy / speed) ** 2
        + (az - along * vz / speed) ** 2
    )

    if "home_team" in df.columns:
        elevation = df["home_team"].map(PARK_ELEVATION_FT).fillna(DEFAULT_ELEVATION_FT)
    else:
        elevation = pd.Series(DEFAULT_ELEVATION_FT, index=df.index)
    rho = RHO_SEA_LEVEL * np.exp(-elevation / AIR_SCALE_HEIGHT_FT)

    speed_m = speed * FT_TO_M
    df["lift_coeff"] = (
        df["magnus_accel"] * FT_TO_M
        / (0.5 * rho * BALL_AREA_M2 * speed_m ** 2 / BALL_MASS_KG)
    )
    omega = df["release_spin_rate"] * 2 * np.pi / 60
    df["spin_factor"] = BALL_RADIUS_M * omega / speed_m

    eff = _invert_lift(df["lift_coeff"]) / df["spin_factor"]
    if "pitch_type" in df.columns:
        eff = eff * _level_factor(df["pitch_type"])
    eff = eff.where(df["spin_factor"] > 0)
    df["spin_efficiency"] = eff.clip(0, 1)

    # Exclude pitch types where spin efficiency is not meaningful
    if "pitch_type" in df.columns:
        df.loc[df["pitch_type"].isin(EXCLUDE_PITCH_TYPES), "spin_efficiency"] = np.nan

    valid = df["spin_efficiency"].notna().sum()
    log.info(
        "compute_spin_efficiency — %d pitches with valid spin_efficiency / %d total",
        valid, len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Savant measured active spin
# ---------------------------------------------------------------------------

def fetch_savant_active_spin(season: int, min_pitches: int = 1) -> pd.DataFrame:
    """
    Savant's measured active spin for one season, long format:
    pitcher, pitch_type, active_spin (0–1). Empty before 2020.
    """
    if season < MEASURED_FROM_SEASON:
        return pd.DataFrame(columns=["pitcher", "pitch_type", "active_spin"])

    import requests
    url = SAVANT_ACTIVE_SPIN_URL.format(season=season, min_pitches=min_pitches)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig")
    if "<html" in text[:500].lower():
        raise RuntimeError(f"Savant returned HTML, not CSV, for {season} active spin")

    wide = pd.read_csv(io.StringIO(text))
    rows = []
    for col, codes in SAVANT_PITCH_COLUMNS.items():
        if col not in wide.columns:
            continue
        part = wide[["entity_id", col]].dropna()
        for code in codes:
            rows.append(pd.DataFrame({
                "pitcher":     part["entity_id"].astype("int64"),
                "pitch_type":  code,
                "active_spin": part[col].astype(float) / 100.0,
            }))
    if not rows:
        raise RuntimeError(f"Savant {season} active spin CSV has no pitch columns")
    out = pd.concat(rows, ignore_index=True)
    log.info("Savant active spin %d: %d pitcher × pitch_type values", season, len(out))
    return out


def load_savant_active_spin(season: int, force: bool = False) -> pd.DataFrame:
    """fetch_savant_active_spin, cached to processed/savant_active_spin_{season}.parquet."""
    if season < MEASURED_FROM_SEASON:
        return fetch_savant_active_spin(season)
    processed = Path(__file__).parent / "processed"
    processed.mkdir(exist_ok=True)
    cache = processed / f"savant_active_spin_{season}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    out = fetch_savant_active_spin(season)
    out.to_parquet(cache, index=False)
    return out


# ---------------------------------------------------------------------------
# Step 2 — Aggregate to pitcher × pitch_type
# ---------------------------------------------------------------------------

def aggregate_pitcher_pitch_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level data to pitcher × pitch_type.

    Input df must already have spin_efficiency column (run compute_spin_efficiency first).

    Returns columns:
      pitcher, pitch_type, pitch_count,
      spin_efficiency_raw,   (median of pitch-level physics estimates)
      spin_source,           ("physics" here; attach_savant_active_spin may change it)
      spin_rate_mean,
      velo_mean,
      pfx_x_mean,           (signed horizontal break — positive = arm side)
      pfx_z_mean,           (signed vertical break)
      spin_axis_mean,
    """
    needed = {"spin_efficiency", "pitcher", "pitch_type"}
    if not needed.issubset(df.columns):
        raise ValueError(f"aggregate_pitcher_pitch_type: need columns {needed}")

    valid = df[df["spin_efficiency"].notna()].copy()

    agg_cols = {
        "pitch_count":         ("spin_efficiency",   "count"),
        "spin_efficiency_raw": ("spin_efficiency",   "median"),
        "spin_rate_mean":      ("release_spin_rate", "mean"),
    }

    # Optional columns — include if present
    if "release_speed" in valid.columns:
        agg_cols["velo_mean"] = ("release_speed", "mean")
    if "pfx_x" in valid.columns:
        agg_cols["pfx_x_mean"] = ("pfx_x", "mean")
    if "pfx_z" in valid.columns:
        agg_cols["pfx_z_mean"] = ("pfx_z", "mean")
    if "spin_axis" in valid.columns:
        agg_cols["spin_axis_mean"] = ("spin_axis", "mean")

    agg = valid.groupby(["pitcher", "pitch_type"]).agg(**agg_cols).reset_index()
    agg["spin_source"] = "physics"

    # Drop rows below minimum threshold (too sparse to be meaningful)
    before = len(agg)
    agg = agg[agg["pitch_count"] >= MIN_PITCHES_TO_INCLUDE].copy()
    log.info(
        "aggregate_pitcher_pitch_type — %d pitcher×pitch_type rows "
        "(%d dropped below min=%d pitches)",
        len(agg), before - len(agg), MIN_PITCHES_TO_INCLUDE,
    )
    return agg


def attach_savant_active_spin(agg: pd.DataFrame, active_spin: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the physics estimate with Savant's measured active spin wherever
    Savant has a value. Rows Savant leaves out (below its pitch minimum) keep
    the physics estimate, which PITCH_TYPE_LEVEL puts on the same scale.
    """
    if active_spin is None or active_spin.empty:
        return agg
    agg = agg.merge(active_spin, on=["pitcher", "pitch_type"], how="left")
    has = agg["active_spin"].notna()
    agg.loc[has, "spin_efficiency_raw"] = agg.loc[has, "active_spin"]
    agg.loc[has, "spin_source"] = "savant"
    log.info("attach_savant_active_spin — %d/%d rows from Savant, %d physics",
             has.sum(), len(agg), (~has).sum())
    return agg.drop(columns="active_spin")


# ---------------------------------------------------------------------------
# Step 3 — Reliability weighting
# ---------------------------------------------------------------------------

def apply_reliability_weighting(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Shrink spin_efficiency_raw toward its pitch type's mean based on sample size.

    reliability = min(1.0, pitch_count / 200)
    weighted    = (reliability × raw) + ((1 - reliability) × pitch_type_mean)

    Shrinking toward a pooled mean would pull every slider toward fastball
    levels, so each pitch type shrinks toward its own mean.

    Adds columns:
      reliability             — 0.0–1.0
      league_mean_spin_eff    — the pitch type's mean used for shrinkage
      spin_efficiency_weighted
    """
    agg = agg.copy()

    agg["league_mean_spin_eff"] = (
        agg.groupby("pitch_type")["spin_efficiency_raw"].transform("mean")
    )

    agg["reliability"] = agg["pitch_count"].apply(
        lambda n: compute_reliability(int(n), "spin_efficiency")
    )

    agg["spin_efficiency_weighted"] = agg.apply(
        lambda row: apply_reliability_weight(
            row["spin_efficiency_raw"],
            row["reliability"],
            row["league_mean_spin_eff"],
        ),
        axis=1,
    )

    log.info(
        "apply_reliability_weighting — reliability: min=%.2f  median=%.2f  max=%.2f",
        agg["reliability"].min(),
        agg["reliability"].median(),
        agg["reliability"].max(),
    )
    return agg


# ---------------------------------------------------------------------------
# Step 4 — Percentile normalization
# ---------------------------------------------------------------------------

def normalize_spin_efficiency(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-rank spin_efficiency_weighted within each pitch type.

    Higher spin efficiency = higher percentile (not inverted).
    Adds column: spin_efficiency_pct (0–100).
    """
    agg = agg.copy()
    agg["spin_efficiency_pct"] = (
        agg.groupby("pitch_type")["spin_efficiency_weighted"]
        .transform(lambda s: normalize_percentile(s, invert=False))
    )
    log.info(
        "normalize_spin_efficiency — pct range: %.1f – %.1f",
        agg["spin_efficiency_pct"].min(),
        agg["spin_efficiency_pct"].max(),
    )
    return agg


# ---------------------------------------------------------------------------
# Step 5 — Pitcher-level rollup (across pitch types)
# ---------------------------------------------------------------------------

def build_pitcher_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up from pitcher × pitch_type to pitcher-level.

    Weights each pitch type by pitch_count.
    Returns one row per pitcher with:
      spin_efficiency_weighted (pitch-weighted active spin — display only;
                                mixes pitch types, so do not rank on it)
      spin_efficiency_type_pct (pitch-weighted mean of within-type percentiles)
      spin_efficiency_pct      (spin_efficiency_type_pct ranked across pitchers)
      spin_savant_share        (share of pitches whose value is Savant-measured)
      dominant_pitch_type      (highest-count pitch type)
      pitch_types              (comma-separated list)
      total_pitches
    """
    def weighted_mean(group: pd.DataFrame) -> pd.Series:
        w = group["pitch_count"]
        dominant = group.loc[group["pitch_count"].idxmax(), "pitch_type"]
        types = ",".join(sorted(group["pitch_type"].tolist()))
        return pd.Series({
            "spin_efficiency_weighted": np.average(group["spin_efficiency_weighted"], weights=w),
            "spin_efficiency_type_pct": np.average(group["spin_efficiency_pct"], weights=w),
            "spin_savant_share":        np.average(group["spin_source"].eq("savant"), weights=w),
            "dominant_pitch_type":      dominant,
            "pitch_types":              types,
            "total_pitches":            w.sum(),
        })

    summary = agg.groupby("pitcher").apply(weighted_mean, include_groups=False).reset_index()
    summary["spin_efficiency_pct"] = normalize_percentile(
        summary["spin_efficiency_type_pct"], invert=False
    )
    log.info("build_pitcher_summary — %d pitchers", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _season_of(df: pd.DataFrame) -> int | None:
    if "game_year" in df.columns and df["game_year"].notna().any():
        return int(df["game_year"].mode().iloc[0])
    if "game_date" in df.columns and df["game_date"].notna().any():
        return int(pd.to_datetime(df["game_date"]).dt.year.mode().iloc[0])
    return None


def build_pitcher_spin_profile(
    df: pd.DataFrame,
    season: int | None = None,
    active_spin: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete spin efficiency pipeline.

    Args:
        df:          pitch-level Statcast DataFrame
        season:      used to look up Savant measured active spin (2020+);
                     inferred from game_year / game_date when omitted
        active_spin: pre-loaded Savant values (pitcher, pitch_type,
                     active_spin); skips the lookup when given

    Returns:
        (pitch_type_level, pitcher_level)
        pitch_type_level — one row per pitcher × pitch_type
        pitcher_level    — one row per pitcher (weighted rollup)
    """
    df = compute_spin_efficiency(df)
    agg = aggregate_pitcher_pitch_type(df)

    if active_spin is None:
        season = season if season is not None else _season_of(df)
        if season is not None and season >= MEASURED_FROM_SEASON:
            try:
                active_spin = load_savant_active_spin(season)
            except Exception as exc:
                log.warning("Savant active spin %s unavailable (%s) — "
                            "using the physics estimate for every row", season, exc)
    agg = attach_savant_active_spin(agg, active_spin)

    agg = apply_reliability_weighting(agg)
    agg = normalize_spin_efficiency(agg)
    summary = build_pitcher_summary(agg)
    return agg, summary


def load_league_spin(
    season: int,
    statcast: pd.DataFrame | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    League-wide pitcher spin-efficiency summary for one season, cached to
    parquet. spin_efficiency_pct here is ranked against ALL pitchers in the
    season — the per-team pipeline only ranks within one team's ~30 arms,
    which is not valid for league-percentile claims (archetype scoring).
    """
    _here = Path(__file__).parent
    processed = _here / "processed"
    processed.mkdir(exist_ok=True)
    cache = processed / f"spin_league_{season}.parquet"
    if cache.exists() and not force:
        cached = pd.read_parquet(cache)
        if ("spin_method_version" in cached.columns
                and (cached["spin_method_version"] == SPIN_METHOD_VERSION).all()):
            log.info("spin_league %d: loading from cache", season)
            return cached
        log.info("spin_league %d: cache built by an older method — rebuilding", season)

    if statcast is None:
        from ingest import pull_statcast_season
        statcast = pull_statcast_season(season)

    log.info("spin_league %d: building from Statcast (%d pitches)…",
             season, len(statcast))
    # A forced rebuild re-pulls Savant too: in-season its values move nightly.
    active_spin = None
    if season >= MEASURED_FROM_SEASON:
        try:
            active_spin = load_savant_active_spin(season, force=force)
        except Exception as exc:
            log.warning("Savant active spin %d unavailable (%s) — "
                        "using the physics estimate for every row", season, exc)
            active_spin = pd.DataFrame(columns=["pitcher", "pitch_type", "active_spin"])
    _, summary = build_pitcher_spin_profile(statcast, season=season, active_spin=active_spin)
    if not summary.empty:
        summary["spin_method_version"] = SPIN_METHOD_VERSION
        summary.to_parquet(cache, index=False)
        log.info("spin_league %d: saved %d pitchers to cache", season, len(summary))
    return summary
