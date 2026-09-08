"""
tunneling.py — Pitch path estimation and tunnel score per pitcher.

A good tunnel: two consecutive pitches from different pitch families share
the same location at the hitter's decision point (23 ft from plate) but
arrive at different plate locations. The hitter must commit before knowing
which pitch is coming.

Method:
  1. Solve each pitch's location at TUNNEL_POINT_FT exactly from the Statcast
     kinematics, anchored on the measured plate location.
  2. Pair consecutive pitches within each at-bat (batter × game × at_bat).
  3. Keep only cross-type pairs (different pitch families).
  4. Compute tunnel_distance (separation at 23 ft) and
     plate_divergence (separation at plate).
  5. tunnel_score = plate_divergence_mean / tunnel_distance_mean
     Higher = same tunnel → different destination = good tunnel.
  6. Aggregate to pitcher level; apply reliability; normalize to 0–100.

GAP 2 is closed. It called for "a physics-based ODE solver using Statcast
initial conditions (vx0, vy0, vz0, ax, ay, az)" to replace the linear
interpolation, but no integration is required: Statcast's published
kinematics ARE a constant-acceleration fit, so solving that quadratic in
closed form is not an approximation of the data, it is the data. A Magnus
integration would model a different trajectory than the one Statcast
reports, which is the thing every other column here is derived from.

What the linear version was actually measuring: its tunnel location
correlated 0.998 with release_pos_z. It was a release-consistency metric.
See estimate_tunnel_location for why, and aggregate_tunnel_scores for the
separate aggregation problem.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ingest import (
    apply_reliability_weight,
    compute_reliability,
    normalize_percentile,
)

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
PROCESSED_DIR = _HERE / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TUNNEL_POINT_FT = 23.0   # feet from plate — hitter decision point
PLATE_FRONT_FT  = 17.0 / 12.0   # where plate_x/plate_z are measured
Y_REF_FT        = 50.0   # plane the Statcast kinematics are referenced to

# Pitch family groupings for cross-type pairing
# Pitches within the same family are NOT cross-type
PITCH_FAMILIES: dict[str, str] = {
    "FF": "fastball",
    "FA": "fastball",
    "SI": "fastball",
    "FC": "fastball",   # cutter counts as fastball family
    "FS": "fastball",
    "FO": "fastball",
    "CH": "offspeed",
    "SC": "offspeed",
    "CU": "breaking",
    "KC": "breaking",
    "SL": "breaking",
    "ST": "breaking",
    "SV": "breaking",
    "CS": "breaking",
}

# Minimum cross-type pitch pairs needed per pitcher for a reliable score
RELIABILITY_THRESHOLD = 150   # matches STABILIZATION['tunnel_score']
MIN_PAIRS_TO_INCLUDE  = 10    # below this → exclude from output


# ---------------------------------------------------------------------------
# Step 1 — Estimate location at tunnel point
# ---------------------------------------------------------------------------

def _time_to_plane(y_target: float, vy0, ay):
    """
    Seconds from the Statcast y=50 ft reference plane to a given distance from
    the plate. Smaller positive root: the ball crosses each plane once going
    forward. NaN where the quadratic has no real solution.
    """
    a = 0.5 * ay
    disc = vy0 ** 2 - 4 * a * (Y_REF_FT - y_target)
    t = np.where(disc >= 0, (-vy0 - np.sqrt(np.clip(disc, 0, None))) / (2 * a), np.nan)
    return np.where(t >= 0, t, np.nan)


def estimate_tunnel_location(
    df: pd.DataFrame,
    distance: float = TUNNEL_POINT_FT,
) -> pd.DataFrame:
    """
    Each pitch's (x, z) at `distance` feet from the plate, solved exactly from
    the Statcast kinematics.

    This replaces a linear interpolation of `pfx` along the path, which was
    wrong twice over. `pfx_x`/`pfx_z` are stored in FEET and the old code
    divided them by 12 as though they were inches, and break does not
    accumulate linearly along the path — it goes as t², so at the decision
    point only 29.7% of a pitch's break has happened, not the 57.5% a linear
    reading implies. Together those understated the movement term by 6.2x,
    leaving it at 0.45 inches against a typical 1.72 inches of release-point
    scatter. The score was measuring how consistently a pitcher releases the
    ball, not how well he tunnels.

    Anchored on plate_x/plate_z and propagated BACKWARD, rather than forward
    from the release point. plate_x/plate_z are measured; release_pos is
    itself an extrapolation, and pairing it with vx0/vy0/vz0 — which are
    referenced to the y=50 ft plane, not to release — mixes a position from
    one plane with a velocity from another. Anchoring at the plate also makes
    tunnel_distance and plate_divergence two readings of one trajectory
    rather than two independent estimates.

    No ODE solver is needed to close GAP 2: Statcast's own fit IS a
    constant-acceleration model, so solving its quadratic exactly is not an
    approximation of the data — it is the data.

    Adds columns:
      tunnel_x, tunnel_z
    """
    df = df.copy()

    required = ["plate_x", "plate_z", "vx0", "vy0", "ax", "ay", "az", "vz0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"estimate_tunnel_location: missing columns {missing}")

    vy0, ay = df["vy0"].to_numpy(), df["ay"].to_numpy()
    t_plate = _time_to_plane(PLATE_FRONT_FT, vy0, ay)
    t_pt    = _time_to_plane(distance,       vy0, ay)

    # x(t) = x50 + vx0·t + ½·ax·t², so the displacement between the two planes
    # needs no knowledge of x50 — it cancels.
    dt, dt2 = t_pt - t_plate, t_pt ** 2 - t_plate ** 2
    df["tunnel_x"] = df["plate_x"] + df["vx0"] * dt + 0.5 * df["ax"] * dt2
    df["tunnel_z"] = df["plate_z"] + df["vz0"] * dt + 0.5 * df["az"] * dt2

    valid = df["tunnel_x"].notna().sum()
    log.info(
        "estimate_tunnel_location — %d / %d pitches have tunnel coords",
        valid, len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Step 2 — Build sequential cross-type pitch pairs
# ---------------------------------------------------------------------------

def _pitch_family(pitch_type: str) -> str:
    return PITCH_FAMILIES.get(pitch_type, "other")


def build_pitch_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find consecutive pitch pairs within each at-bat where the two pitches
    belong to different pitch families.

    Requires columns:
      pitcher, game_pk, at_bat_number, pitch_number,
      pitch_type, tunnel_x, tunnel_z, plate_x, plate_z

    Returns one row per cross-type consecutive pair with:
      pitcher, game_pk, at_bat_number,
      pitch_type_a, pitch_type_b,
      family_a, family_b,
      tunnel_distance   — Euclidean distance at tunnel point (ft)
      plate_divergence  — Euclidean distance at plate (ft)
      tunnel_ratio      — plate_divergence / (tunnel_distance + 0.01)
    """
    needed = {
        "pitcher", "game_pk", "at_bat_number", "pitch_number",
        "pitch_type", "tunnel_x", "tunnel_z", "plate_x", "plate_z",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"build_pitch_pairs: missing columns {missing}")

    valid = df.dropna(
        subset=["pitch_type", "tunnel_x", "tunnel_z", "plate_x", "plate_z"]
    ).copy()
    valid["pitch_family"] = valid["pitch_type"].map(_pitch_family)

    # Sort so consecutive rows within each at-bat are adjacent
    valid = valid.sort_values(
        ["pitcher", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    # Vectorised lag-1 shift — much faster than groupby.apply
    prev = valid.shift(1)

    # Consecutive pair is valid only when both rows share the same at-bat
    same_ab = (
        (valid["pitcher"]       == prev["pitcher"])
        & (valid["game_pk"]     == prev["game_pk"])
        & (valid["at_bat_number"] == prev["at_bat_number"])
    )

    pairs = pd.DataFrame({
        "pitcher":        valid.loc[same_ab, "pitcher"].values,
        "game_pk":        valid.loc[same_ab, "game_pk"].values,
        "at_bat_number":  valid.loc[same_ab, "at_bat_number"].values,
        "pitch_type_a":   prev.loc[same_ab,  "pitch_type"].values,
        "pitch_type_b":   valid.loc[same_ab, "pitch_type"].values,
        "family_a":       prev.loc[same_ab,  "pitch_family"].values,
        "family_b":       valid.loc[same_ab, "pitch_family"].values,
        "tunnel_x_a":     prev.loc[same_ab,  "tunnel_x"].values,
        "tunnel_z_a":     prev.loc[same_ab,  "tunnel_z"].values,
        "tunnel_x_b":     valid.loc[same_ab, "tunnel_x"].values,
        "tunnel_z_b":     valid.loc[same_ab, "tunnel_z"].values,
        "plate_x_a":      prev.loc[same_ab,  "plate_x"].values,
        "plate_z_a":      prev.loc[same_ab,  "plate_z"].values,
        "plate_x_b":      valid.loc[same_ab, "plate_x"].values,
        "plate_z_b":      valid.loc[same_ab, "plate_z"].values,
    })

    # Cross-family filter
    cross = pairs[
        (pairs["family_a"] != pairs["family_b"])
        & (pairs["family_a"] != "other")
        & (pairs["family_b"] != "other")
    ].copy()

    cross["tunnel_distance"] = np.sqrt(
        (cross["tunnel_x_a"] - cross["tunnel_x_b"]) ** 2
        + (cross["tunnel_z_a"] - cross["tunnel_z_b"]) ** 2
    )
    cross["plate_divergence"] = np.sqrt(
        (cross["plate_x_a"] - cross["plate_x_b"]) ** 2
        + (cross["plate_z_a"] - cross["plate_z_b"]) ** 2
    )
    cross["tunnel_ratio"] = cross["plate_divergence"] / (cross["tunnel_distance"] + 0.01)

    log.info(
        "build_pitch_pairs — %d cross-type pairs from %d pitches",
        len(cross), len(valid),
    )
    return cross


# ---------------------------------------------------------------------------
# Step 3 — Aggregate to pitcher level
# ---------------------------------------------------------------------------

def aggregate_tunnel_scores(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cross-type pairs to one row per pitcher.

    The score is a RATIO OF MEANS, not the mean of the per-pair ratio.

    tunnel_ratio has a near-zero denominator by construction — 0.73% of pairs
    tunnel inside an inch, where the ratio runs to 115 — so its mean is set by
    a handful of flukes rather than by how the pitcher generally works.
    Measured by splitting each pitcher's season in half and correlating the
    two halves across 492 pitchers with 150+ pairs:

        mean of the ratio   r = 0.524     <- what this used to return
        median of the ratio r = 0.717
        ratio of the means  r = 0.765     <- what it returns now

    The old aggregate also correlates only 0.60 with the other two, so it was
    not a noisier reading of the same thing; it was reading something else.

    Returns:
      pitcher, pair_count,
      tunnel_distance_mean,    (lower = tighter tunnel)
      plate_divergence_mean,   (higher = more plate separation)
      tunnel_ratio_mean,       (retained for diagnostics only)
      tunnel_score_raw         (= plate_divergence_mean / tunnel_distance_mean)
    """
    if len(pairs) == 0:
        return pd.DataFrame(
            columns=["pitcher", "pair_count", "tunnel_distance_mean",
                     "plate_divergence_mean", "tunnel_ratio_mean", "tunnel_score_raw"]
        )

    agg = pairs.groupby("pitcher").agg(
        pair_count           =("tunnel_ratio",    "count"),
        tunnel_distance_mean =("tunnel_distance", "mean"),
        plate_divergence_mean=("plate_divergence","mean"),
        tunnel_ratio_mean    =("tunnel_ratio",    "mean"),
    ).reset_index()

    agg["tunnel_score_raw"] = (
        agg["plate_divergence_mean"] / agg["tunnel_distance_mean"].clip(lower=1e-6)
    )

    # Drop pitchers below minimum pair threshold
    before = len(agg)
    agg = agg[agg["pair_count"] >= MIN_PAIRS_TO_INCLUDE].copy()
    log.info(
        "aggregate_tunnel_scores — %d pitchers (%d dropped below min=%d pairs)",
        len(agg), before - len(agg), MIN_PAIRS_TO_INCLUDE,
    )
    return agg


# ---------------------------------------------------------------------------
# Step 4 — Reliability weighting
# ---------------------------------------------------------------------------

def apply_tunnel_reliability(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Shrink tunnel_score_raw toward the league mean.

    reliability = min(1.0, pair_count / 150)
    Adds: reliability, league_mean_tunnel, tunnel_score_weighted
    """
    agg = agg.copy()

    league_mean = agg["tunnel_score_raw"].mean()
    agg["league_mean_tunnel"] = league_mean

    agg["reliability"] = agg["pair_count"].apply(
        lambda n: compute_reliability(int(n), "tunnel_score")
    )

    agg["tunnel_score_weighted"] = agg.apply(
        lambda row: apply_reliability_weight(
            row["tunnel_score_raw"],
            row["reliability"],
            league_mean,
        ),
        axis=1,
    )

    log.info(
        "apply_tunnel_reliability — league_mean=%.3f | "
        "reliability: min=%.2f  median=%.2f  max=%.2f",
        league_mean,
        agg["reliability"].min(),
        agg["reliability"].median(),
        agg["reliability"].max(),
    )
    return agg


# ---------------------------------------------------------------------------
# Step 5 — Percentile normalization
# ---------------------------------------------------------------------------

def normalize_tunnel_scores(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile rank tunnel_score_weighted within the pitcher pool.

    Higher tunnel ratio = better tunneling = higher percentile.
    Adds: tunnel_score_pct (0–100).
    """
    agg = agg.copy()
    agg["tunnel_score_pct"] = normalize_percentile(
        agg["tunnel_score_weighted"], invert=False
    )
    log.info(
        "normalize_tunnel_scores — pct range: %.1f – %.1f",
        agg["tunnel_score_pct"].min(),
        agg["tunnel_score_pct"].max(),
    )
    return agg


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_tunnel_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full tunneling pipeline: pitch-level Statcast → pitcher tunnel scores.

    Args:
        df: pitch-level Statcast DataFrame

    Returns:
        One row per pitcher with tunnel metrics and tunnel_score_pct.
    """
    df   = estimate_tunnel_location(df)
    pairs = build_pitch_pairs(df)
    agg   = aggregate_tunnel_scores(pairs)

    if agg.empty:
        log.warning("build_tunnel_profile: no valid pitcher-level tunnel data")
        return agg

    agg = apply_tunnel_reliability(agg)
    agg = normalize_tunnel_scores(agg)
    return agg


def load_league_tunnel(
    season: int,
    statcast: pd.DataFrame | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    League-wide pitcher tunnel profile for one season, cached to parquet.

    Unlike per-team build_tunnel_profile calls, this ranks tunnel_score_pct
    against ALL pitchers in the season — required for any claim like
    "this pitcher is a top-quartile tunneler" (approach labels, P2 scoring,
    team-level tunnel percentiles).

    Args:
        season:   season year
        statcast: pre-loaded full-season Statcast DataFrame (loaded via
                  ingest.pull_statcast_season if omitted)
        force:    rebuild even when the cache exists
    """
    cache = PROCESSED_DIR / f"tunnel_league_{season}.parquet"
    if cache.exists() and not force:
        log.info("tunnel_league %d: loading from cache", season)
        return pd.read_parquet(cache)

    if statcast is None:
        from ingest import pull_statcast_season
        statcast = pull_statcast_season(season)

    log.info("tunnel_league %d: building from Statcast (%d pitches)…",
             season, len(statcast))
    agg = build_tunnel_profile(statcast)
    if not agg.empty:
        agg.to_parquet(cache, index=False)
        log.info("tunnel_league %d: saved %d pitchers to cache", season, len(agg))
    return agg
