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

from ingest import (
    pull_statcast_range,
    pull_statcast_season,
    pull_fg_batting,
    pull_fg_pitching,
    load_chadwick,
    compute_reliability,
    apply_reliability_weight,
    normalize_percentile,
    PROCESSED_DIR,
)
from database import query_batting, query_pitching, season_batting_seeded
from spin_efficiency import build_pitcher_spin_profile
from tunneling import build_tunnel_profile
from park_effects import build_park_profile
from philosophy import build_philosophy_summary
from hitter_archetypes import build_hitter_profile
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

# Minimum PAs for a hitter to be included in team roster
MIN_HITTER_PA    = 30
# Minimum BF for a pitcher to be classified
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
    "whiff_percent":    "SwStr_pct",
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
    "whiff_pct":        "SwStr_pct",
    "chase_pct":        "OSwing_pct",
    "avg":              "AVG",
    "obp":              "OBP",
    "iso":              "ISO",
    "sb":               "SB",
    "cs":               "CS",
}

PITCHING_COL_MAP: dict[str, str] = {
    # Savant column names
    "k_percent":        "K_pct_pitch",
    "bb_percent":       "BB_pct_pitch",
    "hard_hit_percent": "HardHit_allowed",
    "xwoba":            "xwOBA_allowed",
    "pa":               "BF",
    "whiff_percent":    "SwStr_pct_pitch",
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
    "whiff_pct":        "SwStr_pct_pitch",
    "xwoba_allowed":    "xwOBA_allowed",
    "fb_velocity":      "avg_velo",
    "gs":               "GS",
    "bf":               "BF",
}

# Internal normalized key → pitcher_archetypes.py expected key
_PITCHER_KEY_REMAP: dict[str, str] = {
    "K_pct_pitch_pct":     "K_pct_pct",
    "SwStr_pct_pitch_pct": "SwStr_pct_pct",
    "BB_pct_pitch_pct":    "BB_pct_pct",
}


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

    Logic:
      inning_topbot == 'Top'  → away team batting → pitcher is home_team
      inning_topbot == 'Bot'  → home team batting → pitcher is away_team
    """
    sc = statcast.copy()

    # Hitters: rows where batting_team matches
    bat_home = sc[(sc["inning_topbot"] == "Bot") & (sc["home_team"] == team)]
    bat_away = sc[(sc["inning_topbot"] == "Top") & (sc["away_team"] == team)]
    hitter_ids: set[int] = set(
        pd.concat([bat_home, bat_away])["batter"].dropna().astype(int).unique()
    )

    # Pitchers: rows where pitching_team matches
    pit_home = sc[(sc["inning_topbot"] == "Top") & (sc["home_team"] == team)]
    pit_away = sc[(sc["inning_topbot"] == "Bot") & (sc["away_team"] == team)]
    pitcher_ids: set[int] = set(
        pd.concat([pit_home, pit_away])["pitcher"].dropna().astype(int).unique()
    )

    log.info(
        "Team %s — found %d hitters, %d pitchers in Statcast slice",
        team, len(hitter_ids), len(pitcher_ids),
    )
    return hitter_ids, pitcher_ids


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

def _normalize_batting(batting_full: pd.DataFrame, team_bat: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-rank batting metrics within the full-season pool, then
    attach those ranks to the team subset.

    Returns team_bat with _pct columns appended.
    """
    if batting_full.empty or team_bat.empty:
        return team_bat

    # Rename Savant columns to internal names
    full = batting_full.rename(columns=BATTING_COL_MAP)
    team = team_bat.rename(columns=BATTING_COL_MAP)

    metric_invert: dict[str, bool] = {
        "K_pct":  True,   # lower K = better contact → invert
    }

    pct_cols: dict[str, str] = {}
    for metric in BATTING_COL_MAP.values():
        if metric not in full.columns:
            continue
        invert = metric_invert.get(metric, False)
        pct_series = normalize_percentile(full[metric], invert=invert)
        col_name = f"{metric}_pct"
        full[col_name] = pct_series
        pct_cols[metric] = col_name

    # Attach pct columns to the team subset (by key_mlbam)
    pct_col_list = list(pct_cols.values())
    merge_cols = ["key_mlbam"] + [c for c in pct_col_list if c in full.columns]
    team = team.merge(
        full[merge_cols],
        on="key_mlbam",
        how="left",
        suffixes=("", "_pct_drop"),
    )
    team.drop(columns=[c for c in team.columns if c.endswith("_pct_drop")], inplace=True)

    return team


def _normalize_pitching(pitching_full: pd.DataFrame, team_pit: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-rank pitching metrics within the full-season pool.
    """
    if pitching_full.empty or team_pit.empty:
        return team_pit

    full = pitching_full.rename(columns=PITCHING_COL_MAP)
    team = team_pit.rename(columns=PITCHING_COL_MAP)

    metric_invert: dict[str, bool] = {
        "BB_pct_pitch":    True,
        "HardHit_allowed": True,
        "xwOBA_allowed":   True,
        "Barrel_allowed":  True,
    }

    pct_cols: dict[str, str] = {}
    for metric in PITCHING_COL_MAP.values():
        if metric not in full.columns:
            continue
        invert = metric_invert.get(metric, False)
        pct_series = normalize_percentile(full[metric], invert=invert)
        col_name = f"{metric}_pct"
        full[col_name] = pct_series
        pct_cols[metric] = col_name

    pct_col_list = list(pct_cols.values())
    merge_cols = ["key_mlbam"] + [c for c in pct_col_list if c in full.columns]
    team = team.merge(
        full[merge_cols],
        on="key_mlbam",
        how="left",
        suffixes=("", "_pct_drop"),
    )
    team.drop(columns=[c for c in team.columns if c.endswith("_pct_drop")], inplace=True)

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

    pct_cols = [c for c in team_bat.columns if c.endswith("_pct")]
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

    pct_cols = [c for c in team_pit.columns if c.endswith("_pct")]
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

def _classify_hitters(team_bat: pd.DataFrame) -> list[dict]:
    """
    Build hitter profile for each player in team_bat.
    Requires percentile columns (_pct suffix) to be present.
    Returns list of profile dicts from build_hitter_profile().
    """
    profiles = []
    for _, row in team_bat.iterrows():
        player_id = row.get("key_mlbam")
        if pd.isna(player_id):
            continue

        # Build metrics dict from _pct columns.
        # Populate both the stripped key AND the original key so that
        # archetype functions which use either convention get a hit.
        # e.g. ISO_pct → both "ISO" (for TTO/Complete thresholds)
        #                  AND "ISO_pct" (for spectrum power keys)
        metrics: dict[str, float] = {}
        for col in team_bat.columns:
            if col.endswith("_pct"):
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    fval = float(val)
                    metrics[col.removesuffix("_pct")] = fval  # K_pct_pct→K_pct, ISO_pct→ISO
                    metrics[col] = fval                        # also keep ISO_pct, AVG_pct

        # Raw sprint speed for Speed modifier
        sprint_raw = row.get("sprint_speed") or row.get("Sprint Speed")
        sprint_raw = float(sprint_raw) if sprint_raw is not None and not pd.isna(sprint_raw) else None

        # SB/CS — available from BRef merge; Savant-only falls back to 0
        profile = build_hitter_profile(
            player_id=int(player_id),
            metrics=metrics,
            sprint_speed_raw=sprint_raw,
            sb=int(row.get("SB", 0) or 0),
            cs=int(row.get("CS", 0) or 0),
            opportunities=int(row.get("1B", 0) or 0)
            + int(row.get("BB", 0) or 0)
            + int(row.get("HBP", 0) or 0),
        )
        info = _get_player_info_map().get(int(player_id), {})
        profile["name"] = info.get("name")
        profile["age"]  = info.get("age")
        profile["pa"]   = int(row.get("PA") or row.get("pa") or 0)
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
) -> tuple[list[dict], dict]:
    """
    Classify starters individually (build_starter_profile per pitcher) and
    build one collective bullpen profile (build_bullpen_profile for the team).

    Returns (starters_list, bullpen_profile_dict).
    """
    starters: list[dict] = []

    if team_pit.empty:
        return starters, {}

    # Detect BF column for starter vs. bullpen split
    bf_col = next(
        (c for c in ["BF", "bf", "batters_faced", "n", "TBF"] if c in team_pit.columns),
        None,
    )

    bullpen_metrics_acc: list[dict[str, float]] = []
    bullpen_weights: list[float] = []

    for _, row in team_pit.iterrows():
        player_id = row.get("key_mlbam")
        if pd.isna(player_id):
            continue
        player_id = int(player_id)

        bf = int(row.get(bf_col, 0) or 0) if bf_col else 0
        if bf < MIN_PITCHER_BF:
            continue

        # Build metrics dict from _pct columns (strip trailing _pct for key names)
        metrics: dict[str, float] = {}
        for col in team_pit.columns:
            if col.endswith("_pct"):
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    metrics[col] = float(val)   # keep full _pct name for archetypes

        # Attach spin efficiency per pitcher
        if spin_pitcher is not None and "pitcher" in spin_pitcher.columns:
            spin_row = spin_pitcher[spin_pitcher["pitcher"] == player_id]
            if not spin_row.empty:
                se = spin_row.iloc[0].get("weighted_spin_efficiency")
                if se is not None and not np.isnan(float(se or np.nan)):
                    metrics["spin_efficiency_pct"] = float(se)

        # Attach tunneling per pitcher
        if tunneling_pitcher is not None and "pitcher" in tunneling_pitcher.columns:
            tun_row = tunneling_pitcher[tunneling_pitcher["pitcher"] == player_id]
            if not tun_row.empty:
                tr = tun_row.iloc[0].get("tunnel_score_pct")
                if tr is not None and not np.isnan(float(tr or np.nan)):
                    metrics["tunnel_score_pct"] = float(tr)

        # Remap internal keys to what pitcher_archetypes.py expects
        for old_key, new_key in _PITCHER_KEY_REMAP.items():
            if old_key in metrics:
                metrics[new_key] = metrics[old_key]

        # Use GS from BRef when available; fall back to BF threshold for Savant-only data
        gs = int(row.get("GS", 0) or 0)
        if gs > 0:
            is_starter = gs >= 5
        else:
            is_starter = bf >= STARTER_MIN_BF * 3  # ~150 BF ≈ short-season proxy

        if is_starter:
            profile = build_starter_profile(player_id, metrics)
            info = _get_player_info_map().get(player_id, {})
            profile["name"] = info.get("name")
            profile["age"]  = info.get("age")
            profile["bf"]   = bf
            starters.append(profile)
        else:
            bullpen_metrics_acc.append(metrics)
            bullpen_weights.append(float(bf))

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

    return starters, bullpen_profile


# ---------------------------------------------------------------------------
# Step 7 — Philosophy metric assembly
# ---------------------------------------------------------------------------

def _build_philosophy_metrics(
    team_batting_agg: dict[str, Optional[float]],
    team_pitching_agg: dict[str, Optional[float]],
    park: Optional[dict],
    spin_team: Optional[dict],
    tunneling_team: Optional[dict],
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

    # Caller-inverted keys expected by philosophy.py
    metrics["K_inv_pct"]             = _inv("K_pct_pct")
    metrics["Sprint_inv_pct"]        = _inv("sprint_speed_pct")
    metrics["PitchesPerPA_inv_pct"]  = _inv("PitchesPerPA_pct")
    metrics["BB_inv_pct"]            = _inv("BB_pct_pct")
    metrics["BB_inv_pct"]            = _inv("BB_pct_pct")

    # ── Pitching metrics ──────────────────────────────────────────────────
    for key, val in team_pitching_agg.items():
        if key not in metrics:
            metrics[key] = val

    # Aliases for case-convention mismatches between internal and philosophy keys
    metrics["SprintSpeed_pct"] = metrics.get("sprint_speed_pct")
    metrics["AvgVelo_pct"]     = metrics.get("avg_velo_pct")

    # Rename for philosophy dimension B (team-level pitching keys)
    for raw, phil in [
        ("K_pct_pitch_pct",    "TeamK_pct_pct"),
        ("SwStr_pct_pitch_pct","SwStr_pct_pct"),
        ("HardHit_allowed_pct","HardHit_allowed_pct"),
    ]:
        if raw in metrics and phil not in metrics:
            metrics[phil] = metrics[raw]

    # Inverted pitching keys
    metrics["BB_pitch_inv_pct"] = _inv("BB_pct_pitch_pct")

    # ── Park factor ───────────────────────────────────────────────────────
    if park and "pitcher_friendly" in park:
        metrics["ParkPitcherFriendly_pct"] = park["pitcher_friendly"]

    # ── Spin efficiency (B3) ──────────────────────────────────────────────
    if spin_team and "weighted_spin_efficiency_pct" in spin_team:
        metrics["SpinEfficiency_pct"] = spin_team["weighted_spin_efficiency_pct"]

    # ── CSW (called strike + whiff) — approximate from SwStr if CSW absent ─
    if "CSW_pct_pct" not in metrics and "SwStr_pct_pct" in metrics:
        metrics["CSW_pct_pct"] = metrics["SwStr_pct_pct"]

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
# Main entry point
# ---------------------------------------------------------------------------

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
    # Fast path: read from Supabase (pre-seeded full 30-team pool)
    # Fallback: pull from BRef/Savant API if season not seeded
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

    team_bat_raw = batting_full[batting_full["key_mlbam"].isin(hitter_ids)].copy()
    team_pit_raw = pitching_full[pitching_full["key_mlbam"].isin(pitcher_ids)].copy()

    team_bat = _normalize_batting(batting_full, team_bat_raw)
    team_pit = _normalize_pitching(pitching_full, team_pit_raw)

    # ── 4. Spin efficiency ────────────────────────────────────────────────
    pitcher_sc = team_sc[team_sc["pitcher"].isin(pitcher_ids)].copy()
    spin_pitch_level: Optional[pd.DataFrame] = None
    spin_pitcher_level: Optional[pd.DataFrame] = None
    try:
        if not pitcher_sc.empty:
            spin_pitch_level, spin_pitcher_level = build_pitcher_spin_profile(pitcher_sc)
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
    spin_team_summary: Optional[dict] = None
    if spin_pitcher_level is not None and not spin_pitcher_level.empty:
        se_col = "weighted_spin_efficiency"
        if se_col in spin_pitcher_level.columns:
            valid = spin_pitcher_level[se_col].dropna()
            if len(valid) > 0:
                spin_team_summary = {
                    "mean_spin_efficiency":            float(valid.mean()),
                    "weighted_spin_efficiency_pct":    float(
                        normalize_percentile(valid).mean()
                    ),
                    "pitcher_count":                   len(valid),
                }

    # ── 10. Philosophy metrics assembly ───────────────────────────────────
    philosophy_metrics = _build_philosophy_metrics(
        team_batting_agg,
        team_pitching_agg,
        park_result,
        spin_team_summary,
        None,   # tunneling team summary not yet aggregated
    )

    # ── 11. Philosophy scoring ────────────────────────────────────────────
    philosophy = build_philosophy_summary(team, season, philosophy_metrics)

    # ── 12. Hitter archetypes ─────────────────────────────────────────────
    hitter_profiles = _classify_hitters(team_bat)

    # ── 13. Pitcher archetypes ────────────────────────────────────────────
    starters, bullpen_profile = _classify_pitchers(
        team, season, team_pit, spin_pitcher_level, tun_pitcher_level
    )

    # ── 14. Data coverage ─────────────────────────────────────────────────
    coverage = _data_coverage(philosophy_metrics)

    portrait = {
        "team":   team,
        "season": season,
        "players": {
            "hitters":         hitter_profiles,
            "starters":        starters,
            "bullpen_profile": bullpen_profile,
        },
        "team_metrics": {
            "batting":  team_batting_agg,
            "pitching": team_pitching_agg,
        },
        "park":        park_result,
        "philosophy":  philosophy,
        "temporal":    temporal_info,
        "spin":        spin_team_summary,
        "tunneling":   {
            "pitcher_level": (
                tun_pitcher_level.to_dict("records")
                if tun_pitcher_level is not None
                else None
            )
        },
        "data_coverage": coverage,
    }

    log.info(
        "Portrait complete — team=%s season=%d | coverage=%.0f%% | mode=%s | "
        "hitters=%d starters=%d bullpen_profile=%s",
        team, season, coverage * 100, temporal_info["mode"],
        len(hitter_profiles), len(starters), bool(bullpen_profile),
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
        row["bullpen_arms"]  = bool(p.get("players", {}).get("bullpen_profile"))

        rows.append(row)

    return pd.DataFrame(rows)
