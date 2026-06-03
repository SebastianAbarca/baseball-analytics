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
    pull_batting_history,
    pull_pitching_history,
    pull_bwar,
    load_chadwick,
    compute_reliability,
    apply_reliability_weight,
    normalize_percentile,
    PROCESSED_DIR,
)
from database import (
    query_batting, query_pitching,
    season_batting_seeded, query_debut_seasons,
    query_seeded_seasons,
)
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
    # Luck delta: xwOBA − wOBA (positive = unlucky, negative = lucky)
    # Comes from FG batting CSV column est_woba_minus_woba_diff
    "est_woba_minus_woba_diff": "LuckDelta",
    # Spray chart metrics (computed from Statcast hc_x/hc_y)
    "xb_pct":    "XB_pct",       # (2B + 3B) / BIP — gap contact rate
    "gap_pct":   "GapTend_pct",  # fraction of BIP in ±20–50° gap zones
    "pull_pct":  "Pull_pct",     # fraction of BIP to pull side
    "oppo_pct":  "Oppo_pct",     # fraction of BIP to opposite field
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

# Internal normalized key → pitcher_archetypes.py expected key
_PITCHER_KEY_REMAP: dict[str, str] = {
    "K_pct_pitch_pct":     "K_pct_pct",
    "SwStr_pct_pitch_pct": "SwStr_pct_pct",
    "BB_pct_pitch_pct":    "BB_pct_pct",
    "GB_pct_pitch_pct":    "GB_pct_pct",   # P3 / P4 threshold
    "CSW_pct_pct":         "CSW_pct_pct",  # identity — already correct name
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
# Step 1b — All-team hitter mapping + cross-team spread metrics
# ---------------------------------------------------------------------------

def _all_team_hitter_ids(statcast: pd.DataFrame) -> dict[str, set[int]]:
    """Return {team_abbr: set_of_batter_mlbam_ids} for every team in the dataset."""
    all_teams = (
        set(statcast["home_team"].dropna().unique())
        | set(statcast["away_team"].dropna().unique())
    )
    result: dict[str, set[int]] = {}
    for team in all_teams:
        bat_home = statcast[(statcast["inning_topbot"] == "Bot") & (statcast["home_team"] == team)]
        bat_away = statcast[(statcast["inning_topbot"] == "Top") & (statcast["away_team"] == team)]
        ids = set(
            pd.concat([bat_home, bat_away])["batter"].dropna().astype(int).unique()
        )
        result[team] = ids
    return result


def _compute_cross_team_metrics(
    statcast: pd.DataFrame,
    batting_full: pd.DataFrame,
    target_team: str,
    season: int,
    debut_seasons: dict[int, int],
) -> dict[str, float]:
    """
    Compute spread/concentration and tenure metrics for all 30 teams then
    return the target team's percentile rank in each metric.

    Metrics produced:
      ISO_spread_pct, PA_concentration_pct, WAR_concentration_pct,
      WAR_variance_inv_pct, RosterFloor_pct,
      AvgTenure_inv_pct, AvgTenure_pct,   # C3/C4 — MLB years since debut
      NewPlayerShare_pct,                  # C3 — share with tenure <= 2
      VeteranShare_pct,                    # C4 — share with tenure >= 5
      HR_FB_pct
    """
    all_team_ids = _all_team_hitter_ids(statcast)

    # League-wide ISO median — used for PowerContributors threshold
    all_bat = batting_full.rename(columns=BATTING_COL_MAP)
    _league_iso = pd.to_numeric(all_bat.get("ISO", pd.Series(dtype=float)), errors="coerce")
    league_median_iso = float(_league_iso.median()) if _league_iso.notna().sum() > 0 else 0.15

    rows = []
    for team, ids in all_team_ids.items():
        tb = batting_full[batting_full["key_mlbam"].isin(ids)].rename(columns=BATTING_COL_MAP)

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

        # MLB tenure per player = current season − first season in our DB
        # Floored at 2015 (start of Statcast era in our data).
        # A player with debut_season == season has tenure 0 (rookie year).
        mlbam_ids = tb["key_mlbam"].dropna().astype(int).tolist() if "key_mlbam" in tb.columns else list(ids)
        tenures = []
        pa_vals  = []
        for pid in mlbam_ids:
            debut = debut_seasons.get(pid)
            if debut is None:
                continue
            t = int(season) - int(debut)
            # Find PA for this player
            player_pa_series = pa[tb["key_mlbam"].astype(int) == pid] if "key_mlbam" in tb.columns else pd.Series([1])
            player_pa = float(player_pa_series.iloc[0]) if not player_pa_series.empty else 1.0
            tenures.append(t)
            pa_vals.append(player_pa)

        if tenures:
            tenure_arr = np.array(tenures, dtype=float)
            pa_arr     = np.array(pa_vals,  dtype=float)
            avg_tenure      = float(np.average(tenure_arr, weights=pa_arr))
            new_player_share = float((tenure_arr <= 2).sum() / len(tenure_arr))
            veteran_share    = float((tenure_arr >= 5).sum() / len(tenure_arr))
        else:
            avg_tenure = new_player_share = veteran_share = np.nan

        # HR/FB and raw HR count from Statcast for this team's batters
        team_sc    = statcast[statcast["batter"].isin(ids)]
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

        # Gap-zone BIP from Statcast hc_x/hc_y
        team_sc_bip = team_sc[team_sc["hc_x"].notna() & team_sc["hc_y"].notna()].copy()
        if not team_sc_bip.empty:
            HP_X, HP_Y = 126.0, 203.0
            angles = np.degrees(np.arctan2(
                team_sc_bip["hc_x"].astype(float) - HP_X,
                HP_Y - team_sc_bip["hc_y"].astype(float),
            ))
            team_gap_pct = float((angles.abs().between(20, 50)).sum() / len(angles))
        else:
            team_gap_pct = np.nan

        # A4 — Lineup Power metrics
        # PA-weighted SLG and Barrel%
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
            "avg_tenure":         avg_tenure,
            "new_player_share":   new_player_share,
            "veteran_share":      veteran_share,
            "hr_fb":              float(hr_count / fb_count)                 if fb_count > 0            else np.nan,
            # A4
            "team_hr":            float(hr_count),
            "team_slg":           team_slg,
            "power_contributors": power_contributors,
            "team_barrel":        team_barrel,
            # A2 — Gap contact
            "team_xb_rate":       team_xb_rate,
            "team_gap_pct":       team_gap_pct,
            "power_source_ratio": power_source_ratio,   # display metadata, not scored
        })

    cross = pd.DataFrame(rows).set_index("team")

    # ── Cache current season's cross-team DataFrame ────────────────────────
    # Then load all other cached seasons to build a multi-year normalization
    # pool. Normalizing against 30-team × N-season history rather than just
    # 30-team × 1-season means "70" = "better than 70% of Statcast-era teams"
    # and scores are stable year-over-year.
    cache_path = PROCESSED_DIR / f"cross_team_metrics_{season}.csv"
    try:
        cross.reset_index().to_csv(cache_path, index=False)
    except Exception as exc:
        log.warning("Could not save cross-team cache for %d: %s", season, exc)

    pool_frames = [cross.reset_index().assign(pool_season=season)]
    for p in sorted(PROCESSED_DIR.glob("cross_team_metrics_*.csv")):
        try:
            other_season = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        if other_season == season:
            continue
        try:
            df_hist = pd.read_csv(p)
            df_hist["pool_season"] = other_season
            pool_frames.append(df_hist)
        except Exception:
            pass

    if len(pool_frames) > 1:
        pool = pd.concat(pool_frames, ignore_index=True)
        log.info("Cross-team pool: %d team-seasons (%d historical seasons)",
                 len(pool), len(pool_frames))
    else:
        pool = cross.reset_index()

    METRIC_MAP = [
        ("iso_spread",        "ISO_spread_pct",        False),
        ("xwoba_spread",      "wRCplus_spread_pct",    False),
        ("pa_concentration",  "PA_concentration_pct",  False),
        ("war_concentration", "WAR_concentration_pct", False),
        ("war_variance",      "WAR_variance_inv_pct",  True),
        ("roster_floor",      "RosterFloor_pct",       False),
        ("avg_tenure",        "AvgTenure_inv_pct",     True),
        ("avg_tenure",        "AvgTenure_pct",         False),
        ("new_player_share",  "NewPlayerShare_pct",    False),
        ("veteran_share",     "VeteranShare_pct",      False),
        ("hr_fb",             "HR_FB_pct",             False),
        ("team_hr",           "TeamHR_pct",            False),
        ("team_slg",          "TeamSLG_pct",           False),
        ("power_contributors","PowerContributors_pct", False),
        ("team_barrel",       "TeamBarrel_pct",        False),
        # A2 — Gap contact metrics
        ("team_xb_rate",      "TeamXB_pct",            False),
        ("team_gap_pct",      "TeamGap_pct",           False),
    ]

    out: dict[str, float] = {}
    # Target team's current-season values (from the just-computed cross frame)
    if target_team not in cross.index:
        log.warning("Cross-team: target team %s not in current season cross frame", target_team)
        return out

    for col, key, invert in METRIC_MAP:
        pool_col = pool[col] if col in pool.columns else None
        if pool_col is None or pool_col.notna().sum() <= 1:
            continue
        target_val = cross.loc[target_team, col] if col in cross.columns else None
        if target_val is None or (isinstance(target_val, float) and np.isnan(target_val)):
            continue
        # Rank target_val against the full historical pool
        pool_vals = pool_col.dropna()
        if invert:
            pct = float((pool_vals < target_val).sum() / len(pool_vals) * 100)
        else:
            pct = float((pool_vals <= target_val).sum() / len(pool_vals) * 100)
        out[key] = pct

    log.info("Cross-team metrics — %d populated for %s (pool size: %d team-seasons)",
             len(out), target_team, len(pool))
    return out


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

    metric_invert: dict[str, bool] = {
        "K_pct":  True,   # lower K = better contact → invert
    }

    # Metrics to rank: all BATTING_COL_MAP targets + derived OBP_SLG_gap
    metrics_to_rank = list(dict.fromkeys(BATTING_COL_MAP.values()))
    if "OBP_SLG_gap" in full.columns:
        metrics_to_rank.append("OBP_SLG_gap")

    for metric in metrics_to_rank:
        if metric not in full.columns:
            continue
        if metric not in team.columns:
            continue
        invert = metric_invert.get(metric, False)
        col_name = f"{metric}_pct"
        team[col_name] = _pool_rank(full[metric], team[metric], invert=invert)

    # K_pct_raw_pct: non-inverted K% percentile for TTO classifier
    # (K_pct_pct is inverted; TTO classifier needs raw strikeout rank)
    if "K_pct" in full.columns and "K_pct" in team.columns:
        team["K_pct_raw_pct"] = _pool_rank(full["K_pct"], team["K_pct"], invert=False)

    # LuckDelta_pct: normalize xwOBA−wOBA gap from the full batting pool,
    # then map to team players by key_mlbam (team_bat may not have this column
    # since it comes from the DB rather than the FG batting CSV).
    if "LuckDelta" in full.columns and full["LuckDelta"].notna().sum() > 1:
        try:
            from ingest import normalize_percentile as _np
            luck_valid   = full[full["LuckDelta"].notna()].copy()
            luck_pcts    = _np(luck_valid["LuckDelta"])  # higher = more unlucky
            luck_map     = dict(zip(luck_valid["key_mlbam"], luck_pcts))
            team["LuckDelta_pct"] = team["key_mlbam"].map(
                lambda x: luck_map.get(int(x)) if pd.notna(x) else None
            )
        except Exception:
            pass

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

    metric_invert: dict[str, bool] = {
        "BB_pct_pitch":  True,
        "xwOBA_allowed": True,
        "FIP":           True,   # lower FIP = better
        # HardHit_allowed and Barrel_allowed are NOT inverted here.
        # Statcast already expresses them as "higher = better pitcher"
        # (they invert hard-hit/barrel rate before ranking), so pool_rank
        # with invert=False naturally gives higher pct = better suppressor.
    }

    for metric in dict.fromkeys(PITCHING_COL_MAP.values()):
        if metric not in full.columns or metric not in team.columns:
            continue
        invert = metric_invert.get(metric, False)
        col_name = f"{metric}_pct"
        team[col_name] = _pool_rank(full[metric], team[metric], invert=invert)

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
    from ingest import normalize_percentile

    # Pre-compute cross-player attempt_rate percentile for Disruptive/Chaotic gate.
    # opportunities ≈ OBP * PA (times on base proxy — avoids needing raw H/BB/HBP counts).
    def _attempt_rate(row_) -> Optional[float]:
        sb_ = float(row_.get("SB", 0) or 0)
        cs_ = float(row_.get("CS", 0) or 0)
        obp_= float(row_.get("OBP") or row_.get("obp") or 0)
        pa_ = float(row_.get("PA") or row_.get("pa") or 1)
        tob = obp_ * pa_
        if tob <= 0:
            return None
        return (sb_ + cs_) / tob

    attempt_rates = pd.Series(
        [_attempt_rate(r) for _, r in team_bat.iterrows()],
        index=team_bat.index,
    )
    # Only normalize among players who have at least minimum attempts
    min_att = 2
    has_attempts = attempt_rates.notna() & (
        (team_bat.get("SB", pd.Series(0, index=team_bat.index)).fillna(0)
         + team_bat.get("CS", pd.Series(0, index=team_bat.index)).fillna(0)) >= min_att
    )
    att_pct_series = pd.Series(np.nan, index=team_bat.index)
    if has_attempts.sum() > 1:
        att_pct_series[has_attempts] = normalize_percentile(
            attempt_rates[has_attempts]
        ).values

    profiles = []
    for idx, row in team_bat.iterrows():
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
                if val is not None and not pd.isna(val):
                    fval = float(val)
                    metrics[col.removesuffix("_pct")] = fval  # K_pct_pct→K_pct, ISO_pct→ISO
                    metrics[col] = fval                        # also keep ISO_pct, AVG_pct

        # Raw sprint speed (ft/sec) for Speed modifier hard threshold.
        # sprint_speed in DB is stored as raw ft/sec after re-seeding.
        sprint_raw = row.get("sprint_speed") or row.get("Sprint Speed")
        sprint_raw = float(sprint_raw) if sprint_raw is not None and not pd.isna(sprint_raw) else None

        # Attempt rate percentile (cross-player, pre-computed above)
        att_pct = att_pct_series.get(idx)
        att_pct = float(att_pct) if att_pct is not None and not np.isnan(att_pct) else None

        # SB/CS — available from BRef merge; Savant-only falls back to 0
        obp_val = float(row.get("OBP") or row.get("obp") or 0)
        pa_val  = float(row.get("PA") or row.get("pa") or 1)
        profile = build_hitter_profile(
            player_id=int(player_id),
            metrics=metrics,
            sprint_speed_raw=sprint_raw,
            sb=int(row.get("SB", 0) or 0),
            cs=int(row.get("CS", 0) or 0),
            opportunities=int(obp_val * pa_val),
            attempt_rate_pct=att_pct,
        )
        info = _get_player_info_map().get(int(player_id), {})
        profile["name"] = info.get("name")
        # Prefer DB age (BRef season-specific) over Chadwick today-computed age
        db_age = row.get("age") or row.get("Age")
        profile["age"]  = (int(db_age) if db_age is not None and not pd.isna(db_age) else None) or info.get("age")
        profile["pa"]   = int(row.get("PA") or row.get("pa") or 0)
        war_raw = row.get("WAR_bat")
        profile["war"]  = float(war_raw) if war_raw is not None and not pd.isna(war_raw) else None

        # Store curated percentile metrics for the interactive player chart.
        # Each raw_key is the exact key that _classify_hitters puts in `metrics`
        # (the col.removesuffix("_pct") form OR the full col name — both present).
        _DISPLAY_KEYS = [
            ("ISO_pct",           "Power"),          # ISO_pct → ISO percentile
            ("BB_pct_pct",        "Walk Rate"),       # BB_pct_pct → BB% percentile
            ("K_pct_pct",         "Contact"),         # K_pct_pct → K% pct (inverted → higher = better)
            ("xwOBA_pct",         "Quality"),         # xwOBA_pct → xwOBA percentile
            ("sprint_speed_pct",  "Speed"),           # sprint_speed_pct → speed percentile
            ("OBP_SLG_gap_pct",   "Contact-First"),   # OBP_SLG_gap_pct → gap percentile
            ("FPS_pct_pct",       "Aggression"),      # FPS_pct_pct → first-pitch swing% pct
            ("ZSwing_pct_pct",    "Zone Swing"),      # ZSwing_pct_pct → zone swing% pct
            ("Contact_pct_pct",   "Contact Rate"),    # Contact_pct_pct → contact rate pct
            ("HardHit_pct_pct",   "Hard Hit"),        # HardHit_pct_pct → hard-hit% pct
            ("Barrel_pct_pct",    "Barrel Rate"),     # Barrel_pct_pct → barrel% pct
        ]
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

        # hard_hit_pct / barrel_pct from Statcast are already "higher = better"
        # (Statcast inverts before ranking: 100 = best at preventing hard contact).
        # Use them directly — no double-inversion.
        # K_inv_pct: P3 scoring uses low-K as a signal (ground-ball style, not strikeout).
        if "K_pct_pct" in metrics:
            metrics["K_inv_pct"] = 100.0 - metrics["K_pct_pct"]

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
            profile = build_starter_profile(player_id, metrics)
            info = _get_player_info_map().get(player_id, {})
            # Prefer Chadwick name; fall back to DB name column if available
            db_name = row.get("name")
            profile["name"] = info.get("name") or (str(db_name).strip() if db_name else None)
            # Prefer DB age (BRef season-specific) over Chadwick today-computed age
            _db_age = row.get("age")
            profile["age"]  = (int(_db_age) if _db_age is not None and not pd.isna(_db_age) else None) or info.get("age")
            profile["bf"]   = bf
            war_raw = row.get("WAR_pitch")
            profile["war"]  = float(war_raw) if war_raw is not None and not pd.isna(war_raw) else None
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
                "bf":        bf,
                "war":       float(war_raw) if war_raw is not None and not pd.isna(war_raw) else None,
                "metrics_pct": {
                    k: metrics[k] for k in [
                        "avg_velo_pct", "K_pct_pct", "SwStr_pct_pct",
                        "GB_pct_pct", "HardHit_allowed_pct",
                        "Barrel_allowed_pct", "CSW_pct_pct", "BB_pct_pct",
                    ] if k in metrics
                },
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

    # Pitcher GB% — used in B2 (GB_pct_pct) and B4 (GB_pitch_pct)
    pitcher_gb = metrics.get("GB_pct_pitch_pct")
    if pitcher_gb is not None:
        metrics.setdefault("GB_pct_pct",  pitcher_gb)
        metrics.setdefault("GB_pitch_pct", pitcher_gb)

    # Inverted pitching keys
    metrics["BB_pitch_inv_pct"] = _inv("BB_pct_pitch_pct")

    # ── Park factor ───────────────────────────────────────────────────────
    if park and "pitcher_friendly" in park:
        metrics["ParkPitcherFriendly_pct"] = park["pitcher_friendly"]

    # ── Spin efficiency (B3) ──────────────────────────────────────────────
    if spin_team and "weighted_spin_efficiency_pct" in spin_team:
        metrics["SpinEfficiency_pct"] = spin_team["weighted_spin_efficiency_pct"]

    # ── CSW% — prefer Statcast-derived; fall back to SwStr approximation ──
    if "CSW_pct_pct" not in metrics and "SwStr_pct_pct" in metrics:
        metrics["CSW_pct_pct"] = metrics["SwStr_pct_pct"]

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
    For each team, fraction of current-season hitters who were NOT on the
    team the prior season.  Requires prior-season Statcast parquet in cache.

    Results are cached to data/processed/team_turnover_{season}.csv so
    subsequent portrait builds skip the prior-year parquet load.

    Returns {team: turnover_rate} for all teams where prior-season data exists.
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

    # Load only the columns we need (fast — subset of 139 cols)
    prev_sc = pd.read_parquet(
        str(prev_parquet),
        columns=["batter", "home_team", "away_team", "inning_topbot"],
    )

    cur_team_ids  = _all_team_hitter_ids(statcast)
    prev_team_ids = _all_team_hitter_ids(prev_sc)

    rates: dict[str, float] = {}
    for team, cur_ids in cur_team_ids.items():
        prev_ids = prev_team_ids.get(team, set())
        if cur_ids:
            new_players = cur_ids - prev_ids
            rates[team] = len(new_players) / len(cur_ids)

    # Persist so future builds skip the parquet load
    pd.DataFrame({"team": list(rates), "turnover_rate": list(rates.values())}).to_csv(
        cache, index=False
    )
    log.info("Turnover computed — %d teams (season %d vs %d)", len(rates), season, season - 1)
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
    # current-season pool: used for team filtering and cross-team metrics
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

    # Merge bWAR into batting_full / pitching_full — overrides the empty DB war column
    try:
        bwar_bat, bwar_pitch = pull_bwar(season)
        if not bwar_bat.empty:
            bwar_bat["key_mlbam"] = pd.to_numeric(bwar_bat["key_mlbam"], errors="coerce")
            batting_full = batting_full.drop(columns=["war"], errors="ignore").merge(
                bwar_bat.rename(columns={"war": "war"}), on="key_mlbam", how="left"
            )
            log.info("bWAR merged into batting_full — %d players have WAR", batting_full["war"].notna().sum())
        if not bwar_pitch.empty:
            bwar_pitch["key_mlbam"] = pd.to_numeric(bwar_pitch["key_mlbam"], errors="coerce")
            pitching_full = pitching_full.drop(columns=["war"], errors="ignore").merge(
                bwar_pitch.rename(columns={"war": "war"}), on="key_mlbam", how="left"
            )
            log.info("bWAR merged into pitching_full — %d pitchers have WAR", pitching_full["war"].notna().sum())
    except Exception as exc:
        log.warning("bWAR merge failed: %s", exc)

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
                spray[["key_mlbam", "xb_pct", "gap_pct", "pull_pct", "oppo_pct", "hr_per_bip"]],
                on="key_mlbam", how="left",
            )
            log.info("Spray stats merged — %d batters with spray data",
                     batting_full["xb_pct"].notna().sum())
    except Exception as exc:
        log.warning("Spray stats merge failed: %s", exc)

    # ── 3b. Multi-season historical pool for percentile normalization ─────
    # Percentiles are computed against ALL seeded seasons so that "70" means
    # 70th pct of the Statcast era, not just this year's 30 teams.
    try:
        seeded_seasons = query_seeded_seasons()
        # Always include the current season even if not in Supabase
        if season not in seeded_seasons:
            seeded_seasons = sorted(seeded_seasons + [season])
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

    team_bat_raw = batting_full[batting_full["key_mlbam"].isin(hitter_ids)].copy()
    team_pit_raw = pitching_full[pitching_full["key_mlbam"].isin(pitcher_ids)].copy()

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
        se_col = "spin_efficiency_weighted"
        if se_col in spin_pitcher_level.columns:
            valid = spin_pitcher_level[se_col].dropna()
            if len(valid) > 0:
                team_mean_se = float(valid.mean())

                # Compute league-wide team spin efficiency to get a proper percentile.
                # For each pitching team in Statcast, compute mean spin_rate as a proxy
                # (full spin-efficiency model is too slow for all 30 teams; raw spin_rate
                # correlates tightly and is available directly in the Statcast parquet).
                se_rows = []
                sc_spin = statcast[["pitcher", "release_spin_rate",
                                    "inning_topbot", "home_team", "away_team"]].copy()
                sc_spin["pitching_team"] = np.where(
                    sc_spin["inning_topbot"] == "Top",
                    sc_spin["home_team"], sc_spin["away_team"],
                )
                sc_spin["_spin"] = pd.to_numeric(sc_spin["release_spin_rate"], errors="coerce")
                league_team_spin = (
                    sc_spin.dropna(subset=["_spin", "pitching_team"])
                    .groupby("pitching_team")["_spin"]
                    .mean()
                )
                if team in league_team_spin.index and league_team_spin.notna().sum() > 1:
                    se_pct = float(normalize_percentile(league_team_spin).loc[team])
                else:
                    se_pct = 50.0

                spin_team_summary = {
                    "mean_spin_efficiency":         team_mean_se,
                    "weighted_spin_efficiency_pct": se_pct,
                    "pitcher_count":                len(valid),
                }

    # ── 10a. Cross-team spread/concentration + tenure metrics ─────────────
    cross_team = {}
    _power_source_ratio: Optional[float] = None
    try:
        debut_seasons = query_debut_seasons()
        cross_team = _compute_cross_team_metrics(statcast, batting_full, team, season, debut_seasons)
        # Extract raw power_source_ratio for the target team from the cross frame
        # (it's display metadata — stored in team_metrics, not scored)
        _ps_cache = PROCESSED_DIR / f"cross_team_metrics_{season}.csv"
        if _ps_cache.exists():
            _ct_df = pd.read_csv(_ps_cache)
            _row = _ct_df[_ct_df["team"] == team]
            if not _row.empty and "power_source_ratio" in _ct_df.columns:
                _power_source_ratio = float(_row["power_source_ratio"].iloc[0])
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

    # ── 10. Philosophy metrics assembly ───────────────────────────────────
    philosophy_metrics = _build_philosophy_metrics(
        team_batting_agg,
        team_pitching_agg,
        park_result,
        spin_team_summary,
        None,   # tunneling team summary not yet aggregated
        cross_team_metrics=cross_team,
        oaa_val=oaa_val,
        def_runs_val=def_runs_val,
        opener_val=opener_val,
        platoon_val=platoon_val,
        turnover_val=turnover_val,
    )

    # ── 11. Philosophy scoring ────────────────────────────────────────────
    philosophy = build_philosophy_summary(team, season, philosophy_metrics)

    # ── 12. Hitter archetypes ─────────────────────────────────────────────
    hitter_profiles = _classify_hitters(team_bat)

    # ── 13. Pitcher archetypes ────────────────────────────────────────────
    starters, bullpen_profile, bullpen_arms = _classify_pitchers(
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
            "bullpen_arms":    bullpen_arms,
        },
        "team_metrics": {
            "batting":            team_batting_agg,
            "pitching":           team_pitching_agg,
            "power_source_ratio": _power_source_ratio,  # raw HR/(HR+2B+3B) — display only
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
            )
        },
        "data_coverage": coverage,
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
        row["bullpen_arms"]  = bool(p.get("players", {}).get("bullpen_profile"))

        rows.append(row)

    return pd.DataFrame(rows)
