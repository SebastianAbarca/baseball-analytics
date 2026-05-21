"""
seed.py — Seed Supabase with a full season of BRef + Savant data.

Seeds ALL players for a season (not filtered by team) so percentile
rankings are computed against the true 30-team MLB pool.

Usage:
    python3 baseball_construction/seed.py --season 2023
    python3 baseball_construction/seed.py --season 2021 2022 2023 2024
    python3 baseball_construction/seed.py --season 2023 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "modules"))

from ingest import _bref_savant_batting_merge, _bref_savant_pitching_merge
from database import (
    season_batting_seeded,
    season_pitching_seeded,
    upsert_batting,
    upsert_pitching,
    get_client,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BRef team code → our internal team code
# BRef uses different abbreviations for some franchises
# ---------------------------------------------------------------------------

BREF_TEAM_MAP: dict[str, str] = {
    "ARI": "AZ",
    "CHW": "CWS",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WSN": "WSH",
}


def _norm_team(bref_tm: str) -> str:
    return BREF_TEAM_MAP.get(str(bref_tm).strip(), str(bref_tm).strip())


# ---------------------------------------------------------------------------
# Column maps: BRef+Savant merge output → DB schema
# ---------------------------------------------------------------------------

BATTING_COL_MAP = {
    "key_mlbam":        "key_mlbam",
    "Name":             "name",
    "Age":              "age",
    "PA":               "pa",
    "AVG":              "avg",
    "OBP":              "obp",
    "SLG":              "slg",
    "ISO":              "iso",
    "K_rate":           "k_rate",
    "BB_rate":          "bb_rate",
    "SB":               "sb",
    "CS":               "cs",
    "xwoba":            "xwoba",
    "xba":              "xba",
    "hard_hit_percent": "hard_hit_pct",
    "brl_percent":      "barrel_pct",
    "whiff_percent":    "whiff_pct",
    "chase_percent":    "chase_pct",
    "sprint_speed":     "sprint_speed",
    "exit_velocity":    "exit_velocity",
    "join_source":      "join_source",
}

PITCHING_COL_MAP = {
    "key_mlbam":        "key_mlbam",
    "Name":             "name",
    "Age":              "age",
    "GS":               "gs",
    "BF":               "bf",
    "K_rate_pitch":     "k_rate",
    "BB_rate_pitch":    "bb_rate",
    "xwoba":            "xwoba_allowed",
    "hard_hit_percent": "hard_hit_pct",
    "brl_percent":      "barrel_pct",
    "whiff_percent":    "whiff_pct",
    "fb_velocity":      "fb_velocity",
    "join_source":      "join_source",
}


# ---------------------------------------------------------------------------
# Seed functions
# ---------------------------------------------------------------------------

def seed_batting(season: int) -> None:
    log.info("Pulling BRef + Savant batting for %d (full league)...", season)
    df = _bref_savant_batting_merge(season, min_pa=1)

    # Drop BRef "TOT" rows (multi-team totals — keep individual stints)
    if "Tm" in df.columns:
        df = df[df["Tm"] != "TOT"].copy()

    # Normalize team codes
    if "Tm" in df.columns:
        df["Tm"] = df["Tm"].map(_norm_team)

    # Add season column
    df["season"] = season

    # Rename to DB schema
    df = df.rename(columns=BATTING_COL_MAP)

    # Keep only DB columns that exist in the DataFrame
    db_cols = list(BATTING_COL_MAP.values()) + ["season"]
    df = df[[c for c in db_cols if c in df.columns]].copy()

    # Coerce types
    for int_col in ["key_mlbam", "season", "pa", "sb", "cs", "age"]:
        if int_col in df.columns:
            df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")

    df = df[df["key_mlbam"].notna()].copy()

    log.info("Upserting %d batting rows for %d...", len(df), season)
    upsert_batting(df)
    log.info("Batting seed complete for %d", season)


def seed_pitching(season: int) -> None:
    log.info("Pulling BRef + Savant pitching for %d (full league)...", season)
    df = _bref_savant_pitching_merge(season, min_pa=1)

    if "Tm" in df.columns:
        df = df[df["Tm"] != "TOT"].copy()
        df["Tm"] = df["Tm"].map(_norm_team)

    df["season"] = season

    df = df.rename(columns=PITCHING_COL_MAP)

    db_cols = list(PITCHING_COL_MAP.values()) + ["season"]
    df = df[[c for c in db_cols if c in df.columns]].copy()

    for int_col in ["key_mlbam", "season", "gs", "bf", "age"]:
        if int_col in df.columns:
            df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")

    df = df[df["key_mlbam"].notna()].copy()

    log.info("Upserting %d pitching rows for %d...", len(df), season)
    upsert_pitching(df)
    log.info("Pitching seed complete for %d", season)


def seed_season(season: int, force: bool = False) -> None:
    log.info("=== Seeding season %d ===", season)

    if not force and season_batting_seeded(season):
        log.info("Batting already seeded for %d — skipping (use --force to re-seed)", season)
    else:
        seed_batting(season)

    if not force and season_pitching_seeded(season):
        log.info("Pitching already seeded for %d — skipping (use --force to re-seed)", season)
    else:
        seed_pitching(season)

    log.info("=== Season %d complete ===", season)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Supabase with a full MLB season")
    parser.add_argument(
        "--season", type=int, nargs="+", required=True,
        help="Season year(s) to seed, e.g. --season 2023 or --season 2021 2022 2023",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-seed even if data already exists",
    )
    args = parser.parse_args()

    for season in args.season:
        seed_season(season, force=args.force)
