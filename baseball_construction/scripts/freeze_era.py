#!/usr/bin/env python3
"""
freeze_era.py — copy the finished seasons' pool roots into data/era/.

Every portrait ranks its era percentiles against the finished seasons plus its
own (ingest.era_pool_seasons), and that pool is built from each season's
fg_batting_{s}.csv / fg_pitching_{s}.csv. Those files live in data/processed,
which is not in git, so a machine without this one's data — the GitHub runner
— would have to rebuild them from the original sources, pulling a full season
of Statcast for each to compute the pitch aggregates in them (~1 GB, hours),
and would still not be guaranteed byte-identical files.

So the finished seasons' roots are committed: three small CSVs a season. The
nightly job copies them into data/processed before building, so it ranks
against exactly the pool this machine does, and build fingerprints agree.

Run it at finalize, after the season that just ended has been refreshed for the
last time, then commit data/era/.

    python scripts/freeze_era.py            # copy, and report what changed
    python scripts/freeze_era.py --check    # exit 1 if data/era is out of date
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

from ingest import PROCESSED_DIR, season_state  # noqa: E402

ERA_DIR = _HERE.parent / "data" / "era"
# The fg roots behind the player percentiles, and the cross-team rows behind
# the team-construction percentiles (team_portrait.cross_team_path) — both
# ranked against the era pool, and neither rebuildable on the runner without
# a full season of Statcast per season. And each season's service-time totals
# (service_time.service_through): the MLS lists they come from are kept out
# of git, and the runner adds only the season in progress to the last of them.
# And each season's talent counts (talent.py): a portrait's career-to-date
# estimates read every season up to its own, from Statcast the runner lacks.
ROOTS = ("fg_batting_{s}.csv", "fg_pitching_{s}.csv", "cross_team_metrics_v3_{s}.csv",
         "service_through_{s}.csv", "talent_counts_{s}.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report differences and exit 1 if any, without copying")
    args = ap.parse_args()

    finished, _ = season_state()
    ERA_DIR.mkdir(parents=True, exist_ok=True)
    changed, missing = [], []
    for s in finished:
        for pat in ROOTS:
            src, dst = PROCESSED_DIR / pat.format(s=s), ERA_DIR / pat.format(s=s)
            if not src.exists():
                missing.append(src.name)
                continue
            if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                continue
            changed.append(src.name)
            if not args.check:
                shutil.copy2(src, dst)

    verb = "out of date" if args.check else "copied"
    print(f"freeze_era: {len(finished)} finished seasons "
          f"({finished[0]}-{finished[-1]}); {len(changed)} file(s) {verb}"
          + (f": {', '.join(changed)}" if changed else ""))
    if missing:
        print(f"freeze_era: missing in data/processed: {', '.join(missing)}")
    return 1 if (missing or (args.check and changed)) else 0


if __name__ == "__main__":
    sys.exit(main())
