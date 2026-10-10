"""
Upload all pre-generated portrait JSON caches to Supabase Storage.

Run after generate_all_portraits.py completes:
    python3 scripts/upload_portraits.py [--dry-run] [--team HOW] [--season 2023]

Portraits are stored in the 'portraits' Supabase Storage bucket as
{TEAM}_{SEASON}.json — downloaded by the app on production cache miss.
"""
import sys, argparse, logging, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

from database import get_client

PORTRAITS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "portraits"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

UPLOAD_ATTEMPTS = 3


def upload_portraits(team_filter=None, season_filter=None, dry_run=False):
    client = get_client()
    bucket = client.storage.from_("portraits")

    files = sorted(PORTRAITS_DIR.glob("*.json"))

    # league_identity_*.json goes to the same bucket. It was never uploaded,
    # so Storage held 100 portraits and zero identity files while the Identity
    # tab read those from local disk with no fallback — meaning a deploy from a
    # clean clone, where they are gitignored, rendered that tab empty.
    # Unfiltered by team because each file covers all 30.
    identity = sorted(PROCESSED_DIR.glob("league_identity_*.json"))
    if season_filter:
        identity = [f for f in identity if f.stem.endswith(str(season_filter))]
    if team_filter:
        identity = []          # not a per-team artefact

    if not files and not identity:
        log.error("Nothing to upload from %s", PORTRAITS_DIR)
        return

    # Filter
    if team_filter:
        files = [f for f in files if f.stem.startswith(team_filter + "_")]
    if season_filter:
        files = [f for f in files if f.stem.endswith("_" + str(season_filter))]

    files = files + identity
    log.info("Found %d file(s) to upload (%d portraits, %d league identity)",
             len(files), len(files) - len(identity), len(identity))
    ok = fail = skip = 0

    for fp in files:
        name = fp.name
        size_kb = fp.stat().st_size // 1024

        if dry_run:
            log.info("  [DRY RUN] %s (%dKB)", name, size_kb)
            ok += 1
            continue

        # Upsert, and retry — do NOT remove first.
        #
        # This used to delete the existing object and then upload a
        # replacement, which turns any transient network error into a DELETED
        # file rather than a stale one. On a 372-file run five uploads failed
        # with SSL BAD_RECORD_MAC — transport noise, nothing to do with
        # permissions — and those five team-seasons were left absent from the
        # bucket entirely. All five succeeded on the first retry.
        #
        # x-upsert overwrites in place, so a failure leaves the old file
        # standing. Backoff is 1s, 2s, 4s.
        data = fp.read_bytes()
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                bucket.upload(name, data, file_options={
                    "content-type": "application/json",
                    "x-upsert": "true",
                })
                log.info("  ✓ %s (%dKB)%s", name, size_kb,
                         f" [attempt {attempt}]" if attempt > 1 else "")
                ok += 1
                break
            except Exception as exc:
                if attempt == UPLOAD_ATTEMPTS:
                    log.error("  ✗ %s: %s", name, exc)
                    fail += 1
                else:
                    log.warning("  … %s failed (%s) — retrying in %ds",
                                name, str(exc)[:60], 2 ** (attempt - 1))
                    time.sleep(2 ** (attempt - 1))

    log.info("\nDone: %d uploaded, %d failed, %d skipped", ok, fail, skip)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--team",   help="Filter by team (e.g. HOU)")
    p.add_argument("--season", type=int, help="Filter by season (e.g. 2023)")
    args = p.parse_args()
    upload_portraits(args.team, args.season, args.dry_run)
