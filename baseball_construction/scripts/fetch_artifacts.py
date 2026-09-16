#!/usr/bin/env python3
"""
fetch_artifacts.py — pull prebuilt portraits into the local tree.

Run during `docker build` so the image ships with portraits on disk. The web
tier then serves them from the filesystem: instant, and working even when the
Supabase project is paused, which on the free tier it will be.

This is the "bake into the image" half of the hybrid. The other half is the
runtime fallback in callbacks._storage_download, which picks up anything
uploaded after the image was built, so new data does not require a redeploy.

Portraits are gitignored, so a build from a clean clone has none. Without this
step a Render deploy would ship an empty portrait directory and every request
would be a Storage round trip — or, if the project were paused, a dead page.

Reads with whatever key is configured. The portraits bucket is meant to be
anon-readable, so SUPABASE_ANON_KEY is enough and no build secret is needed.

NEVER FAILS THE BUILD. If Storage is unreachable the image is still correct,
just without the local copy, and the runtime fallback covers it. A deploy that
cannot reach Supabase should still produce a running container.

    python scripts/fetch_artifacts.py            # portraits + league identity
    python scripts/fetch_artifacts.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

PORTRAIT_DIR = _HERE.parent / "data" / "processed" / "portraits"
IDENTITY_DIR = _HERE.parent / "data" / "processed"
BUCKET = "portraits"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what is in the bucket, download nothing")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def say(*a):
        if not args.quiet:
            print(*a, flush=True)

    try:
        from database import get_client
        store = get_client().storage.from_(BUCKET)
    except Exception as exc:
        say(f"fetch_artifacts: no Supabase client ({exc}) — "
            "image will rely on the runtime fallback")
        return 0

    try:
        listing = store.list()
    except Exception as exc:
        say(f"fetch_artifacts: cannot list bucket ({exc}) — "
            "image will rely on the runtime fallback")
        return 0

    names = [o.get("name") for o in (listing or []) if o.get("name")]
    portraits = [n for n in names if n.endswith(".json")
                 and not n.startswith("league_identity_")]
    identity = [n for n in names if n.startswith("league_identity_")]
    say(f"fetch_artifacts: bucket holds {len(portraits)} portraits, "
        f"{len(identity)} league identity files")

    if args.dry_run:
        return 0

    PORTRAIT_DIR.mkdir(parents=True, exist_ok=True)
    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)

    ok = failed = 0
    for name, dest_dir in ([(n, PORTRAIT_DIR) for n in portraits]
                           + [(n, IDENTITY_DIR) for n in identity]):
        try:
            blob = store.download(name)
            text = blob.decode("utf-8") if isinstance(blob, bytes) else blob
            json.loads(text)                     # refuse to write a broken file
            (dest_dir / name).write_text(text)
            ok += 1
        except Exception as exc:
            failed += 1
            if failed <= 3:
                say(f"  {name}: {exc}")

    say(f"fetch_artifacts: {ok} written, {failed} failed")
    return 0                                     # never fail the build


if __name__ == "__main__":
    sys.exit(main())
