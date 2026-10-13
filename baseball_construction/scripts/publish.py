#!/usr/bin/env python3
"""
publish.py — put what the build machine made where the dashboard reads it.

The ONLY writer to the serving tables (schema_serving.sql). Runs on the
machine that builds portraits, with the service key, and reads the artefacts
the build already left on disk:

    portraits        data/processed/portraits/*.json            -> portraits
    league identity  data/processed/league_identity_{s}.json    -> league_seasons.identity
    pitch mix        modules/processed/pitch_mix_{s}.parquet    -> league_seasons.pitch_mix
    tunnel average   computed here from the season's portraits  -> league_seasons.tunnel_average
    scout            flattened here from all portraits          -> scout_tags, scout_players

The dashboard reads those tables with the anon key and nothing else. This
replaces upload_portraits.py (Storage bucket) and fetch_artifacts.py (baking
portraits into the Docker image): one place for published data, one way in.

Refuses to publish a portrait that is not current for this machine's pools
and gates (team_portrait.portrait_is_current) — publishing is the step that
makes a build public, so it is where a stale one has to stop.

Idempotent and cheap to re-run: a portrait or league season whose content
hash is unchanged is skipped. Scout is republished whole each time.

    python scripts/publish.py                 # everything
    python scripts/publish.py --seasons 2026  # portraits + league for 2026; scout is always whole
    python scripts/publish.py --only scout
    python scripts/publish.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))
sys.path.insert(0, str(_HERE.parent / "dashboard"))

log = logging.getLogger("publish")

ATTEMPTS = 3              # per request; backoff 1s, 2s, 4s
SCOUT_BATCH = 500         # scout_players rows per upsert


def _retry(what: str, fn):
    """Run fn, retrying transient failures. Raises after the last attempt."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == ATTEMPTS:
                raise
            wait = 2 ** (attempt - 1)
            log.warning("  … %s failed (%s) — retrying in %ds", what, str(exc)[:80], wait)
            time.sleep(wait)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


REQUEST_TIMEOUT_S = 60    # one ~300 KB upsert takes about a second


def _write_client():
    """
    Service-key client with a request timeout. Without one, a connection that
    died mid-request hung the whole publish indefinitely — one did, 25 minutes
    into a full run, with the database answering other queries in 0.4s. With
    it, the dead request fails and _retry sends it again.
    """
    import database
    from supabase import ClientOptions, create_client
    if not database.SUPABASE_SERVICE_KEY:
        raise SystemExit("publish needs SUPABASE_SERVICE_KEY — it is the only writer, "
                         "and the anon key cannot write by design")
    return create_client(database.SUPABASE_URL, database.SUPABASE_SERVICE_KEY,
                         options=ClientOptions(postgrest_client_timeout=REQUEST_TIMEOUT_S))


# ---------------------------------------------------------------------------
# Portraits
# ---------------------------------------------------------------------------

def publish_portraits(root: Path, seasons: set[int], client, dry: bool, force: bool) -> int:
    import team_portrait as tp

    files = sorted(f for f in (root / "data/processed/portraits").glob("*_*.json")
                   if int(f.stem.rsplit("_", 1)[1]) in seasons)
    have = {}
    if not force:
        rows = _retry("read portrait hashes", lambda: client.table("portraits")
                      .select("team, season, content_hash").execute().data)
        have = {(r["team"], r["season"]): r["content_hash"] for r in rows}

    sent = same = refused = 0
    for f in files:
        text = f.read_text()
        p = json.loads(text)
        team, season = p["team"], int(p["season"])
        if not tp.portrait_is_current(p):
            log.error("  ✗ %s %s is not current (schema %s, fingerprint %s; this "
                      "machine is at %s, %s) — rebuild or re-derive it first",
                      team, season, p.get("schema_version"), p.get("build_fingerprint"),
                      tp.PORTRAIT_SCHEMA_VERSION, tp.build_fingerprint(season))
            refused += 1
            continue
        h = _sha(text)
        if have.get((team, season)) == h:
            same += 1
            continue
        if dry:
            sent += 1
            continue
        row = {
            "team": team, "season": season,
            "schema_version": p["schema_version"],
            "build_fingerprint": p.get("build_fingerprint"),
            "gate_fingerprint": p.get("gate_fingerprint"),
            "content_hash": h,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime)),
            "data": p,
        }
        _retry(f"portrait {team} {season}",
               lambda: client.table("portraits").upsert(row).execute())
        sent += 1
    log.info("  portraits: %d %s, %d unchanged, %d refused",
             sent, "would publish" if dry else "published", same, refused)
    return refused


# ---------------------------------------------------------------------------
# League seasons
# ---------------------------------------------------------------------------

def publish_league(root: Path, seasons: set[int], client, dry: bool, force: bool) -> None:
    import pandas as pd
    import charts
    import team_portrait as tp

    have = {}
    if not force:
        rows = _retry("read league hashes", lambda: client.table("league_seasons")
                      .select("season, content_hash").execute().data)
        have = {r["season"]: r["content_hash"] for r in rows}

    for season in sorted(seasons):
        ident_path = root / f"data/processed/league_identity_{season}.json"
        identity = json.loads(ident_path.read_text()) if ident_path.exists() else None

        mix_path = root / f"modules/processed/pitch_mix_{season}.parquet"
        pitch_mix = ({str(k): v for k, v in
                      charts.compute_pitch_mix(pd.read_parquet(mix_path)).items()}
                     if mix_path.exists() else None)

        portraits = [json.loads(f.read_text()) for f in
                     sorted((root / "data/processed/portraits").glob(f"*_{season}.json"))]
        tunnel = charts.compute_tunnel_league_average(portraits) if portraits else None

        missing = [n for n, v in (("identity", identity), ("pitch_mix", pitch_mix),
                                  ("tunnel_average", tunnel)) if not v]
        if missing:
            log.warning("  league %s: no %s on disk", season, ", ".join(missing))
        if not (identity or pitch_mix or tunnel):
            continue

        payload = {"identity": identity, "tunnel_average": tunnel, "pitch_mix": pitch_mix}
        h = _sha(json.dumps(payload, sort_keys=True))
        if have.get(season) == h:
            log.info("  league %s: unchanged", season)
            continue
        if dry:
            log.info("  league %s: would publish", season)
            continue
        row = {"season": season, **payload, "content_hash": h,
               "build_fingerprint": tp.build_fingerprint(season),
               "gate_fingerprint": tp.gate_fingerprint(),
               "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _retry(f"league {season}",
               lambda: client.table("league_seasons").upsert(row).execute())
        log.info("  league %s: published", season)


# ---------------------------------------------------------------------------
# Scout
# ---------------------------------------------------------------------------

def _scout_ord(team: str, season: int, grp_idx: int, idx: int) -> int:
    """
    A row's position in the flattening order, computed from the row alone.

    Every Scout sort breaks ties on `ord`, and the order it encodes is the one
    the in-memory index had: portrait files sorted by name (team, then season),
    then hitters / starters / bullpen_arms, then roster order. It used to be a
    running count over every portrait, which only a whole-league publish could
    compute; the nightly job publishes one season, so the position is built
    from its parts instead. Same order, same ties.
    """
    from build_league_identity import MLB_TEAMS
    teams = sorted(MLB_TEAMS)
    t = teams.index(team) if team in teams else len(teams)
    return t * 10_000_000 + (int(season) - 2000) * 10_000 + grp_idx * 1_000 + idx


def flatten_scout(portrait_dir: Path, seasons: Optional[set[int]] = None) -> dict:
    """
    Flatten portraits into the Scout catalogue and player-season rows — all of
    them, or only `seasons`.

    The portraits already carry everything Scout needs — 18,664 player-seasons,
    every tag with the evidence sentence that earned it — so Scout reads them
    rather than re-deriving traits, which would be a second implementation of
    the tag system free to disagree with every other tab.

      tags    catalogue, tag -> {kind, family, side, n, dir}
      players one row per player-season-group, short keys (the shape the
              dashboard renderers use), tags as [tag, pct, evidence] triples.
              List order is the `ord` every Scout sort breaks ties on.
    """
    from team_portrait import kind_of

    GROUPS = (("hitters", "H"), ("starters", "P"), ("bullpen_arms", "P"))
    cat: dict[str, dict] = {}
    counts: Counter = Counter()
    pcts: defaultdict = defaultdict(list)
    sides: defaultdict = defaultdict(set)
    players: list[dict] = []

    for f in sorted(portrait_dir.glob("*_*.json")):
        if seasons and int(f.stem.rsplit("_", 1)[1]) not in seasons:
            continue
        try:
            p = json.loads(f.read_text())
        except Exception as exc:
            log.warning("  scout: skipping %s (%s)", f.name, exc)
            continue
        team, season = p.get("team"), p.get("season")
        for grp_idx, (grp, side) in enumerate(GROUPS):
            for idx, pl in enumerate((p.get("players") or {}).get(grp) or []):
                tg = []
                for t in (pl.get("traits") or []):
                    tag = t.get("tag")
                    if not tag:
                        continue
                    counts[tag] += 1
                    sides[tag].add(side)
                    if tag not in cat:
                        cat[tag] = {"kind": kind_of(tag) or "other",
                                    "family": t.get("family")}
                    pct = t.get("pct")
                    if isinstance(pct, (int, float)):
                        pcts[tag].append(float(pct))
                    tg.append([tag,
                               round(float(pct), 1) if isinstance(pct, (int, float)) else None,
                               t.get("evidence")])
                players.append({
                    "_ord": _scout_ord(team, season, grp_idx, idx),
                    "i": pl.get("player_id"), "n": pl.get("name"),
                    "t": team, "s": season, "d": side, "g": grp,
                    "a": pl.get("age"),
                    "v": pl.get("pa") if side == "H" else pl.get("bf"),
                    "w": round(w, 2) if isinstance((w := pl.get("war")), (int, float)) else None,
                    "l": bool(pl.get("limited_sample")),
                    "p": pl.get("home_position"), "h": pl.get("handedness"),
                    "pop": pl.get("populations") or [],
                    "tg": tg,
                    "m": {k: round(v, 1) for k, v in (pl.get("metrics_pct") or {}).items()
                          if isinstance(v, (int, float))},
                })

    # Direction. A tag's percentile is the percentile of the METRIC behind it,
    # not of the trait's strength, and for 22 of them low IS the trait:
    # `patient` is chase rate at the 5th percentile, `air-ball bat` is
    # ground-ball rate at the 5th. Rendering a bare 5 beside a 99 says the
    # opposite of what it means.
    #
    # Derivable rather than declared: measured over all 87,133 firings, 22
    # tags never exceed the 50th percentile, 37 never fall below it, and
    # exactly none straddle it. So the observed range settles direction
    # without a hand-maintained list that could drift from the gates.
    for tag, meta in cat.items():
        meta["n"] = counts[tag]
        s = sides[tag]
        meta["side"] = "B" if len(s) > 1 else next(iter(s), None)
        seen = pcts.get(tag)
        if not seen:
            meta["dir"] = None          # categorical: handedness, out-pitch
        elif max(seen) <= 50:
            meta["dir"] = "low"
        elif min(seen) >= 50:
            meta["dir"] = "high"
        else:
            meta["dir"] = "high"        # never observed; treat as ordinary

    return {"tags": cat, "players": players}


def _all_scout_rows(client, columns: str, seasons: Optional[set[int]] = None) -> list[dict]:
    """Every scout_players row (optionally only `seasons`), paging past the cap."""
    out: list[dict] = []

    def fetch(start: int) -> list[dict]:
        # A fresh builder per request: postgrest-py builders accumulate
        # parameters, so reusing one across pages would stack its ranges.
        q = client.table("scout_players").select(columns)
        if seasons:
            q = q.in_("season", sorted(seasons))
        return q.order("ord").range(start, start + 999).execute().data

    while True:
        page = _retry("list scout rows", lambda: fetch(len(out)))
        out.extend(page)
        if len(page) < 1000:
            return out


def publish_scout(root: Path, client, dry: bool,
                  seasons: Optional[set[int]] = None) -> None:
    """
    Publish Scout. Whole (the default) or only `seasons` — the nightly job's
    mode, where the machine holds one season's portraits and must not treat
    every other season's rows as stale. Season-scoped, the tag catalogue keeps
    its kind/family/side/direction and recounts `n` from the published rows.
    """
    import charts

    ix = flatten_scout(root / "data/processed/portraits", seasons)
    cat, players = ix["tags"], ix["players"]

    if seasons:
        # Catalogue as published, so vectors and directions use the same
        # metadata as every other season's rows.
        published = {r["tag"]: r for r in _retry("list tags", lambda: client
                     .table("scout_tags").select("*").execute().data)}
        cat = {**cat, **{t: {k: r[k] for k in ("kind", "family", "side", "n", "dir")}
                         for t, r in published.items()}}

    rows = []
    for p in players:
        vec = charts._comp_vector(p, cat)
        rows.append({
            "season": p["s"], "team": p["t"], "player_id": p["i"], "grp": p["g"],
            "ord": p["_ord"], "side": p["d"], "name": p["n"], "age": p["a"],
            "volume": p["v"], "war": p["w"], "limited": p["l"],
            "position": p["p"], "hand": p["h"], "populations": p["pop"],
            "metrics": p["m"], "tags": [g[0] for g in p["tg"]], "tag_detail": p["tg"],
            "vec": vec, "vec_norm": math.sqrt(sum(x * x for x in vec.values())),
        })

    scope = f"season(s) {', '.join(map(str, sorted(seasons)))}" if seasons else "all seasons"
    if dry:
        log.info("  scout: would publish %d player-seasons (%s)", len(rows), scope)
        return

    keep = {(r["season"], r["team"], r["player_id"], r["grp"]) for r in rows}
    for i in range(0, len(rows), SCOUT_BATCH):
        chunk = rows[i:i + SCOUT_BATCH]
        _retry(f"scout rows {i}", lambda: client.table("scout_players").upsert(chunk).execute())
    # Rows no longer in any portrait — within the published scope only.
    existing = _all_scout_rows(client, "season, team, player_id, grp", seasons)
    stale = [r for r in existing
             if (r["season"], r["team"], r["player_id"], r["grp"]) not in keep]
    for r in stale:
        _retry("delete stale scout row", lambda: client.table("scout_players").delete()
               .eq("season", r["season"]).eq("team", r["team"])
               .eq("player_id", r["player_id"]).eq("grp", r["grp"]).execute())

    if seasons:
        # Incidence across every season, from what is now published.
        counts: Counter = Counter()
        for r in _all_scout_rows(client, "tags"):
            counts.update(r["tags"] or [])
        for t, meta in cat.items():
            meta["n"] = counts.get(t, 0)
    tag_rows = [{"tag": t, **{k: m.get(k) for k in ("kind", "family", "side", "n", "dir")}}
                for t, m in cat.items() if m.get("n")]
    _retry("scout tags", lambda: client.table("scout_tags").upsert(tag_rows).execute())
    live = {r["tag"] for r in tag_rows}
    gone = [t["tag"] for t in _retry("list tags", lambda: client.table("scout_tags")
                                      .select("tag").execute().data)
            if t["tag"] not in live]
    if gone:
        _retry("delete retired tags", lambda: client.table("scout_tags").delete()
               .in_("tag", gone).execute())
    log.info("  scout: published %d player-seasons (%s), %d tags "
             "(%d stale rows, %d retired tags removed)",
             len(rows), scope, len(tag_rows), len(stale), len(gone))


# ---------------------------------------------------------------------------

def _point_at(root: Path) -> None:
    """
    Make the build fingerprint read `root`'s pools, for publishing from a
    checkout other than the one this code lives in. Normal use publishes from
    the checkout that built the portraits, where this is a no-op.
    """
    import team_portrait as tp
    if root.resolve() == (_HERE.parent).resolve():
        return
    tp.PROCESSED_DIR = root / "data" / "processed"
    tp._FINGERPRINT_EXTRA_POOLS = tuple(
        (root / "modules" / "processed", pat) for _d, pat in tp._FINGERPRINT_EXTRA_POOLS)
    tp.clear_fingerprint_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2015-2026", help="e.g. 2026 or 2015-2026")
    ap.add_argument("--only", choices=("portraits", "league", "scout"), action="append",
                    help="publish just these (repeatable); default all")
    ap.add_argument("--data-root", type=Path, default=_HERE.parent,
                    help="the baseball_construction directory holding the built data")
    ap.add_argument("--force", action="store_true", help="ignore content hashes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    a, _, b = args.seasons.partition("-")
    seasons = set(range(int(a), int(b or a) + 1))
    parts = set(args.only or ("portraits", "league", "scout"))
    root = args.data_root.resolve()
    _point_at(root)
    client = _write_client()

    refused = 0
    if "portraits" in parts:
        refused = publish_portraits(root, seasons, client, args.dry_run, args.force)
    if "league" in parts:
        publish_league(root, seasons, client, args.dry_run, args.force)
    if "scout" in parts:
        publish_scout(root, client, args.dry_run)
    return 1 if refused else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
