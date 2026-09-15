# Operations

Running, refreshing and rebuilding. For why it is shaped this way see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Run the dashboard

```bash
python baseball_construction/dashboard/app.py
# http://localhost:8050
```

Add `--debug` for hot reload. In any deployment set `BASEBALL_READ_ONLY=1`
first — see [read-only mode](#read-only-mode).

---

## Refresh the data

One command, and it is the only thing that writes:

```bash
python baseball_construction/scripts/refresh.py
```

Stages run in dependency order. Two are opt-in.

| stage | what it does | default |
|---|---|---|
| `statcast` | tops up the in-progress season | on |
| `sources` | re-pulls FanGraphs/BRef/Savant, OAA, bWAR, def runs, IL | **off** |
| `tunnel` | rebuilds tunnel caches where Statcast moved | on |
| `portraits` | rebuilds whatever the fingerprint marks stale | on |
| `tags` | regenerates TAGS.md, tag_reference, scout_index | on |
| `upload` | pushes portraits to Supabase Storage | **off** |

### Common invocations

```bash
# See what would change. Touches nothing.
python baseball_construction/scripts/refresh.py --dry-run

# Everything, including re-pulling every external feed
python baseball_construction/scripts/refresh.py \
    --stages statcast,sources,tunnel,portraits,tags

# One season
python baseball_construction/scripts/refresh.py --seasons 2026

# Rebuild seasons in parallel (see the memory note below)
python baseball_construction/scripts/refresh.py --jobs 3

# After changing tunnelling CODE rather than data
python baseball_construction/scripts/refresh.py --force-tunnel
```

`--dry-run` is worth running first on anything unfamiliar. It reports staleness
per stage without writing, and it reads the same `ingest.statcast_gap()` the
real run does, so it cannot drift from what would actually happen.

---

## What invalidates what

| you changed | rebuilds |
|---|---|
| a tag threshold, `TAG_KIND`, a zone constant | **all 360 portraits** — the gate digest moved |
| the in-progress season's Statcast | all 360 — the pool digest is global (see below) |
| a chart, a layout, a callback | nothing; restart the dashboard |
| `TAGS.md` or `scout_index` content | run `--stages tags`, about 3s |

**The global-pool caveat.** Topping up 2026 changes
`tunnel_league_2026.parquet`, which changes the fingerprint, which marks every
season stale — including 2015, whose portrait never reads that file. Five days
of new games therefore costs a full rebuild. This is a known limitation, not a
bug in the data; see the fingerprint section in ARCHITECTURE.

---

## Timings

Measured on a 2025 rebuild, M-series laptop:

| operation | time |
|---|---|
| one portrait, cold process | 16.3s |
| one portrait, warm process | 3.6–4.4s |
| one season (30 teams) | ~2 min |
| **full rebuild, 12 seasons, `--jobs 3`** | **~18 min** |
| full rebuild, sequential | ~28 min |
| `sources` for one season | 35s |
| `tags` only | 3s |
| `--dry-run` | 3–5s |

The first team in a process pays ~10s of league-wide setup — runner control
alone is 4.7s — which is memoised in-process and then free for the other 29.

---

## Memory

Each worker holds a full season of Statcast: **1.23 GB** resident for 712k
pitches. On a 17 GB machine `--jobs 3` sits at about 2.6 GB total with memory
pressure around 33% free, which is comfortable. `--jobs 4` is tighter.

Do not browse the dashboard while a rebuild runs. Doing so once took seasons
from 220s to 1382s — a 6× slowdown from memory contention with the browser.

---

## Read-only mode

```bash
BASEBALL_READ_ONLY=1 python baseball_construction/dashboard/app.py
```

Off by default so local development is unchanged. **On in every deployment.**

Without it, a portrait that is missing from both the disk cache and Supabase
Storage causes the dashboard to load a full season of Statcast and build it —
inside the web container, which does not have the raw data and does not have
the gigabyte. With it, a miss returns a banner naming the team and pointing at
`refresh.py`.

It also permanently closes the race that once wrote a portrait built against
half-regenerated pools. `refresh.py` additionally refuses to start while
anything is listening on :8050; `--allow-dashboard` overrides that, and
`--dry-run` does not need it.

---

## Failure modes seen in practice

**FanGraphs returns 403.** Expected since 2024. `pull_fg_batting` falls through
to Baseball Reference + Savant, then to Savant alone. Nothing to do.

**`sources` reports success but changes nothing.** pybaseball's own HTTP cache
served the response. `refresh.py` now disables it for that stage; if you are
calling a puller directly, disable it yourself or you will get stale numbers
back and no indication.

**`FAILED <TEAM>: Server disconnected`.** Supabase dropped concurrent
connections — most likely several workers opening them at once on a free-tier
instance that was cold. The run continues and keeps the old portraits, so it
can report 30/30 while leaving some stale. `refresh.py` now recounts stale
portraits afterwards and warns. Re-running the stage picks up exactly those; it
took 47s for three.

**Supabase cold start.** The free tier auto-pauses. Expect NXDOMAIN → 521 →
schema-cache errors → ready, over a minute or two. Retry rather than debug.

**Portraits stale after a code change you did not expect to matter.** Check
whether the constant you touched is in `_gate_digest()`. If it is not and it
affects tags, that is the bug — the fingerprint should have moved.

---

## Verifying a rebuild landed

```bash
python - <<'PY'
import sys, json, glob, collections
sys.path.insert(0, 'baseball_construction/modules')
import team_portrait as tp
c, stale = collections.Counter(), 0
for f in glob.glob('baseball_construction/data/processed/portraits/*.json'):
    p = json.load(open(f))
    c[p.get('build_fingerprint')] += 1
    if not tp.portrait_is_current(p):
        stale += 1
print('portraits:', sum(c.values()), '| distinct stamps:', len(c), '| stale:', stale)
print('current fingerprint:', tp.build_fingerprint())
PY
```

Healthy output is 360 portraits, **1** distinct stamp, **0** stale. More than
one stamp means a rebuild did not finish or some builds failed.

---

## Adding a season

1. Add it to `SEASON_DATES` in `ingest.py` with real start and end dates.
2. `refresh.py --seasons <year>` — it will pull Statcast in full the first time.
3. Add it to `SEASONS` in `dashboard/layout.py` so it appears in the picker.

---

## Scripts other than refresh

| script | when |
|---|---|
| `build_league_identity.py <season>` | rebuild one season + its identity file; `refresh.py` calls this |
| `tag_reference.py --check` | exits 1 if TAGS.md disagrees with the code; good in CI |
| `repair_metric_units.py` | after discovering a unit bug; purges every derived layer |
| `rederive_traits.py` | replay tags from stored `gate_inputs` — 0.8s/season vs a full rebuild |
| `upload_portraits.py` | push to Supabase Storage |
| `battery_base.py` | diagnostic: pitcher/catcher separability |
