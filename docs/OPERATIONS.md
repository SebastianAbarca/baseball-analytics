# Operations

Running, refreshing and rebuilding. For why it is shaped this way see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Run the dashboard

```bash
python baseball_construction/dashboard/app.py
# http://localhost:8050
```

Add `--debug` for hot reload; `PORT=8061` to run beside another instance.

It reads everything from Supabase with the anon key, locally exactly as in
production: `SUPABASE_URL` and `SUPABASE_ANON_KEY` in `.env` (or the
environment), internet access, and the project awake. There are no data files
to have and nothing to build — what it shows is what was last published. A
team-season that has not been published says so rather than building itself.

---

## Refresh the data

One command, and it is the only thing that writes:

```bash
python baseball_construction/scripts/refresh.py
```

Stages run in dependency order. One is opt-in.

| stage | what it does | default |
|---|---|---|
| `statcast` | tops up the in-progress season; fetches any game MLB's schedule lists that a cache lacks | on |
| `sources` | re-pulls FanGraphs/BRef/Savant, OAA, bWAR, def runs, IL | **off** |
| `seed` | writes the re-pulled season totals to `player_batting`/`player_pitching`, where the portraits read them — **writes production** | **off** |
| `league` | rebuilds every league cache (tunnel, pitch mix, spin, plate discipline, mechanics, sequencing) where Statcast moved | on |
| `portraits` | rebuilds whatever the fingerprint marks stale | on |
| `tags` | regenerates TAGS.md and tag_reference | on |
| `publish` | writes portraits, league seasons and Scout to the serving tables | on |

### Common invocations

```bash
# See what would change. Touches nothing.
python baseball_construction/scripts/refresh.py --dry-run

# Everything, including re-pulling every external feed and re-seeding the
# database from it (sources without seed changes nothing a portrait reads)
python baseball_construction/scripts/refresh.py \
    --stages statcast,sources,seed,league,portraits,tags,publish

# One season
python baseball_construction/scripts/refresh.py --seasons 2026

# Rebuild seasons in parallel (see the memory note below)
python baseball_construction/scripts/refresh.py --jobs 3

# After changing league-cache CODE rather than data (tunnel, pitch mix,
# spin, discipline, mechanics, sequencing); --force-tunnel still works
python baseball_construction/scripts/refresh.py --force-league
```

`--dry-run` is worth running first on anything unfamiliar. It reports staleness
per stage without writing, and it reads the same `ingest.statcast_gap()` the
real run does, so it cannot drift from what would actually happen.

---

## What invalidates what

| you changed | rebuilds |
|---|---|
| a tag threshold, `TAG_KIND`, a zone constant | **all 360 portraits** — the gate digest moved |
| the in-progress season's data | that season's 30 — finished seasons rank against a pool it is not in |
| a finished season's data | all 360 — finished seasons are in every season's era pool |
| raw Statcast for a season (top-up, backfill) | its six league caches and cross-team rows, then its portraits — the `league` stage does the first two |
| a chart, a layout, a callback | nothing; restart the dashboard |
| `TAGS.md` content | run `--stages tags`, about 3s |
| anything published | `publish.py` (or `--stages publish`); the dashboard picks it up within 10 min |

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

## Publishing

`publish.py` is the only thing that writes to the tables the dashboard reads.
It needs `SUPABASE_SERVICE_KEY` and runs where the portraits were built:

```bash
python baseball_construction/scripts/publish.py --dry-run
python baseball_construction/scripts/publish.py
```

It refuses any portrait that is not current for this machine's pools and gates
— rebuild or re-derive it first — and exits 1 if it refused anything. After
`rederive_traits.py`, run it too: a re-derive changes portraits on disk, and
nobody sees them until they are published.

The schema lives in `baseball_construction/schema_serving.sql`, applied by hand
in the Supabase SQL editor and safe to re-run.

Transient SSL errors on uploads (`BAD_RECORD_MAC`, `EOF occurred in violation
of protocol`) are normal on long runs and retried with backoff; a first full
publish hit four and lost none.

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
print('current fingerprint:', {s: tp.build_fingerprint(s) for s in range(2015, 2027)})
PY
```

Healthy output is 360 portraits, **1** distinct stamp, **0** stale. More than
one stamp means a rebuild did not finish or some builds failed.

---

## The nightly job

`.github/workflows/nightly.yml` runs `scripts/nightly.py` at 10:30 UTC every
day from March to October. With a season in progress it:

- tops up that season's Statcast;
- re-pulls its season totals and re-seeds them into the database (strict: it
  refuses to write rows without the pitch aggregates);
- force-rebuilds its 30 portraits and league identity;
- publishes that season's portraits, league data and Scout rows.

Finished seasons are never touched. With no season in progress it exits in
seconds. It fails the run, and GitHub emails, unless all 30 portraits are
current, publish refused none, and nothing else went stale.

Its data lives in GitHub's cache between runs. GitHub drops a cache unread for
7 days, so the first run each spring bootstraps: the Chadwick register, the
frozen era from `data/era/`, last season's Statcast and this season's so far.
Allow an hour or more for that run; later runs take minutes.

```bash
# Run it by hand (the default: whatever season is in progress)
gh workflow run nightly.yml

# Test against a finished season without writing anything
gh workflow run nightly.yml -f season=2026 -f no_publish=true

# The same, locally
python baseball_construction/scripts/nightly.py --season 2026 --no-publish
```

`keepalive.yml` also re-enables `keepalive`, `nightly` and `seed` every run.
GitHub disables scheduled workflows in a public repository after 60 days with
no commits, and the off-season is longer than that.

---

## Every year

**Before Opening Day:** add the season to `SEASON_DATES` in
`modules/ingest.py` (first and last regular-season dates) and to `SEASONS` in
`dashboard/layout.py`. From March the nightly job fails, deliberately, while a
season is missing. The first date is the first regular-season game
**anywhere** — Tokyo, Seoul and other international openers count, and three
seasons once started late because they were missed. Check it against MLB's
schedule (`statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&season=<year>`).
A wrong date no longer loses games silently — the `statcast` stage fetches any
completed game the cache lacks — but the dates still bound the BRef season
totals.

**After the last regular-season game — finalize.** On the machine with all the
data:

1. `python baseball_construction/scripts/refresh.py --seasons <year> --stages statcast,sources,seed,league`
   to take the season to its final game and write its final totals.
2. `python baseball_construction/modules/validation.py` — V16, V17 and V19
   confirm the season is complete, regular-season only, and its caches
   current. Do not continue on a failure.
3. `python baseball_construction/scripts/refresh.py --stages portraits,tags,publish --jobs 3` (about 20
   minutes). Every portrait re-ranks, because the season just joined the era
   pool.
4. `python baseball_construction/scripts/freeze_era.py`, then commit `data/era/`, `TAGS.md`,
   `dashboard/tag_reference.json` and `dashboard/strike_zones.json`.

Check SEASON_DATES' end date against MLB's schedule or Statcast's
`game_type` (`R` regular, `F` wild card, `D` division series) first. The
projected date ran a week into the postseason for 2026; 2024's ran a day
short and lost a doubleheader. Step 1's `statcast` stage logs
`INCOMPLETE vs MLB's schedule` if any completed game is still missing — do
not continue until it is clean.

A season counts as finished from the **second** day after its last game
(`ingest.season_state`): the morning after it, the nightly job still runs and
collects that night's games.

---

## Scripts other than refresh

| script | when |
|---|---|
| `build_league_identity.py <season>` | rebuild one season + its identity file; `refresh.py` calls this |
| `tag_reference.py --check` | exits 1 if TAGS.md disagrees with the code; good in CI |
| `repair_metric_units.py` | after discovering a unit bug; purges every derived layer |
| `rederive_traits.py` | replay tags from stored `gate_inputs` — 0.8s/season vs a full rebuild |
| `nightly.py` | the in-season night: one season, built and published — see the nightly job |
| `freeze_era.py` | at finalize: copy finished seasons' roots into `data/era/` for the nightly job; `--check` to verify |
| `publish.py` | write to the serving tables; `--dry-run`, `--only portraits\|league\|scout`, `--seasons`, `--force` to ignore content hashes. ~5 min for everything, seconds when little changed |
| `battery_base.py` | diagnostic: pitcher/catcher separability |
