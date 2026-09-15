# Baseball Construction Analytics

How a major-league roster is built, read from pitch-level Statcast: what each
team is made of, what each player actually does, and who resembles whom.

Twelve seasons (2015–2026), 30 teams, **18,664 player-seasons**, 2.3M pitches
per season.

```bash
python baseball_construction/dashboard/app.py     # http://localhost:8050
```

---

## What it does

A team's identity is not its record, and a player's identity is not his slash
line. Both are collections of specific, checkable claims — this pitcher hides
his slider behind his fastball, this hitter never chases, this bullpen gets
outs by strikeout rather than by contact. The system makes those claims
explicit, one at a time, with the evidence attached.

The unit of output is a **tag**: a claim that fired because a player cleared a
league-percentile gate, carrying the number that earned it.

```
Juan Soto · NYY 2024
  patient          chase rate 4th pct — lays off pitches outside the zone
  power bat        ISO 91st pct
  walk machine     walk rate 97th pct
```

Ninety-four tags exist. **87,120 of them fire across the twelve seasons, and
every single one carries an evidence string.** A league-average player carries
none — that is the point. There are no boxes and nobody is assigned an
archetype.

---

## The five views

| tab | the question it answers |
|---|---|
| **Identity** | What is this team made of, and who else looks like it? |
| **Through the Years** | What has this franchise kept or abandoned since 2015? |
| **The Players** | Who is on the roster and what does each of them do? |
| **Compare** | How do these player-seasons actually differ? |
| **Scout** | Who has this trait — or who plays like this guy? |

Scout answers a tag query over all 18,664 player-seasons in **6 ms**, and will
also find a player's nearest comps by tag profile. Skubal 2024's closest comp
is Skubal 2025, which is the metric checking its own work.

---

## The taxonomy

Every tag sits on four axes. The one that does the work is **kind**, ordered by
how much control the player has over it:

| kind | what it is | examples |
|---|---|---|
| **attribute** | a fact about the body | throws left · sidearm · deep extension |
| **tool** | a capability that clears a bar | elite velo · ride four-seam · elite speed |
| **behavior** | a choice he makes | patient · sinker-baller · tunneler |
| **result** | what came of it | bat-misser · walk prone · elite command |
| **deployment** | the club's decision about him | high-leverage arm · platoon specialist |
| **noise** | results minus contact quality | lucky · unlucky |

The ordering matters when you read a roster. Three pitchers can all carry
`bat-misser` — a result — while one gets there on `elite velo` (a tool), one on
`tunneler` (a behavior), and one on nothing you can name, which is the
interesting case.

The other three axes: **family** (which part of the game), **side** (hitter or
pitcher — 40 tags are hitter-only, 53 pitcher-only, 1 shared), and
**population**, the denominator a percentile was taken against, because
`elite framer` means nothing without "among catchers with enough takes to
measure".

Full definitions, gates and incidence counts: **[TAGS.md](TAGS.md)** —
generated from the live maps, so it cannot drift from the code.

---

## How it fits together

Raw pitch data is reduced, once, to a 239 KB JSON **portrait** per team-season.
The dashboard only ever reads portraits — it holds no raw data and computes no
aggregates.

```
Statcast (977 MB)  ──  refresh.py  ──▶  360 portraits (88 MB)  ──▶  dashboard
       build machine, 1.23 GB peak            web tier, 200 MB RSS
```

Refresh everything with one command:

```bash
python baseball_construction/scripts/refresh.py
```

---

## Docs

| | |
|---|---|
| **[ARCHITECTURE.md](../docs/ARCHITECTURE.md)** | how data flows, with diagrams — cache layers, the build, the fingerprint, serving, deployment |
| **[OPERATIONS.md](../docs/OPERATIONS.md)** | the runbook — refreshing, rebuilding, timings, failure modes |
| **[DECISIONS.md](../docs/DECISIONS.md)** | why things are the way they are, and the traps behind them |
| **[TAGS.md](TAGS.md)** | every tag, its gate, and how often it fires |

New to the codebase: read ARCHITECTURE first, then skim DECISIONS. Most of what
looks odd in this repo is odd for a reason that is written down there.

---

## Layout

```
baseball_construction/
  modules/       ingest, aggregates, portrait build, tag gates, reliability
  scripts/       refresh.py (the only writer) + maintenance tools
  dashboard/     Dash app — read-only presentation
  data/
    raw/         Statcast parquet, one per season
    processed/   derived pools + portraits/
docs/            architecture, operations, decisions
```

Requires Python 3.12. `pip install -r baseball_construction/requirements.txt`.

Supabase credentials go in `.env` (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`) and
are only needed for Storage upload and the seeding scripts — the dashboard runs
without them off local portraits.
