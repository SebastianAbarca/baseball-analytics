# Baseball Construction Analytics

How a major-league roster is built, read from pitch-level Statcast: what each
team is made of, what each player actually does, and who resembles whom.

Twelve seasons (2015–2026), 30 teams, **18,664 player-seasons**, 2.3M pitches
per season.

```bash
pip install -r baseball_construction/requirements.txt
python baseball_construction/dashboard/app.py     # http://localhost:8050
```

---

## The idea

A team's identity is not its record, and a player's identity is not his slash
line. Both are collections of specific, checkable claims — this pitcher hides
his slider behind his fastball, this hitter never chases, this bullpen gets
outs by strikeout rather than by contact.

The unit of output is a **tag**: a claim that fired because a player cleared a
league-percentile gate, carrying the number that earned it.

```
Juan Soto · NYY 2024
  patient          chase rate 4th pct — lays off pitches outside the zone
  power bat        ISO 91st pct
  walk machine     walk rate 97th pct
```

Ninety-four tags exist, 87,120 firings across twelve seasons, and every one
carries its evidence. A league-average player carries none — that is the point.
There are no boxes and nobody is assigned an archetype.

---

## Where to go

| | |
|---|---|
| **[baseball_construction/README.md](baseball_construction/README.md)** | the system in one page — the five views, the taxonomy, how it fits together |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | how data flows, with diagrams |
| **[docs/OPERATIONS.md](docs/OPERATIONS.md)** | the runbook — refreshing, rebuilding, failure modes |
| **[docs/DECISIONS.md](docs/DECISIONS.md)** | why things are this way, and the traps behind them |
| **[baseball_construction/TAGS.md](baseball_construction/TAGS.md)** | every tag, its gate, how often it fires |

New here? Read `baseball_construction/README.md`, then ARCHITECTURE. Most of
what looks odd in this repo is odd for a reason written down in DECISIONS.

---

## Layout

```
baseball_construction/
  modules/       ingest, aggregates, portrait build, tag gates, reliability
  scripts/       refresh.py (the only writer) + maintenance tools
  dashboard/     Dash app — read-only presentation
  data/raw/      Statcast parquet, one per season
  data/processed/  derived pools + portraits/
docs/            architecture, operations, decisions
Dockerfile       web tier
render.yaml      deployment
```

Python 3.12. Supabase credentials (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`) go
in `.env` and are only needed for Storage upload and seeding — the dashboard
runs without them off local portraits.
