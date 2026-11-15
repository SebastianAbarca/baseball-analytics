---
type: index
status: living
updated: 2026-11-15
tags:
  - vault/home
---

# Home

The front door to this vault. Three kinds of writing live here, and they are
meant to link to each other.

For what the project *is* — the idea, the tag format, the numbers — start at
the [repo README](README.md). This note is about how the vault is wired.

## 1. Project docs — how this thing works

Canonical, committed, public. These describe the system as it actually is.

| | |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | how data flows — cache layers, the build, the fingerprint, serving, deployment |
| [Operations](docs/OPERATIONS.md) | the runbook — refreshing, rebuilding, timings, failure modes |
| [Decisions](docs/DECISIONS.md) | why things are the way they are, and the traps behind them |
| [Code README](baseball_construction/README.md) | what the project is, the five views, the taxonomy, layout |

Reviews — dated snapshots, not living documents:

- [Review — 12 October 2026](docs/REVIEW-2026-10-12.md) — the first full implementation and conceptual pass
- [Review — 14 October 2026](docs/REVIEW-2026-10-14.md) — follow-up, with closure against the above
- [Conceptual review — 14 October 2026](docs/CONCEPTUAL-REVIEW-2026-10-14.md) — mathematics, evidence, and baseball meaning

## 2. Baseball reference — what the concepts mean

[Reference index](reference/README.md) — the domain knowledge the decisions rest
on. Public, because it is the reasoning behind the project, not a scratchpad.

When a decision in [Decisions](docs/DECISIONS.md) turns on what a metric
*means*, the meaning belongs in `reference/` and the decision links to it. That
link is the point: one note on spin efficiency, backlinked from every place the
project had to take a position on it.

## 3. Planning — what to do next

`planning/` is private. It is a separate git repository nested inside this one
and ignored by it, so nothing there is published. It will not appear on GitHub
and it is not in this repo's history.

If you are reading this on GitHub, that is why there is no link here.

## How this vault is wired

- **Markdown links, not wikilinks.** This repo is public, so links are
  `[Decisions](docs/DECISIONS.md)` rather than `[[Decisions]]`. Obsidian builds
  backlinks and graph view from these identically; double brackets would render
  as broken literal text on GitHub.
- **Renames are safe.** `alwaysUpdateLinks` is on, so moving or renaming a note
  rewrites every link to it.
- **Settings are shared, window layout is not.** `.obsidian/app.json` is
  committed so both machines agree; `workspace.json` is ignored.
- **Sync is git.** Pull on whichever machine you sit down at. `planning/` syncs
  through its own private remote.
