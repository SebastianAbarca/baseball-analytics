---
type: reference
status: living
updated: 2026-11-15
tags:
  - reference/methodology
  - reference/data-quality
---

# Percentiles vs ranks

> A percentile rank says *where you stand*. A rate says *what you did*. They
> are not interchangeable, and a column name rarely tells you which you have.

## What it is

A rate is a measurement in physical units — 94.5 mph, 28% whiffs. A percentile
rank maps that measurement onto 1–100 against some comparison pool. The
mapping destroys the units and the shape of the distribution: ranks are
uniformly distributed by construction, whatever the underlying rates look like.

Two consequences follow, and both bite:

1. **An absolute threshold on a rank fires on a fixed share of the pool.** A
   bar of "above 90" selects the top 10%, always, no matter whether the real
   spread is wide or vanishingly narrow.
2. **A rank is only meaningful against its pool.** The same rate ranks
   differently among all players, among qualified players, or within a pitch
   type.

## Why it matters here

This is the single most expensive data-quality trap the project has hit.
Savant's leaderboard columns are *named* like rates and sometimes *are* ranks:

- `exit_velocity` had a median of **49** — not 88 mph, because the column was a
  percentile rank
- `fb_velocity`, likewise, median 49 rather than ~94
- `sprint_speed` was **mixed** — some rows rates, some rows ranks

The result was **26.2% of `elite speed` tags bogus**, including Jacob deGrom in
four separate seasons. A pitcher with elite sprint speed is the kind of error
that is obvious once seen and invisible in aggregate.

The standing rule from that episode: **derive rates from Statcast, and treat
any Savant column named like a rate as suspect until checked.**

## Traps

- **Direction is per metric, not global.** A 5th-percentile chase rate is
  elite; a 95th-percentile velocity is elite. The project stores a tag's `pct`
  as a percentile in the metric's own direction, which means display, sorting
  and the similarity vector all have to invert the low-is-good ones. Getting it
  backwards put the *least* patient hitters top of a search for the most
  patient — and in a similarity metric there is no obviously right answer to
  check against, so it would have gone unnoticed.
- **Pooling across pitch types ranks the pitch mix, not the skill.** See
  [Spin efficiency](spin-efficiency.md) for the case where this produced a
  metric that measured how few sliders a pitcher throws.
- **A median near 49–50 is the tell.** Any column whose median sits there is a
  rank until proven otherwise.

## Open questions

- Is there a cheap invariant worth asserting at ingest — flagging any column
  whose median lands in, say, 45–55 and whose range is 1–100?
- Which Savant columns have been positively confirmed as rates, and is that
  list written down anywhere other than in the code?

## Sources

- [Decisions](../docs/DECISIONS.md) § Unit traps that produced wrong tags →
  *Savant's percentile columns are ranks, not rates*, and *Percentile direction*
- [Conceptual review — 14 October 2026](../docs/reviews/CONCEPTUAL-REVIEW-2026-10-14.md)
  § 6, on defining what each percentile and team average means
- [Reference index](README.md) · [Home](../HOME.md)
