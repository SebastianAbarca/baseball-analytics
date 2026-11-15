---
type: reference
status: living
updated: 2026-11-15
tags:
  - reference/rules
  - reference/roster
---

# Service time

> Time on a major-league roster, counted in days, written as `years.days`. It
> cannot be estimated from a debut date, and the attempt to do so was wrong in
> both directions at once.

## How it is counted

A championship season credits **172 days** as one full year of service, and
service is written `years.days` — so `3.045` is three years and forty-five
days. Service drives arbitration eligibility and free agency, which is why a
handful of days is worth arguing about.

The crucial subtlety: **days on the major-league injured list earn service.** So
do bereavement, paternity, family-emergency and suspended-player days. A player
can accrue service while not playing at all.

## Why it cannot be estimated

The original implementation in `database.py` assumed a player was in the majors
every day from his debut, then **subtracted injured-list days**. Both halves are
wrong, in opposite directions:

- assuming continuous presence since debut **over-credits** anyone who was
  optioned back down
- subtracting IL days **under-credits** anyone who was hurt, since those days
  count

Errors that cancel on average while being wrong for individuals are the worst
kind, because aggregate checks pass.

## How it is derived now

Two independent sources, deliberately:

**MLB's published service lists** (`scripts/pull_mls.py`) — one per January,
2006–2025, giving career service in `years.days` for everyone under reserve.
They carry pre-2006 careers inside their totals, and they credit things a
roster archive cannot see: suspended players keep accruing, which is how
A-Rod's 2014 is handled correctly.

**The 40-man roster ledger** (`scripts/pull_rosters.py`) — every club's roster
on every regular-season day from 2009, which is as far back as MLB's archive is
complete. Active, MLB injured list, bereavement, paternity, family emergency
and suspended days all count. 2020's 67 days count as a full season. From 2015,
a game appearance fills a day the archive dropped (which is how Figgins in 2009
is recovered).

**Combining them:** service through a season is that season's list where the
player appears on it; otherwise his last list plus ledger days since. 2026 is
the 2025 list plus the ledger.

## The identity problem

The service lists carry **names, not player ids**. Matching runs through
Chadwick using position, playing years, nickname groups and typo tolerance.
Players who genuinely share a name are then separated by whether their totals
**chain** — a career total never falls, and grows by at most one season's worth
between consecutive lists. That monotonicity constraint is doing real work; it
is the only thing distinguishing two same-named players with similar careers.

## How well it agrees

Lists and ledger agree **within 10 days for 96–99% of players** in every season
compared. Two independent derivations landing that close is the reason to trust
either.

## Traps

- **Never subtract IL days.** They count.
- **Never infer presence from a debut date.** Options and recalls are invisible
  to that method.
- **2020 is special** — 67 days, credited as a full season.
- **Pre-2009 has no ledger.** Those seasons rest on the published lists alone.

## Open questions

- The ledger starts in 2009 and the lists in 2006. What is the honest
  uncertainty on a career that began before 2006, where only the carried total
  is available?
- Minor-league option days and the 20-day rule do not appear above. Are they
  simply out of scope for this project's use?
- Does anything downstream display service time to a user, or is it only used
  as a gate? That determines how much the residual 1–4% disagreement matters.

## Sources

- [Decisions](../docs/DECISIONS.md) § Reliability → *Service time is MLB's own
  number, checked against every roster day*
- [Operations](../docs/OPERATIONS.md) — `scripts/pull_mls.py` and
  `scripts/pull_rosters.py`
- [Review — 12 October 2026](../docs/reviews/REVIEW-2026-10-12.md) § 11, and
  [Review — 14 October 2026](../docs/reviews/REVIEW-2026-10-14.md), where the IL-day
  subtraction was still recorded as open
- [Reference index](README.md) · [Home](../HOME.md)
