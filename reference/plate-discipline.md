---
type: reference
status: living
updated: 2026-11-15
tags:
  - reference/hitting
---

# Plate discipline

> Swinging *less* is not discipline. Swinging at strikes and not at balls is.
> The distinction is the whole reason this project has a `zone hunter` tag.

## The raw rates

| rate | what it is |
|---|---|
| zone% | share of pitches seen that were in the zone |
| zone-swing% | swings at pitches in the zone, over in-zone pitches |
| chase% | swings at pitches **outside** the zone, over out-of-zone pitches |

All three depend on where the zone's edges are, which is not a fixed question —
see [The strike zone](the-strike-zone.md), and note that the vertical zone
changed meaning in 2026.

## Discrimination is the skill

Raw zone-swing rate is a poor measure of judgment because it is mostly a
measure of aggression. In this project's data it correlates **+0.81 with
first-pitch-strike rate** — a tag built on it would largely restate
`aggressive`.

So `zone hunter` fires on the **gap** between in-zone and out-of-zone swing
rates: swinging at strikes *more than* at balls. That gap is what the name
implies and what raw zone-swing% fails to isolate.

This generalises. Whenever two rates share a driver, the difference between
them often carries the signal and each one alone carries the driver.

## Direction matters, per metric

For `patient`, a **5th-percentile** chase rate is elite. For a velocity tag, a
95th-percentile is elite. A tag's percentile is stored in the metric's own
direction, so display, sorting and the similarity vector all have to invert the
low-is-good ones.

Getting this backwards put the *least* patient hitters at the top of a search
for the most patient. See [Percentiles vs ranks](percentiles-vs-ranks.md) — in
the similarity metric the same error has no obviously right answer to check
against, which is what makes it dangerous.

## Why `spectrum` was removed

Adjacent, and the same lesson. A single power-versus-contact score is a ratio,
and **ratios normalise away the thing you care about**. Martín Maldonado scored
66 on the power side; Aaron Judge, 37. Both are true as ratios and neither is
useful. **43% of hitters above spectrum 80 had an xwOBA below the 40th
percentile** — the score was finding high-variance hitters, not good ones.

It was replaced with a two-axis power/contact plane where both axes are
absolute and results are colour. A hitter who is genuinely both now sits in the
corner instead of averaging to the middle.

## Traps

- **A ratio of two skills measures neither.** Keep axes absolute and let the
  reader see both.
- **Check any new discipline metric against first-pitch-strike rate and overall
  swing rate.** If it correlates above ~0.8 with either, it is an aggression
  metric.
- **2026 breaks the series.** Chase% and zone% both move when the vertical zone
  definition changes, independently of any hitter changing.

## Open questions

- Is chase% computed against the ABS zone (half-width 0.833 ft) or the rulebook
  plate edge (0.7083 ft)? The choice shifts every borderline pitch between the
  numerator and the denominator.
- Does `patient` use chase rate alone, or chase combined with overall swing
  rate?
- Is there a pitch-type breakdown of discrimination — judgment against breaking
  balls is a different skill from judgment against fastballs.

## Sources

- [Decisions](../docs/DECISIONS.md) § What can and cannot be a tag → *`zone
  hunter` is built on discrimination, not zone-swing rate* and *Spectrum was
  removed*; § Unit traps → *Percentile direction*
- [Whiffs, swings and CSW](whiffs-swings-and-csw.md) — the contact side of the
  same swing events
- [Reference index](README.md) · [Home](../HOME.md)
