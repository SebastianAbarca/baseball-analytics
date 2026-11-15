---
type: reference
status: living
updated: 2026-11-15
tags:
  - reference/rules
  - reference/umpiring
---

# The strike zone

> There are at least three zones in play: the rulebook zone, the zone ABS
> actually judges, and the zone umpires used to call. The project keeps
> separate constants for them on purpose.

## The rulebook zone

Home plate is 17 inches wide, so the plate's own half-width is **0.7083 ft**.
Vertically the zone is defined relative to the batter's stance, which is why its
top and bottom arrive per pitch rather than as constants.

This is the zone *drawn* on the 3D arsenal chart, because it is the zone the
rules describe.

## The zone ABS judges

Measured against **2,350 ABS challenge verdicts**, the zone that reproduces them
is:

- the **front plane** of home plate, not the middle or the back
- the plate's 17 inches grown by **a full ball radius on each side**
- half-width **0.833 ft**, with **99.8% agreement**

Both the front-plane detail and the ball-radius buffer materially change
borderline calls, and neither is guessable — they were recovered from the
verdicts rather than assumed.

## Vertical: `sz_top` / `sz_bot` changed meaning in 2026

This is a change in what the *column* means, not in how batters stand:

| | through 2025 | 2026 onward |
|---|---|---|
| source | per-pitch human estimate of the stance | computed from listed height |
| formula | — | 53.5% and 27% of listed height |

League mean top moved from **3.435 ft to 3.214 ft** — the zone lost **2.9
inches** overnight. Inverting the 2026 formula recovers real listed heights
(Judge 79.0 against 79, Altuve 65.6 against 66), which is how the change was
confirmed rather than inferred.

Any season-over-season comparison of zone%, chase%, CSW% or framing crosses a
definitional break at 2026.

## Umpires were already converging on it

The 50% called-strike edge moved **0.945 → 0.843 ft** against ABS's 0.829.
Umpires closed on the rule's zone for years before ABS arrived. This is why
framing skill was compressing *before* the system was introduced, and it is the
reason the project kept framing tags with a data note rather than dropping them.

## Why batter height is not in the portrait schema

It looks like it belongs — ABS is defined per batter, so a league-average zone
is the right average and the wrong individual. But `pitch_aggregates` already
reads the **per-pitch** `sz_top`/`sz_bot`, which *is* that batter's zone, for
zone%, chase%, CSW% and framing — the metrics that feed tags. Statcast omits
them on **0.27%** of pitches, and those now fall back to the median of the same
data, which makes the fallback era-correct without being told the era.

The only two consumers of `charts.strike_zone()` legitimately want a league
average: the era note, and the 3D arsenal chart, which draws one pitcher's
pitches against no particular batter.

## Open questions

- The docs give the ABS half-width as **0.833 ft** in one place and refer to
  "ABS's **0.829**" in another. A ball radius of about 1.45 in added to
  0.7083 ft gives 0.829; 0.833 implies 1.5 in. Worth settling which constant
  the code actually uses, since the difference sits exactly on the borderline
  calls the zone exists to decide.
- Does the 53.5% / 27% formula invert cleanly for every batter, or are there
  listed-height records where it fails?
- Framing has a dashboard data note for the 2026 line. Is there an equivalent
  note for zone% and chase%, which cross the same break?

## Sources

- [Decisions](../docs/DECISIONS.md) § The strike zone — all four sub-decisions,
  and § Caching and staleness for how era-correct fallbacks are kept
- [Reference index](README.md) · [Home](../HOME.md)
