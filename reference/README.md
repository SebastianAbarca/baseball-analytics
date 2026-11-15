---
type: index
status: living
updated: 2026-11-15
tags:
  - reference/index
---

# Baseball reference

What the concepts mean, kept separately from what this project decided to do
about them.

The split is deliberate. [Decisions](../docs/DECISIONS.md) records *positions* —
what was chosen and what breaks if it is undone. These notes record
*meaning* — what a metric is, what it divides by, what it cannot tell you.
A decision that turns on meaning links here, so the meaning is written once and
backlinked from everywhere it mattered.

Public on purpose: this is the reasoning behind the project, not a scratchpad.

## Notes

| | |
|---|---|
| [Percentiles vs ranks](percentiles-vs-ranks.md) | a rank is not a rate, and mixing them silently fabricates tags |
| [Spin efficiency](spin-efficiency.md) | active spin, gyro spin, and why it must be ranked within pitch type |
| [The strike zone](the-strike-zone.md) | ABS, the ball-radius buffer, and the 2026 `sz_top` redefinition |
| [Pitch movement and break](pitch-movement-and-break.md) | `pfx` units, break as t², and why late break is not a tool |
| [Whiffs, swings and CSW](whiffs-swings-and-csw.md) | foul tips, and what each rate divides by |
| [Plate discipline](plate-discipline.md) | chase, zone-swing, and discrimination as the actual skill |
| [Service time](service-time.md) | how MLB counts it, and why it cannot be estimated from a debut date |

## Writing one of these

Each note answers, in order: what it is, why it matters *here*, what the traps
are, and what is still open. Anything taken from the project's own
investigation cites the section of [Decisions](../docs/DECISIONS.md) it came
from — so a reader can tell a measured finding from received wisdom.

Mark what you are unsure about. An "Open questions" section with a real
question in it is worth more than a confident sentence that turns out to be
wrong.

- [Home](../HOME.md)
