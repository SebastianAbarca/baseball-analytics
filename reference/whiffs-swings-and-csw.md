---
type: reference
status: living
updated: 2026-11-15
tags:
  - reference/methodology
  - reference/pitching
---

# Whiffs, swings and CSW

> Every rate here is a fraction, and the only hard part is the denominator.
> "Whiff rate" means two different numbers depending on who is speaking.

## The two denominators

| name in this project | what it is | denominator |
|---|---|---|
| `whiff_per_swing` / `Whiff_pct` | whiffs per **swing** | swings |
| `swstr_pct` | whiffs per **pitch** | pitches |

Savant's arsenal "whiff%" is the **per-swing** figure. A per-pitch rate is
conventionally a swinging-strike rate. They are both legitimate; using one
under the other's name is not.

**CSW** is called strikes plus whiffs, over pitches — a single number for "did
this pitch get a strike without contact".

## A foul tip is a whiff

This is the definitional question that caused the mess. A foul tip is a pitch
the batter swung at, nicked, and did not put in play. Savant counts it as a
whiff, and the evidence is decisive:

- counting foul tips as whiffs, whiffs per swing match Savant's arsenal whiff%
  **within 0.04 points** across 1,228 pitcher × pitch-type rows (2025)
- counting them as contact, the figure is **2.1 points low on every row**

A uniform bias on every row is the signature of a definitional disagreement
rather than a sampling problem.

## What went wrong here

Three names, two rates, two definitions, across three modules:

- `pitch_aggregates` counted a foul tip as **contact**
- `pitch_mix` counted it as a **whiff**, and called its **per-pitch** rate
  `whiff_pct` — the name the other module used for the **per-swing** rate
- `team_portrait` then renamed that to `SwStr_pct`

Now `pitch_aggregates.WHIFF_EVENTS / SWING_EVENTS / CSW_EVENTS` are the only
definitions in the system, and the names above are the only names.

One deliberate exception: the out-pitch whiff floors stay on the **per-pitch**
scale they were originally fitted on, so the gate did not move when the
definitions were unified.

## Where the bars actually sit

The out-pitch gate is a worked example of why per-type matters. A pitcher's out
pitch is chosen by `argmax(whiff_pct × usage_pct)` over his non-fastballs, so
*everyone* with a secondary pitch has one. Ungated, the tag fired on **82% of
pitcher-seasons** (8,029 of 9,738) and answered "which of his pitches is least
bad?" — Chris Bassitt's fired at 10% whiffs.

The gate is now the 80th percentile of whiffs per pitch **for that pitch type**,
among every pitcher-season with 100+ of that pitch, 2015–2026
(`OUT_PITCH_WHIFF_FLOOR`). The bars differ more than you would guess:

| curveball | ... | splitter |
|---|---|---|
| 16.2% | → | 22.8% |

A median splitter outwhiffs a good curveball. One table covers all seasons
because the bars barely move — the slider's stayed within 19.8–22.2% from 2015
to 2026. Rare types (screwball, forkball, knuckleball, slow curve) share the
pooled non-fastball bar.

Result: **20%** of pitcher-seasons (1,915), and no other tag moved.

## Traps

- **Never compare a per-swing rate to a per-pitch rate.** They differ by the
  swing rate, which is itself a skill.
- **Say the denominator in the name.** The project's rule now; it is the only
  thing that prevented a recurrence.
- **Check against an external source at the row level.** A constant offset on
  every row means a definition mismatch; scattered offsets mean something else.

## Open questions

- Is a foul *bunt* attempt handled the same way as a foul tip?
- The out-pitch floors remain on the per-pitch scale for continuity. Is that
  noted where the constant is defined, so a future reader does not "fix" it?

## Sources

- [Decisions](../docs/DECISIONS.md) § Reliability → *A foul tip is a whiff, and
  each rate says what it divides by*; § What can and cannot be a tag →
  *Out-pitch tags need a top-20% whiff rate for their pitch type*
- [Plate discipline](plate-discipline.md) — the swing-decision side of the same
  events
- [Reference index](README.md) · [Home](../HOME.md)
