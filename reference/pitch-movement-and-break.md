---
type: reference
status: living
updated: 2026-11-15
tags:
  - reference/pitching
  - reference/physics
---

# Pitch movement and break

> Break accumulates as the square of time, not linearly along the path. Almost
> every intuitive statement about *when* a pitch breaks is wrong because of
> this, and the project has the scars to prove it.

## Units first

`pfx_x` and `pfx_z` are in **feet**. A four-seam's `pfx_z` of 1.32 is 15.8
inches of induced vertical break, expressed in feet.

`tunneling.py` divided by 12 as though the value were already inches — a **12×
understatement** of the movement term. This is the cheapest possible bug to
write and among the hardest to notice, because the result stays in a plausible
numeric range.

## Break is quadratic in time

A pitch's trajectory is a constant-acceleration fit, so lateral and vertical
displacement go as t². That has a counterintuitive consequence: at the hitter's
commit point, **only 29.7% of the pitch's break has happened** — not the 57.5%
that linear interpolation along the flight path implies.

Combined with the unit error above, the movement term at the tunnel point was
understated **6.2×**: 0.45 inches against a typical **1.72 inches** of
release-point scatter. The movement signal was an order of magnitude below the
noise it was supposed to be compared with.

So `tunnel_score` was not measuring tunnelling. Its tunnel location correlated
**0.998 with release height** — it was a release-consistency metric wearing a
tunnelling label. **39% of pitchers changed `tunneler` status** when it was
fixed.

No ODE solver is needed to do this properly. Statcast's published kinematics
*are* a constant-acceleration fit, so solving the quadratic in closed form is
not an approximation of the data — it is the data.

## Late break is a constant, so it cannot be a tool

Across **708,559 pitches**, the share of total break occurring *after* the
commit point (23 ft) is **70.3%**, and it does not vary:

| curveball | sweeper | four-seam | sinker | cutter |
|---|---|---|---|---|
| 70.4% | 70.4% | 70.3% | 70.3% | 70.2% |

A 79.7 mph curveball and a 94.5 mph four-seam agree to within two-tenths of a
point. The reason is algebraic: if break goes as t², the late share is
(t₁/T)² — a ratio of *times*, fixed by the constant 60.5 ft distance. A slow
curve takes longer to reach the plate and proportionally longer to reach the
commit point, and the two cancel exactly.

**A tool has to vary between players.** This varies by 0.2 points, so it is not
one. What scouts mean by "late break" decomposes into three things already in
the taxonomy:

| scout's phrase | what it actually is | kind |
|---|---|---|
| total movement | how much it breaks | tool |
| velocity | how little time to react | tool |
| separation from the previous pitch | `tunneler` | behavior |

Only the third is genuinely about *lateness*, and it is a property of a **pair**
of pitches, which is why it can never be a per-pitch tool.

## Traps

- **Any per-pitch "when does it break" metric is suspect.** The physics makes
  it a constant. If yours varies between players, find out what it is really
  measuring before trusting it.
- **Check a movement term against the noise it competes with.** 0.45 inches of
  signal against 1.72 inches of release scatter was the tell.
- **A 0.998 correlation with something unrelated is a diagnosis.** Correlate
  any new metric against release height, velocity and pitch mix before
  believing its name.

## Open questions

- Statcast fits constant acceleration *per pitch*, so quadratic break is partly
  baked into the data. Real mid-flight variation would need finer tracking to
  detect at all. Does any public dataset support testing this?
- `tunneler` is a pairwise property. Is the pair chosen by sequence frequency,
  by release proximity, or something else — and is that choice documented
  anywhere outside the code?

## Sources

- [Decisions](../docs/DECISIONS.md) § What can and cannot be a tag → *Late break
  is a constant, so it is not a tool*; § Unit traps that produced wrong tags →
  *`pfx_x` and `pfx_z` are in feet* and *Break is quadratic in time, not linear
  in distance*
- [Reference index](README.md) · [Home](../HOME.md)
