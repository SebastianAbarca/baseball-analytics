---
type: reference
status: living
updated: 2026-11-15
tags:
  - reference/pitching
  - reference/physics
---

# Spin efficiency

> The share of a pitch's spin that actually moves the ball. Spin around the
> axis of travel — gyro spin — produces no Magnus force, so it does nothing to
> the flight path, however fast the ball is spinning.

## What it is

Total spin splits into two parts. **Transverse (active) spin** generates Magnus
force and therefore movement. **Gyro spin** is rotation about the direction of
travel and generates none. Spin efficiency is the ratio of active spin to total
spin.

High efficiency is not a virtue in itself — it is a description of pitch type.
A four-seam fastball wants high efficiency so it rides. A slider is *defined*
by low efficiency: gyro spin is what makes it a slider. This matters enormously
for how the metric can be ranked (below).

A hard ceiling worth knowing: **seam-shifted wake** moves a ball without spin
doing the work. Any method that infers efficiency *from observed movement*
therefore credits seam effects to spin, and changeups and splitters top out
near **0.6–0.65** under every movement-inferred method, Savant's own included.

## Why it matters here

The project's first implementation was wrong in an instructive way. It computed
a `theoretical_max` as `0.00153 × rpm / mph × 12`, and that constant had no
traceable source and was roughly **4× too small**.

The effect: **90% of 2025 pitches sat at or above 1.0 before clipping** (median
ratio 3.0). Every fastball, sinker and changeup read exactly 1.0.

Once most of the arsenal is pinned at the ceiling, the only thing left varying
is the mix. The pooled percentile therefore measured *how few sliders a pitcher
throws* — Spearman **−0.58** with slider share, and **half the pitchers tagged
`high spin efficiency` threw none at all**. Against Savant's measured active
spin it ranked four-seamers at Spearman **0.20**, which is near noise.

This is the clearest case in the project of a metric that looked plausible,
clipped quietly, and ended up measuring something nobody asked about. It is
also a worked example of [Percentiles vs ranks](percentiles-vs-ranks.md): the
ranking was fine, the thing being ranked was not.

## How it is computed now

- **2020 onward — Savant's measured active spin.** The 3D spin vector from the
  active-spin leaderboard. Measured, not inferred.
- **Before 2020 — a physics estimate.** Magnus acceleration from the
  9-parameter fit with gravity and drag removed, air density from park
  elevation, and the Sawicki lift curve inverted. It ranks like Savant's
  measurement (FF 0.92, CU 0.93, SL 0.83) and level with Savant's own
  movement-inferred numbers. One level factor per pitch type, fitted on
  2020–2025 and stable within ±5% by season, puts it on Savant's scale.
- **Percentiles within pitch type, always.** A pitcher's figure is the
  pitch-weighted mean of his within-type percentiles, re-ranked across
  pitchers. Pooling across types ranks arsenal composition instead of skill.
- **Guarded.** V13 holds the physics estimate to per-type Spearman floors and a
  ±15% level band against Savant, so it cannot drift silently again.

## Traps

- **Never pool across pitch types.** It is the original bug in a different
  costume.
- **The lift curve is least certain at slider and cutter spin factors**, which
  is why those carry explicit 1.65 / 1.92 level factors. Treat pre-2020 slider
  efficiency as the softest number in the system.
- **A clip is a silence.** The failure was invisible because clipping produced
  a legal-looking 1.0 rather than an error. Any ratio with a ceiling deserves a
  check on *how often it sits at the ceiling*.

## Open questions

- Is the ±15% level band and the per-type Spearman floor checked on every
  build, or only when spin code changes?
- Pre-2020 seasons depend on park elevation for air density — how is a neutral
  or unknown park handled?
- Is there a defensible way to separate seam-shifted wake from spin at all, or
  is 0.6–0.65 simply the honest ceiling for off-speed pitches?

## Sources

- [Decisions](../docs/DECISIONS.md) § Unit traps that produced wrong tags →
  *Spin efficiency was a clipped constant, not a measurement*
- [Operations](../docs/OPERATIONS.md) — `SPIN_METHOD_VERSION` and rebuilding the
  spin caches
- [Reference index](README.md) · [Home](../HOME.md)
