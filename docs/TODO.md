---
type: todo
status: living
updated: 2026-11-15
tags:
  - project/todo
---

# TODO

Every open item collected in one place: the October 2026 reviews, the questions
the reference notes raised, and anything found since.

> **Most of this is unverified against current code.** The reviews are dated
> 12–14 October 2026. The schema has moved from 20 to 29 since, and a few items
> below are already fixed — those are marked. Anything marked `unverified` was
> true in October and has not been checked. **Verify before acting.**

Status in this file means:

| | |
|---|---|
| `OPEN` | checked on 15 Nov 2026 and still true |
| `CLOSED` | checked on 15 Nov 2026 and fixed |
| `unverified` | asserted in October, not since checked |

The codebase carries **no `TODO` or `FIXME` comments at all** — the convention
here is that open work lives in prose, which is why this file exists.

---

## 1. Confirmed defects

From [Review — 14 October 2026](reviews/REVIEW-2026-10-14.md). These were the
P1/P2 lists; none has been re-run.

### P1

| # | item | status |
|---|---|---|
| 1 | Sprint-speed percentiles still become feet per second | unverified |
| 2 | Build failure still does not prevent publication | unverified |
| 3 | Scoped publishing can still delete unrelated Scout seasons | unverified |
| 4 | Expanded fingerprints still allow stale inputs to look current | unverified — but `beea0b9` *"Portraits detect their own staleness via a build fingerprint"* and the schema-20 cache purge both touch this |

Item 1 is the same family as
[Percentiles vs ranks](../reference/percentiles-vs-ranks.md) — a rank treated as
a rate. Worth checking whether the fix was local to sprint speed or systematic.

### P2

| item | status |
|---|---|
| Pitch outcomes and tunnel adjacency remain inconsistent | unverified |
| Steal-attempt ranks remain teammate-relative | unverified |
| Compare keys still omit hitter/pitcher identity | unverified |
| Service time still subtracts injured-list days | **CLOSED** |

Service time: [Decisions](DECISIONS.md) § Reliability documents `service_time.py`
using MLB's published lists plus the 40-man roster ledger, and records that IL
days earn service. `scripts/pull_mls.py` and `scripts/pull_rosters.py` exist.
Background in [Service time](../reference/service-time.md).

### Found since

Not from the reviews — found by reading the code on 15 November 2026.

| item | status |
|---|---|
| `tag_reference.py` blanks `TAGS.md` when no portraits are present | **OPEN** |

`build()` in `scripts/tag_reference.py` calls `scan_portraits()` and never
checks whether it found anything. Run with `data/processed/portraits/` empty —
the normal state on any machine that has not done a full build, including the
PC this was written on — it does not fail. It writes a header reading `0 tags
across 0 player-seasons in 0 portraits`, a `⚠ Discrepancies` block listing all
80 coded tags as "has a kind but never fires", and empty per-kind tables. The
`--check` path is safe; the default path is destructive.

Fix: refuse and exit non-zero when `scan_portraits()` finds no portraits. Same
failure shape as [Review — 12 Oct](reviews/REVIEW-2026-10-12.md) § 13, where
missing local files could become destructive Scout deletions.

Related but separate: the committed `TAGS.md` is itself stale — it reports 88
tags against the code's 80, because it was generated before tags were retired.
It can only be regenerated where the portraits live.

---

## 2. Documentation that contradicts implementation

From [Review — 12 October 2026](reviews/REVIEW-2026-10-12.md) § Where
documentation differs from implementation. These are the cheapest items to
settle, because each is a specific checkable claim.

| claim in the docs | reality | status |
|---|---|---|
| "Archetype modules are no longer on the portrait path" | `team_portrait.py:128` imports `build_starter_profile` and `build_bullpen_profile`; called at `:1406` | **OPEN** |
| READMEs say 2.3 million pitches per season | both now say "about 680,000 pitches a season" | **CLOSED** |
| `OPERATIONS` calls `build_fingerprint()` without its season argument | `OPERATIONS.md:210` now passes a season | **CLOSED** |
| "Regular-season-only" | cached spring games and BRef postseason totals remain; date bounds omit legitimate regular-season games | unverified |
| "Every derived layer up to date" | only tunnel has raw-data-driven refresh orchestration; five league caches can go stale; refresh does not sync seeded DB totals | unverified |
| "A run must verify it achieved its goal" | checks exist, but general refresh discards their return values and still returns success | unverified |
| "Publish refuses stale portraits" | applies to the portrait branch only; League and Scout can still publish after refusal | unverified |
| "Fingerprints cover current data and current logic" | only a subset of inputs and selected public constants; missing fingerprints are accepted | unverified |
| "Reliability refuses rather than shrinks" | spin and tunneling inputs still shrink; the floor is 0.5 of configured stabilization, not the full point | unverified |
| "A league-average player carries no tags" | handedness attributes fire on essentially everyone — the claim only holds for non-attribute kinds | unverified |
| "Generated references cannot drift" | the check validates vocabulary coverage, not formulas, freshness or denominators | unverified |
| Player-season counts duplicate team stints | needs a stated policy for stints, side and cameo rows | unverified |

The "reliability refuses rather than shrinks" row matters more than it looks:
it is the project's stated philosophy, and
[Decisions](DECISIONS.md) § Reliability leans on it. If spin and tunneling still
shrink, either the code or the claim should move.

---

## 3. Methodology and interpretation limits

Not bugs. Places where a number is weaker than its name, from
[Review — 12 October 2026](reviews/REVIEW-2026-10-12.md) § Methodological limits
and [Review — 14 October 2026](reviews/REVIEW-2026-10-14.md) § Methodology and
documentation still needing attention.

- **Framing is an unadjusted proxy** — one league called-strike rate subtracted
  across a broad shadow band, with no expectation per location and no
  adjustment for count, pitcher, batter, umpire or pitch type. Repeatability
  alone cannot assign the effect to the catcher.
- **Leverage is realized impact, not opportunity** — mean absolute *realized*
  win-expectancy change, so a home run allowed enlarges it. Standard leverage is
  the potential change before the outcome. Either relabel it an impact proxy or
  use pre-event leverage for deployment tags.
- **Expected-stat residuals are not luck** — `wOBA − xwOBA` carries context and
  model residual. "Above/below expected" is supported; "lucky/unlucky" is not.
- **Similarity is a heuristic, not a validated distance** — thresholded sparse
  vectors discard sub-gate differences and jump at gate boundaries; correlated
  tags get repeated influence; zero can mean typical, unmeasured or unsupported.
  Self-neighbours are a sanity check, not validation.
- **Missingness needs its own state** — broad exception handling turns an
  unavailable feed into absent tags. Store per-metric availability, sample
  count, source, cutoff and fallback status. Absence of a tag is not evidence of
  absence.
- **The stored projection is retrospective** — `_compute_projected_profile`
  (`team_portrait.py:1860`, called at `:4057`, **OPEN**) uses the realized roster
  and current-year workloads, and calls `build_hitter_traits` without PA, which
  suppresses gated traits. Checked NYY/LAD 2025 projected densities contained
  only speed tags.
- **Season and era pools answer different questions** — hitter gates mostly use
  current-season ranks; several pitcher outcomes use a pooled era. Historical
  profiles deliberately include later completed seasons: fine retrospectively,
  unsuitable for as-of backtesting unless frozen at the prediction date.
- **Observed roster shape is not construction intent** — describing who played
  does not establish why a front office built it that way.
- **`scripts/battery_base.py`** was reported failing to import because `_FR_TOP`
  was removed. The file still exists; neither `_FR_TOP` nor `_FR_BOT` is
  referenced anywhere now, so this may be resolved — needs an actual import to
  confirm.
- **Dependency reproducibility** — broad minimum versions remain, including a
  pandas floor below the version needed for `include_groups=False`.
- **Validation suite scope** — some checks replicate implementation logic (V8
  repeats the same foul-tip classification; V19 reconciles to the same event
  filter rather than an external ledger). Workflows still do not run an
  independent correctness suite on pull requests.

---

## 4. The talent estimate has no age adjustment

Found 15 November 2026. **Status: OPEN, measurement pending.** Deliberately
left open rather than fixed — the decision needs a number nobody has yet.

`talent.py` estimates what a player is from his decayed recent seasons,
regressed toward league by `k`. It carries **no age term**: `Spec` is
`(stem, num, den, decay, k, typical, complement, season_rank)`, and
`estimate_panel` never looks at age. Age data does exist in the project —
`team_portrait` derives it from Chadwick birth dates, and BRef supplies it — but
nothing in the talent path consumes it.

Why this is a real gap:

- **`decay` cannot absorb it.** `d` is fitted per *metric*, not per *player*. It
  discounts old seasons uniformly; it cannot shift one player because he is 37.
  At best it reproduces the population's average drift as a constant.
- **The fitting objective cannot see it.** `fit_talent.py` chooses `d` and `k` by
  maximising **correlation**, de-meaned per season. Correlation is shift- and
  scale-invariant, so it is structurally insensitive to the systematic bias an
  age curve corrects — it would register only a second-order variance
  reduction. Refitting will never surface this.
- **The purpose is stated two ways.** `talent.py` says "what a player **IS** as
  of a season"; `fit_talent.py` says it "**predicts** each player's NEXT
  season." An age curve is required for the second reading and arguable for the
  first. This should be settled in [Decisions](DECISIONS.md) either way.
- **"We do not invent numbers" is not available as a defence here.**
  `reliability.py` refuses shrinkage because it "invents a number that nobody
  observed" — but `talent.py` already shrinks toward league with `k`. That line
  is deliberately crossed in this module.

Two things limit the exposure: **velocity is already excluded** from the talent
specs ("the season is its best estimate"), and it is the most age-sensitive
attribute in baseball; and pitcher stuff decays at 0.3, leaning hard on the most
recent season.

**The measurement to run:** residual `actual(N+1) − estimate(N)`, bucketed by
seasonal age, per spec. Flat within noise on the specs that feed tags → record
the omission as deliberate in [Decisions](DECISIONS.md) and move on. Bends at
the tails (≤24, ≥35) → add an `age_slope` to `Spec`, which fits the existing
per-metric pattern.

Two snags for whoever runs it:

- `talent_counts_*.csv` and `chadwick_crosswalk.csv` are gitignored caches and
  are **not present on every machine** — this needs a Statcast pull and a counts
  build first.
- `team_portrait`'s age helper computes age **as of today**, not as of the
  season. It cannot be reused; an age curve needs seasonal age, and the
  convention should be fixed (the usual one is age as of 30 June).

**Also worth doing regardless:** have `fit_talent.py` report a calibration
measure alongside correlation. As written it cannot answer this question however
often it is refit.

---

## 5. Questions the reference notes raised

Each sits in its note's `## Open questions` section. Answering one usually means
either a line in [Decisions](DECISIONS.md) or a correction to the note.

**[The strike zone](../reference/the-strike-zone.md)**
- The ABS half-width is written **0.833 ft** in one place and "ABS's **0.829**"
  in another. A 1.45-inch ball radius on 0.7083 ft gives 0.829; 0.833 implies
  1.5 inches. Which does the code use? It sits exactly on the borderline calls
  the constant exists to decide.
- Does the 2026 `sz_top` formula (53.5% / 27% of listed height) invert cleanly
  for every batter?
- Do zone% and chase% carry a dashboard data note for the 2026 break, as framing
  does? They cross the same definitional line.

**[Plate discipline](../reference/plate-discipline.md)**
- Is chase% measured against the ABS zone (0.833 ft) or the rulebook plate edge
  (0.7083 ft)? The choice moves every borderline pitch between numerator and
  denominator, and therefore moves `patient` and `zone hunter`.
- Does `patient` use chase rate alone, or chase combined with swing rate?
- Is there a pitch-type breakdown of discrimination? Judgment against breaking
  balls is a different skill from judgment against fastballs.

**[Spin efficiency](../reference/spin-efficiency.md)**
- Are the per-type Spearman floors and ±15% level band checked on every build,
  or only when spin code changes? A guard that runs when someone remembers is
  the same failure mode as the league caches before `method_version`.
- How is a neutral or unknown park handled, given pre-2020 needs park elevation
  for air density?
- Is separating seam-shifted wake from spin possible at all, or is 0.6–0.65 the
  honest ceiling for off-speed pitches?

**[Percentiles vs ranks](../reference/percentiles-vs-ranks.md)**
- Could a rank-vs-rate check run at ingest — flag any column whose median lands
  in 45–55 with a 1–100 range? Cheap, and it would have caught `sprint_speed`.
- Is there a positive list of Savant columns confirmed to be rates? If it exists
  only in code, it should exist in `reference/` too.

**[Whiffs, swings and CSW](../reference/whiffs-swings-and-csw.md)**
- Is a foul *bunt* attempt treated like a foul tip?
- The out-pitch floors stay on the per-pitch scale for continuity. Is that noted
  where the constant is defined, so a future reader does not "fix" it?

**[Pitch movement and break](../reference/pitch-movement-and-break.md)**
- `tunneler` is a pairwise property. How is the pitch pair chosen — sequence
  frequency, release proximity, something else?
- Statcast fits constant acceleration per pitch, so quadratic break is partly
  baked in. Does any public dataset support testing real mid-flight variation?

**[Service time](../reference/service-time.md)**
- The ledger starts 2009 and the lists 2006. What is the honest uncertainty on a
  career beginning before 2006?
- Are minor-league option days and the 20-day rule simply out of scope?
- Does anything display service time to a user, or is it only a gate? That
  decides how much the residual 1–4% disagreement matters.

---

## 6. Ordered programmes

Three sequences already written down, kept here as pointers rather than
restated. They disagree in emphasis, which is itself worth resolving.

| source | shape |
|---|---|
| [Review — 12 Oct](reviews/REVIEW-2026-10-12.md) § Engineering direction and repair order | contract → fix numbers → reproducible releases → tests → statistical validation → refactor |
| [Review — 14 Oct](reviews/REVIEW-2026-10-14.md) § Suggested repair order | units → fail-before-write → provenance → shared outcomes → service time and proxies → rebuild and verify |
| [Conceptual review — 14 Oct](reviews/CONCEPTUAL-REVIEW-2026-10-14.md) § Recommended sequence | measurement contract → better comparisons → validate interpretation → only then expand vocabulary |

All three put **the measurement contract first** and **vocabulary expansion
last**. `metric_contracts.py` exists, which suggests step one is at least begun.

---

## Graduating an item

When something here is fixed and the reasoning is worth keeping, write it into
[Decisions](DECISIONS.md) in the house format — what was decided, why, and what
breaks if you change it back — then delete it from this file. If the answer is
"we looked and it does not matter", that is also worth recording there, as a
rejected option.

This file should only ever hold what is **not yet done**.

## Related

- [Decisions](DECISIONS.md) · [Architecture](ARCHITECTURE.md) · [Operations](OPERATIONS.md)
- [Reference index](../reference/README.md)
- [Home](../HOME.md)
