# Decisions

Things that are not obvious from the code, each of which cost real debugging.
Written down so they do not get re-litigated or quietly undone.

Format: what was decided, why, and what breaks if you change it back.

---

## What can and cannot be a tag

### Late break is a constant, so it is not a tool

Across 708,559 pitches, the share of a pitch's total break that happens *after*
the hitter's commit point (23 ft) is **70.3%** — and it does not vary:

| curveball | sweeper | four-seam | sinker | cutter |
|---|---|---|---|---|
| 70.4% | 70.4% | 70.3% | 70.3% | 70.2% |

A 79.7 mph curveball and a 94.5 mph four-seam are identical to within
two-tenths of a point. Break accumulates as t², so the late share is (t₁/T)² —
a ratio of times, fixed by the constant 60.5 ft. A slow curve takes longer to
reach the plate and proportionally longer to reach the commit point, and it
cancels.

**A tool has to vary between players.** This varies by 0.2 points. What scouts
mean by "late break" decomposes into three things already in the taxonomy:
total movement (tool), velocity (tool), and separation from the previous pitch
(behavior — `tunneler`). Only the third is genuinely about lateness, and it is
a property of a *pair* of pitches, which is why it can never be a per-pitch
tool.

Caveat: Statcast fits a constant-acceleration model per pitch, so quadratic
break is partly baked in. Real mid-flight variation would need finer tracking
to detect. That does not change the conclusion — you cannot tag what you cannot
measure.

### Spectrum was removed

A single power-versus-contact score is a ratio, and ratios normalise away the
thing you care about. Martín Maldonado came out at 66 on the power side;
Aaron Judge at 37. Both are "true" as ratios and neither is useful. **43% of
hitters above spectrum 80 had an xwOBA below the 40th percentile** — the score
was finding high-variance hitters, not good ones.

Replaced with a two-axis power/contact plane where both axes are absolute, and
results are colour. A hitter who is genuinely both now sits in the corner
instead of averaging to the middle.

### `zone hunter` is built on discrimination, not zone-swing rate

Raw zone-swing rate correlates **+0.81** with first-pitch-strike rate, which
means a tag built on it would mostly restate `aggressive`. The tag fires on the
*gap* between in-zone and out-of-zone swing rates — swinging at strikes more
than balls — which is the skill the name implies.

### Out-pitch tags are `behavior`, and they have no floor

The out pitch is chosen by `argmax(whiff_pct × usage_pct)` over a pitcher's
non-fastballs, and the tag stores `usage_pct`. Its siblings — `sinker-baller`,
`cutter-primary`, `two-pitch`, `deep arsenal` — are all `behavior`, under "what
he chooses to throw". `ride four-seam` is the counter-example that shows where
the line is: it is a `tool` because it clears a threshold (IVB ≥ 1.4).

**Open issue.** Out-pitch has no threshold at all, so it fires on **85% of
pitcher-seasons** — every pitcher with any secondary gets one by definition.
Chris Bassitt's fires at 10% whiffs. Every other tag answers "is this player
notable for X?"; this one answers "which of his pitches is least bad?" It
either needs a whiff floor or should be presented as an arsenal descriptor
rather than a tag.

---

## Unit traps that produced wrong tags

Every one of these was found because a tag fired at an implausible rate.

### Savant's percentile columns are ranks, not rates

`exit_velocity` had a median of **49**. Not 88 mph — 49, because the column is
a percentile rank. Same for `fb_velocity` (median 49, not ~94) and, worse,
`sprint_speed`, which was *mixed*: some rows rates, some ranks. That made
**26.2% of `elite speed` tags bogus**, including Jacob deGrom in four separate
seasons.

Anything ranked against a uniform 1–100 pool by an absolute threshold will fire
on roughly the top N% no matter what the real distribution is. Derive rates
from Statcast; treat any Savant column named like a rate as suspect until
checked.

### Foul balls carry `launch_speed`

A batted-ball event is a ball put **in play**. Statcast records `launch_speed`
on fouls too, and they are weakly hit by nature — counting them drags average
exit velocity about 6 mph below the published figure and roughly halves
hard-hit and barrel rate. Gate on `bb_type`, not on `launch_speed` being
present.

### `pfx_x` and `pfx_z` are in feet

A four-seam's `pfx_z` is 1.32 — that is 15.8 inches of induced vertical break,
in feet. `tunneling.py` divided by 12 as though the value were inches, a **12×
understatement** of the movement term.

### Break is quadratic in time, not linear in distance

At the commit point only **29.7%** of a pitch's break has happened, not the
57.5% that linear interpolation along the path implies. Combined with the unit
bug above, the movement term at the tunnel point was understated **6.2×** —
0.45 inches against a typical 1.72 inches of release-point scatter.

The consequence: `tunnel_score` was not measuring tunnelling. Its tunnel
location correlated **0.998 with release height**. It was a release-consistency
metric wearing a tunnelling label, and 39% of pitchers changed `tunneler`
status when it was fixed.

No ODE solver is needed. Statcast's published kinematics *are* a
constant-acceleration fit, so solving that quadratic in closed form is not an
approximation of the data — it is the data.

### Percentile direction

For `patient`, a **5th** percentile chase rate is elite. For `elite velo`, a
95th percentile is elite. A tag's `pct` is a percentile in the metric's own
direction, and the display, the sort and the similarity vector all have to
invert the low-is-good ones.

Getting this backwards put the *least* patient hitters at the top of a search
for the most patient. It would have been worse in the similarity metric, where
there is no obviously right answer to check against.

---

## The strike zone

### ABS judges at the front of the plate, with a ball-radius buffer

Measured against **2,350 ABS challenge verdicts**: the zone that reproduces
them is the front plane of home plate, with the plate's 17 inches grown by a
full ball radius on each side — half-width **0.833 ft**. Agreement **99.8%**.

The zone *drawn* on the 3D chart uses the plate's own edge (0.7083 ft), because
that is the rulebook zone. These are two different questions and the code keeps
two constants on purpose.

### The definition of `sz_top`/`sz_bot` changed in 2026

Through 2025 they were a per-pitch human estimate of the batter's stance. From
2026 they are computed as 53.5% and 27% of the batter's listed height. The
league mean top moved from **3.435 to 3.214 ft** — the zone lost 2.9 inches,
and that is a change in what the column *means*, not in how batters stand.

Inverting the 2026 formula recovers real listed heights (Judge 79.0 vs 79,
Altuve 65.6 vs 66), which is how the change was confirmed.

### Batter height does not belong in the portrait schema

It looks like it should: ABS is defined per batter, so a league-average zone is
the right average and the wrong individual. But `pitch_aggregates` already
reads the **per-pitch** `sz_top`/`sz_bot` — which *is* that batter's zone — for
zone%, chase%, CSW% and framing, the metrics that feed tags. Statcast omits
them on 0.27% of pitches, and those now fall back to the median of the same
data rather than a constant, so the fallback is era-correct without being told
the era.

`charts.strike_zone()` has exactly two consumers, and both legitimately want a
league average: the era note, and the 3D arsenal chart, which draws one
pitcher's pitches against no particular batter.

---

## Caching and staleness

### Purge every layer or none

Repairing the database is not enough. `team_portrait` ranks players against
pools cached on disk, and those files keep whatever units they were built with.
After the first unit repair the database held true mph while the pool still
held percentiles, so a 93 mph fastball was ranked against a uniform 1–100 pool
and came back as the 93rd percentile: `elite velo` fired on **89.8%** of
pitcher-seasons and `soft tosser` on none.

This trap was sprung twice before the lesson stuck. See the cache stack in
[ARCHITECTURE.md](ARCHITECTURE.md).

### Schema version cannot detect a content change

`PORTRAIT_SCHEMA_VERSION` detects a change of *shape*. Units and thresholds
changing is a change of *content* at the same shape, so a portrait built
minutes before a repair and one built after are indistinguishable to it. Thirty-one
portraits carrying the current version and pre-repair units were skipped as
cache hits.

Hence `build_fingerprint()`, which hashes the pools and every gate constant.

### The gate digest must cover every module that cuts tags

`pitch_aggregates` was missing from it, and it is not a bystander — it owns the
strike-zone geometry behind zone%, chase%, CSW% and framing. A change there
moved tag output while leaving the fingerprint identical. The constants were
also invisible by name: one pair was function-local and both were
underscore-prefixed, which the `isupper()` filter skips.

### An in-progress season is not immutable

`pull_statcast_season` said it plainly: "Loads from cache on subsequent calls —
never re-pulls." Correct for a finished season, wrong for the one being played.
The 2026 file was written on 21 May holding games through 20 May, and every
read after that returned 49 games per team no matter how much of the season had
happened.

Top-ups target **yesterday**, not today — today's games are still being played,
and chasing them would make every read a network call forever.

### A run must verify it achieved its goal

`build_league_identity` catches per-team exceptions and carries on, so a season
reports "30/30 teams" having silently kept portraits it never rewrote. Three
workers opened Supabase connections at the same moment, all three got "Server
disconnected", and ATH 2015–2017 stayed on the previous build while the run
reported success.

Counting what is still stale *afterwards* is the only check that speaks to the
goal rather than the attempt.

---

## The dashboard

### It must never write

On a cache miss the dashboard would load a full season of Statcast (~1.2 GB)
and build the portrait itself, then save it. Two things wrong with that. It
needs the whole ~977 MB of raw data on the web tier, when the artefact a reader
consumes is 239 KB. And it makes the dashboard a **writer** — which is how a
portrait built against half-regenerated pools got saved mid-rebuild and then
looked like a valid cache hit forever.

`BASEBALL_READ_ONLY=1` in any deployment. `refresh.py` is the only writer, and
it refuses to start while anything is listening on :8050.

### The store holds a key, not the portrait

Dash sends a `dcc.Store`'s contents back to the server on *every* callback that
reads it. With the 286 KB portrait in the store, one team switch uploaded
**5.41 MB** — the same payload sixteen times over. With a key it is **11 KB**.

Invisible on localhost, where upload is free, and punishing anywhere real,
since upload is the scarce direction on a home connection.

### `no_update` cannot be a `running=` off-value

Pointing a `running=` spinner and a callback's return at the same Output forces
`no_update` as the spinner's off-value, and Dash renders that sentinel straight
through to the component. React then reports *"Objects are not valid as a React
child (found: object with keys {_dash_no_update})"*. Use separate elements.

### A callback cannot tell a click from a programmatic set

A callback watching a dropdown's value fires when the user changes it *and*
when another callback sets it. In the arsenal chain that meant re-setting a
pitch list that was already correct, and that write re-triggered both figures —
the arsenal was drawn twice on every team switch. The fix is to compare against
what the component already holds and stand down when it matches.

---

## Similarity

### Attributes are excluded from the comp vector

Handedness fires on nearly everyone, so including it does not tilt a comp list
— it collapses it:

| | comps sharing handedness |
|---|---|
| attributes excluded | 3 of 6 |
| attributes included | **6 of 6** |

With them in, Mookie Betts occupies four of Judge's six slots and Votto and
Belt — genuinely similar hitters — drop out entirely. Excluded, the list is
chosen on how they play; you can see it working in that Judge's comps are
right-handed and Soto's are left-handed without either being sorted that way.

They are available as a **filter** instead, which is the honest way to ask for
the constraint.

### A player matching himself is the validation

Other seasons of the same player are deliberately kept in the candidate pool.
If a player's nearest neighbour is not usually himself a year later, the
similarity is not measuring anything stable. Skubal 2024's closest comp is
Skubal 2025 at 0.873; Soto 2024 surfaces Soto 2023.

---

## Reliability

Tags are gated on sample size, and the gate refuses rather than hedges. A
metric below its stabilisation point does not produce a weaker claim — it
produces no claim. `limited_sample` marks the player instead, which is
honest about why nothing fired.

About 45% of hitter-seasons fall under 100 PA. That is not an error: a lot of
players get a cup of coffee. The marker sits on the player, not on the team, so
it does not imply the team's data is thin.
