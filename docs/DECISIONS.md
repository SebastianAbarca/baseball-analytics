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

### Out-pitch tags need a top-20% whiff rate for their pitch type

The out pitch is chosen by `argmax(whiff_pct × usage_pct)` over a pitcher's
non-fastballs, so every pitcher with a secondary pitch has one. Ungated, the
tag fired on **82% of pitcher-seasons** (8,029 of 9,738) and answered "which of
his pitches is least bad?" rather than "is he notable for this?" — Chris
Bassitt's fired at 10% whiffs.

Now the pitch must reach the 80th percentile of whiffs per pitch **for its
type** — the bat-misser bar — among every pitcher-season with 100+ of that
pitch, 2015–2026 (`OUT_PITCH_WHIFF_FLOOR` in pitcher_traits). Per type, because
the types differ more than the gate does: the bars run from 16.2% (curveball)
to 22.8% (splitter), and a median splitter outwhiffs a good curveball. One table
for all seasons, because the bars barely move — the slider's stays within
19.8–22.2% from 2015 to 2026. Rare types (screwball, forkball, knuckleball,
slow curve) share the pooled non-fastball bar. The whiff rate also passes the
reliability gate as a swinging-strike rate.

Result: **20%** of pitcher-seasons (1,915), no other tag moved. Bassitt keeps
his sweeper in 2021, 2023 and 2024 and loses the 10%-whiff curveballs;
2025's tagged pitches include Skubal's changeup (28%) and Yesavage's and
Fernando Cruz's splitters (32%, 29%).

**Kind: still `behavior`, decided 2026-10-12.** The reasoning that made
`ride four-seam` a `tool` — it clears a threshold — applies here too, and the
threshold is a whiff rate, which reads like a result. Kept as `behavior`
anyway: the tag names which pitch he chooses to finish hitters with, beside
`sinker-baller` and `deep arsenal`, and the whiff bar decides only whether
that choice is notable enough to say. `result` would have moved it beside
`bat-misser` in every fingerprint.

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

### Spin efficiency was a clipped constant, not a measurement

`theoretical_max = 0.00153 × rpm / mph × 12` left **90% of 2025 pitches at or
above 1.0** before the clip (median ratio 3.0): the constant had no traceable
source and was ~4× too small. Every fastball, sinker and changeup read 1.0, so
the pooled percentile measured one thing — how few sliders a pitcher throws
(Spearman −0.58 with slider share; half of `high spin efficiency` pitchers
threw none). Against Savant's measured active spin it ranked four-seamers at
Spearman 0.20.

Now:

- **2020+ is Savant's measured active spin** (the 3D spin vector, from the
  active-spin leaderboard). Anything inferred from movement credits
  seam-shifted wake to spin — changeups and splitters top out near 0.6–0.65
  for *any* movement method, Savant's own included.
- **Before 2020, a physics estimate**: Magnus acceleration from the
  9-parameter fit (gravity and drag removed), air density from park
  elevation, the Sawicki lift curve inverted. It ranks like Savant measured
  (FF 0.92, CU 0.93, SL 0.83) and level with Savant's own movement-inferred
  numbers. One level factor per pitch type, fitted on 2020–2025 and stable
  ±5% by season, puts its values on Savant's scale. The lift curve is least
  certain at slider/cutter spin factors, hence their 1.65 / 1.92 factors.
- **Percentiles are within pitch type.** Gyro spin is what makes a slider;
  pooling ranks pitch mix. A pitcher's percentile is the pitch-weighted mean
  of their within-type percentiles, re-ranked across pitchers.

V13 holds the physics estimate to per-type Spearman floors and a ±15% level
band against Savant, so it cannot drift silently again.

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

### Framing tags are kept through ABS, with a note

Measured on qualified catchers (500+ taken pitches within 3 in of the edge),
2022–2026:

| | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| spread between catchers (SD, points) | 3.06 | 3.18 | 2.94 | 2.87 | 2.22 |
| elite framer bar (top 15%) | +3.4 | +3.4 | +2.7 | +2.6 | +2.3 |
| odd vs even games, full-season | 0.77 | 0.80 | 0.74 | 0.69 | 0.60 |

Year to year: 0.69, 0.44, 0.65, then 0.36 into 2026.

The skill is compressing, and it started before ABS: umpires have been closing
on the rule's zone for years (the 50% called-strike edge went 0.945 → 0.843 ft
against ABS's 0.829), which leaves less to steal. 2026 also records an
overturned challenge as the machine's call — at least 2,662 challenges, 47%
overturned, counting only those visible on the pitch that ended the PA.

Kept anyway, because the signal is still mostly real: 0.60 means most of the
2026 spread repeats within the season, and the poor end is stable (Pérez,
O'Hoppe, Garver, Langeliers, McCann were bottom-quarter or near it in 2025).
The elite end is shakier — Ruiz and Will Smith went from 10th/12th percentile
to elite. The tags rank within a season, so they fire at the same rate; what
moved is how far an outlier stands out. A dashboard data note says so, and
says not to compare framing across the 2026 line. Rejected: doubling the
sample floor for ABS seasons (more trustworthy, far fewer catchers) and
dropping the tags.

---

## Caching and staleness

### Finished seasons are frozen, and only the season in progress moves

The nightly job rebuilds one season, which needs the other 330 portraits not
to depend on it. Two things made every portrait depend on every season:

- the build fingerprint hashed every pool file of every season, so a night of
  2027 games would have moved 2015's fingerprint;
- the era pool behind display percentiles (and some pitcher gates) contained
  the season in progress, so new games genuinely re-ranked old seasons.

Now a season is either finished or in progress (`ingest.season_state`, read
from SEASON_DATES), a portrait's era pool is every finished season plus its
own, and its fingerprint hashes only the roots of that pool and its own tunnel
file. Finished seasons are frozen until the next one finishes and joins them —
a once-a-year re-rank, run by hand as the finalize.

The cost is a fixed era during a season: May 2027's portraits rank against
2015–2026 plus 2027-so-far, rather than updating 2015's percentiles as 2027
goes. A 12-season era barely moves in one season's games, and a frozen past is
what the "historical is fixed, current is live" model asked for anyway.

The era pool had also been arbitrary. It came from `query_seeded_seasons()`,
which read `player_batting` through the API's 1,000-row cap and returned
whichever seasons came back first: 2015–2025, missing 2026, when replaced. So
2026 portraits had ranked against a different era than every other season.

### The finished seasons' roots are committed

The nightly job runs on GitHub, which has none of this machine's data. The era
pool is built from 24 `fg_*` CSVs, and rebuilding them from source would mean a
full season of Statcast each for the pitch aggregates in them — about a
gigabyte, hours, and still not guaranteed byte-identical. So the finished
seasons' copies are committed in `data/era/` (7 MB), refreshed at finalize by
`freeze_era.py`. The runner ranks against exactly the pool this machine does,
and the fingerprints agree.


### "Season" means the regular season, for every input

Statcast was filtered to `game_type == "R"` only when a season was first
pulled, so whatever a cache already held was served as-is: 2024's carried
20,583 spring-training pitches. And the BRef season totals came from
pybaseball helpers that request March 1 to November 30, postseason included —
Soto 2022 had 716 PA and 165 G against MLB's 664 and 153, and from 2021 on
about a hundred hitters and a hundred pitchers per season carried October
plate appearances into workloads and rate denominators.

Now every Statcast load filters and rewrites a cache that held anything else,
the BRef pulls are bounded by `SEASON_DATES`, and V18 compares each season's
totals with the PA/BF regular-season Statcast counts — a leak shows up as
totals running ahead, without naming any player.

### The roster is everyone who appeared, and a team gets its stint

A team portrait used to select whole-season player rows by membership, so a
player traded mid-season was credited in full to every club: Soto 2022 had
716 PA and 5.47 WAR on both SD and WSH, Scherzer 2021 was a `workhorse` on
both LAD and WSH, and NYY 2025's hitters summed to 7,560 PA against the
team's 6,235.

Decided 2026-10-13:

- **The roster is everyone who appeared for the team**, sorted by work with
  the team. Under 30 PA / 20 BF with the team a player is listed under
  `players.also_appeared` — on the roster, no tags, no weight.
- **Workload, WAR and team weights are the team's stint**, counted from
  regular-season Statcast (PA/BF/G/GS) and BRef's per-team WAR. The season
  total sits beside it (`pa_total`, `bf_total`, `war_total`): the card reads
  "436 PA / 664 total".
- **Deployment tags are earned with the team** — `workhorse`,
  `everyday player`, `heavy usage`, leverage, `platoon specialist`,
  `super-utility` rank the stint against every player-team stint in the
  league.
- **Every other tag stays season-wide**, because it describes the player, and
  a traded player's card says so ("full season: WSH + SD"). Rejected:
  recomputing every rate per stint. 16–43% of a traded regular's stints are
  under 100 PA/BF, so most would fall under the reliability floors and
  traded players would systematically lose tags their season supports; and
  half the rates come from BRef season totals that have no per-team line.

Team weighting now reconciles: NYY 2025 is 6,210 qualified + 25 also-appeared
= 6,235 PA, exactly the team's plate appearances, and its pitchers' BF sum to
the team's 6,080. The cross-team rows (team HR, WAR concentration, tenure)
had the same double count — 2022's summed to 5,719 home runs against the
league's 5,215 — and are now built per season before any portrait, ranked
against the era pool, and frozen in `data/era/` with the fg roots.

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

On a cache miss the dashboard used to load a full season of Statcast (~1.2 GB)
and build the portrait itself, then save it. Two things wrong with that. It
needs the whole ~977 MB of raw data on the web tier, when the artefact a reader
consumes is 239 KB. And it makes the dashboard a **writer** — which is how a
portrait built against half-regenerated pools got saved mid-rebuild and then
looked like a valid cache hit forever.

The first fix was a `BASEBALL_READ_ONLY` flag, with portraits baked into the
Docker image and a Storage bucket behind them. It worked, but as a stack of
workarounds: a mode to remember, copies of the data in three places, league
files with a fallback at one of four call sites, a pitch mix the image never
had, Scout reading a 13 MB file committed to git. And the baked copies made the
image reject every portrait, because the fingerprint hashed files the image
left out.

### It reads one database, with a key that cannot write

So the dashboard reads published tables in Supabase and nothing else, through
the REST API Supabase already serves for every table — no API service of our
own to run. Its credential is the anon key; Row Level Security limits that key
to SELECT on the serving tables, so "read-only" is what the credential can do,
not a flag. `publish.py`, with the service key, is the only writer. Local
development reads the same way, deliberately: one path, no modes, no fallbacks.

Chosen against our own API service (more control, but a second thing to deploy
and keep awake, with no second consumer to justify it yet) and against keeping
files as a fallback (a second code path is the problem being removed). The cost
is that the site depends on the database being awake; the free tier pauses
after a week idle, so `keepalive.yml` reads a row every two days.

Published JSON is stored as `json`, not `jsonb`, because `jsonb` re-sorts
object keys and the dashboard reads meaning into key order.

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

### A foul tip is a whiff, and each rate says what it divides by

`pitch_aggregates` counted a foul tip as contact; `pitch_mix` counted it as a
whiff, and called its per-PITCH rate `whiff_pct` — the name the other module
gave the per-SWING rate, which `team_portrait` then renamed `SwStr_pct`. Three
names, two rates, two definitions. Savant settles the definition: with foul
tips as whiffs, whiffs per swing match its arsenal whiff% within 0.04 points on
1,228 pitcher x pitch-type rows (2025); as contact, 2.1 points low on every row.

`pitch_aggregates.WHIFF_EVENTS / SWING_EVENTS / CSW_EVENTS` are now the only
definitions. A per-swing rate is `whiff_per_swing` / `Whiff_pct`; a per-pitch
rate is `swstr_pct`. The out-pitch floors stay on the per-pitch scale they were
fitted on, so they did not move.

### A league cache knows which method built it

Each league cache only knew whether it existed: a change to how pitch mix,
plate discipline, sequencing or tunneling is computed kept serving the old
numbers until someone remembered `--force-league`. Each now stamps
`method_version`; bump the module's constant and every older cache rebuilds —
on its next load and in refresh's `league` stage. Spin already worked this way.
