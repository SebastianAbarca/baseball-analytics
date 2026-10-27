# Conceptual review: mathematics, evidence, and baseball meaning

Reviewed October 14, 2026. Source revision: `88602951528cbda32d2baee16bb7174313d3b5b5`, together with the local derived data. This is a review of the model's meaning and statistical validity. Production code and data were not changed.

**Verdict: keep the independent, evidence-bearing tags. Strengthen the measurement model underneath them before adding more tags.** The project is a promising description of observed baseball. Some wording and downstream calculations currently promote those descriptions into claims about skill, intent, or certainty that the evidence does not establish.

The most consequential changes are: distinguish unknown from ordinary; define comparison and exposure populations explicitly; separate continuous measurements from display tags; and validate claims on data that did not choose their thresholds.

## What deserves to stay

- Independent tags allow a player to combine several characteristics without forcing an archetype. Keep this flexibility.
- Separating behavior, tools, results, and deployment makes the product more interpretable. Keep these as categories, without presenting them as a universal ordering of control.
- Team-stint workloads alongside full-season player evidence are a defensible solution to traded players. Keep both scopes explicit.
- Evidence attached to each tag, repeatable builds, and published portraits are good foundations. The portrait can remain the serving artifact even as its underlying model becomes more precise.
- A conservative eligibility rule is a reasonable product choice. It should be described as eligibility, not proof that a statistic has become reliable.

## 1. High priority: a missing tag must not imply an ordinary player

**Observed design.** Tags are stored as positive assertions. An absent tag can mean average performance, insufficient sample, missing source data, an inapplicable metric, or a failed gate. The player-level `limited_sample` flag cannot distinguish those situations per metric. Team comparison fills absent tag densities with zero. Fingerprints can label a category `league-typical` when no tag survives selection, even when the obstacle was coverage or the breadth requirement.

Evidence: [reliability.py](../baseball_construction/modules/reliability.py:1), [population membership](../baseball_construction/modules/team_portrait.py:2648), [fingerprint construction](../baseball_construction/scripts/build_league_identity.py:113).

**Measured example.** In the 2025 portraits, 89 hitter/team rows belong to the shared `catchers` population, but only 77 have a non-null framing percentile. Pop-time data alone can qualify a player for that shared population. STL's elite-framer share is 38.55% when weighted by all those catchers' batting PA, versus 69.84% when the denominator is restricted to the framing-qualified rows. This is a denominator diagnostic, not a recommended final framing measure: batting PA is itself the wrong exposure if the question is who received the team's pitches.

Sprint-speed membership also varies substantially: 472/524 hitter/team rows in 2015 versus 301/626 in 2025. That does not establish why the coverage differs; it establishes why coverage cannot safely be treated as absence of speed.

**Recommendation.** Record a per-metric/per-tag state: `present`, `absent`, `insufficient_sample`, `missing_data`, or `not_applicable`. Separate the reference cohort used to rank a player from the set eligible for a team denominator. Define catcher populations separately for framing, blocking, and pop time. Publish coverage alongside conditional tag density. Use `insufficient evidence for a distinctive label` rather than `league-typical` when typicality has not been established.

## 2. High priority: the reliability rules are eligibility heuristics

**Observed design.** The gate uses `n / stabilization_point >= 0.5`; hitter tags generally pass season PA, and pitcher tags season BF. The chosen 0.5 fraction was calibrated by how many tags disappeared. That measures output sparsity, not tag accuracy or repeatability. Different measurements also have different opportunities: swings, outside-zone pitches, batted balls, pitch pairs, and steal attempts.

Evidence: [thresholds and calibration](../baseball_construction/modules/reliability.py:57), [common PA gate](../baseball_construction/modules/hitter_traits.py:438).

Stabilization is gradual, not a boundary at which noise becomes skill. The underlying sample-size literature also distinguishes PA, AB, and balls in play. A PA proxy can be calibrated for a particular metric, but it cannot be assumed to work for every statistic. [FanGraphs sample-size discussion](https://library.fangraphs.com/principles/sample-size/).

**Concrete illustration.** Eight successful steals in ten attempts passes the 80% `high steal rate` gate. A simple 95% Wilson interval is approximately 49%–94%, even under an independent, constant-probability binomial model. The observed 8/10 is a fact; a durable “safe when he goes” identity is a stronger inference.

**Recommendation.** Keep raw observed rates. Store their actual opportunity counts and measurement-specific eligibility. Add an internal uncertainty/replication check before claiming a stable tendency; this does not require showing a probability on every chip. Evaluate tag agreement on separated game blocks and future seasons, conditional on workload and role. Use shrinkage when estimating underlying ability, while preserving raw observations as a separate field. The current rejection of shrinkage as an “invented” number confuses an estimate with an observation; moreover, [tunneling already shrinks its scores](../baseball_construction/modules/tunneling.py:318).

Do not treat a healthy-looking number of tags, or a few convincing famous-player examples, as statistical validation.

## 3. High priority: tag names sometimes overstate the measured concept

These are changes to the claims, not requests to discard the underlying measurements.

- **`elite command` / `plus command`:** the gate measures walk avoidance. It does not observe the intended target or miss distance. Prefer `low walk rate` or explicit control wording; reserve command for a target/location model. [Outcome gates](../baseball_construction/modules/pitcher_traits.py:535).
- **`power bat` described as “elite raw power”:** the gate measures ISO rank, which represents extra-base production. The local Bellinger 2025 row has ISO at the 88.4th percentile and a barrel/hard-hit composite around the 43rd, yet receives “elite raw power.” Keep the production tag, and distinguish it from a contact-quality or physical-power tool. [Power gate](../baseball_construction/modules/hitter_traits.py:200).
- **`invisible ball`:** this is a swinging-strike residual from a one-variable fastball-velocity fit. Pitch mix, movement, location, role, opponents, and model misspecification remain in the residual. Call it `whiffs above velocity baseline` unless deception is validated independently. [Residual calculation](../baseball_construction/modules/pitch_mix.py:236).
- **`lucky` / `unlucky`:** a wOBA–xwOBA difference is a model residual; it does not isolate randomness. Prefer `results above expected` / `results below expected`. MLB describes xwOBA's limited inputs; interpreting everything outside them as luck is an additional assumption. [MLB xwOBA definition](https://www.mlb.com/glossary/statcast/expected-woba), [current gates](../baseball_construction/modules/hitter_traits.py:425).
- **`elite framer`:** the implementation compares called-strike rates in a broad edge band with one league average. Catchers receiving different locations within that band can differ without any receiving effect. Use `high shadow-zone strike rate` for this proxy, or model per-pitch expectations. Published framing research adjusts for location, count, participants, and umpire effects. [Current proxy](../baseball_construction/modules/pitch_aggregates.py:687), [framing model research](https://arxiv.org/abs/1704.00823).
- **`extra base taker` as willingness:** dropping ambiguous/out outcomes does not turn successful advancement into a measurement of attempted advancement. Exclusion can depend on the very aggression being measured. Track attempts, successes, outs, and unresolved plays separately. [Advancement reconstruction](../baseball_construction/modules/pitch_aggregates.py:780).

The same rule applies above the player level: “Pitch Design and Analytics Driven” is an explanation of organizational intent, while the inputs are arsenal diversity, platoon deployment, opener use, CSW, and tunneling. Those can describe the roster without identifying why the organization built it. [Philosophy definition](../baseball_construction/modules/philosophy.py:116).

## 4. High priority: similarity currently measures the vocabulary as much as baseball

**Player comparisons.** The vector consists of emitted tags, each weighted by percentile strength; categorical tags receive weight 1. A player just below the elite-velocity threshold occupies the `plus velo` dimension, while a nearly identical player just above it occupies `elite velo`. Those coordinates have no overlap. In a one-feature illustrative case their cosine is zero despite essentially identical velocity. Conversely, two profiles carrying only the same tag have cosine one, regardless of differences in strength.

Correlated tags get multiple votes, categorical tags have maximal amplitude, and missing measurements remove dimensions. The current vector includes results, deployment, and luck as well as style. Similarity to another season of the same player is useful face validity, but does not establish that the geometry measures the intended concept. [Player vector](../baseball_construction/dashboard/charts.py:4901).

**Team comparisons — measured sensitivity.** Team fingerprint selection standardizes deviations, but nearest-team comparisons use raw density deviations. Across the 30 local 2025 offenses, the five catcher tags contribute **38.93% of total squared centered density variation**. Elite speed alone contributes another **15.31%**. These are shares of vector variation, not exact shares of every individual cosine score.

Rescaling each nonconstant tag dimension to unit cross-team standard deviation changes **20 of 30 nearest offense matches**. This is a sensitivity experiment, not evidence that standardization alone is the correct solution. It shows that feature scale is a consequential modeling decision. [Team comparison geometry](../baseball_construction/scripts/build_league_identity.py:241).

**Recommendation.** Compute similarity from continuous underlying features; use tags to explain the result. Give baseball domains explicit weight budgets and avoid counting several manifestations of one trait as independent evidence. Require adequate shared coverage. Distinguish hitter offense from catcher defense in both naming and vector construction. Offer separate style, production, and deployment comparisons if each is useful. Validate ranking stability under resampling, modest threshold changes, and removal of one feature family.

## 5. High priority: the sequencing ratio does not have its advertised interpretation

The current formula is:

`H(next pitch | previous pitch, within-PA pairs) / H(all pitches)`.

The numerator and denominator use different populations. First pitches are represented in the all-pitch distribution but not in the next-pitch marginal for the transitions. Therefore this is not the usual conditional-entropy ratio on a common joint distribution and need not lie between zero and one.

**Measured.** In the cached 2025 sequencing pool, **75/573 pitchers exceed 1**, with a maximum of **1.1561**. Calling the actual function with 200 two-pitch PAs, each starting with FF and with the second pitch evenly divided between FF and SL, returns **1.2326**. The next pitch has one bit of entropy; the all-pitch distribution is 75% FF and has only about 0.8113 bits. No data-fetch or cache explanation is needed for this counterexample. [Sequencing calculation](../baseball_construction/modules/pitcher_traits.py:208).

**Recommendation.** For the narrow question “how much does knowing the previous pitch reduce uncertainty?”, use the next-pitch marginal from the same transition sample, with an explicit undefined state for zero entropy. Then assess bias against a suitable shuffled baseline. To claim predictability to a hitter, test held-out prediction after accounting for count, handedness, and other available context. A first-order transition statistic alone does not capture all predictability or strategic intent. The adjacent-pitch issue from the code review should also be fixed before interpreting either sequencing or tunneling.

## 6. Medium priority: define what each percentile and team average means

The system mixes within-season hitter gates, era-based pitcher outcomes/display ranks, within-pitch-type ranks, raw thresholds, and some roster-local ranks. Each can answer a useful question, but a bare `pct` does not identify which one it answers. A historical percentile also changes when the reference era expands, even if the player's observed season does not.

Evidence: [hitter season ranks](../baseball_construction/modules/team_portrait.py:3129), [era ranks](../baseball_construction/modules/team_portrait.py:822), [pitcher normalization](../baseball_construction/modules/team_portrait.py:921).

Keep two explicit concepts where needed: `rank_among_contemporaries` and `rank_in_reference_era`, with cohort/version/eligibility attached. Separate sample eligibility from reference-cohort inclusion, then document why any small-sample players belong in the reference distribution. Align pitcher and hitter policies deliberately. The teammate-relative steal-attempt issue in the earlier review is one concrete violation of this contract.

**Team calculations need equally explicit estimands.** An average of player percentiles is not the percentile of a team statistic. PA-weighting a barrel rate is not the team's barrel rate; its natural denominator is eligible batted balls. Both a workload-weighted roster profile and an opportunity-weighted performance rate can be useful, but they should have different names. [Current aggregation](../baseball_construction/modules/team_portrait.py:946).

For a tag, distinguish:

- **Roster contribution:** share of total team workload supplied by carriers.
- **Conditional prevalence:** share of eligible, measured workload supplied by carriers.
- **Exposure coverage:** eligible, measured workload divided by the relevant total workload.

For receiving, fielding, and running, choose the relevant defensive or baserunning opportunities instead of defaulting to batting PA. Keep the full-season player tags on traded players if desired, but describe the team result as the composition of players used, not necessarily what they did during that particular stint.

## 7. Medium priority: the data model needs typed evidence, stable identities, and explicit order

The shared trait contract says `value` is a raw metric. Many producers instead store its percentile in both `value` and `pct`. `ISO`, `ISO_pct`, and nested `_pct` names also change meaning between frames. The actual raw evidence cannot consistently be recovered from a trait record alone. [Trait contract](../baseball_construction/modules/pitcher_traits.py:10), [metric alias construction](../baseball_construction/modules/team_portrait.py:1093).

Keep the serving JSON, but introduce these logical records beneath it:

1. **Player-season-side:** stable identity `(player_id, season, side)` and full-season observations.
2. **Team stint:** `(player_id, season, team, side)` with workload, deployment, and references to season observations. Add stint dates or a stint ID if multiple separate spells must be represented.
3. **Metric observation:** metric ID/version, value, unit, numerator/denominator or sample count, scope, time interval, coverage, source/method, and uncertainty where modeled.
4. **Reference cohort:** ID/version, seasons, role/position constraints, eligibility rule, sample size, tie policy, and direction.
5. **Tag definition and evaluation:** stable tag ID, display label, predicate/version, semantic family/kind, cohort reference, evaluation state, and evidence references.

These are logical distinctions, not a requirement for five new database services or an immediate storage rewrite. They can first be represented as validated structures inside the existing pipeline. Derive all tag documentation, gate configuration, and display metadata from one registry; declare direction instead of inferring it from observed firings.

Use side in comparison identifiers, deduplicate player-season identities for season-level comparisons, and make team-specific deployment an explicit comparison scope. The serving table already has `grp`, but the comparison lookup loses that distinction. [Serving records](../baseball_construction/schema_serving.sql:87), [comparison query](../baseball_construction/schema_serving.sql:279).

Ordered display items belong in arrays with explicit ranks. The current decision to preserve JSON object-key order to control which identity items appear is a presentation rule encoded in a storage accident. [Ordering dependency](../baseball_construction/schema_serving.sql:13).

## 8. Medium priority: retain multi-dimensional baseball concepts

The taxonomy is useful, but `kind` is not an ordered scale of controllability: tools can change, mechanics can be trained, results have several causes, and deployment is controlled by a different actor. Rename its conceptual definition to “what type of claim is this?” Add actor or mechanism separately only where useful.

Some exclusive groups mix different measurements. Pull share and opposite-field share can both be relatively high if center-field share is low. ISO production and poor contact quality can coexist. Gap production and power need not be mutually exclusive. Suppressing one label for display readability is different from asserting that two baseball facts cannot coexist. [Exclusive groups](../baseball_construction/modules/team_portrait.py:2357), [power/gap suppression](../baseball_construction/modules/hitter_traits.py:214).

Keep logical exclusion for genuinely incompatible levels of the same variable. Give display de-duplication its own rule. Consider separate continuous power and contact axes instead of the surviving power/contact ratio: weak in both and strong in both can have the same ratio. [Spectrum calculation](../baseball_construction/modules/hitter_archetypes.py:244).

Finally, “everyday player” should describe appearance frequency while available to that club if that is the intended meaning. Total stint PA instead measures accumulated contribution and penalizes a daily player traded midseason. The existing policy is coherent for season-long team contribution; the label should say that.

## 9. Medium priority: separate descriptive indices from confidence and forecasts

The philosophy scores are weighted averages of percentiles. Their values are constructed indices, not measured probabilities or necessarily percentiles of team philosophy. Reweighting missing components means two teams can receive the same score from different constructs. Correlated ingredients also receive multiple votes.

The function named `display_confidence` rescales the gap between the two leading indices; this is score separation, not statistical confidence. With only one available candidate it produces maximum confidence. That function-level issue is confirmed; this review does not claim every current dashboard view displays that field. [Scoring](../baseball_construction/modules/philosophy.py:192), [confidence calculation](../baseball_construction/modules/philosophy.py:287).

The projected profile uses prior-year statistics but actual current-year roster membership and PA. It is a useful retrospective baseline, not an ex-ante forecast. Label it “prior-year performance, weighted by this year's usage.” [Projection construction](../baseball_construction/modules/team_portrait.py:1874).

For real forecast evaluation, freeze the information available at a stated cutoff, estimate future roles/playing time, address new players, and hold out future data. For descriptive indices, disclose component coverage and call score separation what it is.

## Recommended sequence

**First: establish the measurement contract.** Add explicit state, units, denominators, scope, and reference cohort; fix the entropy denominator and tag-specific eligibility; narrow labels that currently overclaim. These changes make the existing product more truthful without replacing its central idea.

**Second: improve comparisons.** Build continuous, coverage-aware feature vectors with deliberate domain weights. Show which shared measurements explain a match. Keep display tags as a readable summary.

**Third: validate interpretation.** Use game-block resampling and held-out seasons to assess tag replication, rank uncertainty, and nearest-neighbor stability. Refit any thresholds or calibration only on training data. Same-player recurrence is one check alongside domain ablation and role/era-matched baselines. Do not optimize every descriptive trait solely for prediction; distinguish stable style from a season outcome that is intentionally retrospective.

Only then expand the vocabulary. Each proposed tag should state its exact claim, observation unit, numerator/denominator, cohort, uncertainty policy, overlap with existing tags, and what evidence would contradict its interpretation.

## Scope and reproducibility

Inspected the current trait, reliability, aggregation, sequencing, tunneling, philosophy, identity, comparison, and schema paths, plus the project's documented design decisions and earlier reviews. Read all 360 local portraits: **15,965 hitter/pitcher profile rows**, representing **14,681 distinct player-season-side identities**; `also_appeared` is excluded from those counts. Local stored outputs are evidence of this build, not a fresh reconciliation with official statistics or proof of what the deployed database serves.

The similarity experiment used the stored 2025 offense density vectors, filled absent entries with zero to reproduce the current method, centered each coordinate across the 30 teams, and compared raw cosine neighbors with neighbors after dividing each nonconstant coordinate by its population standard deviation. Variation shares are sums of squared centered coordinates divided by the total across all teams and coordinates. The experiment demonstrates sensitivity; no replacement weighting was selected or deployed.

The sequencing counterexample ran the actual `_entropy` and `_pitcher_sequencing` function definitions in an isolated namespace with their current constants. The framing check used saved `populations`, `pa`, tags, and `gate_inputs.catching.framing_pct`, counting team-stint rows rather than unique catchers. The steal interval is illustrative and does not model selection of steal opportunities.

No full statistical calibration, future-season validation, production rebuild, or deployment was performed. Known implementation findings from the code reviews remain separate; resolving them alone would not resolve the conceptual issues here.
