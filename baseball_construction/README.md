# Baseball Team Construction Analysis System

Visual-first analytics application showing how MLB teams are constructed — philosophy, player archetypes, strengths, weaknesses — using Statcast-era data (2015+).

**Master join key:** MLBAM ID (`key_mlbam`). Never join on player name strings.

---

## Module Status

| # | Module | File | Status |
|---|--------|------|--------|
| 1 | Data ingest, cache, reliability | `modules/ingest.py` | **COMPLETE** |
| 2 | Spin efficiency | `modules/spin_efficiency.py` | **COMPLETE** |
| 3 | Pitch tunneling | `modules/tunneling.py` | **COMPLETE** |
| 4 | Park effects | `modules/park_effects.py` | **COMPLETE** |
| 5 | Philosophy scoring | `modules/philosophy.py` | **COMPLETE** |
| 6 | Hitter archetypes | `modules/hitter_archetypes.py` | **COMPLETE** |
| 7 | Pitcher archetypes | `modules/pitcher_archetypes.py` | **COMPLETE** |
| 8 | Temporal modes | `modules/temporal.py` | **COMPLETE** |
| 9 | Team portrait | `modules/team_portrait.py` | **COMPLETE** |
| 10 | Validation | `modules/validation.py` | **COMPLETE** |

---

## Module 1 — ingest.py

**Purpose:** Data pull, cache, normalize, reliability scoring.

### Functions

| Function | Description |
|----------|-------------|
| `load_chadwick()` | Load Chadwick Bureau crosswalk. Returns key_mlbam, key_fangraphs, key_bbref, name_first, name_last. |
| `pull_statcast_season(season)` | Full season Statcast pull. Cached to `data/raw/statcast_{season}.parquet`. Never re-pulls. |
| `pull_statcast_range(start, end, label)` | Arbitrary date range pull. Cached to `data/raw/statcast_{label}.parquet`. |
| `pull_fg_batting(season, qual=100)` | Season batting aggregates. Tries FanGraphs; falls back to Baseball Savant. Cached to `data/processed/fg_batting_{season}.csv`. |
| `pull_fg_pitching(season, qual=30)` | Season pitching aggregates. Tries FanGraphs; falls back to Baseball Savant. Cached to `data/processed/fg_pitching_{season}.csv`. |
| `attach_mlbam_ids(df, chadwick)` | Join df to Chadwick via FanGraphs ID, then name fallback. Adds `key_mlbam` and `join_source`. Does not drop failed rows. |
| `compute_reliability(sample_size, metric)` | Returns `min(1.0, sample / threshold)` using STABILIZATION dict. |
| `apply_reliability_weight(observed, reliability, league_mean)` | Shrinks toward league mean: `(r × obs) + ((1-r) × mean)`. |
| `normalize_percentile(series, invert)` | Percentile rank 0–100 within series. `invert=True` for lower-is-better metrics. |
| `normalize_df(df, columns)` | Batch normalize; adds `{col}_pct` columns. Auto-inverts known lower-is-better metrics. |

### Data Sources & Known Issues

**FanGraphs 403:** As of 2024+, FanGraphs returns 403 on pybaseball scrape requests. `pull_fg_batting` and `pull_fg_pitching` automatically fall back to Baseball Savant aggregate endpoints (`statcast_batter_expected_stats`, `statcast_batter_percentile_ranks`, etc.). Savant data carries `player_id` = MLBAM ID directly — no crosswalk join needed.

**Chadwick crosswalk:** The register was split across 16 shards (`people-0.csv` through `people-f.csv`) as of 2025. `chadwick_crosswalk.csv` is the combined file. 516K total rows, 127K with MLBAM IDs.

### Smoke Test Results (2023-06-01 → 2023-06-07)

```
Statcast rows:          25,714
Required columns:       26/26 present
Pitcher MLBAM coverage: 100.0%
Batter  MLBAM coverage: 100.0%
Batting rows (Savant):  656 — join rate 100.0%
Pitching rows (Savant): 863 — join rate 100.0%
PASS: True
```

---

## Module 2 — spin_efficiency.py

**Purpose:** Magnus-force spin efficiency per pitcher × pitch type. Higher efficiency = spin is oriented to produce Magnus-force movement. Lower = gyro component present (slider, cutter).

**Formula (Nathan 2008):** `spin_efficiency = sqrt(pfx_x² + pfx_z²) / (MAGNUS_K × spin_rate / speed × 12)`, clipped [0, 1]

### Functions

| Function | Description |
|----------|-------------|
| `compute_spin_efficiency(df)` | Pitch-level: adds `movement_mag`, `theoretical_max_mov`, `spin_efficiency`. Excludes KN/EP/PO/IN/FO. |
| `aggregate_pitcher_pitch_type(df)` | Groups to pitcher × pitch_type. Drops rows with < 10 pitches. |
| `apply_reliability_weighting(agg)` | Shrinks `spin_efficiency_raw` toward league mean; threshold = 200 pitches. Adds `reliability`, `spin_efficiency_weighted`. |
| `normalize_spin_efficiency(agg)` | Percentile rank 0–100 → `spin_efficiency_pct`. |
| `build_pitcher_summary(agg)` | Weighted rollup to pitcher level; adds `dominant_pitch_type`. |
| `build_pitcher_spin_profile(df)` | Full pipeline; returns `(pitch_type_level, pitcher_level)`. |

### Smoke Test Results (2023-06-01 → 2023-06-07)

```
Pitches with valid SE:         25,552 / 25,714
pitcher × pitch_type rows:     930 (609 below min-10 threshold dropped)
Unique pitchers:               375
Reliability max (1-week data): 0.59 (expected — full season needed for floor=1.0)
Percentile range:              0.1 – 100.0
Physics check — SL (0.81) < CU (0.97) < FF (1.00) ✓
PASS: True
```

**Physics interpretation:** Four-seam fastball backspin is nearly perfectly oriented for Magnus lift → SE ≈ 1.0. Sliders carry significant gyro spin → SE ≈ 0.81. Curveballs sit between. This ordering confirms the formula is computing correctly.

---

## Module 3 — tunneling.py

**Purpose:** Estimate how well a pitcher "tunnels" consecutive pitches — sharing the same location at the hitter's decision point (23 ft) while arriving at different plate destinations.

**Method:** Linear interpolation along the pitch path (see GAP 2 for Magnus upgrade path).

### Functions

| Function | Description |
|----------|-------------|
| `estimate_tunnel_location(df, distance=23)` | Adds `tunnel_x`, `tunnel_z` for each pitch at the tunnel point. |
| `build_pitch_pairs(df)` | Vectorised lag-1 shift to find consecutive cross-family pairs within each at-bat. Adds `tunnel_distance`, `plate_divergence`, `tunnel_ratio`. |
| `aggregate_tunnel_scores(pairs)` | Groups to pitcher level. Drops pitchers with < 10 cross-type pairs. |
| `apply_tunnel_reliability(agg)` | Shrinks toward league mean; threshold = 150 pairs. Adds `reliability`, `tunnel_score_weighted`. |
| `normalize_tunnel_scores(agg)` | Percentile rank 0–100 → `tunnel_score_pct`. |
| `build_tunnel_profile(df)` | Full pipeline; returns one row per pitcher. |

### Smoke Test Results (2023-06-01 → 2023-06-07)

```
Pitches with tunnel coords:  25,649 / 25,714
Cross-type pairs:            9,214
Unique pitchers with pairs:  393 (314 above min-10 threshold)
tunnel_distance mean:        0.235 ft  (separation at 23-ft point)
plate_divergence mean:       1.708 ft  (separation at plate)
Physics check: divergence > distance: True ✓
Percentile range:            0.3 – 100.0
PASS: True
```

**tunnel_ratio interpretation:** `plate_divergence / (tunnel_distance + 0.01)`. A ratio of 10 means the pitches diverge 10× more at the plate than they were apart at the decision point — good tunneling. League mean in this sample: 10.3.

---

## Module 4 — park_effects.py

**Purpose:** Compute wOBA-based and HR-based park factors per team from Statcast PA outcomes. Produces `pitcher_friendly` (0–100 percentile) for use in B4 Defensive Infrastructure scoring.

**Method:** Home/road wOBA split per team → raw PF = home_rate / road_rate → regress 1/3 toward 1.0 → invert for pitcher-friendly percentile.

### Functions

| Function | Description |
|----------|-------------|
| `extract_pa_outcomes(df)` | Reduces pitch-level data to one row per PA (terminal event, `woba_denom == 1`). |
| `compute_raw_park_factors(pa)` | Home vs road wOBA and HR/PA splits per team. Requires >= 200 home and road PAs. |
| `apply_regression(teams, weight=1/3)` | Regresses raw PF toward neutral (1.0) to reduce single-season noise. |
| `normalize_park_factors(teams)` | `pitcher_friendly` = inverted wOBA PF percentile; `hr_pf_pct` = HR PF percentile. |
| `build_park_profile(df, season)` | Full pipeline; returns one row per team. |

### Smoke Test Results (2023-06-01 → 2023-06-07)

```
PA outcomes extracted:   6,489 from 25,714 pitches
league_woba:             0.317  (consistent with 2023 MLB ~0.318)
Teams at ≥200 PA:        3 (expected — full season gives all 30)
woba_pf range:           0.988 – 1.095 (post-regression)
Physics check — KC (pitcher-friendly) < HOU < TOR (hitter-friendly): True ✓
Percentile range valid:  True
PASS: True
```

**Regression rationale:** Single-season park factors carry ~±15 point noise. A 1/3 regression toward 1.0 is standard practice (Tango/Lichtman). Multi-season averaging in `temporal.py` will further stabilise values.

---

## Module 5 — philosophy.py

**Purpose:** Score all 11 team philosophies (0–100 each) across three dimensions, compute per-dimension confidence, and identify primary/secondary philosophies.

**Design:** Accepts a flat dict of pre-normalised metrics (0–100 each). Missing metrics are handled by reweighting to available coverage — scores stay on the 0–100 scale regardless of data gaps.

### Philosophy Definitions

| Code | Name | Dimension | Key Metrics |
|------|------|-----------|-------------|
| A1 | Three True Outcomes | offense | ISO, BB%, K%, HR/FB, low sprint |
| A2 | Contact and Pressure | offense | low K%, OBP>SLG gap, P/PA, sprint, contact |
| A3 | Aggressive Early Count | offense | FPS%, low P/PA, Z-swing, low BB |
| A4 | Power Concentration | offense | WAR concentration, wRC+ spread, ISO spread |
| B1 | Stuff Dominant | pitching | K%, velo, SwStr%, spin rate |
| B2 | Command and Contact Mgmt | pitching | low BB%, zone%, GB%, defense |
| B3 | Pitch Design / Analytics | pitching | spin eff., arsenal diversity, platoon opt., opener usage, CSW% |
| B4 | Defensive Infrastructure | pitching | OAA, DRS, GB%, park pitcher-friendly |
| C1 | Star and Support | roster | WAR concentration, payroll concentration |
| C2 | Roster Balance | roster | low WAR variance, roster floor, low Gini |
| C3 | Youth and Development | roster | low avg age, pre-arb share, pipeline |
| C4 | Veteran Experience | roster | high avg age, post-arb share, core retention |

### Functions

| Function | Description |
|----------|-------------|
| `score_philosophy(metrics, weights)` | Weighted mean with coverage tracking. Returns `(score, coverage)`. |
| `compute_all_philosophies(metrics)` | Scores all 11 philosophies for one team. |
| `display_confidence(raw, ceiling=0.35)` | Converts `(top-second)/100` gap to 0–100 display value. |
| `compute_all_confidences(scores)` | Per-dimension primary/secondary/confidence/hybrid. |
| `build_philosophy_summary(team, season, metrics)` | Full profile dict for one team × season. |
| `scores_to_dataframe(summaries)` | Batch list of summaries → flat DataFrame. |

### Smoke Test Results (synthetic TTO-heavy team)

```
A1 (Three True Outcomes): 85.1  ← correctly tops offense dimension
A2 (Contact / Pressure):  36.5  ← correctly suppressed
A4 (Power Concentration): 74.3  ← secondary
C4 (Veteran Experience):  76.8  ← tops roster dimension
C3 (Youth / Dev):         27.5  ← correctly suppressed
B1 (Stuff Dominant):      73.8  ← tops pitching
Hybrid flag offense:       True  (A1=85, A4=74 both > 60)
Confidence math:           ✓ (display_confidence verified at 5 breakpoints)
Missing-metric reweight:   ✓ (single metric → score = that metric)
PASS: True
```

---

## Module 6 — hitter_archetypes.py

**Purpose:** Classify every hitter into a primary archetype with optional modifiers. Classification is sequential — most restrictive type wins.

### Classification Order

| Priority | Type | Code | Gate |
|----------|------|------|------|
| 1 | Complete Hitter | T1 | All 5 thresholds: wRC+ ≥75th, OBP ≥70th, ISO ≥70th, BB% ≥65th, ZContact% ≥55th |
| 2 | Three True Outcomes | T2 | K% ≥50th, BB% ≥65th, ISO ≥65th; AND display_confidence ≥30 |
| 3 | Pure Contact | T3 | Spectrum score < 50 |
| 4 | Pure Power | T4 | Spectrum score > 50 |
| — | Grey zone | — | Spectrum 45–55; low confidence flag |

### Spectrum Formula

```
power_score   = ISO(0.30) + HR/FB(0.25) + Barrel%(0.25) + EV_90(0.20)
contact_score = Contact%(0.25) + (100-K%)(0.20) + AVG(0.20) + OBP_ISO_gap(0.10)
spectrum      = power_score / (power_score + contact_score + 0.001) × 100
```

### Modifiers (additive, independent)

| Modifier | Gate |
|----------|------|
| Free Swinger | FPS% ≥65th AND OSwing% ≥60th AND BB_inv% ≥65th (= raw BB% <35th) |
| Speed | sprint_speed_raw ≥ 28.0 ft/s AND any one: SB_efficiency_pct ≥50, XBT_pct ≥55, HP_to_1B_inv_pct ≥25 |
| Disruptive | efficiency > 0.5, SB% > 72%, attempt_rate > 40% |
| Chaotic | efficiency < -0.5, SB% < 65%, attempt_rate > 40% |

### Smoke Test Results

```
Trout-like  → T1 Complete Hitter      conf=43%  ✓
Gallo-like  → T2 Three True Outcomes  conf=60%  ✓
Arraez-like → T3 Pure Contact         conf=73%  spectrum=13.1  ✓
Borderline TTO (conf<30) → T4 Pure Power  spectrum=64.9  ✓
Free Swinger flag: True/False correct  ✓
Speed modifier: 29.0 ft/s=Speed, 27.5=None  ✓
Disruptive/Chaotic/insufficient_sample  ✓
PASS: True
```

**Calibration note:** `BB_inv_pct` (inverted BB percentile) uses threshold 65 — equivalent to raw BB_pct < 35th percentile. The caller is responsible for supplying the inverted value.

---

## Module 7 — pitcher_archetypes.py

**Purpose:** Classify starters into P1–P4 archetypes with velocity routing rules, modifiers, and a 7-dimension bullpen collective profile.

**Classification:** All four types are scored; the qualifying type with the highest composite score wins.

### Starter Type Gates

| Type | Min Thresholds | Ceilings |
|------|---------------|---------|
| P1 Power Ace | avg_velo ≥60th, K% ≥65th, SwStr% ≥65th | none |
| P2 Craft Strikeout | K% ≥55th, SwStr% ≥55th | avg_velo ≤65th |
| P3 GB Craftsman | GB% ≥50th, HardHit_allowed ≤50th | K% ≤60th |
| P4 Stuff to Contact | avg_velo ≥55th, GB% ≥45th | K% ≤65th |

**Key routing rule:** `avg_velo > 65th AND K% > 65th` → P1 only, P2 blocked.  
**P3 vs P4 distinguisher:** P3 requires `HardHit_allowed ≤ 50th` (suppresses hard contact). A pitcher who induces GB% but allows above-average hard contact routes to P4, not P3.

### Modifiers

| Modifier | Gate |
|----------|------|
| Platoon Vulnerable | wOBA_vs_opposite − wOBA_vs_same > 0.040 |
| Elite Command | BB_pct < 20th AND Zone_pct > 65th |
| Poor Command | BB_pct > 65th |
| Deep Arsenal | 3+ pitch types thrown >10% with above-avg run value |
| One Dimensional | ≤1 above-avg pitch type |

### Bullpen Dimensions (0–100 vs league)

velocity, attack_philosophy, damage_prevention, platoon_balance, leverage_structure, K_out_mechanism, GB_out_mechanism → derived labels: out_mechanism, leverage_structure.

### Smoke Test Results

```
deGrom-like   → P1 Power Ace          score=90.5  conf=68%  ✓
Greinke-like  → P2 Craft Strikeout    score=82.5  conf=42%  ✓
Hendricks-like→ P3 GB Craftsman       score=74.1  conf=62%  ✓
Sinker-P4     → P4 Stuff to Contact   score=55.6  conf=47%  ✓
Velocity routing (velo=70+K=70 → P2 blocked): ✓
Platoon/Command/Arsenal modifiers: ✓
Bullpen: Strikeout-driven, One-arm dominant (NYY): ✓
PASS: True
```

---

## Module 8 — temporal.py

**Purpose:** Three-mode temporal processing applied universally to every metric for every player and team.

### Mode Logic

| Mode | Condition | Behaviour |
|------|-----------|-----------|
| historical | PA < 15 OR games < 15 | Use 3-year weighted average of prior seasons |
| early | PA 15–199 OR games 15–49 | Blend historical + current; current weight ramps 0→50% |
| current | PA ≥ 200 AND games ≥ 50 | Use current season value (upstream reliability-weighted) |

### Historical Smoothing Weights

`lag 0 (prior season) = 0.50 · lag 1 = 0.30 · lag 2 = 0.20`  
Missing seasons are handled by reweighting to available lags.

### Early-Mode Blend

```
progress          = (games - 15) / (50 - 15)  clipped [0, 1]
current_weight    = progress × 0.50            (max 0.50 at game 50)
historical_weight = 1 - current_weight
```

### Functions

| Function | Description |
|----------|-------------|
| `determine_mode(pa, games)` | Returns 'historical' \| 'early' \| 'current'. |
| `blend_weights(games)` | Returns (historical_weight, current_weight) for early mode. |
| `compute_historical_smooth(season_values, reference_season)` | 3-year weighted average; reweights for missing seasons. |
| `apply_temporal_mode(mode, current, historical, games)` | Returns temporally-adjusted scalar value; falls back gracefully. |
| `compute_delta(current, historical)` | current − historical. Positive = improvement over expectation. |
| `process_metrics(pa, games, current_dict, hist_dict)` | Applies temporal processing to every metric in a dict. Returns mode, adjusted, delta per metric. |
| `apply_temporal_to_df(df, pa_col, games_col, metric_cols)` | Batch DataFrame version; adds `_adj`, `_delta`, and `mode` columns. |

### Smoke Test Results

```
determine_mode: 7/7 cases correct  ✓
blend_weights: G=15 curr=0.0, G=50 curr=0.50  ✓
compute_historical_smooth: 3-season, 2-season reweighted, empty → None  ✓
apply_temporal_mode: all 5 edge cases correct  ✓
compute_delta: +15.0, -25.0, None for missing  ✓
process_metrics: early-mode blend correct  ✓
apply_temporal_to_df: 4-player DataFrame, modes/adj/delta all correct  ✓
PASS: True
```

---

## Module 9 — team_portrait.py

**Purpose:** Integration layer. Combines all prior modules into a unified team portrait for one team × season.

**Team-player affiliation logic:** `inning_topbot == 'Top'` → pitcher is `home_team`; `inning_topbot == 'Bot'` → pitcher is `away_team`. All affiliation inferred from Statcast — no external roster file required.

### Functions

| Function | Description |
|----------|-------------|
| `build_team_portrait(team, season, statcast, statcast_label)` | Full portrait. Loads data, runs all modules, returns nested portrait dict. |
| `portraits_to_dataframe(portraits)` | Flatten list of portrait dicts to one row per team × season. |

### Portrait Structure

```python
{
  team, season,
  players: {
    hitters:        [build_hitter_profile(...) per player],
    starters:       [build_starter_profile(...) per pitcher with GS > 0],
    bullpen_profile: build_bullpen_profile(team_aggregate),
  },
  team_metrics: {batting: {...agg pct metrics}, pitching: {...agg pct metrics}},
  park:         build_park_profile() row for this team,
  philosophy:   build_philosophy_summary() result (scores A1–C4, confidences),
  temporal:     {mode, pa, games},
  spin:         {mean_spin_efficiency, weighted_spin_efficiency_pct, pitcher_count},
  tunneling:    {pitcher_level: [...]},
  data_coverage: float (fraction of philosophy metric slots populated),
}
```

### Smoke Test Results (2023-06-01 → 2023-06-07, HOU/NYY/ATL)

```
team=HOU  temporal=historical (7 games/258 PA)  hitters=13  data_coverage=20.8%
team=NYY  temporal=historical (7 games/~PA)      hitters=14  data_coverage=18.8%
team=ATL  temporal=historical (7 games/~PA)      hitters=10  data_coverage=18.8%

HOU offense_primary=A2 (Contact and Pressure)   conf=50
HOU pitching_primary=B1 (Stuff Dominant)         conf=2
NYY offense_primary=A1 (Three True Outcomes)     conf=53
ATL offense_primary=A2 (Contact and Pressure)    conf=29
portraits_to_dataframe: 3 rows, all columns present
PASS: True
```

**Note:** 20% data coverage on a 1-week slice is expected. Full season provides C-dimension roster metrics (WAR, payroll, age) which require separate data sources (GAP 4, GAP 5). Starters show 0 in short slice because starter-BF threshold requires ≥150 BF.

---

## Module 10 — validation.py

**Purpose:** 7 validation tests against known analytical outputs. Verifies every layer of the system — temporal math, reliability math, philosophy scoring, archetype routing, and end-to-end portrait integration.

### Tests

| Code | Name | What it verifies |
|------|------|-----------------|
| V1 | Temporal mode determination | All 7 PA/games combos → correct mode; blend weights at boundary and midpoint |
| V2 | Reliability weighting | `compute_reliability` output; `apply_reliability_weight` shrinkage math |
| V3 | Historical smoothing | 3-year weighted average hand calculation; missing-season reweighting; all-missing → None |
| V4 | Philosophy scoring | TTO-heavy metric dict → A1 score ≈ 79.5; primary offense = A1 |
| V5 | Hitter archetypes | Complete Hitter routing; TTO low-conf → spectrum; power spectrum; ordering (Complete > TTO > Spectrum) |
| V6 | Pitcher archetypes | P1 qualification; P2 velo ceiling enforcement; P3 K%/HardHit ceiling; P1 wins when multiple qualify |
| V7 | End-to-end portrait | HOU/NYY/ATL portraits build without error on cached 2023 data; structure check; DataFrame output |

### Smoke Test Results

```
PASS  V1  Temporal mode determination  — all cases correct
PASS  V2  Reliability weighting        — all cases correct
PASS  V3  Historical smoothing         — all cases correct
PASS  V4  Philosophy scoring           — all cases correct
PASS  V5  Hitter archetypes            — all cases correct
PASS  V6  Pitcher archetypes           — all cases correct
PASS  V7  End-to-end portrait          — 3 portraits built, DataFrame 3 rows
ALL PASS  7/7 tests passed
```

---

## Known Gaps

| Gap | Description |
|-----|-------------|
| GAP 1 | Retrosheet baserunning (XBT%, first-to-third). Using Statcast proxies. Leave None where unavailable. |
| GAP 2 | True pitch tunneling (linear interpolation used; Magnus force upgrade path documented). |
| GAP 3 | MiLB data (MLB 2015+ only; architecture ready for MiLB Statcast when released). |
| GAP 4 | Payroll data (manual CSV import from Spotrac; `data/reference/payroll_{season}.csv`). |
| GAP 5 | Prospect pipeline rankings (manual CSV from Baseball America / MLB Pipeline; `data/reference/pipeline_rankings_{season}.csv`). |

---

## Phase 2 — Dashboard

**Stack:** Dash 4 + dash-bootstrap-components + Plotly 6. Dark theme.

**Run:**
```bash
python baseball_construction/dashboard/app.py
# Open http://localhost:8050
```

**Files:**

| File | Purpose |
|------|---------|
| `dashboard/app.py` | Dash app init, layout wiring, server entry point |
| `dashboard/layout.py` | Bootstrap component builders: navbar, tabs, cards, controls |
| `dashboard/charts.py` | Pure Plotly figure builders (one function per chart, no Dash) |
| `dashboard/callbacks.py` | Callback wiring: load-btn → portrait-store → each chart |

**Architecture:** A `dcc.Store` holds the serialized portrait JSON. One callback (button click → portrait-store) runs all analytics. Every chart reads from the store independently — adding a new visualization is 3 steps: figure function in `charts.py`, component in `layout.py`, callback in `callbacks.py`.

**Charts implemented:**

| Chart | Function | Tab |
|-------|----------|-----|
| Philosophy radar (all 12) | `philosophy_radar(portrait, "all")` | Overview |
| Dimension confidence bars | `dimension_confidence_bars(portrait)` | Overview |
| Park factor gauge | `park_factor_gauge(portrait)` | Overview |
| Spin efficiency bar | `spin_efficiency_bar(portrait)` | Overview |
| Offense radar | `philosophy_radar(portrait, "offense")` | Offense |
| Batting metrics percentile bars | `team_batting_bars(portrait)` | Offense |
| Hitter archetype pie | `hitter_archetype_pie(portrait)` | Offense |
| Hitter roster table | `hitter_archetype_table(portrait)` | Offense |
| Pitching radar | `philosophy_radar(portrait, "pitching")` | Pitching |
| Starter archetype bars | `starter_archetype_bars(portrait)` | Pitching |
| Roster radar | `philosophy_radar(portrait, "roster")` | Roster |

**Smoke test:** All 12 chart functions PASS with HOU 2023 portrait data.

---

## Data Directory

```
data/
├── raw/
│   └── statcast_{season}.parquet     # pitch-level Statcast, one file per season
├── processed/
│   ├── fg_batting_{season}.csv       # season batting aggregates
│   └── fg_pitching_{season}.csv      # season pitching aggregates
└── reference/
    ├── chadwick_crosswalk.csv         # Chadwick Bureau ID register (combined)
    ├── payroll_{season}.csv           # manual import from Spotrac
    └── pipeline_rankings_{season}.csv # manual import from BA / MLB Pipeline
```
