"""
metric_contracts.py — what every stored percentile means.

A percentile is only a number until three things are said about it: WHAT was
ranked (the raw quantity and its unit), AGAINST WHOM (the cohort), and WHICH
WAY (whether a high rank means more of the thing or less). Those facts used
to live in three places — the rank-invert tables, the column names, and the
gates that read them — and the names lied: a hitter's `K` was the raw
strikeout rate while `K_pct` was its percentile, `ISO` and `AVG` were
percentiles under raw names, and `season_metrics["K_pct_raw"]` was a season
rank while `metrics["K_pct_raw"]` was an era one.

Since schema 25 a percentile has ONE name: the ranked column's, which is
`{metric}_pct` (`ISO_pct`, `BB_pct_pct`, `K_pct_raw_pct`). The scope it is
stored in says the cohort — `metrics` era, `season_metrics` this season —
and the same metric keeps the same name in both. The rank tables in
team_portrait are derived from here, and validation V27 checks every numeric
key in every portrait against it: declared, in range, and pointing the way
it says (against the raw values where both are stored).

Since schema 26 pitcher `metrics` follow the same rule: one name per
percentile (`K_pct_pitch_pct`, not the old `K_pct_pct` alias, which for a
pitcher meant the opposite-side metric's name) and no raw fractions.
`BB_pct_pct` is now only ever the hitter's walk percentile.

COHORTS name the population a rank is taken in. Era cohorts pool every
finished Statcast season plus the portrait's own (ingest.era_pool_seasons);
season cohorts are this season only, league-wide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

COHORTS: dict[str, str] = {
    "era_hitters":   "hitters with 100+ PA in any Statcast-era season (2015 through this one)",
    "era_pitchers":  "pitchers with 30+ BF in any Statcast-era season (2015 through this one)",
    "season_hitters": "this season's hitters, league-wide",
    "season_stints": "every player-team stint this season",
    "season_mechanics": "this season's pitchers with 150+ pitches with release data",
    "season_sequencing": "this season's pitchers with 300+ pitches",
    "season_discipline": "this season's pitchers with 150+ pitches (first-pitch rate: 50+ first pitches)",
    "season_tunnel": "this season's pitchers with 10+ qualifying pitch pairs",
    "season_spin": "this season's pitchers, by pitch type (10+ of a type)",
    "savant_qualified": "players on Baseball Savant's qualified leaderboard this season",
    "season_framing": "this season's catchers with 500+ taken pitches near the edge",
    "season_steal_opps": "this season's hitters with 50+ times on base, league-wide",
    "season_xbt": "this season's runners with 10+ extra-base chances",
    "season_runner_control": "this season's pitchers with runner opportunities",
    "era_talent_hitters": "career-to-date estimates of hitters with 100+ PA in a "
                          "Statcast-era season, each as of that season",
    "era_talent_pitchers": "career-to-date estimates of pitchers with 30+ BF in a "
                           "Statcast-era season, each as of that season",
    "season_talent_hitters": "career-to-date estimates of this season's hitters with 100+ PA",
}


@dataclass(frozen=True)
class Percentile:
    scope: str             # where it is stored (see SCOPES)
    key: str               # its key there
    label: str             # what was ranked, in words
    cohort: str            # a COHORTS key
    high: str              # what a HIGH percentile means
    inverted: bool = False  # ranked 100 - rank, so a LOW raw value ranks high
    raw: Optional[str] = None   # the raw value's key in the same scope, if stored
    ranks: Optional[str] = None  # internal column name the rank tables key on
    aliases: tuple = ()     # other keys in the same scope holding this value


# Where values live in a portrait player profile.
SCOPES = {
    "H.metrics":   "hitter gate_inputs.metrics",
    "H.season":    "hitter gate_inputs.season_metrics",
    "H.catching":  "hitter gate_inputs.catching",
    "H.adv":       "hitter gate_inputs.advancement",
    "H.gi":        "hitter gate_inputs",
    "P.metrics":   "pitcher gate_inputs.metrics",
    "P.tempo":     "pitcher gate_inputs.tempo",
    "P.rc":        "pitcher gate_inputs.runner_control",
    "P.pool.league_mechanics":  "pitcher pool row, mechanics",
    "P.pool.league_sequencing": "pitcher pool row, sequencing",
    "P.pool.league_discipline": "pitcher pool row, plate discipline",
    "P.pool.league_tunnel":     "pitcher pool row, tunneling",
    "P.pool.league_spin":       "pitcher pool row, spin efficiency",
    "H.talent":    "hitter gate_inputs.talent_metrics",
    "H.talent_season": "hitter gate_inputs.talent_season_metrics",
    "P.talent":    "pitcher gate_inputs.talent_metrics",
}


def _h_era(stem: str, label: str, high: str, inverted: bool = False,
           ranks: Optional[str] = None) -> Percentile:
    """An era hitter percentile, stored as `{stem}_pct`."""
    return Percentile("H.metrics", f"{stem}_pct", label, "era_hitters", high,
                      inverted, None, ranks or stem)


def _h_szn(stem: str, label: str, high: str, cohort: str = "season_hitters") -> Percentile:
    """A this-season hitter percentile: the same name as its era twin."""
    return Percentile("H.season", f"{stem}_pct", label, cohort, high)


PERCENTILES: list[Percentile] = [
    # ── hitters, era (team_portrait._normalize_batting) ────────────────────
    # The two K% percentiles keep a raw pair: k_rate_raw, so V27 still sees
    # the inverted one point down.
    Percentile("H.metrics", "K_pct_pct", "strikeout rate", "era_hitters", "fewer strikeouts",
               inverted=True, raw="k_rate_raw", ranks="K_pct"),
    # ranked separately (not through the rank table) for the TTO classifier
    Percentile("H.metrics", "K_pct_raw_pct", "strikeout rate", "era_hitters", "more strikeouts",
               raw="k_rate_raw"),
    _h_era("BB_pct", "walk rate", "more walks"),
    _h_era("HardHit_pct", "hard-hit rate", "more hard contact"),
    _h_era("Barrel_pct", "barrel rate", "more barrels"),
    _h_era("Whiff_pct", "whiff rate per swing", "more whiffs"),
    _h_era("OSwing_pct", "chase rate", "more chasing"),
    _h_era("ZSwing_pct", "zone swing rate", "more zone swings"),
    _h_era("Contact_pct", "contact rate per swing", "more contact"),
    _h_era("FPS_pct", "first-pitch swing rate", "more first-pitch swings"),
    _h_era("BatGB_pct", "ground-ball rate", "more ground balls"),
    _h_era("Pull_pct", "pull rate", "more pulled balls"),
    _h_era("Oppo_pct", "opposite-field rate", "more opposite-field balls"),
    _h_era("GapTend_pct", "gap tendency", "more gap hits"),
    _h_era("XB_pct", "extra-base hit rate per ball in play", "more extra-base hits"),
    _h_era("HR_FB_pct", "home runs per fly ball", "more home runs per fly ball"),
    _h_era("ISO", "isolated power", "more isolated power"),
    _h_era("AVG", "batting average", "higher average"),
    _h_era("OBP", "on-base percentage", "higher on-base percentage"),
    _h_era("SLG", "slugging percentage", "higher slugging"),
    _h_era("xBA", "expected batting average", "higher expected average"),
    _h_era("xwOBA", "expected wOBA", "higher expected wOBA"),
    _h_era("xSLG", "expected slugging", "higher expected slugging"),
    _h_era("WAR_bat", "batting WAR", "more WAR"),
    _h_era("EV_avg", "average exit velocity", "higher exit velocity"),
    _h_era("sprint_speed", "sprint speed", "faster"),
    _h_era("PA", "plate appearances", "more playing time"),
    _h_era("SB", "stolen bases", "more stolen bases"),
    _h_era("CS", "times caught stealing", "caught more often"),
    _h_era("PitchesPerPA", "pitches seen per PA", "more pitches per PA"),
    _h_era("OBP_SLG_gap", "OBP minus SLG", "on-base outweighs power"),
    # ── hitters, season (tag gates; the _szn columns) ──────────────────────
    _h_szn("K_pct_raw", "strikeout rate", "more strikeouts"),
    _h_szn("BB_pct", "walk rate", "more walks"),
    _h_szn("OSwing_pct", "chase rate", "more chasing"),
    _h_szn("FPS_pct", "first-pitch swing rate", "more first-pitch swings"),
    _h_szn("ISO", "isolated power", "more isolated power"),
    _h_szn("AVG", "batting average", "higher average"),
    _h_szn("OBP", "on-base percentage", "higher on-base percentage"),
    _h_szn("Barrel_pct", "barrel rate", "more barrels"),
    _h_szn("HardHit_pct", "hard-hit rate", "more hard contact"),
    _h_szn("BatGB_pct", "ground-ball rate", "more ground balls"),
    _h_szn("sprint_speed", "sprint speed", "faster"),
    _h_szn("SwingDiscrim", "zone swing minus chase", "better swing decisions"),
    _h_szn("OAA", "outs above average", "better defense", "savant_qualified"),
    _h_szn("ArmStrength", "throwing arm strength", "stronger arm", "savant_qualified"),
    _h_szn("PAvol", "plate appearances with this team", "more playing time", "season_stints"),
    # ── hitters, other ranks ───────────────────────────────────────────────
    Percentile("H.gi", "attempt_rate_pct", "steal attempts per time on base", "season_steal_opps", "runs more"),
    Percentile("H.adv", "xbt_pct", "extra-base attempts per chance", "season_xbt",
               "goes for the extra base more", raw="attempt_rate"),
    Percentile("H.catching", "framing_pct", "called strikes above expected", "season_framing",
               "steals more strikes", raw="framing_above"),
    Percentile("H.catching", "blocking_pct", "blocks above average", "savant_qualified",
               "blocks more", raw="blocks_above_average"),
    Percentile("H.catching", "pop_pct", "pop time to second", "savant_qualified",
               "slower pop time", raw="pop_2b"),
    # ── pitchers, era (team_portrait._normalize_pitching) ──────────────────
    Percentile("P.metrics", "K_pct_pitch_pct", "strikeout rate", "era_pitchers", "more strikeouts",
               ranks="K_pct_pitch"),
    Percentile("P.metrics", "BB_pct_pitch_pct", "walk rate", "era_pitchers", "fewer walks",
               inverted=True, ranks="BB_pct_pitch"),
    Percentile("P.metrics", "HardHit_allowed_pct", "hard-hit rate allowed", "era_pitchers",
               "less hard contact allowed", inverted=True, ranks="HardHit_allowed"),
    Percentile("P.metrics", "Barrel_allowed_pct", "barrel rate allowed", "era_pitchers",
               "fewer barrels allowed", inverted=True, ranks="Barrel_allowed"),
    Percentile("P.metrics", "xwOBA_allowed_pct", "expected wOBA allowed", "era_pitchers",
               "less damage allowed", inverted=True, ranks="xwOBA_allowed"),
    Percentile("P.metrics", "FIP_pct", "FIP", "era_pitchers", "better FIP", inverted=True, ranks="FIP"),
    Percentile("P.metrics", "Whiff_pct_pitch_pct", "whiff rate per swing", "era_pitchers", "more whiffs",
               ranks="Whiff_pct_pitch"),
    Percentile("P.metrics", "GB_pct_pitch_pct", "ground-ball rate", "era_pitchers", "more ground balls",
               ranks="GB_pct_pitch"),
    Percentile("P.metrics", "Zone_pct_pct", "zone rate", "era_pitchers", "more pitches in the zone",
               ranks="Zone_pct"),
    Percentile("P.metrics", "CSW_pct_pct", "called plus swinging strike rate", "era_pitchers",
               "more called and swinging strikes", ranks="CSW_pct"),
    Percentile("P.metrics", "avg_velo_pct", "average fastball velocity", "era_pitchers", "more velocity",
               ranks="avg_velo"),
    Percentile("P.metrics", "AvgSpinRate_pct", "average spin rate", "era_pitchers", "more spin",
               ranks="AvgSpinRate"),
    Percentile("P.metrics", "ArsenalDiversity_pct", "arsenal diversity", "era_pitchers", "deeper arsenal",
               ranks="ArsenalDiversity"),
    Percentile("P.metrics", "PitchesPerBF_pct", "pitches per batter", "era_pitchers", "more pitches per batter",
               ranks="PitchesPerBF"),
    Percentile("P.metrics", "BF_pct", "batters faced", "era_pitchers", "more workload", ranks="BF"),
    Percentile("P.metrics", "GS_pct", "games started", "era_pitchers", "more starts", ranks="GS"),
    Percentile("P.metrics", "WAR_pitch_pct", "pitching WAR", "era_pitchers", "more WAR", ranks="WAR_pitch"),
    Percentile("P.metrics", "K_inv_pct", "strikeout rate", "era_pitchers", "fewer strikeouts",
               inverted=True),
    # ── pitchers, season pools (attached to metrics and stored as pool rows)
    Percentile("P.metrics", "TunnelScore_pct", "tunnel score", "season_tunnel", "better tunneling"),
    Percentile("P.metrics", "SwStr_per_velo_pct", "whiff rate above what velocity predicts",
               "season_discipline", "more whiffs than velocity predicts"),
    Percentile("P.metrics", "FirstPitchStrike_pct", "first-pitch strike rate", "season_discipline",
               "more first-pitch strikes"),
    Percentile("P.pool.league_mechanics", "arm_angle_pct", "arm angle", "season_mechanics",
               "higher arm slot", raw="arm_angle"),
    Percentile("P.pool.league_mechanics", "extension_pct", "release extension", "season_mechanics",
               "longer extension", raw="extension"),
    Percentile("P.pool.league_mechanics", "rel_x_abs_pct", "release width (within handedness)",
               "season_mechanics", "wider release", raw="rel_x_abs"),
    Percentile("P.pool.league_sequencing", "entropy_ratio_pct", "pitch-sequence entropy",
               "season_sequencing", "less predictable", raw="entropy_ratio"),
    Percentile("P.pool.league_sequencing", "count_shift_pct", "mix change ahead vs behind",
               "season_sequencing", "more count-dependent", raw="count_shift"),
    Percentile("P.pool.league_discipline", "first_pitch_strike_pct", "first-pitch strike rate",
               "season_discipline", "more first-pitch strikes", raw="fps_rate"),
    Percentile("P.pool.league_discipline", "swstr_per_velo_pct", "whiff rate above what velocity predicts",
               "season_discipline", "more whiffs than velocity predicts", raw="swstr_resid"),
    Percentile("P.pool.league_tunnel", "tunnel_score_pct", "tunnel score", "season_tunnel",
               "better tunneling", raw="tunnel_score_weighted"),
    # Ranked within each pitch type, then usage-weighted — not a rank of
    # spin_efficiency_weighted (they agree only at rho ~0.6), so no raw pair.
    Percentile("P.pool.league_spin", "spin_efficiency_pct",
               "spin efficiency against the same pitch types, usage-weighted", "season_spin",
               "more efficient spin"),
    Percentile("P.pool.league_spin", "spin_efficiency_type_pct", "spin efficiency of the main pitch",
               "season_spin", "more efficient spin"),
    Percentile("P.tempo", "tempo_pct", "seconds between pitches, bases empty", "savant_qualified",
               "slower", raw="tempo_empty_s"),
    Percentile("P.rc", "advance_pct", "runner advances allowed per opportunity",
               "season_runner_control", "more advances allowed", raw="advance_rate"),
]

# ── career-to-date talent (talent.py) ──────────────────────────────────────
# Each talent percentile ranks a career-to-date estimate of the SAME stat as
# its twin — same name, label and direction, a different scope and cohort —
# and stores the estimate beside it as `{stem}_est`, so V27 checks the rank
# points the way the twin says. V30 checks these lists match talent.SPECS.
TALENT_STEMS: dict[str, tuple] = {
    "H": ("K_pct_raw", "K_pct", "BB_pct", "OSwing_pct", "ZSwing_pct", "FPS_pct", "Whiff_pct",
          "Contact_pct", "BatGB_pct", "Pull_pct", "Oppo_pct", "HardHit_pct", "Barrel_pct",
          "EV_avg", "ISO", "HR_FB_pct", "XB_pct", "xwOBA"),
    "P": ("K_pct_pitch", "BB_pct_pitch", "Whiff_pct_pitch", "CSW_pct", "Zone_pct",
          "GB_pct_pitch", "HardHit_allowed", "Barrel_allowed", "xwOBA_allowed"),
}
TALENT_SEASON_STEMS_H: tuple = ("K_pct_raw", "BB_pct", "OSwing_pct", "FPS_pct", "BatGB_pct",
                                "HardHit_pct", "Barrel_pct", "ISO", "SwingDiscrim")


def _talent(scope: str, twin_scope: str, stem: str, cohort: str) -> Percentile:
    t = next(c for c in PERCENTILES if c.scope == twin_scope and c.key == f"{stem}_pct")
    return Percentile(scope, t.key, t.label, cohort, t.high, t.inverted, f"{stem}_est")


PERCENTILES += [_talent("H.talent", "H.metrics", s, "era_talent_hitters")
                for s in TALENT_STEMS["H"]]
PERCENTILES += [_talent("H.talent_season", "H.season" if s in ("SwingDiscrim",) else "H.metrics",
                        s, "season_talent_hitters") for s in TALENT_SEASON_STEMS_H]
PERCENTILES += [_talent("P.talent", "P.metrics", s, "era_talent_pitchers")
                for s in TALENT_STEMS["P"]]

# Numeric keys that are NOT percentiles, by scope, with what they are. V27
# requires every numeric key in a portrait to be in PERCENTILES or here.
RAW: dict[str, dict[str, str]] = {
    "H.metrics": {"k_rate_raw": "strikeout rate (fraction of PA)"},
    "H.adv": {"opps": "count", "attempts": "count", "advances": "count", "thrown_out": "count",
              "attempt_rate": "fraction", "xbt_rate": "fraction",
              "first_to_third_n": "count", "first_to_third_opps": "count"},
    "H.catching": {"framing_above": "called strikes above expected per 100 edge pitches",
                   "framing_taken": "count", "blocks_above_average": "count",
                   "n_pbwp": "count", "pop_2b": "seconds"},
    "H.gi": {"sprint_speed_raw": "ft/s", "talent_n": "decayed PA behind the talent estimates"},
    "P.gi": {"talent_n": "decayed BF behind the talent estimates"},
    "H.talent": {f"{s}_est": "career-to-date estimate" for s in TALENT_STEMS["H"]},
    "H.talent_season": {f"{s}_est": "career-to-date estimate" for s in TALENT_SEASON_STEMS_H},
    "P.talent": {f"{s}_est": "career-to-date estimate" for s in TALENT_STEMS["P"]},
    "P.tempo": {"tempo_empty_s": "seconds"},
    "P.rc": {"opps": "count", "advances_allowed": "count", "advance_rate": "fraction"},
    "P.pool.league_mechanics": {"pitcher": "id", "method_version": "version", "n_pitches": "count",
                                "arm_angle": "degrees", "extension": "ft", "rel_x": "ft",
                                "rel_x_abs": "ft", "rel_z": "ft"},
    "P.pool.league_sequencing": {"pitcher": "id", "method_version": "version", "n_pitches": "count",
                                 "n_transitions": "count", "count_shift": "distance",
                                 "entropy_ratio": "ratio"},
    "P.pool.league_discipline": {"pitcher": "id", "method_version": "version", "n_pitches": "count",
                                 "n_first": "count", "fb_velo": "mph", "fps_rate": "fraction",
                                 "swstr_rate": "fraction", "swstr_resid": "fraction"},
    "P.pool.league_tunnel": {"pitcher": "id", "method_version": "version", "pair_count": "count",
                             "league_mean_tunnel": "score", "plate_divergence_mean": "ft",
                             "reliability": "fraction", "tunnel_distance_mean": "ft",
                             "tunnel_ratio_mean": "ratio", "tunnel_score_raw": "score",
                             "tunnel_score_weighted": "score"},
    "P.pool.league_spin": {"pitcher": "id", "spin_method_version": "version",
                           "spin_efficiency_weighted": "fraction", "spin_savant_share": "fraction",
                           "total_pitches": "count"},
}


# The curated percentiles a player card charts (profile["metrics_pct"]),
# display label -> the metrics key it shows. "Avoids Ks" is the inverted
# strikeout percentile; it was labelled "Contact" beside "Contact Rate".
HITTER_DISPLAY: dict[str, str] = {
    "Power": "ISO_pct", "Walk Rate": "BB_pct_pct", "Avoids Ks": "K_pct_pct",
    "Quality": "xwOBA_pct", "Speed": "sprint_speed_pct",
    "Contact-First": "OBP_SLG_gap_pct", "Aggression": "FPS_pct_pct",
    "Zone Swing": "ZSwing_pct_pct", "Contact Rate": "Contact_pct_pct",
    "Hard Hit": "HardHit_pct_pct", "Barrel Rate": "Barrel_pct_pct",
}
PITCHER_DISPLAY: tuple = (
    "avg_velo_pct", "K_pct_pitch_pct", "Whiff_pct_pitch_pct", "GB_pct_pitch_pct",
    "HardHit_allowed_pct", "Barrel_allowed_pct", "CSW_pct_pct", "BB_pct_pitch_pct",
)


def describe_display(label: str, pitcher: bool = False) -> Optional[str]:
    """describe() for a card's display label (hitters) or key (pitchers)."""
    if pitcher:
        return describe("P.metrics", label)
    key = HITTER_DISPLAY.get(label)
    return describe("H.metrics", key) if key else None


def lookup(scope: str, key: str) -> Optional[Percentile]:
    """The contract for `key` stored in `scope`, by key or alias."""
    return _INDEX.get((scope, key))


def describe(scope: str, key: str) -> Optional[str]:
    """One line a reader can trust, e.g. 'K% percentile among hitters with
    100+ PA in any Statcast-era season — high means fewer strikeouts'."""
    c = lookup(scope, key)
    if c is None:
        return None
    return f"{c.label} percentile among {COHORTS[c.cohort]} — high means {c.high}"


def rank_invert(scope: str) -> dict[str, bool]:
    """The rank-invert table for an era scope, keyed by the internal column
    the ranking reads (team_portrait.BATTING/PITCHING_RANK_INVERT)."""
    return {c.ranks: c.inverted for c in PERCENTILES
            if c.scope == scope and c.ranks and c.cohort.startswith("era_")}


_INDEX: dict[tuple, Percentile] = {}
for _c in PERCENTILES:
    for _k in (_c.key, *_c.aliases):
        _INDEX[(_c.scope, _k)] = _c
