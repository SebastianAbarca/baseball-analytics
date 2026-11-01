"""
reliability.py — Sample sufficiency as a GATE, never as a displayed number.

WHY A GATE
----------
A rate computed from a tiny sample is not a weak signal, it is a different
kind of thing: it describes what happened, not what the player does. Measured
across the 360 portraits before this existed, 20.4% of all hitter tags fired
on fewer than 50 plate appearances — and the median carrier of `free swinger`
had NINE. That tag was not describing an approach; it was describing a
September call-up who swung at two balls.

The fix is refusal, not estimation. Where the sample cannot support a claim,
no tag is emitted. This is deliberately different from
ingest.apply_reliability_weight, which SHRINKS a value toward the league mean:
shrinkage invents a number that nobody observed, and this product is meant to
be base-descriptive. A tag is a claim about a player, so either the evidence
carries it or it stays silent — and silence is already a meaningful state in
this vocabulary, since a league-average player carries no tags either.

Reliability is never shown to a reader and never stored as a per-metric score.
Its only outputs are (a) whether a tag is allowed to fire and (b) one
player-level `limited_sample` marker.

HOW THE THRESHOLDS ARE CHOSEN
-----------------------------
They are not chosen. Each tag declares which METRIC its evidence rests on, and
the stabilization point for that metric — the published sample at which the
statistic starts describing the player rather than the sequence — supplies the
number. A tag fires when the player has at least RELIABILITY_FLOOR of that
sample. So there is one tunable knob in the whole system rather than ninety
hand-set cutoffs, which is the same reason the team headline labels were
retired.

Tags absent from TAG_EVIDENCE are ungated ON PURPOSE, and each one says why
in UNGATED_TAGS. Attributes (handedness), counts (super-utility's four
positions) and tags that already carry a purpose-built floor in their own gate
(`rarely strikes out`, the steal tags, platoon splits) do not need a second
one. validate_tag_evidence checks that every tag in the vocabulary is in
exactly one of the two maps, so a renamed or new tag cannot slip through.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stabilization points
# ---------------------------------------------------------------------------
# Sample at which a metric starts describing the player rather than the
# sequence of events. Canonical home for these values — ingest.py imports this
# dict rather than defining its own, so the shrinkage path and the gate path
# can never drift apart.
STABILIZATION: dict[str, int] = {
    # Hitter metrics
    "K_pct":           60,
    "BB_pct":         120,
    "SwStr_pct":      150,
    "Chase_pct":      100,
    "Contact_pct":    150,
    "ZContact_pct":   150,
    "xwOBA":          200,
    "EV_90":          200,
    "HardHit_pct":    200,
    "Barrel_pct":     300,
    "xBA":            200,
    "BABIP":          800,
    "HR_per_FB":      300,
    "ISO":            300,
    "wRC_plus":       300,
    "OBP":            300,
    "FPS_pct":        150,
    "OSwing_pct":     100,
    "PitchesPerPA":   150,
    # Statcast pitch-aggregate hitter metrics
    "BatGB_pct":      200,
    "FPS_pct_agg":    150,
    "pitches_per_pa": 150,
    # Pitcher metrics
    "spin_rate":      200,
    "spin_efficiency":200,
    "tunnel_score":   150,
    "K_pct_pitch":    150,
    "BB_pct_pitch":   150,
    "GB_pct":         200,
    "HardHit_allowed":200,
    "Barrel_allowed": 200,
    "FIP":            150,
    # Statcast pitch-aggregate pitcher metrics
    "GB_pct_pitch":   200,
    "Zone_pct":       150,
    "CSW_pct":        150,
    "pitches_per_bf": 150,
    "WAR":            162,
    # Catcher
    "framing":       1000,
    # Movement
    "sprint_speed":    50,
    "SB_attempts":      8,
}

# Fraction of the stabilization sample a player must reach before a tag
# resting on that metric may fire. Calibrated empirically rather than assumed:
# see the note on _CALIBRATION below.
RELIABILITY_FLOOR: float = 0.5

# Measured across all 360 portraits (29,468 hitter tags), suppression by floor:
#
#     0.25 -> 13.8%      0.50 -> 21.3%      0.75 -> 26.2%      1.00 -> 30.0%
#
# The curve is steepest below 0.5 and flattens after: going 0.5 -> 0.75 costs
# another 1,451 tags to remove only 90 more `free swinger`, because by then the
# tag is already down to players with real playing time. 0.5 is where the
# suppressed population stops looking like players and starts looking like
# call-ups. At that setting `free swinger` keeps 694 of 2,793 (needs 50 PA),
# `high-K` 699 of 1,827 (30 PA), `power bat` 1,297 of 2,063 (150 PA), and
# `lucky` is untouched because it already rested on a sufficient sample.
_CALIBRATION = "measured on 360 portraits, 2026-08-22"

# ---------------------------------------------------------------------------
# Which metric each tag's claim rests on
# ---------------------------------------------------------------------------
# Only tags whose evidence is a RATE appear here. See the module docstring for
# why the others are deliberately absent.
TAG_EVIDENCE: dict[str, str] = {
    # ── hitters ──────────────────────────────────────────────────────────
    "power bat":       "HardHit_pct",  # barrel + hard-hit composite
    "plus power":      "HardHit_pct",
    "gap hitter":      "ISO",          # extra-base production per ball in play
    "weak contact":    "Barrel_pct",   # barrel + hard-hit composite
    "high-K":          "K_pct",
    "walk machine":    "BB_pct",
    "patient":         "OSwing_pct",
    "free swinger":    "OSwing_pct",
    "zone hunter":     "OSwing_pct",   # differences two swing-decision rates
    "aggressive":      "FPS_pct",
    "pull-heavy":      "BatGB_pct",    # batted-ball direction share
    "oppo bat":        "BatGB_pct",
    "air-ball bat":    "BatGB_pct",
    "ground-ball bat": "BatGB_pct",
    # ── pitchers ─────────────────────────────────────────────────────────
    "bat-misser":        "K_pct_pitch",
    "pitch-to-contact":  "K_pct_pitch",
    "ground-baller":     "GB_pct_pitch",
    "fly-ball prone":    "GB_pct_pitch",
    "contact suppressor":"HardHit_allowed",
    "elite control":     "BB_pct_pitch",
    "plus control":      "BB_pct_pitch",
    "walk prone":        "BB_pct_pitch",
    "whiffs above velo": "SwStr_pct",     # velocity-adjusted whiff rate
    "first-pitch attacker": "FPS_pct",
    "high spin efficiency": "spin_efficiency",
    "tunneler":          "tunnel_score",
}

# Every other tag in the vocabulary, and why it is not gated here. Explicit,
# because "absent means ungated" is how `walk prone` and `fly-ball prone`
# went ungated: the map keyed `wild`, a name nothing has ever emitted, while
# the shipped tag was `walk prone`, and `fly-ball prone` was never added
# beside `ground-baller` — 751 walk-prone firings below the 75 BF floor.
# validate_tag_evidence fails on any vocabulary tag that is in neither map,
# or in both. Candidate tags that do not exist yet (`wild`, `zone-filler`,
# `nibbler`) live in scripts/tag_reference.PROSPECTIVE_TAGS, not here.
_ATTRIBUTE   = "an attribute of the player, not a rate"
_MECHANICS   = "release geometry; the mechanics pool requires 150 pitches"
_ARSENAL     = "pitch usage and shape; pitch_mix requires 150 pitches and 50 of a type"
_VELO        = "a physical measurement, stable within the 20-BF roster minimum"
_SEQUENCING  = "the sequencing pool requires 300 pitches"
_TEMPO       = "the tempo pool requires 300 bases-empty pitches"
_SAVANT_QUAL = "a Savant leaderboard that publishes qualified players only"
_OWN_FLOOR   = "its gate carries a purpose-built floor"
_DEPLOYMENT  = "how the club used the player; a count or a usage share"

UNGATED_TAGS: dict[str, str] = {
    **dict.fromkeys(["left-handed hitter", "right-handed hitter", "switch hitter",
                     "left-handed pitcher", "right-handed pitcher"], _ATTRIBUTE),
    **dict.fromkeys(["over-the-top", "sidearm", "submarine", "wide release",
                     "deep extension", "short extension"], _MECHANICS),
    **dict.fromkeys(["ride four-seam", "deep arsenal", "two-pitch", "cutter-primary",
                     "sinker-baller", "knuckleballer"], _ARSENAL),
    **dict.fromkeys(["elite velo", "plus velo", "soft tosser"], _VELO),
    **dict.fromkeys(["count-shifter", "steady mix", "patterned", "unpredictable"],
                    _SEQUENCING),
    **dict.fromkeys(["quick pitcher", "slow pitcher"], _TEMPO),
    **dict.fromkeys(["elite speed", "fast", "station-to-station",
                     "elite defender", "plus defender", "defensive liability",
                     "good blocker", "bad blocker", "cannon arm", "quick pop"],
                    _SAVANT_QUAL),
    "elite framer":       "framing requires 500 taken pitches near the edge",
    "poor framer":        "framing requires 500 taken pitches near the edge",
    "rarely strikes out": f"{_OWN_FLOOR}: 200 PA",
    "high steal rate":    f"{_OWN_FLOOR}: 10 attempts",
    "high steal attempts": f"{_OWN_FLOOR}: ranked league-wide among hitters with 50+ times on base",
    "low steal attempts":  f"{_OWN_FLOOR}: ranked league-wide among hitters with 50+ times on base",
    "extra base taker":   f"{_OWN_FLOOR}: 10 advancement chances",
    "controls runners":   f"{_OWN_FLOOR}: 100 runner opportunities",
    "platoon-vulnerable": f"{_OWN_FLOOR}: 50 BF on each side",
    **dict.fromkeys(["everyday player", "heavy usage", "high-leverage arm",
                     "mop-up duty", "multi-inning reliever", "short-outing starter",
                     "swingman", "workhorse", "platoon specialist", "super-utility"],
                    _DEPLOYMENT),
}


def validate_tag_evidence(vocabulary) -> list[str]:
    """
    Problems with the gate map against the live tag vocabulary
    (team_portrait.TAG_KIND): a tag in neither TAG_EVIDENCE nor UNGATED_TAGS
    passes ungated by accident, one in both is ambiguous, and a key the
    vocabulary no longer has is a retired name that gates nothing. Every
    evidence metric must have a stabilization point, or reliability() reads
    it as 0 and the tag can never fire. Empty list = consistent.
    """
    vocab = set(vocabulary)
    gated, exempt = set(TAG_EVIDENCE), set(UNGATED_TAGS)
    problems = []
    if vocab - gated - exempt:
        problems.append(f"neither gated nor exempt: {sorted(vocab - gated - exempt)}")
    if gated & exempt:
        problems.append(f"both gated and exempt: {sorted(gated & exempt)}")
    if (gated | exempt) - vocab:
        problems.append(f"not in the vocabulary (retired?): {sorted((gated | exempt) - vocab)}")
    no_stab = sorted({m for m in list(TAG_EVIDENCE.values()) + [OUT_PITCH_EVIDENCE]
                      if m not in STABILIZATION})
    if no_stab:
        problems.append(f"evidence metric with no stabilization point: {no_stab}")
    return problems

# The `<pitch> out pitch` tags, matched by suffix in tag_supported.
OUT_PITCH_EVIDENCE = "SwStr_pct"

# A player under this share of a full workload carries the `limited_sample`
# marker. One marker, player-level, not per metric — a reader needs to know
# "don't lean on this profile", not to audit thirteen denominators.
LIMITED_SAMPLE_PA = 100
LIMITED_SAMPLE_BF = 100


def reliability(metric: str, n: Optional[float]) -> float:
    """Fraction of the metric's stabilization sample this player reached,
    capped at 1.0. Unknown metric or missing sample → 0.0 (claim unsupported)."""
    threshold = STABILIZATION.get(metric)
    if threshold is None:
        log.debug("Metric '%s' not in STABILIZATION — treating as unsupported", metric)
        return 0.0
    if n is None:
        return 0.0
    try:
        return min(1.0, max(0.0, float(n) / threshold))
    except (TypeError, ValueError):
        return 0.0


def tag_supported(tag: str, n: Optional[float],
                  floor: float = RELIABILITY_FLOOR) -> bool:
    """
    May this tag fire on a player with `n` of the relevant sample?

    True for any tag not in TAG_EVIDENCE — UNGATED_TAGS says why for each
    (validate_tag_evidence keeps the two maps covering the vocabulary).
    """
    metric = TAG_EVIDENCE.get(tag)
    if metric is None and tag.endswith(" out pitch"):
        # Named per pitch ("slider out pitch"), so it cannot be a TAG_EVIDENCE
        # key; its claim is a whiff rate, which rests on swinging strikes.
        metric = OUT_PITCH_EVIDENCE
    if metric is None:
        return True
    return reliability(metric, n) >= floor


def filter_traits(traits: list[dict], n: Optional[float],
                  floor: float = RELIABILITY_FLOOR) -> list[dict]:
    """Drop the traits whose evidence `n` cannot support. Order preserved."""
    return [t for t in traits if tag_supported(t.get("tag", ""), n, floor)]


def is_limited_sample(n: Optional[float], threshold: int) -> bool:
    """The player-level marker. True when the whole profile is thin."""
    if n is None:
        return True
    try:
        return float(n) < threshold
    except (TypeError, ValueError):
        return True
