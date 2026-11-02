"""
hitter_traits.py — Trait tags for hitters: evidence-based, independent,
non-exclusive.

Same philosophy as pitcher_traits.py: no boxes. Each trait is tagged when the
evidence clears a league-percentile gate; a league-average hitter may carry
zero tags.

Every tag carries four axes (see team_portrait.TAG_POPULATION and the tag
axis spec): family (subject) x side x kind (who controls it: attribute /
tool / behavior / result / deployment / noise) x population (the eligible
denominator it is scored against).

Families:
    bat          — power bat / plus power / gap hitter / weak contact
    bat-to-ball  — high-K / rarely strikes out
    approach     — patient / free swinger / zone hunter / walk machine /
                   aggressive
    batted-ball  — pull-heavy / oppo bat / air-ball bat / ground-ball bat
    athleticism  — elite speed / fast / station-to-station
    running-game — high|low steal attempts / high steal rate /
                   extra base taker
    fielding     — elite|plus defender / defensive liability / cannon arm /
                   super-utility
    catcher      — elite|poor framer / good|bad blocker / quick pop
    handedness   — left|right-handed hitter / switch hitter
    handedness   — left|right-handed / switch hitter
    role         — everyday player

The power-contact spectrum survives as a CONTINUOUS field (`spectrum`,
0=extreme contact, 100=extreme power) but no longer produces tags: a ratio
cannot express a hitter who is moderate in both, which is where doubles
hitters live.

Retired: `complete` / `table setter` (evaluative verdicts), `elite
discipline` (behavior/result composite), `contact bat` (ratio negative
space), `hard contact` (82% contained by power bat), `disruptive` /
`chaotic` (blended the choice to run with the result of running).

Inputs are the same pre-normalised metric dict + raw values that
hitter_archetypes.build_hitter_profile consumes — reuses its compute_*
functions directly so the underlying evidence definitions stay single-sourced.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from reliability import filter_traits, LIMITED_SAMPLE_PA  # noqa: F401
from hitter_archetypes import (
    compute_spectrum,
    compute_aggressive,
    compute_speed_tier,
    compute_gap_hitter,
    compute_disruptiveness,
)

log = logging.getLogger(__name__)

# Power gates on the PROCESS — barrel and hard-hit rates, how hard and how
# squarely a hitter strikes the ball — not on ISO, the OUTCOME (extra bases,
# after park, defense, gaps and luck). The process repeats: year-to-year rank
# correlation 0.81 for the barrel + hard-hit composite against 0.61 for ISO
# (300+ PA, 2022-25). ISO stays in the evidence. The composite is the mean of
# the two percentiles, which bunches toward the middle, so the tiers are set
# to fire about as often as the ISO gates did (85th / 60th percentile).
POWER_BAT_CONTACT    = 80.0   # barrel + hard-hit composite ≥ → power bat
PLUS_POWER_CONTACT   = 60.0   # composite ≥ (below power bat) → plus power
WEAK_CONTACT_COMPOSITE = 24.0  # composite ≤ → weak contact

# `gap hitter` means extra-base production WITHOUT home-run power. The HR/FB
# ceiling was removed once to avoid mutual exclusivity with plus power; the
# result was a tag that fired on 38% of home-run hitters and only 28% of
# doubles hitters. Mutual exclusivity here is the point, not an artifact.
GAP_HITTER_HRFB_CEILING = 70.0   # HR/FB percentile ≤ → power isn't the source
# Doubles are noisy (extra-base rate repeats at rho 0.21 at 100+ PA, 0.29 at
# 300+); the tag repeated 16% / 20% of the time at those floors.
GAP_HITTER_MIN_PA = 300

# Recalibrated for schema 28 (talent.py). The gates below that read a Statcast
# stat now read a career-to-date estimate, whose percentiles spread evenly
# across regulars; the single-season ranks they were tuned on were bunched —
# small samples and pitchers batting held the extremes of the era pool, so no
# regular reached the 90th percentile of strikeout rate. Each threshold was
# moved to keep its tag's count where it was (a calibration replay,
# 2026-11-01): high-K 80 -> 94, rarely strikes out 10 -> 7, oppo bat
# 85 -> 90.5, pull-heavy 85 -> 86, ground-ball bat 90 -> 94.5, walk machine
# 80 -> 75.5, weak contact 30 -> 24, patient 15 -> 16.5, zone hunter
# 85 -> 83.5. Same counts, better tags, judged against the NEXT season's
# actual rate (Statcast counts): 84% of high-K hitters are in the following
# season's top fifth of strikeout rates (69% before), rarely strikes out 93%
# (86%); walk machine and pull-heavy gain a point or two.

# Approach gates — zone judgment only. `patient` / `free swinger` read chase
# rate ALONE: the old composite was 60% walk rate, and walk rate is a result,
# not a decision. Chase and K rate correlate +0.02, so this axis is genuinely
# independent of bat-to-ball.
PATIENT_CHASE_PCT      = 16.5   # OSwing (chase) pct ≤ → patient
FREE_SWINGER_CHASE_PCT = 85.0   # OSwing (chase) pct ≥ → free swinger
WALK_MACHINE_PCT = 75.5
# Swing discrimination (zone-swing rate − chase rate) percentile ≥ → zone
# hunter. See the tag for why the raw zone-swing rate is not used.
ZONE_HUNTER_PCT = 83.5

# bat-to-ball gates (K rate correlates −0.88 with contact rate — this is a
# contact axis, not an approach one)
HIGH_K_PCT       = 94.0   # K_pct_raw percentile (non-inverted: higher = more Ks)
RARELY_K_PCT     = 7.0   # ≤ → "rarely strikes out" (min PA gate below)
RARELY_K_MIN_PA  = 200    # small samples produce fake 8% K rates
STATION_SPEED_PCT = 10.0  # sprint speed season pct ≤ → "station-to-station"
EVERYDAY_PA_PCT  = 85.0   # PA season pct ≥ → "everyday player"

# Running game — the decision to run and the success of running are separate
# claims, so they get separate tags. `disruptive`/`chaotic` blended them.
STEAL_ATTEMPTS_HIGH_PCT = 85.0   # attempt-rate percentile ≥ → runs often
STEAL_ATTEMPTS_LOW_PCT  = 15.0   # ≤ → stays put
STEAL_RATE_HIGH         = 0.80   # success rate ≥ (league ~0.78)
# `low steal rate` is retired: success rate repeats year to year at rho
# ~0.2 at any attempt floor, and runners who keep getting caught stop being
# sent, so the tag empties out as the floor rises (65 -> 21 -> 6 -> 0 at
# 10 / 15 / 20 / 30 attempts, 2015-26). The high end stays: it is rarer
# noise and a team keeps sending a runner who makes it.
STEAL_RATE_MIN_ATTEMPTS = 10     # below this a rate is noise

# Batted-ball direction gates (spray percentiles are season-ranked upstream)
PULL_HEAVY_PCT      = 86.0
OPPO_BAT_PCT        = 90.5
AIR_BALL_GB_PCT     = 10.0   # batter GB% at/below → lives in the air
GROUND_BALL_GB_PCT  = 94.5   # batter GB% at/above → hits it into the ground

# Platoon gates (raw wOBA gaps from statcast splits; league platoon gap ~.020)
PLATOON_MIN_PA        = 50
# `platoon liability` and `reverse split` are retired. A season's split is a
# difference of two ~150-450 PA wOBAs, about .045 of noise against a .040-.060
# gate, and it does not carry forward: over 2,620 hitter-seasons the split
# predicts the hitter's later seasons at rho +0.07 — +0.06 pooled over his
# career to date, +0.06 shrunk to his handedness's average. What persists is
# the handedness effect (lefties .036 worse vs LHP, righties .018 worse vs
# RHP), which the handedness tags already state. The split itself stays
# visible as numbers on the team split card.

# Fielding gates (season percentiles of OAA / arm strength)
ELITE_DEFENDER_PCT     = 95.0
PLUS_DEFENDER_PCT      = 85.0
DEF_LIABILITY_PCT      = 10.0
# OAA is a counting stat, so part-timers sit at the extremes: the tag repeated
# 24% of the time at 100+ PA, 28% at 300+, 31% at 450+ (2015-26).
DEF_LIABILITY_MIN_PA   = 300
CANNON_ARM_PCT         = 90.0

# Catcher gates (percentiles among qualified catchers)
ELITE_FRAMER_PCT   = 85.0
POOR_FRAMER_PCT    = 15.0
QUICK_POP_PCT      = 15.0   # pop time percentile — LOWER time = better
GOOD_BLOCKER_PCT   = 85.0   # blocks above average percentile
BAD_BLOCKER_PCT    = 15.0

# Baserunning advancement (extra bases taken on hits; base-state derived)
XBT_TAKER_PCT      = 85.0   # attempt-rate percentile among qualified runners
XBT_MIN_OPPS       = 10

# Versatility
SUPER_UTILITY_MIN_POSITIONS = 4


def _trait(tag: str, family: str, value, pct: Optional[float],
           evidence: str) -> dict:
    return {
        "tag":      tag,
        "family":   family,
        "value":    round(float(value), 2) if value is not None else None,
        "pct":      round(float(pct), 1) if pct is not None else None,
        "evidence": evidence,
    }


def build_hitter_traits(
    metrics:          dict[str, float],
    sprint_speed_raw: Optional[float] = None,
    sb:               int = 0,
    cs:               int = 0,
    opportunities:    int = 0,
    games:            int = 162,
    season_games:     int = 162,
    attempt_rate_pct: Optional[float] = None,
    season_metrics:   Optional[dict[str, float]] = None,
    splits:           Optional[dict] = None,
    catching:         Optional[dict] = None,
    pa:               Optional[int] = None,
    advancement:      Optional[dict] = None,
    positions:        Optional[dict] = None,
    pa_team:          Optional[int] = None,
    talent_metrics:   Optional[dict[str, float]] = None,
    talent_season_metrics: Optional[dict[str, float]] = None,
) -> tuple[list[dict], Optional[float]]:
    """
    Full trait set + spectrum score for one hitter-season.

    pa      — season PA, all teams: the sample the season-wide rates rest on,
              so the reliability gate's n.
    pa_team — PA for the team whose portrait this is. Deployment tags are
              earned with a team (`everyday player`'s PAvol gate is already
              ranked on it upstream); defaults to pa for a one-team season.

    season_metrics — within-season percentile overrides for TAG GATES
    (hybrid normalization: gates fire vs this year's peers so incidence
    doesn't drift with the run environment, while display percentiles and
    the spectrum score stay era-based).

    talent_metrics / talent_season_metrics — career-to-date percentiles
    (talent.py), era- and season-ranked, under the same names. A gate reads
    the season-ranked talent, then the season rank, then the era-ranked
    talent, then the era rank: talent replaces the one-season sample, and the
    cohort each gate was cut against stays the same.

    Returns (traits, spectrum) — spectrum is the continuous 0–100
    power–contact score (None if inputs are missing).
    """
    traits: list[dict] = []

    # Gate metrics: talent before season, season cohort before era.
    def _known(d):
        return {k: v for k, v in (d or {}).items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))}
    gm = {**metrics, **_known(talent_metrics), **_known(season_metrics),
          **_known(talent_season_metrics)}

    # ── bat: power (barrel + hard-hit — the process, not ISO) ─────────────
    # The spectrum is still computed and returned as a continuous field, but
    # nothing is tagged off its tails any more. Blank on a limited sample: it
    # is power / (power + contact), undefined when both are near zero, and
    # those hitters are exactly the ones without enough PA — a pitcher batting
    # .000 with no extra bases read 99.7 ("extreme power") before the on-base
    # fix and 1.7 ("extreme contact") after, and neither meant anything.
    spectrum = compute_spectrum(metrics)
    if pa is not None and pa < LIMITED_SAMPLE_PA:
        spectrum = None
    iso = gm.get("ISO_pct")
    vals = [float(v) for v in (gm.get("Barrel_pct_pct"), gm.get("HardHit_pct_pct")) if v is not None]
    contact = float(np.mean(vals)) if vals else None
    iso_note = f" (ISO {iso:.0f}th pct)" if iso is not None else ""
    power_bat = contact is not None and contact >= POWER_BAT_CONTACT
    if power_bat:
        # `hard contact` was 82% contained by this tag — one skill shouldn't
        # wear two chips, and "power" reads to a casual fan where "hard
        # contact" doesn't.
        traits.append(_trait("power bat", "bat", contact, contact,
                             f"barrel + hard-hit {contact:.0f}th pct — elite "
                             f"contact quality{iso_note}"))

    # ── bat: gap hitter — doubles, not homers ─────────────────────────────
    # `power bat` suppresses this outright. The HR/FB ceiling alone was not
    # enough: a hitter can clear elite ISO while his extra-base production
    # still skews doubles, so both tags fired on the same player and asserted
    # opposite things — "elite raw power" beside "without home-run power".
    # Measured on 2023, 2 of 4 gap hitters across HOU/ATL/LAD/NYY carried both
    # (Kyle Tucker, Freddie Freeman). Suppression mirrors `plus power`, which
    # both tags already silence.
    #
    # This also closes the ceiling's null hole: HR/FB is absent for ~2 of 3
    # hitters, and `hrfb is None` skipped the power check entirely. ISO is
    # known far more often, so power_bat now carries that case.
    hrfb = gm.get("HR_FB_pct_pct")
    gap_hitter = (not power_bat) and compute_gap_hitter(gm) and (
        hrfb is None or float(hrfb) <= GAP_HITTER_HRFB_CEILING) and (
        pa is not None and pa >= GAP_HITTER_MIN_PA)
    if gap_hitter:
        xb = gm.get("XB_pct_pct")
        traits.append(_trait("gap hitter", "bat", xb, xb,
                             "spray-confirmed extra-base production into the "
                             "gaps, without home-run power"
                             + (f" (HR/FB {hrfb:.0f}th pct)" if hrfb is not None else "")))
    if (not power_bat and not gap_hitter and contact is not None
            and contact >= PLUS_POWER_CONTACT):
        traits.append(_trait("plus power", "bat", contact, contact,
                             f"barrel + hard-hit {contact:.0f}th pct — real "
                             f"thump{iso_note}"))

    # ── bat: weak contact (no `hard contact` counterpart — see above) ──────
    # The composite is the archetype layer's (hitter_archetypes
    # .compute_contact_quality), but the line is this module's, so it can be
    # recalibrated and replayed without a rebuild.
    vals = [v for v in [gm.get("Barrel_pct_pct"), gm.get("HardHit_pct_pct")]
            if v is not None]
    comp = float(np.mean(vals)) if vals else None
    if comp is not None and comp <= WEAK_CONTACT_COMPOSITE:
        traits.append(_trait("weak contact", "bat", comp, comp,
                             f"barrel + hard-hit composite {comp:.0f}th pct"))

    # ── approach: zone judgment only ───────────────────────────────────────
    # `walk machine` is a RESULT and stands on its own; `patient` / `free
    # swinger` are DECISIONS read from chase rate alone. The old composite
    # mixed the two, which is why free swinger fired on 30% of the league and
    # 60% of high-K hitters despite chase and K being uncorrelated.
    bb = gm.get("BB_pct_pct")
    if bb is not None and float(bb) >= WALK_MACHINE_PCT:
        traits.append(_trait("walk machine", "approach", bb, bb,
                             f"walk rate {bb:.0f}th pct"))

    chase = gm.get("OSwing_pct_pct")   # higher = chases more
    if chase is not None and not (isinstance(chase, float) and np.isnan(chase)):
        if float(chase) <= PATIENT_CHASE_PCT:
            traits.append(_trait("patient", "approach", chase, chase,
                                 f"chase rate {chase:.0f}th pct — lays off "
                                 "pitches outside the zone"))
        elif float(chase) >= FREE_SWINGER_CHASE_PCT:
            traits.append(_trait("free swinger", "approach", chase, chase,
                                 f"chase rate {chase:.0f}th pct — expands the zone"))

    # `zone hunter` — swing DISCRIMINATION, not swing volume. Zone-swing rate
    # on its own was not worth a tag: it correlates +0.81 with first-pitch
    # swing rate, so it would have restated `aggressive`. Differencing it
    # against chase drops that to +0.26 and isolates a real, repeatable skill
    # — Corey Seager clears this gate in three separate seasons, Kyle Tucker
    # and Brandon Belt in two, while Javier Baez sits at the bottom twice.
    # Distinct from `patient`, which is chase alone and says nothing about
    # whether the hitter then does damage on the strikes he does swing at.
    discrim = gm.get("SwingDiscrim_pct")
    if (discrim is not None and not (isinstance(discrim, float) and np.isnan(discrim))
            and float(discrim) >= ZONE_HUNTER_PCT):
        traits.append(_trait("zone hunter", "approach", discrim, discrim,
                             f"swing discrimination {discrim:.0f}th pct — "
                             "attacks strikes and lays off balls"))

    # ── bat-to-ball ────────────────────────────────────────────────────────
    k = gm.get("K_pct_raw_pct")
    if k is not None and float(k) >= HIGH_K_PCT:
        traits.append(_trait("high-K", "bat-to-ball", k, k,
                             f"strikeout rate {k:.0f}th pct (more Ks than "
                             f"{k:.0f}% of this season's hitters)"))
    elif (k is not None and float(k) <= RARELY_K_PCT
            and pa is not None and pa >= RARELY_K_MIN_PA):
        traits.append(_trait("rarely strikes out", "bat-to-ball", k, k,
                             f"strikeout rate {k:.0f}th pct of the season — "
                             "elite bat-to-ball"))

    if compute_aggressive(gm):
        fps = gm.get("FPS_pct_pct")
        traits.append(_trait("aggressive", "approach", fps, fps,
                             "attacks first pitches and expands the zone "
                             "(high FPS% + high chase%)"))

    # ── athleticism / baserunning ──────────────────────────────────────────
    speed = compute_speed_tier(sprint_speed_raw)
    if speed == "Elite":
        traits.append(_trait("elite speed", "athleticism", sprint_speed_raw, None,
                             f"{sprint_speed_raw:.1f} ft/s sprint speed (top ~10%)"))
    elif speed == "Fast":
        traits.append(_trait("fast", "athleticism", sprint_speed_raw, None,
                             f"{sprint_speed_raw:.1f} ft/s sprint speed (top ~35%)"))
    _spd_szn = (season_metrics or {}).get("sprint_speed_pct")
    if (_spd_szn is not None and not (isinstance(_spd_szn, float) and np.isnan(_spd_szn))
            and float(_spd_szn) <= STATION_SPEED_PCT):
        traits.append(_trait("station-to-station", "athleticism",
                             sprint_speed_raw, _spd_szn,
                             (f"{sprint_speed_raw:.1f} ft/s — " if sprint_speed_raw else "")
                             + "bottom decile of the season's runners"))

    # ── running game: the decision to run, and the result of running ───────
    # Split from the old `disruptive`/`chaotic` pair, which blended volume with
    # success rate into a single verdict.
    attempts = int(sb) + int(cs)
    if attempt_rate_pct is not None and not (
            isinstance(attempt_rate_pct, float) and np.isnan(attempt_rate_pct)):
        rate = attempts / max(int(opportunities), 1)
        if float(attempt_rate_pct) >= STEAL_ATTEMPTS_HIGH_PCT:
            traits.append(_trait("high steal attempts", "running-game",
                                 rate, attempt_rate_pct,
                                 f"{attempts} attempts on {opportunities} times "
                                 f"aboard ({attempt_rate_pct:.0f}th pct) — runs often"))
        elif float(attempt_rate_pct) <= STEAL_ATTEMPTS_LOW_PCT:
            traits.append(_trait("low steal attempts", "running-game",
                                 rate, attempt_rate_pct,
                                 f"{attempts} attempts on {opportunities} times "
                                 f"aboard ({attempt_rate_pct:.0f}th pct) — stays put"))

    if attempts >= STEAL_RATE_MIN_ATTEMPTS:
        succ = sb / attempts
        if succ >= STEAL_RATE_HIGH:
            traits.append(_trait("high steal rate", "running-game", succ, None,
                                 f"{sb} of {attempts} ({succ:.0%}) — "
                                 "safe when he goes"))
    
    # `extra base taker` is a BEHAVIOR: ranked on how often he tries for the
    # extra base (made it or thrown out, compute_batter_advancement); how
    # often he makes it is shown as evidence, not gated on.
    adv = advancement or {}
    if (adv.get("xbt_pct") is not None and adv.get("opps", 0) >= XBT_MIN_OPPS
            and float(adv["xbt_pct"]) >= XBT_TAKER_PCT):
        att, made = adv.get("attempts", 0), adv.get("advances", 0)
        ft_n, ft_o = adv.get("first_to_third_n", 0), adv.get("first_to_third_opps", 0)
        ft = f" — took 3rd on {ft_n}/{ft_o} singles" if ft_o else ""
        traits.append(_trait("extra base taker", "running-game",
                             adv.get("attempt_rate"), adv["xbt_pct"],
                             f"went for the extra base on {att} of "
                             f"{adv.get('opps', 0)} chances "
                             f"({adv['xbt_pct']:.0f}th pct), made it {made} "
                             f"of {att}{ft}"))

    # ── batted-ball direction ──────────────────────────────────────────────
    pull = gm.get("Pull_pct_pct")
    if pull is not None and float(pull) >= PULL_HEAVY_PCT:
        traits.append(_trait("pull-heavy", "batted-ball", pull, pull,
                             f"pull-side BIP rate {pull:.0f}th pct"))
    oppo = gm.get("Oppo_pct_pct")
    if oppo is not None and float(oppo) >= OPPO_BAT_PCT:
        traits.append(_trait("oppo bat", "batted-ball", oppo, oppo,
                             f"opposite-field BIP rate {oppo:.0f}th pct"))
    bat_gb = gm.get("BatGB_pct_pct")
    if bat_gb is not None:
        if float(bat_gb) <= AIR_BALL_GB_PCT:
            traits.append(_trait("air-ball bat", "batted-ball", bat_gb, bat_gb,
                                 f"ground-ball rate {bat_gb:.0f}th pct — "
                                 "lives in the air"))
        elif float(bat_gb) >= GROUND_BALL_GB_PCT:
            traits.append(_trait("ground-ball bat", "batted-ball", bat_gb, bat_gb,
                                 f"ground-ball rate {bat_gb:.0f}th pct — "
                                 "hits it into the ground"))

    # ── platoon structure (from statcast splits) ───────────────────────────
    traits.extend(_platoon_traits(splits))

    # ── fielding (season OAA / arm strength percentiles) ───────────────────
    oaa = (season_metrics or {}).get("OAA_pct")
    if oaa is not None and not (isinstance(oaa, float) and np.isnan(oaa)):
        if float(oaa) >= ELITE_DEFENDER_PCT:
            traits.append(_trait("elite defender", "fielding", oaa, oaa,
                                 f"outs above average {oaa:.0f}th pct"))
        elif float(oaa) >= PLUS_DEFENDER_PCT:
            traits.append(_trait("plus defender", "fielding", oaa, oaa,
                                 f"outs above average {oaa:.0f}th pct"))
        elif float(oaa) <= DEF_LIABILITY_PCT and (pa or 0) >= DEF_LIABILITY_MIN_PA:
            traits.append(_trait("defensive liability", "fielding", oaa, oaa,
                                 f"outs above average {oaa:.0f}th pct"))
    arm = (season_metrics or {}).get("ArmStrength_pct")
    if (arm is not None and not (isinstance(arm, float) and np.isnan(arm))
            and float(arm) >= CANNON_ARM_PCT):
        traits.append(_trait("cannon arm", "fielding", arm, arm,
                             f"arm strength {arm:.0f}th pct"))

    # ── catcher craft (framing / pop time; only for catchers) ──────────────
    traits.extend(_catcher_traits(catching))

    # ── role / versatility ─────────────────────────────────────────────────
    posn = positions or {}
    if (posn.get("positions_played") or 0) >= SUPER_UTILITY_MIN_POSITIONS:
        detail = posn.get("pos_detail") or {}
        shown = " / ".join(f"{p} {v:.0%}" for p, v in list(detail.items())[:5])
        traits.append(_trait("super-utility", "fielding",
                             posn["positions_played"], None,
                             f"{posn['positions_played']} positions with real "
                             f"innings — {shown}"))

    # ── role / availability ────────────────────────────────────────────────
    _pa_szn = (season_metrics or {}).get("PAvol_pct")
    if (_pa_szn is not None and not (isinstance(_pa_szn, float) and np.isnan(_pa_szn))
            and float(_pa_szn) >= EVERYDAY_PA_PCT):
        _pa_t = pa_team if pa_team is not None else pa
        traits.append(_trait("everyday player", "role", _pa_t, _pa_szn,
                             f"{_pa_t or 0:,} PA with the team — {_pa_szn:.0f}th "
                             "pct of the season (in the lineup daily)"))

    # Luck (`lucky` / `unlucky`, wOBA − xwOBA tails) is retired: the gap is a
    # model residual — park, sprint speed, xwOBA's own blind spots — not
    # randomness, and season to season it is mostly noise. The delta stays in
    # the metrics; no tag is cut from it.

    # ── reliability gate ──────────────────────────────────────────────────
    # Last step on purpose: every tag above states its own baseball logic, and
    # this asks the separate question of whether the sample can carry it. PA is
    # the sample for all of them — the stabilization points in reliability.py
    # are expressed in plate appearances, and a hitter's swing, batted-ball and
    # plate-discipline opportunities all scale with PA. Tags with their own
    # purpose-built floor (rarely strikes out, the steal tags, platoon splits)
    # are absent from TAG_EVIDENCE and pass through untouched.
    traits = filter_traits(traits, pa)

    return traits, (float(spectrum) if spectrum is not None else None)


def _catcher_traits(catching: Optional[dict]) -> list[dict]:
    """
    Catcher-craft tags. `catching` holds whichever readings this catcher has
    (framing_taken for anyone who caught an edge pitch; the percentiles only
    where he qualified):
      framing_pct — percentile among qualified catchers of called strikes
                    above expected on edge pitches (expected from location,
                    batter side and count — compute_catcher_framing)
      framing_above — strikes above expected per 100 edge pitches (evidence)
      pop_pct     — pop time percentile among catchers (lower time = better,
                    so the QUICK gate is a LOW percentile)
    """
    out: list[dict] = []
    c = catching or {}

    fr = c.get("framing_pct")
    if fr is not None and not (isinstance(fr, float) and np.isnan(fr)):
        above = c.get("framing_above")
        detail = (f" ({above:+.1f} strikes per 100 edge pitches vs expected)"
                  if above is not None else "")
        if float(fr) >= ELITE_FRAMER_PCT:
            out.append(_trait("elite framer", "catcher", fr, fr,
                              f"strikes above expected {fr:.0f}th pct "
                              f"among catchers{detail}"))
        elif float(fr) <= POOR_FRAMER_PCT:
            out.append(_trait("poor framer", "catcher", fr, fr,
                              f"strikes above expected {fr:.0f}th pct "
                              f"among catchers{detail}"))

    bl = c.get("blocking_pct")
    if bl is not None and not (isinstance(bl, float) and np.isnan(bl)):
        baa = c.get("blocks_above_average")
        npb = c.get("n_pbwp")
        det = ""
        if baa is not None:
            det = f" ({baa:+.0f} blocks above average"
            det += f", {npb:.0f} PB+WP allowed)" if npb is not None else ")"
        if float(bl) >= GOOD_BLOCKER_PCT:
            out.append(_trait("good blocker", "catcher", baa, bl,
                              f"blocking {bl:.0f}th pct among catchers{det}"))
        elif float(bl) <= BAD_BLOCKER_PCT:
            out.append(_trait("bad blocker", "catcher", baa, bl,
                              f"blocking {bl:.0f}th pct among catchers{det}"))

    pop = c.get("pop_pct")
    if (pop is not None and not (isinstance(pop, float) and np.isnan(pop))
            and float(pop) <= QUICK_POP_PCT):
        pop_s = c.get("pop_2b")
        out.append(_trait("quick pop", "catcher", pop_s or pop, pop,
                          (f"{pop_s:.2f}s pop to second — " if pop_s else "")
                          + "among the fastest exchanges in the league"))

    return out


def _platoon_traits(splits: Optional[dict]) -> list[dict]:
    """
    Platoon tags from statcast batter splits:
      left- / right-handed hitter, switch hitter (≥15% of PAs from each side)

    The split tags that used to follow (platoon liability, reverse split) are
    retired — see PLATOON_MIN_PA.
    """
    out: list[dict] = []
    sp = splits or {}
    bats = sp.get("bats")

    # Handedness is an ATTRIBUTE — a fixed fact, not a measurement. On its own
    # it says nothing; as a team density ("this lineup is 42% left-handed") it
    # is one of the more meaningful things a roster can show.
    if bats == "L":
        out.append(_trait("left-handed hitter", "handedness", None, None,
                          "bats left-handed"))
    elif bats == "R":
        out.append(_trait("right-handed hitter", "handedness", None, None,
                          "bats right-handed"))
    elif bats == "S":
        out.append(_trait("switch hitter", "handedness", sp.get("stand_l_share"), None,
                          "takes PAs from both sides of the plate"))
        return out
    return out
