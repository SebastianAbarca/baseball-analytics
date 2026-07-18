"""
hitter_traits.py — Trait tags for hitters: evidence-based, independent,
non-exclusive.

Same philosophy as pitcher_traits.py: no boxes. Each trait is tagged when the
evidence clears a league-percentile gate; a league-average hitter may carry
zero tags. The power–contact spectrum survives as a CONTINUOUS field
(`spectrum`, 0=extreme contact, 100=extreme power) — only the labels that
used to sit on top of it retire.

Collapse rule ("complete"): when a hitter independently earns the power,
on-base, discipline, and hard-contact evidence (with the K% ceiling), the
constituent tags collapse into a single `complete` tag whose evidence
enumerates what it absorbed. Mirrors the modifier-suppression style in
hitter_archetypes.build_hitter_profile.

Families:
    bat          — power bat / contact bat / plus power / gap hitter /
                   hard contact / weak contact
    approach     — walk machine / high-K / elite discipline / free swinger /
                   aggressive
    athleticism  — elite speed / fast / disruptive / chaotic / table setter
    luck         — lucky / unlucky
    outcome      — complete (collapse badge)

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
from hitter_archetypes import (
    compute_spectrum,
    compute_aggressive,
    compute_speed_tier,
    compute_contact_quality,
    compute_plate_discipline,
    compute_table_setter,
    compute_plus_power,
    compute_gap_hitter,
    compute_disruptiveness,
    COMPLETE_K_CEILING,
)

log = logging.getLogger(__name__)

# Spectrum tail gates (absolute 0–100 spectrum score; 50 = balanced).
# The old boxes cut at 45/55; tags mark the tails only, so gate wider.
# The spectrum is a RATIO (power vs contact), so both tails carry an absolute
# floor: a .190 hitter with no contact skill reads as "extreme power" on the
# ratio alone (Maldonado effect), and an elite ISO with elite contact quality
# compresses toward 50 (Yordan effect) — hence the ISO override.
POWER_BAT_SPECTRUM   = 60.0
POWER_BAT_ISO_FLOOR  = 60.0   # ISO pct floor when qualifying via spectrum
POWER_BAT_ISO_ELITE  = 85.0   # elite absolute ISO = power bat regardless of ratio
CONTACT_BAT_SPECTRUM = 40.0
CONTACT_BAT_AVG_FLOOR = 50.0  # must actually hit for the contact-bat tag

# Approach gates (league percentiles)
WALK_MACHINE_PCT = 80.0
HIGH_K_PCT       = 80.0   # K_pct_raw percentile (non-inverted: higher = more Ks)
RARELY_K_PCT     = 10.0   # ≤ → "rarely strikes out" (min PA gate below)
RARELY_K_MIN_PA  = 200    # small samples produce fake 8% K rates
STATION_SPEED_PCT = 10.0  # sprint speed season pct ≤ → "station-to-station"
EVERYDAY_PA_PCT  = 85.0   # PA season pct ≥ → "everyday player"

# Luck tags mark the TAILS of the luck-delta distribution. The old 25/75
# gates tagged half the league — a tag that common is noise, not identity.
# Sign: the source column (est_woba_minus_woba_diff) is empirically
# wOBA − xwOBA, so HIGH percentile = results outran contact = lucky.
LUCKY_TAIL_PCT   = 90.0
UNLUCKY_TAIL_PCT = 10.0

# "complete" constituents: on-base percentile floor
COMPLETE_OBP_PCT = 70.0

# Batted-ball direction gates (spray percentiles are season-ranked upstream)
PULL_HEAVY_PCT      = 85.0
OPPO_BAT_PCT        = 85.0
AIR_BALL_GB_PCT     = 10.0   # batter GB% at/below → lives in the air
GROUND_BALL_GB_PCT  = 90.0   # batter GB% at/above → hits it into the ground

# Platoon gates (raw wOBA gaps from statcast splits; league platoon gap ~.020)
PLATOON_MIN_PA        = 50
PLATOON_LIABILITY_GAP = 0.060   # wOBA vs same-hand this far below vs opp-hand
REVERSE_SPLIT_GAP     = 0.040   # better vs same-hand by this much (unusual)

# Fielding gates (season percentiles of OAA / arm strength)
ELITE_DEFENDER_PCT     = 95.0
PLUS_DEFENDER_PCT      = 85.0
DEF_LIABILITY_PCT      = 10.0
CANNON_ARM_PCT         = 90.0

# Catcher gates (percentiles among qualified catchers)
ELITE_FRAMER_PCT   = 85.0
POOR_FRAMER_PCT    = 15.0
QUICK_POP_PCT      = 15.0   # pop time percentile — LOWER time = better
GOOD_BLOCKER_PCT   = 85.0   # blocks above average percentile
BAD_BLOCKER_PCT    = 15.0

# Baserunning advancement (extra bases taken on hits; base-state derived)
XBT_TAKER_PCT      = 85.0   # xbt_rate percentile among qualified runners
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
) -> tuple[list[dict], Optional[float]]:
    """
    Full trait set + spectrum score for one hitter-season.

    season_metrics — within-season percentile overrides for TAG GATES
    (hybrid normalization: gates fire vs this year's peers so incidence
    doesn't drift with the run environment, while display percentiles and
    the spectrum score stay era-based).

    Returns (traits, spectrum) — spectrum is the continuous 0–100
    power–contact score (None if inputs are missing).
    """
    traits: list[dict] = []

    # Gate metrics: season percentiles where available, era fallback.
    gm = {**metrics, **{k: v for k, v in (season_metrics or {}).items()
                        if v is not None and not (isinstance(v, float) and np.isnan(v))}}

    # ── bat: spectrum tails (with absolute floors — see constants) ────────
    # Spectrum itself stays era-based (continuous display axis); only the
    # ISO/AVG floors use season gates.
    spectrum = compute_spectrum(metrics)
    iso = gm.get("ISO")
    avg = gm.get("AVG")
    power_bat = False
    if iso is not None and float(iso) >= POWER_BAT_ISO_ELITE:
        power_bat = True
        traits.append(_trait("power bat", "bat", iso, iso,
                             f"ISO {iso:.0f}th pct — elite raw power"))
    elif (spectrum is not None and spectrum >= POWER_BAT_SPECTRUM
          and iso is not None and float(iso) >= POWER_BAT_ISO_FLOOR):
        power_bat = True
        traits.append(_trait("power bat", "bat", spectrum, iso,
                             f"power–contact spectrum {spectrum:.0f}/100 with "
                             f"ISO {iso:.0f}th pct — production skews to damage"))
    elif (spectrum is not None and spectrum <= CONTACT_BAT_SPECTRUM
          and (avg is None or float(avg) >= CONTACT_BAT_AVG_FLOOR)):
        traits.append(_trait("contact bat", "bat", spectrum, avg,
                             f"power–contact spectrum {spectrum:.0f}/100 — "
                             "production skews to contact"))

    # ── bat: gap hitter + plus power ───────────────────────────────────────
    # plus power is implied by power bat (suppressed); otherwise the original
    # rule holds: gap hitter (spray-confirmed, more specific) suppresses it.
    gap_hitter = compute_gap_hitter(gm)
    if gap_hitter:
        traits.append(_trait("gap hitter", "bat", gm.get("XB_pct"),
                             gm.get("XB_pct"),
                             "spray-confirmed extra-base production into the gaps"))
    if not power_bat and not gap_hitter and compute_plus_power(gm):
        traits.append(_trait("plus power", "bat", iso, iso,
                             f"ISO {iso:.0f}th pct — real thump"))

    # ── bat: contact quality ───────────────────────────────────────────────
    cq = compute_contact_quality(gm)
    if cq == "Plus Contact":
        vals = [v for v in [gm.get("Barrel_pct"), gm.get("HardHit_pct")]
                if v is not None]
        comp = float(np.mean(vals)) if vals else None
        traits.append(_trait("hard contact", "bat", comp, comp,
                             f"barrel + hard-hit composite {comp:.0f}th pct"))
    elif cq == "Weak Contact":
        vals = [v for v in [gm.get("Barrel_pct"), gm.get("HardHit_pct")]
                if v is not None]
        comp = float(np.mean(vals)) if vals else None
        traits.append(_trait("weak contact", "bat", comp, comp,
                             f"barrel + hard-hit composite {comp:.0f}th pct"))

    # ── approach ───────────────────────────────────────────────────────────
    bb = gm.get("BB_pct")
    if bb is not None and float(bb) >= WALK_MACHINE_PCT:
        traits.append(_trait("walk machine", "approach", bb, bb,
                             f"walk rate {bb:.0f}th pct"))
    k = gm.get("K_pct_raw")
    if k is not None and float(k) >= HIGH_K_PCT:
        traits.append(_trait("high-K", "approach", k, k,
                             f"strikeout rate {k:.0f}th pct (more Ks than "
                             f"{k:.0f}% of this season's hitters)"))
    elif (k is not None and float(k) <= RARELY_K_PCT
            and pa is not None and pa >= RARELY_K_MIN_PA):
        traits.append(_trait("rarely strikes out", "approach", k, k,
                             f"strikeout rate {k:.0f}th pct of the season — "
                             "elite bat-to-ball"))

    disc = compute_plate_discipline(gm)
    if disc == "Elite Discipline":
        # The stronger two-way claim absorbs walk machine (73% co-occurrence —
        # one skill shouldn't wear two chips). Same collapse pattern as
        # `complete`; the walks live on in this tag's evidence.
        _wm = next((t for t in traits if t["tag"] == "walk machine"), None)
        if _wm is not None:
            traits.remove(_wm)
        ev = "walk rate + chase avoidance composite in the top tier"
        if _wm is not None and bb is not None:
            ev += f" (absorbs walk machine — BB {bb:.0f}th pct)"
        traits.append(_trait("elite discipline", "approach", bb, bb, ev))
    elif disc == "Free Swinger":
        traits.append(_trait("free swinger", "approach", bb, bb,
                             "walk rate + chase avoidance composite in the bottom tier"))

    if compute_aggressive(gm):
        fps = gm.get("FPS_pct")
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
    _spd_szn = (season_metrics or {}).get("SprintSpeed")
    if (_spd_szn is not None and not (isinstance(_spd_szn, float) and np.isnan(_spd_szn))
            and float(_spd_szn) <= STATION_SPEED_PCT):
        traits.append(_trait("station-to-station", "athleticism",
                             sprint_speed_raw, _spd_szn,
                             (f"{sprint_speed_raw:.1f} ft/s — " if sprint_speed_raw else "")
                             + "bottom decile of the season's runners"))

    disr = compute_disruptiveness(sb, cs, opportunities, games, season_games,
                                  attempt_rate_pct=attempt_rate_pct)
    if disr.get("modifier") == "Disruptive":
        traits.append(_trait("disruptive", "athleticism", disr.get("sb_pct"), None,
                             f"{sb} SB at {disr['sb_pct']:.0%} success — "
                             "net-positive baserunning chaos"))
    elif disr.get("modifier") == "Chaotic":
        traits.append(_trait("chaotic", "athleticism", disr.get("sb_pct"), None,
                             f"{sb+cs} attempts at {disr['sb_pct']:.0%} success — "
                             "runs into outs"))

    adv = advancement or {}
    if (adv.get("xbt_pct") is not None and adv.get("opps", 0) >= XBT_MIN_OPPS
            and float(adv["xbt_pct"]) >= XBT_TAKER_PCT):
        ft_n, ft_o = adv.get("first_to_third_n", 0), adv.get("first_to_third_opps", 0)
        ft = f" — took 3rd on {ft_n}/{ft_o} singles" if ft_o else ""
        traits.append(_trait("extra base taker", "athleticism",
                             adv.get("xbt_rate"), adv["xbt_pct"],
                             f"{adv.get('advances', 0)} extra bases on "
                             f"{adv.get('opps', 0)} chances "
                             f"({adv['xbt_pct']:.0f}th pct){ft}"))

    if compute_table_setter(gm, sprint_speed_raw):
        obp = gm.get("OBP")
        traits.append(_trait("table setter", "athleticism", obp, obp,
                             f"OBP {obp:.0f}th pct + speed, without power-threat ISO"))

    # ── batted-ball direction ──────────────────────────────────────────────
    pull = gm.get("Pull_pct")
    if pull is not None and float(pull) >= PULL_HEAVY_PCT:
        traits.append(_trait("pull-heavy", "batted-ball", pull, pull,
                             f"pull-side BIP rate {pull:.0f}th pct"))
    oppo = gm.get("Oppo_pct")
    if oppo is not None and float(oppo) >= OPPO_BAT_PCT:
        traits.append(_trait("oppo bat", "batted-ball", oppo, oppo,
                             f"opposite-field BIP rate {oppo:.0f}th pct"))
    bat_gb = gm.get("BatGB_pct")
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
    oaa = (season_metrics or {}).get("OAA")
    if oaa is not None and not (isinstance(oaa, float) and np.isnan(oaa)):
        if float(oaa) >= ELITE_DEFENDER_PCT:
            traits.append(_trait("elite defender", "fielding", oaa, oaa,
                                 f"outs above average {oaa:.0f}th pct"))
        elif float(oaa) >= PLUS_DEFENDER_PCT:
            traits.append(_trait("plus defender", "fielding", oaa, oaa,
                                 f"outs above average {oaa:.0f}th pct"))
        elif float(oaa) <= DEF_LIABILITY_PCT:
            traits.append(_trait("defensive liability", "fielding", oaa, oaa,
                                 f"outs above average {oaa:.0f}th pct"))
    arm = (season_metrics or {}).get("ArmStrength")
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
        traits.append(_trait("super-utility", "role",
                             posn["positions_played"], None,
                             f"{posn['positions_played']} positions with real "
                             f"innings — {shown}"))

    # ── availability ───────────────────────────────────────────────────────
    _pa_szn = (season_metrics or {}).get("PAvol")
    if (_pa_szn is not None and not (isinstance(_pa_szn, float) and np.isnan(_pa_szn))
            and float(_pa_szn) >= EVERYDAY_PA_PCT):
        traits.append(_trait("everyday player", "availability", pa, _pa_szn,
                             f"{pa or 0:,} PA — {_pa_szn:.0f}th pct of the "
                             "season (in the lineup daily)"))

    # ── luck (distribution tails, season percentile ONLY) ──────────────────
    # Era fallback deliberately not used here: the era luck map had a
    # cross-season leakage bug and luck is a season-local phenomenon anyway.
    luck = (season_metrics or {}).get("LuckDelta")
    if luck is not None and not (isinstance(luck, float) and np.isnan(luck)):
        if float(luck) >= LUCKY_TAIL_PCT:
            traits.append(_trait("lucky", "luck", luck, luck,
                                 "wOBA far above xwOBA (top decile of luck delta) — "
                                 "results outran the contact"))
        elif float(luck) <= UNLUCKY_TAIL_PCT:
            traits.append(_trait("unlucky", "luck", luck, luck,
                                 "xwOBA far above wOBA (bottom decile of luck delta) — "
                                 "contact deserved better"))

    # ── collapse: complete ─────────────────────────────────────────────────
    traits = _collapse_complete(traits, gm)

    return traits, (float(spectrum) if spectrum is not None else None)


def _catcher_traits(catching: Optional[dict]) -> list[dict]:
    """
    Catcher-craft tags. `catching` is present only for players who caught a
    qualifying number of taken pitches:
      framing_pct — percentile among qualified catchers of shadow-zone
                    called-strike rate above league expectation
      framing_above — raw pp above league (for evidence)
      pop_pct     — pop time percentile among catchers (lower time = better,
                    so the QUICK gate is a LOW percentile)
    """
    out: list[dict] = []
    c = catching or {}

    fr = c.get("framing_pct")
    if fr is not None and not (isinstance(fr, float) and np.isnan(fr)):
        above = c.get("framing_above")
        detail = (f" ({above:+.1f} pp vs league on edge pitches)"
                  if above is not None else "")
        if float(fr) >= ELITE_FRAMER_PCT:
            out.append(_trait("elite framer", "catcher", fr, fr,
                              f"shadow-zone strike-stealing {fr:.0f}th pct "
                              f"among catchers{detail}"))
        elif float(fr) <= POOR_FRAMER_PCT:
            out.append(_trait("poor framer", "catcher", fr, fr,
                              f"shadow-zone strike rate {fr:.0f}th pct "
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
      switch hitter     — took ≥15% of PAs from each side
      platoon liability — wOBA vs same-hand pitching well below vs opp-hand
      reverse split     — better vs same-hand (unusual; matchup-proof)

    Same/opposite is defined off the batter's side: a RHB's same-hand
    pitcher is a RHP. Switch hitters are excluded from split tags (they
    never face a same-hand disadvantage by design).
    """
    out: list[dict] = []
    sp = splits or {}
    bats = sp.get("bats")

    if bats == "S":
        out.append(_trait("switch hitter", "platoon", sp.get("stand_l_share"), None,
                          "takes PAs from both sides of the plate"))
        return out
    if bats not in ("L", "R"):
        return out

    same_lbl = "vs_lhp" if bats == "L" else "vs_rhp"
    opp_lbl  = "vs_rhp" if bats == "L" else "vs_lhp"
    same, opp = sp.get(same_lbl) or {}, sp.get(opp_lbl) or {}
    if (same.get("pa", 0) < PLATOON_MIN_PA or opp.get("pa", 0) < PLATOON_MIN_PA):
        return out
    w_same, w_opp = same.get("woba"), opp.get("woba")
    if w_same is None or w_opp is None:
        return out

    gap = float(w_opp) - float(w_same)   # positive = normal platoon direction
    if gap >= PLATOON_LIABILITY_GAP:
        out.append(_trait("platoon liability", "platoon", gap, None,
                          f"wOBA {w_same:.3f} vs same-hand, {w_opp:.3f} vs "
                          f"opposite — a {gap:.3f} gap the opponent can exploit"))
    elif -gap >= REVERSE_SPLIT_GAP:
        out.append(_trait("reverse split", "platoon", gap, None,
                          f"wOBA {w_same:.3f} vs same-hand, {w_opp:.3f} vs "
                          "opposite — matchup-proof the wrong way around"))
    return out


def _collapse_complete(traits: list[dict], metrics: dict[str, float]) -> list[dict]:
    """
    When the power + on-base + discipline + hard-contact evidence all fires
    (and the K% ceiling holds), collapse the constituent tags into a single
    `complete` badge that lists what it absorbed.
    """
    tags = {t["tag"] for t in traits}

    has_power      = bool(tags & {"power bat", "plus power"})
    has_contactq   = "hard contact" in tags
    has_discipline = bool(tags & {"elite discipline", "walk machine"})
    obp = metrics.get("OBP")
    has_onbase = obp is not None and float(obp) >= COMPLETE_OBP_PCT

    k_raw = metrics.get("k_rate_raw")
    k_ok = k_raw is None or float(k_raw) <= COMPLETE_K_CEILING

    if not (has_power and has_contactq and has_discipline and has_onbase and k_ok):
        return traits

    absorbed = [t for t in traits
                if t["tag"] in {"power bat", "plus power", "hard contact",
                                "elite discipline", "walk machine"}]
    kept = [t for t in traits if t not in absorbed]
    names = ", ".join(t["tag"] for t in absorbed)
    kept.insert(0, _trait("complete", "outcome", obp, obp,
                          f"does everything — absorbs: {names}; "
                          f"OBP {obp:.0f}th pct, K% under {COMPLETE_K_CEILING:.0%}"))
    return kept
