"""
tag_reference.py — Generate the tag vocabulary reference from the code.

WHY GENERATED
-------------
The vocabulary is spread across five structures (TAG_KIND, TAG_POPULATION,
TAG_EVIDENCE, TAG_EXCLUSIVE_GROUPS, and the `family` each _trait() call
passes) plus the gate constants in hitter_traits/pitcher_traits. The only
human-readable version was the dashboard's Trait Guide, and hand-written docs
drift: that card was still describing `complete`, `elite discipline` and
`gyro-heavy` months after all three were retired, and still defined
`free swinger` as a walk-rate composite after it became chase-rate alone.

So this reads the live maps rather than restating them, and cross-checks them
against what the built portraits actually contain. A tag that exists in code
but never fires, or fires but has no kind, shows up as a discrepancy rather
than as a quiet inconsistency.

USAGE
-----
    python tag_reference.py                 # write TAGS.md
    python tag_reference.py --check         # exit 1 if code and data disagree
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

from team_portrait import (TAG_KIND, TAG_KINDS, TAG_POPULATION, POP_ALL,  # noqa: E402
                           TAG_EXCLUSIVE_GROUPS, kind_of)
from reliability import (TAG_EVIDENCE, STABILIZATION,  # noqa: E402
                         RELIABILITY_FLOOR, UNGATED_TAGS)

_PORTRAITS = _HERE.parent / "data" / "processed" / "portraits"
_OUT = _HERE.parent / "TAGS.md"

_SIDE_GROUP = {"hitters": "H", "starters": "P", "bullpen_arms": "P"}

# Tags that are NOT in the vocabulary — nothing emits them — kept as
# candidates to review. Listed here rather than in TAG_EVIDENCE or TAG_KIND
# because a name there is a promise the portraits keep: the gate map carried
# these three from 2026-08-22, and no code in the repository has ever
# emitted them. Their only effect was that the walk gate keyed `wild` while
# the shipped tag was `walk prone`, which went ungated. --check fails if one
# of these turns up in TAG_KIND (ship it, then delete it here).
#   tag -> (side, claim, evidence it would rest on, notes for the review)
PROSPECTIVE_TAGS: dict[str, tuple[str, str, str, str]] = {
    "wild": (
        "P", "walks far more batters than peers",
        "BB% allowed (BB_pct_pitch, 75 BF floor)",
        "Same claim as the live `walk prone` (walk avoidance ≤ 15th pct). "
        "Only worth keeping as a stronger tier, e.g. ≤ 5th pct."),
    "zone-filler": (
        "P", "throws in the strike zone far more than peers",
        "zone% (Zone_pct, 75 BF floor); Zone_pct_pct is already in every "
        "pitcher's metrics",
        "A behavior, not a result. Zone% vs BB% ρ = −0.44 across 1,838 "
        "pitcher-seasons with 200+ BF (2021–25), so it would not just restate "
        "the command tags; check it against `first-pitch attacker` first."),
    "nibbler": (
        "P", "works the edges — lives out of the zone more than peers",
        "zone% (Zone_pct), low end",
        "The low end of the same axis as `zone-filler`; ship both or neither "
        "and make them an exclusive group."),
}


def scan_portraits() -> tuple[dict, dict, dict, int, dict]:
    """What the tags actually DO, as opposed to what the code says they are:
    family, which side carries them, how often each fires, and one real
    evidence string per tag — the sentence the gate itself wrote when it
    fired, which is the most honest available answer to "how is this earned".
    """
    family: dict[str, str] = {}
    side: dict[str, set] = collections.defaultdict(set)
    count: dict[str, int] = collections.Counter()
    evidence: dict[str, str] = {}
    players = 0
    for f in sorted(glob.glob(str(_PORTRAITS / "*.json"))):
        d = json.load(open(f))
        pl = d.get("players") or {}
        for group, s in _SIDE_GROUP.items():
            for p in pl.get(group) or []:
                players += 1
                for t in p.get("traits") or []:
                    tag = t.get("tag")
                    if not tag:
                        continue
                    count[tag] += 1
                    side[tag].add(s)
                    if t.get("family"):
                        family[tag] = t["family"]
                    ev = t.get("evidence")
                    # Keep the longest example seen: the fuller sentences
                    # spell out the whole gate rather than just the number.
                    if ev and len(ev) > len(evidence.get(tag, "")):
                        evidence[tag] = ev
    return family, dict(side), count, players, evidence


def sample_floor(tag: str, why: bool = False) -> str:
    """The reliability gate in the unit a reader thinks in — PA or BF. With
    `why`, an ungated tag says what covers it instead (TAGS.md only — the
    dashboard footer shows a floor only when there is one)."""
    metric = TAG_EVIDENCE.get(tag)
    if metric is None:
        return f"— ({UNGATED_TAGS[tag]})" if why and tag in UNGATED_TAGS else "—"
    stab = STABILIZATION.get(metric)
    if not stab:
        return metric
    return f"{stab * RELIABILITY_FLOOR:.0f} ({metric})"


def exclusive_of(tag: str) -> str:
    for grp in TAG_EXCLUSIVE_GROUPS:
        if tag in grp:
            others = sorted(grp - {tag})
            return ", ".join(others)
    return "—"


KIND_DEF = {
    "attribute": "Which side a player bats or throws from, and the geometry "
                 "of a pitcher's delivery — arm angle, release extension, "
                 "release width. Fixed characteristics rather than outcomes.",
    "tool": "Physical capacity — what the body can do, measured directly "
            "rather than inferred from outcomes.",
    "behavior": "Swing decisions and where the ball is hit, which pitches a "
                "pitcher throws and how he sequences them, and the choice to "
                "run. What the player does, separate from how it turns out.",
    "result": "What actually came out. The consequence of tools and choices "
              "meeting major-league pitching.",
    "deployment": "How the club used the player — workload, leverage, "
                  "starting or relieving, platoon usage, and how many "
                  "positions they were asked to cover.",
    "noise": "The gap between results and expected outcomes — wOBA measured "
             "against xwOBA. Positive means the results outran the contact "
             "quality; negative means the contact deserved better.",
}


def write_json(family, side, count, evidence) -> Path:
    """Machine-readable twin of TAGS.md, for the dashboard's tag footer.

    The footer used to be hand-written and drifted — it described `complete`,
    `elite discipline` and `gyro-heavy` long after all three were retired.
    Generating it from the same scan means the UI cannot disagree with the
    code about what the vocabulary is.
    """
    out = {"kinds": [], "generated_from": "scripts/tag_reference.py"}
    for kind in TAG_KINDS:
        tags = sorted(t for t in count if kind_of(t) == kind)
        if not tags:
            continue
        out["kinds"].append({
            "kind": kind,
            "definition": KIND_DEF.get(kind, ""),
            "tags": [{
                "tag": t,
                "side": "".join(sorted(side.get(t, ""))),
                "family": family.get(t, ""),
                "fires": count[t],
                "population": TAG_POPULATION.get(t, POP_ALL),
                "floor": sample_floor(t),
                "earned": evidence.get(t, ""),
            } for t in tags],
        })
    path = _HERE.parent / "dashboard" / "tag_reference.json"
    path.write_text(json.dumps(out, indent=1))
    return path


def build(check_only: bool) -> int:
    family, side, count, players, evidence = scan_portraits()
    seen = set(count)
    coded = set(TAG_KIND) | {t for t in seen if kind_of(t)}

    # ── discrepancies: the reason this is generated ──────────────────────
    no_kind = sorted(t for t in seen if kind_of(t) is None)
    never_fires = sorted(t for t in TAG_KIND if t not in seen)
    shipped = sorted(t for t in PROSPECTIVE_TAGS if t in TAG_KIND or t in seen)
    problems = len(no_kind) + len(never_fires) + len(shipped)

    if check_only:
        for t in no_kind:
            print(f"  NO KIND: {t} (fires {count[t]}x)")
        for t in never_fires:
            print(f"  NEVER FIRES: {t} (has a kind, absent from every portrait)")
        for t in shipped:
            print(f"  STILL PROSPECTIVE: {t} is live — remove it from PROSPECTIVE_TAGS")
        print(f"\n{len(seen)} tags in data, {len(TAG_KIND)} keyed in TAG_KIND, "
              f"{problems} discrepancies")
        return 1 if problems else 0

    lines = [
        "# Tag vocabulary",
        "",
        "**Generated by `scripts/tag_reference.py` — do not edit by hand.**",
        "Regenerate after any change to the tag maps or gates.",
        "",
        f"{len(seen)} tags across {players:,} player-seasons in "
        f"{len(glob.glob(str(_PORTRAITS / '*.json')))} portraits.",
        "",
        "## The four axes",
        "",
        "Every tag is described by four things. `kind` is the one the UI "
        "colours by, because it answers what a reader asks first.",
        "",
        "| axis | question | source |",
        "|---|---|---|",
        "| **family** | what is it about? | the `family` each tag is cut with |",
        "| **side** | hitter or pitcher? | which rosters carry it |",
        "| **kind** | who controls it? | `TAG_KIND` |",
        "| **population** | who was eligible to earn it? | `TAG_POPULATION` |",
        "",
        "`kind` runs least to most controllable: **attribute** (nobody chose "
        "it) → **tool** (physical capacity) → **behavior** (a choice) → "
        "**result** (what came out) → **deployment** (what the club did) → "
        "**noise** (luck).",
        "",
    ]

    if problems:
        lines += ["## ⚠ Discrepancies", ""]
        for t in no_kind:
            lines.append(f"- `{t}` fires {count[t]}x but has no kind")
        for t in never_fires:
            lines.append(f"- `{t}` has a kind but never fires")
        for t in shipped:
            lines.append(f"- `{t}` is live but still listed as prospective")
        lines.append("")

    for kind in TAG_KINDS:
        tags = sorted(t for t in seen if kind_of(t) == kind)
        if not tags:
            continue
        total = sum(count[t] for t in tags)
        lines += [
            f"## {kind}  ·  {len(tags)} tags, {total:,} firings",
            "",
            "| tag | side | family | fires | population | reliability floor | exclusive with |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in tags:
            lines.append(
                f"| `{t}` | {''.join(sorted(side.get(t, ''))) or '—'} "
                f"| {family.get(t, '—')} | {count[t]:,} "
                f"| {TAG_POPULATION.get(t, POP_ALL)} | {sample_floor(t, why=True)} "
                f"| {exclusive_of(t)} |"
            )
        lines.append("")

    lines += [
        "## Reading the columns",
        "",
        "- **fires** — occurrences across every stored portrait. A tag firing "
        "0 times, or on nearly everyone, is usually a broken gate rather than "
        "a fact about baseball.",
        "- **population** — the eligible denominator. Team density divides by "
        "this, not by the whole unit, so a catcher tag is not diluted by the "
        "eight players who never crouch.",
        f"- **reliability floor** — plate appearances or batters faced needed "
        f"before the tag may fire, from the metric's stabilization point × "
        f"{RELIABILITY_FLOOR}. `—` means the tag is not gated on PA/BF, and "
        "the note says what covers it instead (`reliability.UNGATED_TAGS`).",
        "- **exclusive with** — tags carving up the same axis. Only one member "
        "can appear in a fingerprint row, and a presence beats an absence.",
        "",
        "## Prospective tags",
        "",
        "Not in the vocabulary: nothing emits these, and no portrait carries "
        "them. Candidates to review, kept in `PROSPECTIVE_TAGS` in "
        "`scripts/tag_reference.py`. Shipping one means adding it to "
        "`TAG_KIND`, its gate to `reliability.TAG_EVIDENCE`, and deleting it "
        "here.",
        "",
        "| tag | side | claim | evidence | notes |",
        "|---|---|---|---|---|",
        *[f"| `{t}` | {sd} | {claim} | {ev} | {note} |"
          for t, (sd, claim, ev, note) in PROSPECTIVE_TAGS.items()],
        "",
    ]
    _OUT.write_text("\n".join(lines))
    jp = write_json(family, side, count, evidence)
    print(f"Wrote {_OUT} — {len(seen)} tags, {problems} discrepancies")
    print(f"Wrote {jp} — dashboard tag footer")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report code/data disagreements, exit 1 if any")
    args = ap.parse_args()
    sys.exit(build(args.check))
