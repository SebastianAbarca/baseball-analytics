"""
charts.py — Plotly figure builders and Dash component helpers.

Most functions return a plotly Figure; philosophy_breakdown_card returns a
Dash html.Div component tree (for use in collapsible breakdown panels).
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import html

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

# Tag kind (attribute / tool / behavior / result / deployment / noise) — the
# vocabulary axis, used here to tell a genuinely empty trait cell from one
# holding only attributes that fire regardless of sample.
from team_portrait import kind_of  # noqa: E402

# ---------------------------------------------------------------------------
# Strike zone geometry
#
# The zone ends AT the plate: 17 inches wide, so ±0.7083 ft. This was ±0.833,
# which is a different quantity wearing the zone's name.
#
# A pitch is a strike if ANY part of the ball passes through the zone, so a
# ball merely TANGENT to the edge still counts — its centre sitting one ball
# RADIUS outside. The locus of strike-producing centres is therefore the zone
# grown by a radius on every side: half-width 8.5" + 1.45" = 9.95", which is
# ±0.8291 ft. Equivalently, the FULL width goes 17" -> 19.9", one whole
# diameter. Halves and fulls do not mix: ±(0.7083 + 2R) would put the ball
# entirely clear of the line, which is not the rule.
#
# Measured, not assumed. On 2,350 ABS-challenge verdicts from 2026 — where the
# final recorded call IS the machine's — a front-plane zone grown by exactly
# one ball radius on all four sides agrees 99.8% of the time. Zero buffer
# scores 56.9%. "Any part outside = ball", i.e. the ball fully inside the line,
# scores 29.6%: the worst of every option tried, in both directions.
#
# Only the zone itself is drawn. Worth knowing when reading the chart, though:
# it plots CENTRES — plate_x/plate_z and every trajectory endpoint are ball
# centres — so a legitimate strike's marker can sit up to 1.45" outside the
# box. A second box at ±0.8291 showing that was tried and removed as clutter.
#
# (The old comment's arithmetic was wrong twice over: 1.44" is the ball's
# radius, not its diameter, and "½ ball radius" gives 9.22", not the 10" the
# constant actually held.)
#
# In-zone FLAGS in modules/pitch_aggregates.py are the same question and
# correctly use the grown box — see _ABS_WIDTH and _FR_W.
# ---------------------------------------------------------------------------
PLATE_HALF_WIDTH = 0.7083  # ft — half the 17" plate; the zone's own edge

# Height fallbacks, used only when strike_zone() has no measured season.
# Named ABS_, but they are not ABS values — they are pre-ABS league averages.
# Real ABS is 53.5% / 27% of the batter's height, which for a 6'1" batter is
# 3.26 / 1.64 ft; the 3.38 below is 55.6% of that height, not 53.5%.
ABS_SZ_TOP    = 3.38   # ft
ABS_SZ_BOT    = 1.59   # ft
# Front of home plate. Statcast measures plate_x/plate_z here and the zone is
# judged here, so trajectories are solved to this plane and the zone must be
# drawn on it — not at y=0, which is the back tip of the plate.
#
# That ABS judges HERE and not at the plate's midpoint is measured, against the
# same 2,350 challenge verdicts: agreement peaks sharply at this plane (99.8%)
# and falls off either side — 97.2% at y=1.30, 98.5% at y=1.50, and only 84.2%
# at the midpoint y=0.7083. Reporting on ABS had suggested the midpoint; the
# data says otherwise, so the crossing marker belongs on this plane.
PLATE_Y       = 1.4167  # ft

# Home plate on the ground, as (x, y) feet: the 17" front edge faces the
# pitcher at y=PLATE_Y, two 8.5" sides run back from it, and two 12" sides
# close on the tip at y=0. The parallel sides sit at ±PLATE_HALF_WIDTH, so the
# zone rectangle rises exactly off the plate's own edges.
PLATE_VERTS = [
    (0.0,               0.0),               # back tip, toward the catcher
    (-PLATE_HALF_WIDTH, PLATE_HALF_WIDTH),
    (-PLATE_HALF_WIDTH, PLATE_Y),
    (PLATE_HALF_WIDTH,  PLATE_Y),
    (PLATE_HALF_WIDTH,  PLATE_HALF_WIDTH),
]
# A real plate is ~1" thick and set flush with the ground, which at this
# scene's scale is invisible. Exaggerated so the slab still reads when the
# catcher's-eye camera looks at it nearly edge-on.
PLATE_THICKNESS = 0.18  # ft


def _home_plate_prism(thickness: float = PLATE_THICKNESS):
    """
    Home plate as a solid slab — (x, y, z) point lists for a Mesh3d.

    The outline is a convex pentagon, so extruding it upward gives a convex
    prism, and Mesh3d can derive the faces itself from the ten corner points
    with alphahull=0 (convex hull). That beats spelling out the twenty-six
    triangles by hand, which is twenty-six chances to transpose an index.
    """
    xs = [v[0] for v in PLATE_VERTS] * 2
    ys = [v[1] for v in PLATE_VERTS] * 2
    zs = [0.0] * len(PLATE_VERTS) + [thickness] * len(PLATE_VERTS)
    return xs, ys, zs


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color to an rgba() string with the given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

COLORS = {
    "primary":    "#1a56db",
    "success":    "#0e9f6e",
    "warning":    "#ff5a1f",
    "neutral":    "#6b7280",
    "background": "#111827",
    "surface":    "#1f2937",
    "text":       "#f9fafb",
    "subtext":    "#9ca3af",
    "border":     "#374151",
    # Dimension accent colors
    "offense":  "#3b82f6",
    "pitching": "#10b981",
    "roster":   "#f59e0b",
    # Archetype colors
    "Complete Hitter":      "#6366f1",
    "Three True Outcomes":  "#ef4444",
    "Power":                "#f97316",
    "Contact":              "#22c55e",
    "Balanced":             "#06b6d4",   # cyan — no strong lean, modifiers define identity
    "Undetermined":         "#6b7280",
    # Pitcher type colors
    "P1": "#ef4444",
    "P2": "#8b5cf6",
    "P3": "#10b981",
    "P4": "#f59e0b",
    "P5": "#06b6d4",   # cyan — Power Contact (distinct from P3 green and P4 amber)
    "P6": "#ec4899",   # pink — Finesse Control (command/location-based)
    "Unclassified": "#6b7280",
}

_DARK_LAYOUT = dict(
    paper_bgcolor=COLORS["background"],
    plot_bgcolor=COLORS["surface"],
    font_color=COLORS["text"],
    margin=dict(l=24, r=24, t=40, b=24),
)


# ---------------------------------------------------------------------------
# 1. Philosophy radar
# ---------------------------------------------------------------------------

PHIL_LABELS = {
    "A1": "TTO",
    "A2": "Contact+Pressure",
    "A3": "Aggressive Early",
    "A4": "Lineup Power",
    "B1": "Stuff Dominant",
    "B2": "Command+Contact Mgmt",
    "B3": "Pitch Design",
    "B4": "Defensive Infra",
    "C1": "bWAR Distribution",
    "C2": "Roster Continuity",
    "C3": "Youth+Dev",
    "C4": "Veteran Experience",
}

DIM_CODES = {
    "offense":  ["A1", "A2", "A3", "A4"],
    "pitching": ["B1", "B2", "B3", "B4"],
    "roster":   ["C1", "C2", "C3", "C4"],
}


def philosophy_radar(portrait: dict, dimension: str = "all") -> go.Figure:
    """
    Spider/radar chart of philosophy scores for one dimension or all 12.

    Args:
        portrait:   team portrait dict
        dimension:  'offense' | 'pitching' | 'roster' | 'all'
    """
    scores = portrait.get("philosophy", {}).get("scores", {})

    if dimension == "all":
        codes = list(PHIL_LABELS.keys())
    else:
        codes = DIM_CODES.get(dimension, list(PHIL_LABELS.keys()))

    if dimension == "all":
        # "All" view: skip spokes with no data to avoid false zeros pulling the shape in
        pairs = [
            (PHIL_LABELS.get(c, c), scores.get(c, {}).get("score"))
            for c in codes
            if scores.get(c, {}).get("score") is not None
        ]
        if not pairs:
            return empty_figure("No philosophy data available")
        labels = [p[0] for p in pairs]
        values = [float(p[1]) for p in pairs]
    else:
        # Per-dimension view: always show all 4 spokes so the shape is stable.
        # Missing spokes are drawn at 0 with "(no data)" label so they're identifiable.
        labels, values = [], []
        for c in codes:
            s = scores.get(c, {}).get("score")
            base_label = PHIL_LABELS.get(c, c)
            labels.append(base_label if s is not None else f"{base_label} (no data)")
            values.append(float(s) if s is not None else 0.0)

    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    dim_color = {
        "offense":  COLORS["offense"],
        "pitching": COLORS["pitching"],
        "roster":   COLORS["roster"],
        "all":      COLORS["primary"],
    }.get(dimension, COLORS["primary"])

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=_hex_to_rgba(dim_color, 0.15),
        line=dict(color=dim_color, width=2),
        marker=dict(size=5, color=dim_color),
        hovertemplate="%{theta}: <b>%{r:.1f}</b><extra></extra>",
    ))

    fig.update_layout(
        **_DARK_LAYOUT,
        polar=dict(
            bgcolor=COLORS["surface"],
            angularaxis=dict(
                linecolor=COLORS["border"],
                gridcolor=COLORS["border"],
                tickfont=dict(size=10, color=COLORS["subtext"]),
            ),
            radialaxis=dict(
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(size=8, color=COLORS["subtext"]),
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
            ),
        ),
        showlegend=False,
        title=dict(
            text=f"{dimension.title()} Philosophy" if dimension != "all" else "Philosophy Profile",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Dimension confidence bars
# ---------------------------------------------------------------------------

def dimension_confidence_bars(portrait: dict) -> go.Figure:
    """
    Horizontal bar chart showing every philosophy dimension score (0–100).
    Grouped by offense / pitching / roster with dimension accent colors.
    League average reference line at 50.
    """
    scores_data = portrait.get("philosophy", {}).get("scores", {})

    # All dimensions in display order
    all_codes = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"]
    dim_color_map = {
        "offense":  COLORS["offense"],
        "pitching": COLORS["pitching"],
        "roster":   COLORS["roster"],
    }

    labels, values, colors, hover, bar_text = [], [], [], [], []
    for code in all_codes:
        s = scores_data.get(code, {})
        score = s.get("score")
        name  = s.get("name") or PHIL_LABELS.get(code, code)
        dim   = s.get("dimension", "offense")
        cov   = s.get("coverage", 0)
        label = s.get("label", "")
        if score is None:
            continue
        labels.append(name)
        values.append(float(score))
        colors.append(dim_color_map.get(dim, COLORS["primary"]))
        hover.append(f"<b>{code} — {name}</b><br>Score: {score:.1f} · {label}<br>Coverage: {cov:.0%}")
        bar_text.append(f"{score:.0f}  {label}")

    if not labels:
        return empty_figure("No philosophy scores available")

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=bar_text,
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))

    # League average reference
    fig.add_shape(
        type="line", x0=50, x1=50, y0=-0.5, y1=len(labels) - 0.5,
        line=dict(color=COLORS["subtext"], width=1, dash="dash"),
    )
    fig.add_annotation(
        x=50, y=len(labels) - 0.3, text="Historical avg",
        showarrow=False, font=dict(color=COLORS["subtext"], size=9),
    )

    layout = {**_DARK_LAYOUT}
    layout["margin"] = dict(l=180, r=60, t=40, b=40)
    fig.update_layout(
        **layout,
        xaxis=dict(range=[0, 115], gridcolor=COLORS["border"],
                   title="Score (0–100, historical percentile)"),
        yaxis=dict(gridcolor=COLORS["border"], autorange="reversed"),
        showlegend=False,
        title=dict(
            text="Philosophy Scores by Dimension",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Hitter archetype distribution
# ---------------------------------------------------------------------------

# Trait family → display color (shared by trait charts and chips)
TRAIT_FAMILY_COLORS = {
    "bat":          "#f97316",  # orange
    "bat-to-ball":  "#fb923c",  # light orange — sits beside bat
    "approach":     "#22c55e",  # green
    "batted-ball":  "#84cc16",  # lime
    "athleticism":  "#06b6d4",  # cyan
    "running-game": "#14b8a6",  # teal
    "fielding":     "#0ea5e9",  # sky
    "catcher":      "#6366f1",  # indigo
    "handedness":   "#94a3b8",  # slate — attributes, deliberately quiet
    "platoon":      "#a855f7",  # violet
    "role":         "#d946ef",  # fuchsia
    "luck":         "#6b7280",  # grey
    "outcome":      "#f59e0b",  # amber
    "mechanics":   "#8b5cf6",   # purple
    "sequencing":  "#06b6d4",   # cyan
    "deception":   "#f59e0b",   # amber
    "arsenal":     "#3b82f6",   # blue
}

# tag → family color, populated lazily as density charts render (they see
# tag+family together); drift charts fall back to a neutral palette for
# tags not yet seen this process.
TRAIT_TAG_COLORS: dict[str, str] = {}


def _pa_trait_density(players: list[dict], weight_key: str = "pa") -> dict[str, tuple[float, str]]:
    """
    Weighted share of PA/BF carried by players holding each trait tag.
    Returns {tag: (share, family)}, sorted descending by share.
    """
    tag_w: dict[str, tuple[float, str]] = {}
    total = 0.0
    for p in players:
        w = float(p.get(weight_key) or 0)
        if w <= 0:
            continue
        total += w
        for t in p.get("traits") or []:
            tag = t.get("tag")
            if tag:
                prev = tag_w.get(tag, (0.0, t.get("family", "")))
                tag_w[tag] = (prev[0] + w, prev[1])
                fam = t.get("family")
                if fam and tag not in TRAIT_TAG_COLORS:
                    c = TRAIT_FAMILY_COLORS.get(fam)
                    if c:
                        TRAIT_TAG_COLORS[tag] = c
    if total <= 0:
        return {}
    return {k: (v / total, fam)
            for k, (v, fam) in sorted(tag_w.items(), key=lambda kv: -kv[1][0])}


def hitter_trait_density(portrait: dict, baseline: dict | None = None) -> go.Figure:
    """
    Horizontal bars: PA-weighted share of the lineup carrying each trait tag,
    over a ghosted league-average bar so every share reads against the norm.
    """
    hitters = portrait.get("players", {}).get("hitters", [])
    return _unit_trait_density_figure(hitters, weight_key="pa",
                                      x_title="% of team PA",
                                      empty_msg="No hitter trait data",
                                      baseline=baseline)


# ---------------------------------------------------------------------------
# 4. Hitter archetype table
# ---------------------------------------------------------------------------

def _kind_groups(traits: list[dict]) -> dict[str, list[dict]]:
    """Split a player's traits by kind, preserving the order they were cut in."""
    out: dict[str, list[dict]] = {}
    for t in traits or []:
        out.setdefault(kind_of(t.get("tag", "")) or "other", []).append(t)
    return out


def _player_tag_rows(traits: list[dict], limited: bool):
    """
    A player's tags, one row per kind.

    Grouped rather than listed flat because a flat list makes the reader do
    the sorting: "elite framer, right-handed hitter, high-leverage arm" mixes
    a skill, an accident of birth and a usage decision in one breath. One row
    per kind puts them in the same frame the fingerprint uses.
    """
    groups = _kind_groups(traits)
    if not groups:
        return [html.Span(
            "limited sample" if limited else "league-average",
            style={"color": COLORS["subtext"], "fontSize": "0.72rem",
                   "fontStyle": "italic", "opacity": "0.75"})]
    rows = []
    for k in KIND_ORDER + ["other"]:
        ts = groups.get(k)
        if not ts:
            continue
        rows.append(html.Div([
            html.Span(k, style={
                "display": "inline-block", "width": "76px",
                "color": COLORS["subtext"], "fontSize": "0.62rem",
                "textTransform": "uppercase", "letterSpacing": "0.05em",
                "verticalAlign": "top", "paddingTop": "2px"}),
            html.Span([_chip(t["tag"], k, t.get("evidence")) for t in ts]),
        ], style={"marginBottom": "2px"}))
    return rows


def _tags_inline(traits: list[dict], limited: bool):
    """
    A player's tags on one line, ordered by kind so the colours block up.

    The kind labels live in the legend rather than on every row: with six
    fixed colours the palette already says which kind a chip is, and
    repeating the word for all 27 hitters costs a screen of vertical space
    to restate what the legend said once.
    """
    groups = _kind_groups(traits)
    if not groups:
        return html.Span(
            "limited sample" if limited else "league-average",
            style={"color": COLORS["subtext"], "fontSize": "0.7rem",
                   "fontStyle": "italic", "opacity": "0.75"})
    out = []
    for k in KIND_ORDER + ["other"]:
        for t in groups.get(k, []):
            out.append(_chip(t["tag"], k, t.get("evidence")))
    return html.Span(out)


ROSTER_PAGE_SIZE = 12

# ---------------------------------------------------------------------------
# Arsenal — the full pitch mix, not the two-pitch summary
# ---------------------------------------------------------------------------
# The portrait stores arsenal_profile["display"], which is hardcoded to
# "{fastball} · {out_pitch}" — exactly two pitches. Spencer Arrighetti throws
# SIX (FF, CU, CH, FC, SI, ST) and displayed as "4-Seam · Curveball". The
# names of the rest live in `supporting`, but the usage percentages were lost:
# `pitch_rows` is a DataFrame and does not survive JSON serialisation.
#
# So the mix is read from the per-season pitch_mix parquet the pipeline
# already writes. No portrait rebuild needed, and it carries velo and whiff
# per pitch as a bonus.
_PITCH_MIX_CACHE: dict[int, dict] = {}

# Grouped by what the pitch DOES, so a reader can see the shape of an arsenal
# without knowing the codes: fastballs warm, breaking balls cool, offspeed
# green. Within a group the colours are close enough to read as one family.
_PITCH_COLORS = {
    "FF": "#f87171", "FA": "#f87171", "FT": "#fb923c", "SI": "#fb923c",
    "FC": "#fbbf24",                                    # fastball family
    "SL": "#60a5fa", "ST": "#38bdf8", "SV": "#818cf8",
    "CU": "#a78bfa", "KC": "#c084fc", "CS": "#c084fc",  # breaking
    "CH": "#34d399", "FS": "#2dd4bf", "FO": "#2dd4bf",
    "SC": "#4ade80", "KN": "#94a3b8", "EP": "#94a3b8",  # offspeed / oddity
}


_STRIKE_ZONE_CACHE: dict[int, tuple[float, float]] = {}


def strike_zone(season: int) -> tuple[float, float]:
    """
    (top, bottom) of the season's average strike zone in feet.

    Statcast records sz_top/sz_bot per pitch, set from the batter's stance, so
    the zone is measured rather than assumed. The ABS constants used before
    were a fixed 3.38/1.59 and neither matched a season nor moved with one:
    the real average is 3.361/1.594 in 2023 and 3.435/1.605 in 2025, drifting
    as the batter population changes.

    Width is NOT computed — the plate is 17 inches by rule, so the half-width
    of 0.833 ft (plate plus a ball radius each side) is the same every year.

    Read from a precomputed table rather than from raw Statcast. This used to
    open statcast_{season}.parquet at render time — a 105 MB file, to produce
    two floats — which meant serving the dashboard required the whole ~977 MB
    of raw pitch data on the web tier for twenty-four numbers. refresh.py
    computes the table where the raw data already lives.
    """
    if season in _STRIKE_ZONE_CACHE:
        return _STRIKE_ZONE_CACHE[season]
    top, bot = ABS_SZ_TOP, ABS_SZ_BOT

    table = _strike_zone_table()
    row = table.get(str(season))
    if row:
        top, bot = float(row["top"]), float(row["bot"])
    else:
        # Development fallback: a season the table has not been built for yet.
        # Never reached on a deployment that ships no raw data, which is the
        # point — it degrades to the ABS constants instead of failing.
        try:
            import pandas as _pd
            path = _HERE.parent / "data" / "raw" / f"statcast_{season}.parquet"
            if path.exists():
                df = _pd.read_parquet(path, columns=["sz_top", "sz_bot"])
                t = _pd.to_numeric(df["sz_top"], errors="coerce").mean()
                b = _pd.to_numeric(df["sz_bot"], errors="coerce").mean()
                if _pd.notna(t) and _pd.notna(b) and 2.0 < t < 5.0 and 0.5 < b < 2.5:
                    top, bot = float(t), float(b)
        except Exception:
            pass

    _STRIKE_ZONE_CACHE[season] = (top, bot)
    return top, bot


_STRIKE_ZONE_TABLE_CACHE: dict | None = None


def _strike_zone_table() -> dict:
    """Season → measured zone, written by scripts/refresh.py. Read once."""
    global _STRIKE_ZONE_TABLE_CACHE
    if _STRIKE_ZONE_TABLE_CACHE is None:
        try:
            import json
            _STRIKE_ZONE_TABLE_CACHE = json.loads(
                (_HERE / "strike_zones.json").read_text())
        except Exception:
            _STRIKE_ZONE_TABLE_CACHE = {}
    return _STRIKE_ZONE_TABLE_CACHE


def _pitch_mix(season: int) -> dict[int, list[dict]]:
    """{pitcher_id: [{pitch, usage, velo, whiff}, ...]} sorted by usage."""
    if season in _PITCH_MIX_CACHE:
        return _PITCH_MIX_CACHE[season]
    out: dict[int, list[dict]] = {}
    try:
        import pandas as _pd
        path = (_HERE.parent / "modules" / "processed"
                / f"pitch_mix_{season}.parquet")
        if path.exists():
            df = _pd.read_parquet(path)
            for pid, grp in df.groupby("pitcher"):
                rows = grp.sort_values("usage_pct", ascending=False)
                out[int(pid)] = [{
                    "pitch": r.pitch_type,
                    "usage": float(r.usage_pct or 0),
                    "velo":  float(r.avg_velo) if _pd.notna(r.avg_velo) else None,
                    "whiff": float(r.whiff_pct) if _pd.notna(r.whiff_pct) else None,
                } for r in rows.itertuples()]
    except Exception:
        out = {}
    _PITCH_MIX_CACHE[season] = out
    return out


_ARSENAL_MIN_USAGE = 0.03   # below this it is a show-me pitch, not a weapon

# The trajectory data keys pitches by their long Statcast name; the roster's
# arsenal column keys by code. One map so both use the same colour per pitch.
# Fixed scene bounds for the 3D trajectory view, in feet. Pinned so the strike
# zone is identical in every selection.
#
# SCENE_X has to hold the widest release in the league, not the widest in the
# selection, or the scene would move again. Across all 33k arsenal entries
# release_x runs to ±4.87 (Donnie Hart, BAL 2018 — a sidearm lefty); 7% clear
# ±3.0 and 2.3% clear ±3.5, so ±4 is the honest floor. Same for height: 5.7%
# of releases clear 6.5 ft.
SCENE_X = 4.0    # half-width; 0.6% of releases fall outside this
SCENE_Y = 56.0   # release (~54 ft) to the plate
SCENE_Z = 7.0    # ground to above the tallest release

# The scene box in the space Plotly's camera lives in. `aspectratio` is applied
# by the renderer as the box's half-extent along each axis, and `camera.eye` is
# measured in those same units — so the camera is only OUTSIDE the box while
# |eye.y| > ASPECT_Y. Getting that wrong is what broke the view before.
#
# ASPECT_X is tied to ASPECT_Z by the ratio of the ranges they cover, so one
# foot across the plate is one foot of height on screen and the strike zone
# renders square rather than as a letterbox.
ASPECT_Z = 1.4
ASPECT_X = ASPECT_Z * (2 * SCENE_X) / SCENE_Z
ASPECT_Y = 3.0   # 56 ft of depth over 3.0 vs 7 ft of height over 1.4 — the
                 # flight path is foreshortened 3.7x so the break stays legible
EYE_GAP  = 1.2   # how far beyond the near face the camera sits. Closer fills
                 # the frame but crowds the box against the scene's edges, and
                 # Plotly silently drops an axis title it cannot fit.

_PITCH_NAME_TO_CODE = {
    "4-Seam Fastball": "FF", "Four-Seam Fastball": "FF", "2-Seam Fastball": "FT",
    "Sinker": "SI", "Cutter": "FC", "Slider": "SL", "Sweeper": "ST",
    "Slurve": "SV", "Curveball": "CU", "Knuckle Curve": "KC",
    "Slow Curve": "CS", "Changeup": "CH", "Split-Finger": "FS",
    "Forkball": "FO", "Screwball": "SC", "Knuckleball": "KN", "Eephus": "EP",
}


def _arsenal_cell(player: dict, season: int):
    """A pitcher's whole mix as usage-ordered chips."""
    mix = _pitch_mix(season).get(int(player.get("player_id") or -1)) or []
    mix = [m for m in mix if m["usage"] >= _ARSENAL_MIN_USAGE]
    if not mix:
        # Fall back to the stored two-pitch summary rather than showing
        # nothing when the season's pitch-mix cache is absent.
        disp = (player.get("arsenal_profile") or {}).get("display")
        return html.Span(disp or "·",
                         style={"color": COLORS["border"], "fontSize": "0.68rem"})
    chips = []
    for m in mix:
        c = _PITCH_COLORS.get(m["pitch"], COLORS["subtext"])
        tip = f"{m['pitch']} — {m['usage']:.0%} of pitches"
        if m["velo"] is not None:
            tip += f", {m['velo']:.1f} mph"
        if m["whiff"] is not None:
            tip += f", {m['whiff']:.0%} whiff"
        chips.append(html.Span(
            f"{m['pitch']} {m['usage']*100:.0f}", title=tip,
            style={
                "display": "inline-block", "backgroundColor": f"{c}22",
                "color": c, "border": f"1px solid {c}55",
                "borderRadius": "3px", "padding": "0px 4px",
                "marginRight": "3px", "marginBottom": "2px",
                "fontSize": "0.62rem", "fontWeight": "700",
                "fontFamily": "ui-monospace, SFMono-Regular, monospace",
                "cursor": "help",
            }))
    return html.Span(chips)


_HAND_MARK = {
    "left-handed hitter": "L", "right-handed hitter": "R", "switch hitter": "S",
    "left-handed pitcher": "L", "right-handed pitcher": "R",
}


def _handedness_mark(attribute_traits: list[dict], is_pitcher: bool):
    """Bats/throws as a quiet marker beside the name.

    Rendered even for pitchers, who also keep the attribute column — there it
    sits alongside arm slot and extension, which is where a reader looking at
    `platoon-vulnerable` wants it.
    """
    for t in attribute_traits:
        mark = _HAND_MARK.get(t.get("tag", ""))
        if mark:
            verb = "Throws" if is_pitcher else "Bats"
            full = {"L": "left", "R": "right", "S": "both sides"}[mark]
            return html.Span(
                mark, title=f"{verb} {full}",
                style={"color": KIND_COLORS["attribute"], "fontSize": "0.6rem",
                       "fontWeight": "700", "marginLeft": "5px",
                       "border": f"1px solid {KIND_COLORS['attribute']}55",
                       "borderRadius": "3px", "padding": "0 3px",
                       "cursor": "help", "verticalAlign": "middle"})
    return None


_HITTER_PLANE_MIN_PA = 100


def hitter_plane(portrait: dict) -> go.Figure:
    """
    The lineup on two ABSOLUTE axes — power against contact — rather than the
    one ratio between them.

    The power-contact spectrum collapses both into a single 0-100 score, and
    a ratio cannot tell "elite at both" from "bad at both": Martin Maldonado
    2023 scores 89 of 100 on a scale labelled power while hitting .191,
    because he has almost no contact skill in the denominator. Measured on
    hitters with 400+ PA, 43% of those above spectrum 80 sit below the 40th
    percentile in xwOBA — nearly half the "power" end is not power at all.

    Two axes keep the magnitude the ratio throws away. Power and contact
    correlate at only +0.19, so the plane genuinely spreads: Arraez lands top
    left, Judge top right, Maldonado bottom left where he belongs rather than
    beside Stanton.

    Results (xwOBA) is the COLOUR, not a third axis — it correlates +0.63 with
    power, so it is largely downstream of the other two rather than
    independent of them. Position is how the hitter is built; colour is how
    well it worked.
    """
    hitters = [h for h in ((portrait.get("players") or {}).get("hitters") or [])
               if (h.get("pa") or 0) >= _HITTER_PLANE_MIN_PA]
    pts = []
    for h in hitters:
        m = (h.get("gate_inputs") or {}).get("metrics") or {}
        power, contact, result = m.get("ISO"), m.get("Contact_pct"), m.get("xwOBA")
        if power is None or contact is None:
            continue
        pts.append((h, float(power), float(contact),
                    float(result) if result is not None else None))
    if not pts:
        return empty_figure("No hitters with enough sample for the plane")

    names = [(p[0].get("name") or "?").split()[-1] for p in pts]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    cs = [p[3] if p[3] is not None else 50 for p in pts]
    pas = [p[0].get("pa") or 0 for p in pts]
    full = [(p[0].get("name") or "?") for p in pts]

    fig = go.Figure()
    # The quadrant split is the whole point of the chart, so it is drawn to be
    # seen. At border grey on a dark ground the crosshair was invisible and
    # the plane read as one undifferentiated cloud.
    _AX = "#6b7280"
    fig.add_hline(y=50, line_color=_AX, line_width=1.5, line_dash="dash")
    fig.add_vline(x=50, line_color=_AX, line_width=1.5, line_dash="dash")
    # Shade the two "one-sided" quadrants faintly so the four regions read as
    # regions rather than as a pair of crossing lines.
    for x0, x1, y0, y1 in ((-4, 50, 50, 104), (50, 104, -4, 50)):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor="#ffffff", opacity=0.022,
                      line_width=0, layer="below")
    for x, y, label in ((2, 101, "CONTACT, NO POWER"), (98, 101, "BOTH"),
                        (2, -1, "NEITHER"), (98, -1, "POWER, NO CONTACT")):
        fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                           xanchor="left" if x < 50 else "right",
                           yanchor="top" if y > 50 else "bottom",
                           font=dict(size=9.5, color=_AX, family="monospace"))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=names,
        textposition="top center",
        textfont=dict(size=9, color=COLORS["subtext"]),
        customdata=list(zip(full, pas, cs)),
        marker=dict(
            size=[max(9, min(26, 9 + (pa / 60))) for pa in pas],
            color=cs, colorscale="RdYlBu", cmin=0, cmax=100,
            line=dict(width=1, color=COLORS["background"]),
            colorbar=dict(
                title=dict(text="xwOBA<br>pct", font=dict(size=10, color=COLORS["subtext"])),
                tickfont=dict(size=9, color=COLORS["subtext"]),
                thickness=10, len=0.7, outlinewidth=0),
        ),
        hovertemplate=("<b>%{customdata[0]}</b><br>"
                       "power (ISO) %{x:.0f}th pct<br>"
                       "contact %{y:.0f}th pct<br>"
                       "results (xwOBA) %{customdata[2]:.0f}th pct<br>"
                       "%{customdata[1]} PA<extra></extra>"),
        showlegend=False,
    ))
    # _DARK_LAYOUT already carries a margin; override it rather than passing
    # the keyword twice.
    layout = {**_DARK_LAYOUT, "margin": dict(l=76, r=10, t=34, b=62)}
    _axis = dict(
        range=[-4, 104],
        # Ticks only at the quarters: the numbers are percentiles, and the
        # reader needs the halfway mark far more than a dense scale.
        tickmode="array", tickvals=[0, 25, 50, 75, 100],
        ticktext=["0", "25", "50", "75", "100"],
        tickfont=dict(size=10, color=COLORS["subtext"]),
        gridcolor="#252d3b", zeroline=False,
        showline=True, linecolor=_AX, linewidth=1.5, mirror=False,
        ticks="outside", tickcolor=_AX, ticklen=4,
    )
    fig.update_layout(
        **layout,
        xaxis={**_axis, "title": dict(
            text="<b>POWER</b>  ·  ISO percentile →",
            font=dict(size=11.5, color=COLORS["text"]))},
        yaxis={**_axis, "title": dict(
            text="<b>CONTACT</b>  ·  contact-rate percentile →",
            font=dict(size=11.5, color=COLORS["text"]))},
    )
    return fig


def _roster_meta(unit: str) -> dict:
    """Per-unit differences: what the rows are, how they sort, what the
    non-tag columns say."""
    if unit == "hitters":
        return {"key": "hitters", "weight": "pa", "weight_label": "PA",
                "slot_label": "Pos", "empty": "No hitter data"}
    if unit == "starters":
        return {"key": "starters", "weight": "bf", "weight_label": "BF",
                "slot_label": "GS", "empty": "No starter data"}
    return {"key": "bullpen_arms", "weight": "bf", "weight_label": "BF",
            "slot_label": "G", "empty": "No bullpen data"}


def roster_table(portrait: dict, unit: str, page: int = 0):
    """
    One roster as an HTML table with a column per KIND.

    HTML rather than Plotly because go.Table can only hold flat strings,
    which is why traits used to read as an undifferentiated comma list. A
    column per kind lets the reader scan DOWN a kind — that a lineup is full
    of `free swinger` is a roster-construction fact, and it was invisible
    when every player's tags were mixed together on one line.

    Paginated rather than scrolled: a scroll box keeps all 27 rows in the DOM
    and hides the row count, while pages make the size of the roster legible
    and keep the card a predictable height.
    """
    meta = _roster_meta(unit)
    players = (portrait.get("players") or {}).get(meta["key"]) or []
    if not players:
        return html.Div(meta["empty"], className="text-secondary p-3"), 0, 0

    players = sorted(players, key=lambda x: -(x.get(meta["weight"]) or 0))
    n_pages = max(1, -(-len(players) // ROSTER_PAGE_SIZE))
    page = max(0, min(page, n_pages - 1))
    shown = players[page * ROSTER_PAGE_SIZE:(page + 1) * ROSTER_PAGE_SIZE]

    def th(label, w=None, align="left", color=None):
        # Kind headers carry the kind's colour, which makes the header row
        # its own legend — no separate swatch strip needed above the card.
        return html.Th(label, style={
            "color": color or COLORS["subtext"], "fontSize": "0.6rem",
            "textTransform": "uppercase", "letterSpacing": "0.04em",
            "fontWeight": "700" if color else "600",
            "textAlign": align, "padding": "4px 6px",
            "borderBottom": f"1px solid {COLORS['border']}",
            # Headers must not break mid-word ("DEPLOYME/NT", "NOIS/E").
            "whiteSpace": "nowrap", "overflow": "hidden",
            **({"width": w} if w else {}),
        })

    def td(child, align="left", color=None, size="0.74rem", weight="400"):
        return html.Td(child, style={
            "padding": "5px 6px", "textAlign": align, "fontSize": size,
            "color": color or COLORS["text"], "fontWeight": weight,
            "borderBottom": f"1px solid {COLORS['border']}44",
            "verticalAlign": "top",
        })

    # Width follows how much each kind actually holds: behavior and result
    # carry 32 and 29 of the 94 tags, attribute is usually one chip, and noise
    # is at most one. Pitchers give up some of it to the arsenal column.
    is_pitcher = unit in ("starters", "bullpen_arms")
    # Handedness moves next to the name and drops out of the kind columns.
    # Measured across 18,483 player-seasons, the attribute cell held ONLY
    # handedness on 81% of rows — a full column spending ~11% of the table's
    # width to repeat one letter. It is identifying information, not a
    # finding. Pitchers keep the column because delivery geometry (arm slot,
    # extension, release width) genuinely varies there; for hitters the kind
    # is folded away entirely and its width goes to behavior and result,
    # which are the crowded ones.
    if is_pitcher:
        shown_kinds = KIND_ORDER
        kind_w = {"attribute": "9%", "tool": "8%", "behavior": "16%",
                  "result": "16%", "deployment": "10%", "noise": "4%"}
    else:
        shown_kinds = [k for k in KIND_ORDER if k != "attribute"]
        kind_w = {"tool": "11%", "behavior": "24%",
                  "result": "24%", "deployment": "13%", "noise": "8%"}
    season = int(portrait.get("season") or 0)

    body = []
    for p in shown:
        limited = bool(p.get("limited_sample"))
        name = (p.get("name") or "").strip() or f"ID {p.get('player_id', '?')}"
        war = p.get("war")
        dim = COLORS["subtext"] if limited else COLORS["text"]
        groups = _kind_groups(p.get("traits") or [])
        slot = (p.get("home_position") if unit == "hitters"
                else p.get("gs") if unit == "starters" else p.get("g"))
        # Bats/throws as a marker on the name — the fact a reader wants while
        # looking at a platoon tag, without a column of its own.
        hand = _handedness_mark(groups.get("attribute") or [], is_pitcher)
        cells = [
            td(html.Span([
                html.Span(name, style={"color": dim, "fontWeight": "700"}),
                hand,
            ]), size="0.78rem"),
            td(str(slot) if slot not in (None, "") else "—",
               align="center", color=COLORS["subtext"]),
            td(str(p.get(meta["weight"]) or "—"), align="right", color=dim),
            td(f"{war:+.1f}" if war is not None else "—", align="right"),
        ]
        if is_pitcher:
            cells.append(td(_arsenal_cell(p, season)))
        for k in shown_kinds:
            ts = groups.get(k) or []
            cells.append(td(
                html.Span([_chip(t["tag"], k, t.get("evidence")) for t in ts])
                if ts else
                # An empty cell is a real statement here — nothing of this
                # kind separates the player — so it gets a mark rather than
                # blank space the eye reads as missing data.
                html.Span("·", style={"color": COLORS["border"]})))
        body.append(html.Tr(cells, style={"opacity": "0.7" if limited else "1"}))

    table = html.Table([
        html.Thead(html.Tr(
            [th("Player", "12%" if is_pitcher else "13%"),
             th(meta["slot_label"], "4%", "center"),
             th(meta["weight_label"], "5%", "right"), th("bWAR", "5%", "right")]
            + ([th("Arsenal", "16%")] if is_pitcher else [])
            + [th(k, kind_w[k], color=KIND_COLORS[k]) for k in shown_kinds]
        )),
        html.Tbody(body),
    ], style={"width": "100%", "borderCollapse": "collapse",
              "tableLayout": "fixed"})
    return table, page, n_pages


def hitter_roster_card(portrait: dict):
    """Back-compat single-page render (no pager)."""
    table, _, _ = roster_table(portrait, "hitters", 0)
    return table


def hitter_archetype_table(portrait: dict) -> go.Figure:
    """
    Table: one row per hitter with spectrum score and trait tags.
    """
    hitters = portrait.get("players", {}).get("hitters", [])

    if not hitters:
        fig = go.Figure(go.Table(
            header=dict(values=["No hitter data"], fill_color=COLORS["surface"]),
            cells=dict(values=[[]], fill_color=COLORS["background"]),
        ))
        fig.update_layout(**_DARK_LAYOUT)
        return fig

    names, poss, ages, pas, wars, specs, traits_col = [], [], [], [], [], [], []
    dim = []   # per-row: is this profile too thin to lean on?

    for h in sorted(hitters, key=lambda x: -(x.get("pa") or 0)):
        name = (h.get("name") or "").strip() or f"ID {h.get('player_id', '?')}"
        war  = h.get("war")
        limited = bool(h.get("limited_sample"))
        dim.append(limited)

        names.append(name)
        poss.append(h.get("home_position") or "—")
        ages.append(str(h.get("age")) if h.get("age") else "—")
        pas.append(str(h.get("pa") or "—"))
        wars.append(f"{war:.1f}" if war is not None else "—")

        # Spectrum: 0 = extreme contact, 100 = extreme power
        spec = h.get("spectrum")
        specs.append(f"{spec:.0f}" if spec is not None else "—")

        # An empty trait cell used to mean two opposite things. In this
        # vocabulary carrying no tag is a real finding — the player is
        # league-average, which is why the middle of every distribution is
        # deliberately untagged. But a player with 20 PA also carries no tags,
        # because the reliability gate refused to make a claim. Both rendered
        # as "—", so the reader could not tell "unremarkable" from "unknown".
        #
        # "Empty" has to mean no SUBSTANTIVE tag, not no tag at all. Every
        # hitter carries a handedness attribute, and attributes are ungated by
        # reliability because a sample size cannot make someone bat left. So a
        # pitcher with 12 PA still showed "right-handed hitter" and never fell
        # through to either state — the distinction was unreachable.
        tags = [t["tag"] for t in h.get("traits") or []]
        substantive = [t for t in tags if kind_of(t) != "attribute"]
        if substantive:
            traits_col.append(", ".join(tags))
        elif limited:
            traits_col.append("limited sample")
        else:
            traits_col.append("league-average")

    n = len(names)
    row_colors = [COLORS["surface"] if i % 2 == 0 else COLORS["background"] for i in range(n)]
    # Thin profiles recede rather than disappear: they are still roster facts,
    # they just should not read with the same weight as a regular's.
    name_colors = [COLORS["subtext"] if d else COLORS["text"] for d in dim]
    trait_colors = [COLORS["subtext"] if (d or t in ("limited sample", "league-average"))
                    else COLORS["text"] for d, t in zip(dim, traits_col)]

    fig = go.Figure(go.Table(
        columnwidth=[3, 0.8, 1, 1, 1, 1.2, 4],
        header=dict(
            values=["<b>Player</b>", "<b>Pos</b>", "<b>Age</b>", "<b>PA</b>",
                    "<b>bWAR</b>", "<b>Spectrum</b>", "<b>Traits</b>"],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["subtext"], size=11),
            align=["left", "center", "center", "center", "center", "center", "left"],
            line_color=COLORS["border"],
            height=32,
        ),
        cells=dict(
            values=[names, poss, ages, pas, wars, specs, traits_col],
            fill_color=[row_colors] * 7,
            font=dict(
                color=[
                    name_colors,                # dimmed when sample is thin
                    [COLORS["subtext"]] * n,    # position
                    [COLORS["subtext"]] * n,    # age
                    name_colors,                # PA — the reason for the dim
                    [COLORS["text"]] * n,       # bWAR
                    [COLORS["text"]] * n,       # spectrum
                    trait_colors,
                ],
                size=11,
            ),
            align=["left", "center", "center", "center", "center", "center", "left"],
            line_color=COLORS["border"],
            height=28,
        ),
    ))
    tbl_layout = {**_DARK_LAYOUT, "margin": dict(l=0, r=0, t=40, b=0)}
    fig.update_layout(
        **tbl_layout,
        title=dict(
            text="Hitter Roster Detail",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 4b. Hitter archetype affinity heatmap
# ---------------------------------------------------------------------------

# Composite archetype affinities — computed from metrics_pct stored per player.
# Each entry: (display label, [positive contributors], [negative contributors])
# Score = mean of positives + mean of (100 - negatives), then averaged.
# All values are 0–100 percentiles; negative contributors are inverted.
_ARCHETYPE_AFFINITIES: list[tuple[str, list[str], list[str]]] = [
    (
        "Power",
        ["Power", "Hard Hit", "Barrel Rate"],
        [],
    ),
    (
        "TTO",
        ["Power", "Walk Rate"],
        ["Contact"],           # TTO loves Ks → invert Contact (Contact = inv K%)
    ),
    (
        "Contact",
        ["Contact", "Contact Rate", "Contact-First"],
        [],
    ),
    (
        "Discipline",
        ["Walk Rate", "Zone Swing"],
        ["Aggression"],        # disciplined hitters don't chase first pitches
    ),
    (
        "Speed",
        ["Speed"],
        [],
    ),
    (
        "Quality",
        ["Quality"],           # xwOBA percentile — overall offensive value
        [],
    ),
]


def _archetype_affinity_score(metrics: dict, positives: list[str], negatives: list[str]) -> float | None:
    """Average of positive contributors + inverted negatives. Returns None if no data."""
    vals = []
    for k in positives:
        v = metrics.get(k)
        if v is not None:
            vals.append(float(v))
    for k in negatives:
        v = metrics.get(k)
        if v is not None:
            vals.append(100.0 - float(v))
    return round(sum(vals) / len(vals), 1) if vals else None


def hitter_archetype_heatmap(portrait: dict) -> go.Figure:
    """
    Full-roster heatmap: players (Y) × archetype affinities (X).
    Color encodes 0–100 score. Sorted by PA descending.
    """
    hitters = portrait.get("players", {}).get("hitters", [])
    hitters_sorted = sorted(hitters, key=lambda h: -(h.get("pa") or 0))

    affinity_labels = [a[0] for a in _ARCHETYPE_AFFINITIES]
    player_labels   = []
    z_matrix        = []   # rows = players, cols = affinities

    for h in hitters_sorted:
        name    = (h.get("name") or f"ID {h.get('player_id', '?')}").strip()
        pa      = h.get("pa") or 0
        metrics = h.get("metrics_pct", {})

        row = []
        for _, positives, negatives in _ARCHETYPE_AFFINITIES:
            score = _archetype_affinity_score(metrics, positives, negatives)
            row.append(score)

        # Only include players with at least one affinity score
        if any(v is not None for v in row):
            player_labels.append(f"{name}  ({pa} PA)")
            z_matrix.append([v if v is not None else 0 for v in row])

    if not z_matrix:
        return empty_figure("No player metric data — load a portrait with Statcast coverage")

    # Hover: show player + archetype + score
    hover_text = []
    for i, h in enumerate([h for h in hitters_sorted
                            if any(v is not None for v in [
                                _archetype_affinity_score(h.get("metrics_pct", {}), pos, neg)
                                for _, pos, neg in _ARCHETYPE_AFFINITIES
                            ])]):
        metrics = h.get("metrics_pct", {})
        row_hover = []
        for label, positives, negatives in _ARCHETYPE_AFFINITIES:
            score = _archetype_affinity_score(metrics, positives, negatives)
            score_str = f"{score:.0f}" if score is not None else "—"
            parts = positives + [f"inv({n})" for n in negatives]
            row_hover.append(
                f"<b>{label}</b>: {score_str}<br>"
                f"<span style='font-size:10px;color:#9ca3af'>from: {', '.join(parts)}</span>"
            )
        hover_text.append(row_hover)

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=affinity_labels,
        y=player_labels,
        zmin=0,
        zmax=100,
        colorscale=[
            [0.0,  "#1a2233"],   # dark background for 0
            [0.3,  "#1e3a5f"],   # low blue
            [0.5,  "#1a56db"],   # mid primary blue
            [0.75, "#22c55e"],   # high green
            [1.0,  "#f59e0b"],   # top amber/gold
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Score", font=dict(color=COLORS["subtext"], size=11)),
            tickfont=dict(color=COLORS["subtext"], size=10),
            thickness=12,
            len=0.8,
        ),
        text=[[f"{v:.0f}" for v in row] for row in z_matrix],
        texttemplate="%{text}",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
    ))

    # Mark primary archetype per player with a right-side annotation
    height_px = max(300, 32 * len(player_labels) + 60)

    layout = {**_DARK_LAYOUT, "margin": dict(l=160, r=80, t=50, b=40)}
    fig.update_layout(
        **layout,
        height=height_px,
        xaxis=dict(
            side="top",
            tickfont=dict(size=11, color=COLORS["text"]),
            gridcolor=COLORS["border"],
        ),
        yaxis=dict(
            tickfont=dict(size=10, color=COLORS["text"]),
            autorange="reversed",
            gridcolor=COLORS["border"],
        ),
        title=dict(
            text="Hitter Skill Affinity Scores  (0–100, historical percentile)",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 4c. Interactive player metrics radar
# ---------------------------------------------------------------------------

# Metric display order and group labels for the player radar
_PLAYER_METRIC_GROUPS = {
    "Power":        "Offense",
    "Barrel Rate":  "Offense",
    "Hard Hit":     "Offense",
    "Quality":      "Offense",
    "Walk Rate":    "Patience",
    "Contact":      "Contact",
    "Contact Rate": "Contact",
    "Contact-First":"Contact",
    "Aggression":   "Approach",
    "Zone Swing":   "Approach",
    "Speed":        "Speed",
}

_PLAYER_METRIC_ORDER = list(_PLAYER_METRIC_GROUPS.keys())

# Map archetype type name → color
_ARCHETYPE_COLORS = [
    "#3b82f6",  # blue
    "#ef4444",  # red
    "#10b981",  # green
    "#f59e0b",  # amber
    "#8b5cf6",  # purple
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#f97316",  # orange
]


def player_options_from_portrait(portrait: dict) -> list[dict]:
    """
    Return list of {label, value} dicts for the player-selection dropdown.
    Sorted by PA descending; value is player_id (int as str).
    """
    hitters = portrait.get("players", {}).get("hitters", [])
    options = []
    for h in sorted(hitters, key=lambda x: -(x.get("pa") or 0)):
        pid  = h.get("player_id")
        name = (h.get("name") or "").strip() or f"ID {pid}"
        pa   = h.get("pa") or 0
        arc  = next((t["tag"] for t in h.get("traits") or []), "")
        if pid is not None:
            options.append({
                "label": f"{name}  ({arc + ', ' if arc else ''}{pa} PA)",
                "value": str(pid),
            })
    return options


def player_metrics_radar(portrait: dict, selected_ids: list[str]) -> go.Figure:
    """
    Polar radar showing each selected player's percentile ranks across
    key batting metrics. One trace per player; zero-data axes are hidden.
    """
    hitters = portrait.get("players", {}).get("hitters", [])
    id_to_hitter = {str(h.get("player_id")): h for h in hitters if h.get("player_id") is not None}

    if not selected_ids:
        return empty_figure("Select players from the dropdown above to compare their metrics")

    traces = []
    for idx, pid in enumerate(selected_ids):
        h = id_to_hitter.get(pid)
        if h is None:
            continue
        metrics = h.get("metrics_pct", {})
        name = (h.get("name") or f"ID {pid}").strip()
        arc  = next((t["tag"] for t in h.get("traits") or []), "")

        # Collect values in metric display order; skip entirely absent metrics
        labels, values = [], []
        for metric in _PLAYER_METRIC_ORDER:
            val = metrics.get(metric)
            if val is not None:
                labels.append(metric)
                values.append(float(val))

        if not labels:
            continue

        color = _ARCHETYPE_COLORS[idx % len(_ARCHETYPE_COLORS)]
        labels_closed = labels + [labels[0]]
        values_closed = values + [values[0]]

        traces.append(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.12),
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            name=f"{name} ({arc})",
            hovertemplate="<b>%{theta}</b><br>Percentile: <b>%{r:.0f}</b><extra></extra>",
        ))

    if not traces:
        return empty_figure("No metric data for selected players")

    fig = go.Figure(traces)
    fig.update_layout(
        **_DARK_LAYOUT,
        polar=dict(
            bgcolor=COLORS["surface"],
            angularaxis=dict(
                linecolor=COLORS["border"],
                gridcolor=COLORS["border"],
                tickfont=dict(size=10, color=COLORS["subtext"]),
            ),
            radialaxis=dict(
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(size=8, color=COLORS["subtext"]),
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(color=COLORS["subtext"], size=11),
            orientation="h",
            yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
        ),
        title=dict(
            text="Player Metrics Comparison (Historical Percentile)",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


def player_metrics_bars(portrait: dict, selected_ids: list[str]) -> go.Figure:
    """
    Horizontal grouped bars comparing selected players across key metrics.
    Each group = one metric, bars = players.  Complementary view to the radar.
    """
    hitters = portrait.get("players", {}).get("hitters", [])
    id_to_hitter = {str(h.get("player_id")): h for h in hitters if h.get("player_id") is not None}

    if not selected_ids:
        return empty_figure("Select players from the dropdown above")

    player_data: list[tuple[str, str, dict]] = []
    for pid in selected_ids:
        h = id_to_hitter.get(pid)
        if h is None:
            continue
        name = (h.get("name") or f"ID {pid}").strip()
        arc  = next((t["tag"] for t in h.get("traits") or []), "")
        player_data.append((pid, f"{name} ({arc})", h.get("metrics_pct", {})))

    if not player_data:
        return empty_figure("No metric data for selected players")

    # Collect all metrics present in at least one selected player
    all_metrics = []
    for m in _PLAYER_METRIC_ORDER:
        if any(m in metrics_dict for _, _, metrics_dict in player_data):
            all_metrics.append(m)

    traces = []
    for idx, (pid, label, metrics) in enumerate(player_data):
        color = _ARCHETYPE_COLORS[idx % len(_ARCHETYPE_COLORS)]
        vals  = [metrics.get(m) for m in all_metrics]
        traces.append(go.Bar(
            name=label,
            y=all_metrics,
            x=vals,
            orientation="h",
            marker_color=color,
            hovertemplate="<b>%{y}</b><br>%{x:.0f}th percentile<extra></extra>",
        ))

    fig = go.Figure(traces)
    fig.add_shape(
        type="line", x0=50, x1=50, y0=-0.5, y1=len(all_metrics) - 0.5,
        line=dict(color=COLORS["subtext"], width=1, dash="dash"),
    )
    layout = {**_DARK_LAYOUT, "margin": dict(l=120, r=40, t=40, b=24)}
    fig.update_layout(
        **layout,
        barmode="group",
        xaxis=dict(range=[0, 105], gridcolor=COLORS["border"],
                   title="Percentile (0–100)"),
        yaxis=dict(gridcolor=COLORS["border"], autorange="reversed"),
        showlegend=True,
        legend=dict(font=dict(color=COLORS["subtext"], size=11)),
        title=dict(
            text="Player Metrics Comparison",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Starter archetype bars
# ---------------------------------------------------------------------------

def starter_archetype_bars(portrait: dict) -> go.Figure:
    """
    Table: one row per starter with name, age, BF, bWAR, arsenal display,
    approach, ace badge, and trait tags. Trait-based — no archetype boxes.
    """
    starters = portrait.get("players", {}).get("starters", [])

    if not starters:
        return empty_figure("No starter data — load a full-season portrait")

    # Approach → short display label
    _APPROACH_SHORT = {
        "fastball-first":    "FB-first",
        "secondary-led":     "2nd-led",
        "balanced":          "Balanced",
        "tunnel-dependent":  "Tunnel",
        "knuckleball":       "Knuckleball",
    }

    names, ages, bfs, wars = [], [], [], []
    arsenals, approaches, traits_col = [], [], []

    for s in sorted(starters, key=lambda x: -(x.get("bf") or 0)):
        name = (s.get("name") or "").strip() or f"ID {s.get('player_id', '?')}"
        war  = s.get("war")
        ap   = s.get("arsenal_profile") or {}
        traits = s.get("traits") or []

        names.append(name)
        ages.append(str(s.get("age")) if s.get("age") else "—")
        bfs.append(str(s.get("bf") or "—"))
        wars.append(f"{war:.1f}" if war is not None else "—")

        arsenals.append(ap.get("display") or "—")
        approach_raw = ap.get("approach", "")
        approaches.append(_APPROACH_SHORT.get(approach_raw, approach_raw or "—"))

        # Arsenal facts already have their own column
        shown = [t["tag"] for t in traits if t.get("family") != "arsenal"]
        # See hitter_archetype_table: an empty cell has to say WHICH kind of
        # empty it is — nothing distinctive, or not enough to judge — and
        # "empty" means no SUBSTANTIVE tag, since handedness and arm slot are
        # attributes that fire regardless of sample.
        if [t for t in shown if kind_of(t) != "attribute"]:
            traits_col.append(", ".join(shown))
        else:
            traits_col.append("limited sample" if s.get("limited_sample")
                              else "league-average")

    n = len(names)
    row_colors = [COLORS["surface"] if i % 2 == 0 else COLORS["background"] for i in range(n)]

    fig = go.Figure(go.Table(
        columnwidth=[3, 1, 1, 1, 2.5, 1.5, 4],
        header=dict(
            values=[
                "<b>Pitcher</b>", "<b>Age</b>", "<b>BF</b>", "<b>bWAR</b>",
                "<b>Arsenal</b>", "<b>Approach</b>", "<b>Traits</b>",
            ],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["subtext"], size=11),
            align=["left", "center", "center", "center", "left", "center", "left"],
            line_color=COLORS["border"],
            height=32,
        ),
        cells=dict(
            values=[names, ages, bfs, wars, arsenals, approaches, traits_col],
            fill_color=[row_colors] * 7,
            font=dict(
                color=[
                    [COLORS["text"]] * n,   # names
                    [COLORS["subtext"]] * n, # ages
                    [COLORS["subtext"]] * n, # bfs
                    [COLORS["text"]] * n,    # wars
                    [COLORS["text"]] * n,    # arsenals
                    [COLORS["subtext"]] * n, # approaches
                    [COLORS["text"]] * n,    # traits
                ],
                size=11,
            ),
            align=["left", "center", "center", "center", "left", "center", "left"],
            line_color=COLORS["border"],
            height=28,
        ),
    ))

    tbl_layout = {**_DARK_LAYOUT, "margin": dict(l=0, r=0, t=40, b=0)}
    fig.update_layout(
        **tbl_layout,
        title=dict(
            text="Starter Roster Detail",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 5b. Bullpen detail table
# ---------------------------------------------------------------------------

def bullpen_detail_table(portrait: dict) -> go.Figure:
    """
    Table: one row per reliever sorted by BF desc.
    Shows name, age, BF, velocity%, K%, whiff%, GB%, hard-contact-supp%, bWAR.
    """
    arms = portrait.get("players", {}).get("bullpen_arms", [])

    if not arms:
        return empty_figure("No bullpen data — load a full-season portrait")

    def _fmt(val, decimals=0):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        return f"{val:.{decimals}f}"

    rows = sorted(arms, key=lambda x: -(x.get("bf") or 0))

    names, ages, bfs, wars = [], [], [], []
    velos, ks, whiffs, gbs, hcs, traits_col = [], [], [], [], [], []

    for arm in rows:
        m = arm.get("metrics_pct", {})
        traits = arm.get("traits") or []
        name = (arm.get("name") or "").strip() or f"ID {arm.get('player_id', '?')}"
        names.append(name)
        ages.append(str(arm.get("age")) if arm.get("age") else "—")
        bfs.append(str(arm.get("bf") or "—"))
        wars.append(_fmt(arm.get("war"), 1))
        velos.append(_fmt(m.get("avg_velo_pct")))
        ks.append(_fmt(m.get("K_pct_pct")))
        whiffs.append(_fmt(m.get("SwStr_pct_pct")))
        gbs.append(_fmt(m.get("GB_pct_pct")))
        hcs.append(_fmt(m.get("HardHit_allowed_pct")))

        shown = [t["tag"] for t in traits]
        if [t for t in shown if kind_of(t) != "attribute"]:
            traits_col.append(", ".join(shown))
        else:
            traits_col.append("limited sample" if arm.get("limited_sample")
                              else "league-average")

    n = len(rows)
    row_colors = [COLORS["surface"] if i % 2 == 0 else COLORS["background"] for i in range(n)]

    fig = go.Figure(go.Table(
        columnwidth=[2.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.8, 1, 4],
        header=dict(
            values=[
                "<b>Reliever</b>", "<b>Age</b>", "<b>BF</b>", "<b>bWAR</b>",
                "<b>Velo%</b>", "<b>K%</b>", "<b>Whiff%</b>",
                "<b>GB%</b>", "<b>HC Supp%</b>", "<b>Traits</b>",
            ],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["subtext"], size=11),
            align=["left", "center", "center", "center",
                   "center", "center", "center", "center", "center",
                   "left"],
            line_color=COLORS["border"],
            height=32,
        ),
        cells=dict(
            values=[names, ages, bfs, wars, velos, ks, whiffs, gbs, hcs,
                    traits_col],
            fill_color=[row_colors] * 10,
            font=dict(
                color=[[COLORS["text"]] * n] * 10,
                size=11,
            ),
            align=["left", "center", "center", "center",
                   "center", "center", "center", "center", "center",
                   "left"],
            line_color=COLORS["border"],
            height=26,
        ),
    ))

    h = min(400, 60 + n * 26)
    tbl_layout = {**_DARK_LAYOUT, "margin": dict(l=0, r=0, t=40, b=0), "height": h}
    fig.update_layout(
        **tbl_layout,
        title=dict(
            text="Bullpen Roster Detail  <sup style='font-size:10px;color:#6b7280'>"
                 "percentiles vs. historical pool | HC Supp = hard-contact suppression</sup>",
            font=dict(size=13, color=COLORS["text"]),
            x=0.02, xanchor="left",
        ),
    )
    return fig


def _unit_trait_density_figure(arms: list, weight_key: str,
                               x_title: str, empty_msg: str,
                               baseline: dict | None = None) -> go.Figure:
    """
    Shared horizontal-bar builder for the identity-tab density trio.

    baseline — {tag: league mean density 0–1}. Rendered as a ghosted bar
    behind the team's solid bar so every share reads against the league norm.
    """
    density = _pa_trait_density(arms, weight_key=weight_key)
    if not density:
        return empty_figure(empty_msg)

    if baseline:
        # Order by deviation from league: what most DEFINES this team at the
        # top, what it most lacks at the bottom (top 13 over + 5 most under).
        ranked = sorted(density.keys(),
                        key=lambda t: density[t][0] - float(baseline.get(t, 0) or 0),
                        reverse=True)
        tags = ranked[:13] + (ranked[-5:] if len(ranked) > 18 else ranked[13:])
    else:
        tags = list(density.keys())[:18]
    shares = [density[t][0] * 100 for t in tags]
    colors = kind_shaded_colors(tags)
    # Which kinds are actually on this chart, in canonical order — the legend
    # below is built from these so it never advertises a colour that is absent.
    kinds_present = [k for k in KIND_ORDER
                     if any((kind_of(t) or "noise") == k for t in tags)]

    fig = go.Figure()
    xmax = max(shares)

    if baseline:
        lg = [100 * float(baseline.get(t, 0) or 0) for t in tags]
        xmax = max(xmax, max(lg) if lg else 0)
        fig.add_trace(go.Bar(
            x=lg, y=tags, orientation="h",
            name="League average",
            width=0.78,
            marker_color="#9ca3af",
            opacity=0.28,
            hovertemplate="<b>%{y}</b><br>league average %{x:.1f}%<extra></extra>",
        ))

    fig.add_trace(go.Bar(
        x=shares, y=tags, orientation="h",
        name="This team",
        width=0.5,
        marker_color=colors,
        text=[f"{s:.0f}%" for s in shares],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["subtext"]),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}" + f"{x_title[1:]}<extra></extra>",
    ))

    # Kind legend: one zero-width entry per kind actually on the chart, so the
    # colours are named rather than left to be inferred. Bars are shaded within
    # a kind, so the swatch shows that kind's base hue.
    for k in kinds_present:
        fig.add_trace(go.Bar(
            # A real y category rather than None: a fully-empty trace still
            # claims a legend slot in Plotly, which is how the legend ended up
            # advertising kinds that were not on the chart.
            x=[0], y=[tags[0]], orientation="h", name=k,
            marker_color=KIND_COLORS[k], showlegend=True, hoverinfo="skip",
            width=0.001, opacity=0,
        ))

    fig.update_layout(
        **{**_DARK_LAYOUT, "margin": dict(l=118, r=42, t=44, b=30)},
        barmode="overlay",
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.10,
                    font=dict(size=10, color=COLORS["subtext"]),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title=x_title, range=[0, xmax * 1.25],
                   gridcolor=COLORS["border"],
                   tickfont=dict(color=COLORS["subtext"]),
                   title_font=dict(color=COLORS["subtext"], size=11)),
        yaxis=dict(autorange="reversed",
                   tickfont=dict(color=COLORS["text"], size=10)),
    )
    return fig


_DRIFT_MIN_SEASONS = 3
_DRIFT_MIN_DENSITY = 0.02

# Hard cap on simultaneous lines. Beyond roughly eight categories no palette
# stays distinguishable, so the limit is enforced in the selector rather than
# papered over with more hues — the reader swaps a tag out to bring one in.
DRIFT_MAX_TAGS = 8

# Drift lines do NOT use the kind palette. On the rosters and the density bars
# colour carries meaning and repetition is fine, because position and labels
# still identify the row. On a line chart the colour IS the identity: two
# behaviour tags in near-identical cyan are unfollowable where they cross. So
# drift gets eight maximally separated hues, assigned by slot, and the same
# eight in the same order on all three unit charts — the third line is the
# same colour on offense, rotation and bullpen.
#
# Hues run roughly evenly around the wheel (0/30/50/130/165/210/265/330) at
# lightness that holds up on the #111827 ground.
DRIFT_LINE_COLORS = [
    "#ff6b6b",  # coral
    "#ffa94d",  # orange
    "#ffd43b",  # yellow
    "#69db7c",  # green
    "#38d9a9",  # teal
    "#4dabf7",  # blue
    "#b197fc",  # purple
    "#f783ac",  # pink
]


def drift_deviations(team: str, unit: str,
                     seasons_data: dict[int, dict]) -> dict[str, dict]:
    """tag -> {year: (deviation_pp, density_pp)} for one team and unit."""
    dev: dict[str, dict[int, tuple[float, float]]] = {}
    for yr, league in sorted(seasons_data.items()):
        ident = (league.get(team) or {}).get(unit) or {}
        density = ident.get("trait_density") or {}
        base = (league.get("_baselines") or {}).get(unit) or {}
        for tag in set(density) | set(base):
            d = float(density.get(tag, 0) or 0)
            b = float(base.get(tag, 0) or 0)
            if d < _DRIFT_MIN_DENSITY and b < _DRIFT_MIN_DENSITY:
                continue   # micro-tags: noise, not identity
            dev.setdefault(tag, {})[yr] = ((d - b) * 100, d * 100)
    return dev


def drift_tag_choices(team: str, unit: str, seasons_data: dict[int, dict],
                      kind: str | None = None) -> list[tuple[str, float]]:
    """
    Tags this team has a real multi-season history for, most-defining first.

    Ranked by the largest deviation the tag ever reached, which is what the
    chart used to pick with silently. Exposing the ranking is the point: the
    old chart drew its own top 6 and gave the reader no way to see why those
    six, or to ask a different question.
    """
    dev = drift_deviations(team, unit, seasons_data)
    out = []
    for tag, series in dev.items():
        if len(series) < _DRIFT_MIN_SEASONS:
            continue
        if kind and kind != "all" and kind_of(tag) != kind:
            continue
        out.append((tag, max(abs(v[0]) for v in series.values())))
    return sorted(out, key=lambda kv: -kv[1])


def team_drift_chart(team: str, unit: str,
                     seasons_data: dict[int, dict],
                     tags: list[str] | None = None) -> go.Figure:
    """
    "Through the Years" — one unit's identity drift across seasons.

    seasons_data — {year: league_identity dict for that season}. For each
    season we compute deviation = team density − league baseline per tag,
    then draw the team's most-defining tags (largest |deviation| anywhere
    in the window) as lines against the league zero-line.
    """
    dev = drift_deviations(team, unit, seasons_data)
    if not dev:
        return empty_figure("No multi-season data for this team")

    if tags is None:
        # Unfiltered fallback: the team's most-defining tags.
        ranked = [t for t, _ in drift_tag_choices(team, unit, seasons_data)][:6]
    else:
        ranked = [t for t in tags if t in dev]
    if not ranked:
        return empty_figure("No tags selected")

    years = sorted(seasons_data.keys())
    fig = go.Figure()
    fig.add_hline(y=0, line_color=COLORS["subtext"], line_width=1,
                  line_dash="dot",
                  annotation_text="league average",
                  annotation_font=dict(size=10, color=COLORS["subtext"]),
                  annotation_position="bottom right")

    # Slot colour, not kind colour — see DRIFT_LINE_COLORS. Every line on the
    # chart is a different hue so crossings stay followable.
    for i, tag in enumerate(ranked):
        color = DRIFT_LINE_COLORS[i % len(DRIFT_LINE_COLORS)]
        xs = [yr for yr in years if yr in dev[tag]]
        ys = [dev[tag][yr][0] for yr in xs]
        dens = [dev[tag][yr][1] for yr in xs]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=tag,
            line=dict(color=color, width=2.4),
            marker=dict(size=6),
            connectgaps=False,
            customdata=dens,
            hovertemplate=(f"<b>{tag}</b> · %{{x}}<br>"
                           "%{y:+.0f} pts vs league · %{customdata:.0f}% of unit"
                           "<extra></extra>"),
        ))

    fig.update_layout(
        **{**_DARK_LAYOUT, "margin": dict(l=50, r=20, t=28, b=36)},
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.12,
                    font=dict(size=11, color=COLORS["text"]),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(tickmode="array", tickvals=years,
                   tickfont=dict(color=COLORS["subtext"]),
                   gridcolor=COLORS["border"]),
        yaxis=dict(title="pts vs league",
                   tickfont=dict(color=COLORS["subtext"]),
                   title_font=dict(color=COLORS["subtext"], size=11),
                   gridcolor=COLORS["border"], zeroline=False),
        hovermode="x unified",
    )
    return fig


def rotation_trait_density(portrait: dict, baseline: dict | None = None) -> go.Figure:
    """BF-weighted trait density for the rotation (identity-tab trio)."""
    starters = portrait.get("players", {}).get("starters", [])
    return _unit_trait_density_figure(starters, weight_key="bf",
                                      x_title="% of rotation BF",
                                      empty_msg="No rotation trait data",
                                      baseline=baseline)


def bullpen_trait_density(portrait: dict, baseline: dict | None = None) -> go.Figure:
    """BF-weighted trait density for the bullpen (identity-tab trio)."""
    arms = portrait.get("players", {}).get("bullpen_arms", [])
    return _unit_trait_density_figure(arms, weight_key="bf",
                                      x_title="% of bullpen BF",
                                      empty_msg="No bullpen trait data",
                                      baseline=baseline)


# ---------------------------------------------------------------------------
# 5c. Bullpen dimension bars (collective profile)
# ---------------------------------------------------------------------------

def bullpen_dimension_bars(portrait: dict) -> go.Figure:
    """
    Horizontal bars showing the team's collective bullpen scores on each
    available dimension vs. the league.
    """
    # Team-level collective (top-level since schema 14; players-nested before)
    bp = portrait.get("bullpen_collective") \
        or portrait.get("players", {}).get("bullpen_profile", {})

    if not bp or not bp.get("scores"):
        return empty_figure("No bullpen profile — load a full-season portrait")

    scores = bp.get("scores", {})
    out_mechanism    = bp.get("out_mechanism", "")
    lev_structure    = bp.get("leverage_structure", "")

    DIM_LABELS = {
        "velocity":          "Velocity Tier",
        "attack_philosophy": "Zone Attack Rate",
        "damage_prevention": "Damage Prevention",
        "K_out_mechanism":   "K% Out Mechanism",
        "GB_out_mechanism":  "GB% Out Mechanism",
        "platoon_balance":   "Platoon Balance",
        "leverage_structure":"Leverage Structure",
    }

    # Only show dimensions we actually have scores for
    labels, vals, colors_bar = [], [], []
    for dim, label in DIM_LABELS.items():
        v = scores.get(dim)
        if v is not None:
            labels.append(label)
            vals.append(round(float(v), 1))
            if v >= 70:
                colors_bar.append("#22c55e")   # green
            elif v >= 50:
                colors_bar.append("#1a56db")   # blue
            elif v >= 35:
                colors_bar.append("#f59e0b")   # amber
            else:
                colors_bar.append("#6b7280")   # grey

    if not labels:
        return empty_figure("Bullpen dimension scores not available")

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker_color=colors_bar,
        text=[f"{v:.0f}" for v in vals],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="%{y}: %{x:.0f}<extra></extra>",
        cliponaxis=False,
    ))

    # League-average reference line
    fig.add_vline(x=50, line_dash="dash", line_color=COLORS["border"], line_width=1)

    # Subtitle annotation with out mechanism + leverage label
    subtitle_parts = []
    if out_mechanism:
        subtitle_parts.append(f"Out mechanism: <b>{out_mechanism}</b>")
    if lev_structure:
        subtitle_parts.append(f"Leverage: <b>{lev_structure}</b>")
    subtitle = "  ·  ".join(subtitle_parts) if subtitle_parts else ""

    _layout = {**_DARK_LAYOUT, "margin": dict(l=8, r=60, t=54, b=8)}
    fig.update_layout(
        **_layout,
        height=max(240, 60 + len(labels) * 38),
        xaxis=dict(
            range=[0, 110],
            showgrid=False, zeroline=False,
            tickvals=[], showticklabels=False,
            color=COLORS["subtext"],
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color=COLORS["text"], size=11),
            showgrid=False,
        ),
        title=dict(
            text=f"Bullpen Collective Profile"
                 + (f"<br><sup style='color:#9ca3af;font-size:10px'>{subtitle}</sup>" if subtitle else ""),
            font=dict(size=13, color=COLORS["text"]),
            x=0.02, xanchor="left",
        ),
        bargap=0.35,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Roster control / service-time breakdown
# ---------------------------------------------------------------------------

def roster_control_chart(portrait: dict) -> go.Figure:
    """
    Horizontal stacked bar showing the three service-time tiers:
      Pre-Arb (< 3 yrs EST)  |  Arb-Eligible (3-6 yrs)  |  FA-Eligible (≥ 6 yrs)

    Below the bar: player counts and a one-line identity label
    ("Rebuilding", "Window", "Veteran-Heavy", "Transitioning").

    Data lives in portrait["team_metrics"]["roster_control"] as raw fractions
    populated by build_team_portrait() from the cross-team cache.
    """
    rc = (portrait.get("team_metrics") or {}).get("roster_control") or {}

    pre_arb     = rc.get("pre_arb")
    arb         = rc.get("arb")
    fa_eligible = rc.get("fa_eligible")
    avg_tenure  = rc.get("avg_tenure")

    if pre_arb is None and arb is None and fa_eligible is None:
        return empty_figure("Roster control data unavailable — rebuild portrait")

    # Fill any single missing tier from the others (shares should sum to ~1)
    shares = {"pre_arb": pre_arb, "arb": arb, "fa_eligible": fa_eligible}
    known  = {k: v for k, v in shares.items() if v is not None}
    if len(known) == 2:
        missing_key = next(k for k in shares if k not in known)
        shares[missing_key] = max(0.0, 1.0 - sum(known.values()))
    elif len(known) < 2:
        return empty_figure("Insufficient roster control data")

    pre_arb     = shares["pre_arb"]
    arb         = shares["arb"]
    fa_eligible = shares["fa_eligible"]

    # Total player count from hitter + starter lists (approx roster size for counts)
    hitters  = portrait.get("players", {}).get("hitters", [])
    starters = portrait.get("players", {}).get("starters", [])
    arms     = portrait.get("players", {}).get("bullpen_arms", [])
    n_total  = len(hitters) + len(starters) + len(arms)
    if n_total == 0:
        n_total = 26  # MLB active roster default

    n_pre = round(pre_arb * n_total)
    n_arb = round(arb * n_total)
    n_fa  = round(fa_eligible * n_total)

    # Identity label based on dominant tier and shape
    if pre_arb >= 0.45:
        identity = "Rebuilding"
        identity_color = "#60a5fa"   # blue — future-oriented
    elif fa_eligible >= 0.50:
        identity = "Veteran-Heavy"
        identity_color = "#f87171"   # red — aging/declining risk
    elif arb >= 0.45:
        identity = "Window"
        identity_color = "#34d399"   # green — prime controlled years
    elif pre_arb >= 0.30 and fa_eligible >= 0.30:
        identity = "Transitioning"
        identity_color = COLORS["warning"]
    else:
        identity = "Balanced"
        identity_color = COLORS["subtext"]

    # Tier colors
    COL_PRE = "#60a5fa"   # blue  — youth/cheap
    COL_ARB = "#34d399"   # green — window/controlled prime
    COL_FA  = "#f87171"   # red   — veteran/expensive

    # Labels with counts
    label_pre = f"Pre-Arb<br><b>{n_pre}</b> players"
    label_arb = f"Arb-Eligible<br><b>{n_arb}</b> players"
    label_fa  = f"FA-Eligible<br><b>{n_fa}</b> players"

    fig = go.Figure()

    # Stacked horizontal bars — one trace per tier
    for pct, label, color, customdata in [
        (pre_arb * 100,     label_pre, COL_PRE, f"{pre_arb*100:.0f}% Pre-Arb · {n_pre} players"),
        (arb * 100,         label_arb, COL_ARB, f"{arb*100:.0f}% Arb-Eligible · {n_arb} players"),
        (fa_eligible * 100, label_fa,  COL_FA,  f"{fa_eligible*100:.0f}% FA-Eligible · {n_fa} players"),
    ]:
        fig.add_trace(go.Bar(
            x=[pct],
            y=["Roster"],
            orientation="h",
            marker_color=color,
            text=f"{pct:.0f}%" if pct >= 10 else "",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=12, color="#111827"),
            customdata=[customdata],
            hovertemplate="%{customdata}<extra></extra>",
            name=label,
            showlegend=True,
        ))

    # Average tenure annotation
    tenure_text = f"Avg service time: <b>{avg_tenure:.1f} yrs</b>" if avg_tenure else ""

    layout = {
        **_DARK_LAYOUT,
        "barmode": "stack",
        "height": 180,
        "margin": dict(l=16, r=16, t=52, b=16),
        "xaxis": dict(
            range=[0, 100],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        "yaxis": dict(
            showticklabels=False,
            showgrid=False,
        ),
        "legend": dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.08,
            font=dict(size=10, color=COLORS["subtext"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        "plot_bgcolor":  COLORS["surface"],
        "paper_bgcolor": COLORS["background"],
        "title": dict(
            text=(
                f"Roster Control  ·  "
                f"<span style='color:{identity_color}'><b>{identity}</b></span>"
                + (f"  ·  {tenure_text}" if tenure_text else "")
            ),
            font=dict(size=13, color=COLORS["text"]),
            x=0.5,
        ),
        "annotations": [],
    }

    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 6. Park factor gauge
# ---------------------------------------------------------------------------

def park_factor_gauge(portrait: dict) -> go.Figure:
    """
    Gauge showing pitcher-friendliness (0 = hitter-friendly, 100 = pitcher-friendly).
    """
    park = portrait.get("park") or {}
    pf   = park.get("pitcher_friendly")
    woba_pf = park.get("woba_pf")

    val = float(pf) if pf is not None else 50.0

    subtitle = f"wOBA PF: {woba_pf:.3f}" if woba_pf is not None else ""

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val,
        delta=dict(reference=50, valueformat=".0f"),
        title=dict(text=f"Park — Pitcher Friendly<br><sub>{subtitle}</sub>",
                   font=dict(size=13, color=COLORS["text"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color=COLORS["subtext"])),
            bar=dict(color=COLORS["pitching"]),
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            steps=[
                dict(range=[0,  40], color=_hex_to_rgba("#ef4444", 0.2)),
                dict(range=[40, 60], color=COLORS["surface"]),
                dict(range=[60, 100], color=_hex_to_rgba("#10b981", 0.2)),
            ],
            threshold=dict(
                line=dict(color=COLORS["subtext"], width=2),
                thickness=0.75,
                value=50,
            ),
        ),
        number=dict(font=dict(color=COLORS["text"])),
    ))
    fig.update_layout(**{**_DARK_LAYOUT, "height": 300, "margin": dict(l=24, r=24, t=60, b=24)})
    return fig


# ---------------------------------------------------------------------------
# 7. Team batting metrics bar
# ---------------------------------------------------------------------------

def team_batting_bars(portrait: dict) -> go.Figure:
    """
    Horizontal bar chart of key PA-weighted batting percentile metrics.
    Each bar is 0–100, dashed line at 50 (league average).
    """
    batting_agg = portrait.get("team_metrics", {}).get("batting", {})

    DISPLAY: list[tuple[str, str]] = [
        ("K_pct_pct",       "K%"),
        ("BB_pct_pct",      "BB%"),
        ("HardHit_pct_pct", "Hard Hit%"),
        ("xwOBA_pct",       "xwOBA"),
        ("Barrel_pct_pct",  "Barrel%"),
        ("SwStr_pct_pct",   "SwStr%"),
        ("FPS_pct_pct",     "FPS%"),
        ("OSwing_pct_pct",  "O-Swing%"),
        ("Contact_pct_pct", "Contact%"),
        ("sprint_speed_pct","Sprint Speed"),
    ]

    labels, values, colors = [], [], []
    for key, label in DISPLAY:
        v = batting_agg.get(key)
        if v is None:
            continue
        labels.append(label)
        values.append(float(v))
        colors.append(COLORS["offense"] if float(v) >= 50 else COLORS["warning"])

    if not labels:
        fig = go.Figure()
        fig.add_annotation(
            text="No batting aggregate data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=COLORS["subtext"], size=13),
        )
        fig.update_layout(**_DARK_LAYOUT)
        return fig

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
        hovertemplate="%{y}: <b>%{x:.1f}</b> pct<extra></extra>",
    ))
    # League average reference line
    fig.add_shape(
        type="line", x0=50, x1=50, y0=-0.5, y1=len(labels) - 0.5,
        line=dict(color=COLORS["subtext"], width=1, dash="dash"),
    )
    fig.update_layout(
        **_DARK_LAYOUT,
        xaxis=dict(range=[0, 110], title="Percentile vs. league", gridcolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"]),
        title=dict(
            text="Team Batting Profile (percentile ranks)",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Spin efficiency summary bar
# ---------------------------------------------------------------------------

def spin_efficiency_bar(portrait: dict) -> go.Figure:
    """
    Single bar showing team mean weighted spin efficiency vs. league.
    """
    spin = portrait.get("spin") or {}
    pct  = spin.get("weighted_spin_efficiency_pct")
    raw  = spin.get("mean_spin_efficiency")
    n    = spin.get("pitcher_count")

    if pct is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Spin efficiency not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=COLORS["subtext"], size=13),
        )
        fig.update_layout(**_DARK_LAYOUT)
        return fig

    label = f"Mean SE (n={n})" if n else "Mean Spin Efficiency"
    color = COLORS["pitching"] if pct >= 50 else COLORS["warning"]

    fig = go.Figure(go.Bar(
        x=[float(pct)],
        y=[label],
        orientation="h",
        marker_color=color,
        text=[f"{pct:.0f}th pct  |  raw={raw:.3f}" if raw else f"{pct:.0f}th pct"],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig.add_shape(
        type="line", x0=50, x1=50, y0=-0.5, y1=0.5,
        line=dict(color=COLORS["subtext"], width=1, dash="dash"),
    )
    fig.update_layout(
        **_DARK_LAYOUT,
        xaxis=dict(range=[0, 110], title="Percentile vs. league", gridcolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"]),
        height=180,
        title=dict(
            text="Team Spin Efficiency",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Philosophy metric breakdown (Dash component, not Plotly figure)
# ---------------------------------------------------------------------------

METRIC_LABELS: dict[str, str] = {
    # A1
    "ISO_pct":               "Isolated Power",
    "BB_pct_pct":            "Walk Rate",
    "K_pct_pct":             "Strikeout Rate",
    "HR_FB_pct":             "HR/FB Ratio",
    "Sprint_inv_pct":        "Sprint Speed (low = TTO)",
    # A2
    "K_inv_pct":             "Contact Rate (inv. K%)",
    "OBP_SLG_gap_pct":       "OBP–SLG Gap",
    "PitchesPerPA_pct":      "Pitches per PA",
    "SprintSpeed_pct":       "Sprint Speed",
    "Contact_pct_pct":       "Contact Rate",
    # A3
    "FPS_pct_pct":           "First-Pitch Strike Swing%",
    "PitchesPerPA_inv_pct":  "Pitches per PA (inv.)",
    "ZSwing_pct_pct":        "Zone Swing%",
    "BB_inv_pct":            "Low Walk Rate (inv. BB%)",
    # A4
    "TeamHR_pct":            "Team HR Total",
    "TeamSLG_pct":           "Team SLG",
    "PowerContributors_pct": "Power Contributors (PA share)",
    "TeamBarrel_pct":        "Team Barrel Rate",
    # B1
    "TeamK_pct_pct":         "Team Strikeout Rate",
    "AvgVelo_pct":           "Avg Fastball Velocity",
    "SwStr_pct_pct":         "Swinging Strike Rate",
    "AvgSpinRate_pct":       "Avg Spin Rate",
    # B2
    "BB_pitch_inv_pct":      "Walk Rate (inv.)",
    "Zone_pct_pct":          "Zone Rate",
    "GB_pct_pct":            "Ground Ball Rate",
    "TeamDefense_pct":       "Team Defense (OAA + Def Runs)",
    # B3
    "SpinEfficiency_pct":    "Spin Efficiency",
    "ArsenalDiversity_pct":  "Arsenal Diversity",
    "PlatoonOptimization_pct": "Platoon Optimization",
    "OpenerUsage_pct":       "Opener Usage",
    "CSW_pct_pct":           "Called Strike + Whiff%",
    # B4
    "OAA_pct":               "Outs Above Average",
    "DRS_pct":               "Def. Runs Saved (BRef)",
    "GB_pitch_pct":          "Ground Ball Rate (pitch.)",
    "ParkPitcherFriendly_pct": "Park Pitcher-Friendliness",
    # C1
    "WAR_concentration_pct": "bWAR Concentration",
    "WAR_variance_inv_pct":  "bWAR Variance (inv.)",
    "RosterFloor_pct":       "Roster Floor (bWAR > 0)",
    # C2
    "CoreRetention_pct":     "Core Player Retention",
    # C3
    "AvgTenure_inv_pct":     "Avg Est. Service Time (inv.)",
    "PreArbShare_pct":       "Pre-Arb Share (EST < 3.0 yrs)",
    # C4
    "AvgTenure_pct":         "Avg Est. Service Time",
    "FAShare_pct":           "Free Agent Share (EST ≥ 6.0 yrs)",
    # Additional EST metrics (informational)
    "ArbShare_pct":          "Arb-Eligible Share (3.0 ≤ EST < 6.0)",
}

METRIC_MISSING_REASON: dict[str, str] = {
    # C2 — requires prior-season Statcast parquet; will be missing for seasons
    # where statcast_{season-1}.parquet has not been cached locally.
    "CoreRetention_pct": "Requires prior-season Statcast cache (statcast_{season-1}.parquet)",
}


def philosophy_breakdown_card(portrait: dict, code: str) -> html.Div:
    """
    Dash component tree for one philosophy dimension's metric breakdown table.
    Shows each metric's percentile, weight, and whether it is present or missing.
    """
    from philosophy import PHILOSOPHY_DEFS

    defn     = PHILOSOPHY_DEFS.get(code, {})
    weights  = defn.get("weights", {})
    metrics  = portrait.get("philosophy_metrics", {})
    score_data = portrait.get("philosophy", {}).get("scores", {}).get(code, {})
    score    = score_data.get("score")
    coverage = score_data.get("coverage", 0)
    score_str = f"{score:.0f}" if score is not None else "N/A"

    rows = []
    for metric_key, weight in weights.items():
        label = METRIC_LABELS.get(metric_key, metric_key)
        value = metrics.get(metric_key)

        if value is None:
            reason   = METRIC_MISSING_REASON.get(metric_key, "Data not available")
            pct_cell = html.Td("—", className="text-secondary text-center",
                               style={"fontSize": "0.85rem"})
            status_cell = html.Td(
                html.Small(reason, className="text-warning"),
                style={"fontSize": "0.75rem"},
            )
        else:
            bar_w = f"{max(0, min(100, int(value)))}%"
            pct_cell = html.Td(
                html.Div([
                    html.Span(f"{value:.0f}", className="me-2 fw-semibold",
                              style={"minWidth": "28px", "display": "inline-block",
                                     "textAlign": "right"}),
                    html.Div(
                        html.Div(style={"width": bar_w, "height": "6px",
                                        "backgroundColor": COLORS["primary"],
                                        "borderRadius": "3px"}),
                        style={"width": "72px", "backgroundColor": COLORS["border"],
                               "borderRadius": "3px", "display": "inline-block",
                               "verticalAlign": "middle"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
            )
            status_cell = html.Td(
                html.Small("Present", className="text-success"),
                style={"fontSize": "0.75rem"},
            )

        rows.append(html.Tr([
            html.Td(label, className="text-white", style={"fontSize": "0.85rem"}),
            pct_cell,
            html.Td(f"{weight:.0%}", className="text-secondary text-center",
                    style={"fontSize": "0.85rem"}),
            status_cell,
        ]))

    # Label badge
    label_str   = score_data.get("label", "No Data")
    label_color = {
        "Defining": "success",
        "Strong":   "primary",
        "Notable":  "info",
        "Moderate": "secondary",
        "Low":      "dark",
    }.get(label_str, "secondary")

    import dash_bootstrap_components as dbc
    header_style = {"color": COLORS["subtext"], "fontSize": "0.72rem",
                    "textTransform": "uppercase", "letterSpacing": "0.06em",
                    "fontWeight": "600"}
    return html.Div([
        # Identity label at the top
        html.Div([
            dbc.Badge(label_str, color=label_color,
                      className="me-2 fs-6",
                      style={"fontSize": "0.8rem !important", "letterSpacing": "0.04em"}),
            html.Small(
                f"Score: {score_str} · Coverage: {coverage:.0%}",
                className="text-secondary",
                style={"fontSize": "0.78rem"},
            ),
        ], className="d-flex align-items-center mb-2 pt-1 px-1"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Metric",     style={**header_style, "width": "38%"}),
                html.Th("Percentile", style={**header_style, "width": "30%"}),
                html.Th("Weight",     style={**header_style, "width": "10%",
                                             "textAlign": "center"}),
                html.Th("Status",     style={**header_style, "width": "22%"}),
            ])),
            html.Tbody(rows),
        ], className="table table-dark table-sm mb-0",
           style={"borderCollapse": "collapse"}),
    ], style={"padding": "12px 16px"})


# ---------------------------------------------------------------------------
# Utility — empty placeholder figure
# ---------------------------------------------------------------------------

def team_spray_heatmap(portrait: dict) -> go.Figure:
    """
    Real dot spray chart with directional split bars and power source donut.

    Three panels:
      Top (full width): dot spray chart — actual hc_x/hc_y from Statcast,
        colored by outcome (HR=red, 2B/3B=orange, 1B=blue, out=grey).
        Field diagram with foul lines, arc, diamond, pitcher's mound.
        Batter's box silhouette(s) based on team handedness split.
      Bottom-left: directional split bars (Pull/Gap/Center/Oppo vs league avg)
      Bottom-right: power source donut (HR / 2B+3B / 1B share of hits)
    """
    import math
    from plotly.subplots import make_subplots as _make_subplots

    spray  = portrait.get("spray_data", {})
    team   = portrait.get("team", "")
    season = portrait.get("season", "")

    if not spray or not spray.get("hc_x"):
        return empty_figure("Spray data unavailable — reload portrait")

    # ── Subplots: spray field (top, full-width) + bars + donut ───────────────
    fig = _make_subplots(
        rows=2, cols=2,
        row_heights=[0.62, 0.38],
        specs=[
            [{"colspan": 2, "type": "scatter"}, None],
            [{"type": "bar"},                    {"type": "pie"}],
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    # ── Field geometry (Statcast coords: HP≈(126,203), CF≈(126,15)) ──────────
    HP_X, HP_Y = 126.0, 203.0

    # Outfield arc (approx fence at 300–350 Statcast units from HP)
    arc_angles = [a for a in range(-45, 46)]
    r_fence    = 185  # units from HP to fence (calibrated to HR landing zone)
    arc_x = [HP_X + r_fence * math.sin(math.radians(a)) for a in arc_angles]
    arc_y = [HP_Y - r_fence * math.cos(math.radians(a)) for a in arc_angles]
    arc_x = [HP_X - r_fence * math.sin(math.radians(45))] + arc_x + \
            [HP_X + r_fence * math.sin(math.radians(45))]
    arc_y = [HP_Y - r_fence * math.cos(math.radians(45))] + arc_y + \
            [HP_Y - r_fence * math.cos(math.radians(45))]

    # Foul lines (extend beyond fence)
    r_line = 210
    fig.add_trace(go.Scatter(
        x=[HP_X, HP_X - r_line * math.sin(math.radians(45))],
        y=[HP_Y, HP_Y - r_line * math.cos(math.radians(45))],
        mode="lines", line=dict(color="#6b7280", width=1),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[HP_X, HP_X + r_line * math.sin(math.radians(45))],
        y=[HP_Y, HP_Y - r_line * math.cos(math.radians(45))],
        mode="lines", line=dict(color="#6b7280", width=1),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # Outfield arc + grass fill
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y, mode="lines", fill="toself",
        fillcolor="rgba(22,101,52,0.4)",
        line=dict(color="#4ade80", width=1.5),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # Infield dirt circle (approximate)
    infield_r = 58
    theta_if = [math.radians(a) for a in range(0, 361, 5)]
    fig.add_trace(go.Scatter(
        x=[HP_X + infield_r * math.sin(t) for t in theta_if],
        y=[HP_Y - infield_r * math.cos(t) for t in theta_if],
        mode="lines", fill="toself",
        fillcolor="rgba(120,80,40,0.3)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # Infield diamond (bases ≈ 34 units apart at 45°)
    B = 34  # base-to-base in Statcast units
    diamond_x = [HP_X, HP_X + B, HP_X, HP_X - B, HP_X]
    diamond_y = [HP_Y, HP_Y - B, HP_Y - 2*B, HP_Y - B, HP_Y]
    fig.add_trace(go.Scatter(
        x=diamond_x, y=diamond_y, mode="lines",
        line=dict(color="#d1fae5", width=1.5),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # Pitcher's mound
    mound_r = 5
    theta_m  = [math.radians(a) for a in range(0, 361, 10)]
    fig.add_trace(go.Scatter(
        x=[HP_X + mound_r * math.sin(t) for t in theta_m],
        y=[HP_Y - B + mound_r * math.cos(t) for t in theta_m],
        mode="lines", fill="toself",
        fillcolor="rgba(150,100,50,0.6)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # ── Batted ball dots ──────────────────────────────────────────────────────
    hc_x       = spray["hc_x"]
    hc_y       = spray["hc_y"]
    event_type = spray["event_type"]

    _OUTCOME = {
        "hr":     dict(color="#f97316", size=7,  opacity=0.90, name="Home Run"),
        "triple": dict(color="#22c55e", size=6,  opacity=0.85, name="Triple"),
        "double": dict(color="#06b6d4", size=5,  opacity=0.75, name="Double"),
        "single": dict(color="#60a5fa", size=4,  opacity=0.65, name="Single"),
        "out":    dict(color="#ef4444", size=3,  opacity=0.18, name="Out"),
    }
    for etype in ["out", "single", "double", "triple", "hr"]:  # out first so HRs render on top
        xs = [x for x, e in zip(hc_x, event_type) if e == etype]
        ys = [y for y, e in zip(hc_y, event_type) if e == etype]
        if not xs:
            continue
        cfg = _OUTCOME[etype]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color=cfg["color"], size=cfg["size"],
                        opacity=cfg["opacity"], line=dict(width=0)),
            name=cfg["name"],
            hovertemplate=f"{cfg['name']}<extra></extra>",
        ), row=1, col=1)

    # ── Batter's box silhouettes ──────────────────────────────────────────────
    # ── Home plate marker ────────────────────────────────────────────────────
    # Simple pentagon at HP position so the eye anchors to the origin point
    fig.add_trace(go.Scatter(
        x=[HP_X], y=[HP_Y],
        mode="markers",
        marker=dict(color="#ffffff", size=7, symbol="pentagon",
                    line=dict(color="#6b7280", width=1)),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # ── Directional split bars (5 field-location zones) ──────────────────────
    # Pull/Oppo are handedness-aware; LC Gap, Center, RC Gap are absolute field locations.
    DIRS = [
        ("Pull",    "pull"),
        ("RC Gap",  "rc_gap"),
        ("Center",  "center"),
        ("LC Gap",  "lc_gap"),
        ("Oppo",    "oppo"),
    ]
    dirs      = [label for label, _ in DIRS]
    team_vals = [spray.get(f"{key}_pct", 0) * 100 for _, key in DIRS]
    lg_vals   = [spray.get(f"lg_{key}_pct", 0) * 100 for _, key in DIRS]

    fig.add_trace(go.Bar(
        x=dirs, y=team_vals, name="Team",
        marker_color=COLORS["primary"],
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=dirs, y=lg_vals, name="Lg Avg",
        marker_color="#4b5563",
        marker_line=dict(color="#6b7280", width=1),
        hovertemplate="Lg avg %{x}: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    # ── Power source donut ────────────────────────────────────────────────────
    hr_n  = spray.get("hr_count", 0)
    xbh_n = spray.get("xbh_count", 0)
    s_n   = spray.get("single_count", 0)
    if hr_n + xbh_n + s_n > 0:
        fig.add_trace(go.Pie(
            values=[hr_n, xbh_n, s_n],
            labels=["HR", "2B/3B", "1B"],
            hole=0.52,
            marker=dict(colors=["#ef4444", "#f97316", "#60a5fa"],
                        line=dict(color=COLORS["background"], width=1)),
            textfont=dict(color=COLORS["text"], size=11),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            showlegend=False,
        ), row=2, col=2)
        # Center annotation
        fig.add_annotation(
            text="Hits", xref="paper", yref="paper",
            x=0.88, y=0.12, showarrow=False,
            font=dict(color=COLORS["subtext"], size=10),
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(**{
        **_DARK_LAYOUT,
        "title": dict(
            text=f"{team} {season} — Batted Ball Profile",
            font=dict(size=13, color=COLORS["text"]), x=0.5,
        ),
        "margin": dict(l=10, r=10, t=45, b=10),
        "barmode": "group",
        "bargap": 0.25,
        "bargroupgap": 0.08,
        "showlegend": True,
        "legend": dict(
            orientation="h",
            font=dict(color=COLORS["subtext"], size=10),
            bgcolor="rgba(17,24,39,0.85)",
            bordercolor="#374151", borderwidth=1,
            x=0.5, y=0.375,
            xanchor="center", yanchor="top",
            traceorder="reversed",
        ),
        "plot_bgcolor": "#0d1117",
        "paper_bgcolor": COLORS["background"],
    })

    # Field axes (flipped y so HP is at bottom)
    fig.update_xaxes(range=[0, 252], showgrid=False, zeroline=False,
                     showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[215, 5], showgrid=False, zeroline=False,
                     showticklabels=False, row=1, col=1)

    # Bar chart axes
    fig.update_yaxes(title_text="% of BIP", ticksuffix="%",
                     gridcolor="#374151",
                     tickfont=dict(color=COLORS["subtext"]),
                     title_font=dict(color=COLORS["subtext"]),
                     row=2, col=1)
    fig.update_xaxes(tickfont=dict(color=COLORS["text"]), row=2, col=1)

    return fig


def team_split_card(portrait: dict) -> html.Div:
    """
    Team Split Resistance — 4 mini-cards (AVG / OBP / SLG / OPS) showing
    PA-weighted team performance vs LHP and RHP plus the gap.

    Gap colour:
      < .020  → muted green  (split-resistant team)
      .020–.050 → amber      (moderate platoon split)
      > .050  → bright red   (platoon-vulnerable team)
    """
    import dash_bootstrap_components as dbc

    hitters = portrait.get("players", {}).get("hitters", [])
    if not hitters:
        return html.Div("No hitter data", className="text-secondary small p-2")

    # PA-weighted aggregation of individual splits
    lhp_totals: dict[str, float] = {"avg": 0, "obp": 0, "slg": 0, "ops": 0}
    rhp_totals: dict[str, float] = {"avg": 0, "obp": 0, "slg": 0, "ops": 0}
    lhp_pa = rhp_pa = 0.0

    for h in hitters:
        sp = h.get("splits", {})
        lhp = sp.get("vs_lhp")
        rhp = sp.get("vs_rhp")
        if lhp:
            pa = lhp.get("pa", 0)
            lhp_pa += pa
            for m in lhp_totals:
                v = lhp.get(m)
                if v is not None:
                    lhp_totals[m] += v * pa
        if rhp:
            pa = rhp.get("pa", 0)
            rhp_pa += pa
            for m in rhp_totals:
                v = rhp.get(m)
                if v is not None:
                    rhp_totals[m] += v * pa

    if lhp_pa == 0 and rhp_pa == 0:
        return html.Div("No split data available", className="text-secondary small p-2")

    def _avg(totals: dict, pa: float, metric: str) -> float | None:
        return (totals[metric] / pa) if pa > 0 else None

    METRICS = [
        ("avg", "AVG"),
        ("obp", "OBP"),
        ("slg", "SLG"),
        ("ops", "OPS"),
    ]

    def _gap_color(gap: float) -> str:
        if gap < 0.020:
            return "#10b981"   # green — resistant
        if gap < 0.050:
            return "#f59e0b"   # amber — moderate
        return "#ef4444"       # red — vulnerable

    def _gap_label(gap: float) -> str:
        if gap < 0.020: return "Resistant"
        if gap < 0.050: return "Moderate"
        return "Vulnerable"

    def _mini_card(metric: str, label: str) -> dbc.Col:
        lv = _avg(lhp_totals, lhp_pa, metric)
        rv = _avg(rhp_totals, rhp_pa, metric)

        if lv is None or rv is None:
            body = html.Div("—", className="text-secondary")
        else:
            gap   = abs(rv - lv)
            color = _gap_color(gap)
            glabel = _gap_label(gap)
            body = html.Div([
                dbc.Row([
                    dbc.Col(html.Small("vs LHP", className="text-secondary",
                                       style={"fontSize": "0.65rem"}), width=6),
                    dbc.Col(html.Span(f"{lv:.3f}",
                                      style={"color": "#f9fafb", "fontSize": "0.95rem",
                                             "fontWeight": "600"}), width=6),
                ], className="g-0 mb-1"),
                dbc.Row([
                    dbc.Col(html.Small("vs RHP", className="text-secondary",
                                       style={"fontSize": "0.65rem"}), width=6),
                    dbc.Col(html.Span(f"{rv:.3f}",
                                      style={"color": "#f9fafb", "fontSize": "0.95rem",
                                             "fontWeight": "600"}), width=6),
                ], className="g-0 mb-2"),
                html.Hr(style={"borderColor": "#374151", "margin": "4px 0"}),
                dbc.Row([
                    dbc.Col(html.Small("Split Δ", className="text-secondary",
                                       style={"fontSize": "0.65rem"}), width=6),
                    dbc.Col(html.Span(f"{gap:.3f}",
                                      style={"color": color, "fontSize": "0.95rem",
                                             "fontWeight": "700"}), width=6),
                ], className="g-0 mb-1"),
                html.Div(html.Small(glabel,
                                    style={"color": color, "fontSize": "0.65rem",
                                           "fontWeight": "600",
                                           "textTransform": "uppercase",
                                           "letterSpacing": "0.05em"})),
            ])

        return dbc.Col(dbc.Card([
            dbc.CardHeader(
                html.Small(label, className="fw-bold text-uppercase",
                           style={"fontSize": "0.75rem", "letterSpacing": "0.08em",
                                  "color": "#9ca3af"}),
                style={"backgroundColor": "#1a2233",
                       "borderBottom": "1px solid #374151",
                       "padding": "6px 12px"},
            ),
            dbc.CardBody(body, style={"padding": "10px 12px"}),
        ], style={"backgroundColor": "#1f2937", "border": "1px solid #374151",
                  "borderRadius": "6px"}),
        md=3, xs=6, className="mb-2")

    mini_cards = dbc.Row(
        [_mini_card(m, lbl) for m, lbl in METRICS],
        className="g-2",
    )

    # Footer: sample sizes
    footer = html.Small(
        f"vs LHP: {int(lhp_pa):,} PA  ·  vs RHP: {int(rhp_pa):,} PA  "
        f"·  PA-weighted team averages",
        className="text-secondary",
        style={"fontSize": "0.65rem"},
    )

    return html.Div([mini_cards, html.Div(footer, className="mt-1 px-1")])


# ---------------------------------------------------------------------------
# Tag chips, coloured by KIND
# ---------------------------------------------------------------------------
# Tags used to be coloured by FAMILY — 17 colours for what a tag is about
# (bat, approach, catcher...), which is too many to hold in your head and
# answers a question the reader is not asking. Kind is six values and answers
# the one that matters at a glance: is this something the player IS, something
# he CHOSE, something that HAPPENED, or something the club decided?
#
# Ordered by control, so the palette itself runs cool-to-warm as agency
# increases: an attribute is slate and inert, a behavior is bright, luck is
# deliberately washed out.
KIND_COLORS: dict[str, str] = {
    "attribute":  "#c4b5fd",   # light purple
    "tool":       "#86efac",   # light green
    "behavior":   "#67e8f9",   # light blue / cyan
    "result":     "#f87171",   # red
    # Not specified, chosen to stay clear of the four above: amber is the
    # remaining warm hue that red does not swallow, and luck stays grey so it
    # reads as the absence of a claim rather than as another category.
    "deployment": "#fbbf24",   # amber
    "noise":      "#9ca3af",   # grey
}
KIND_ORDER = ["attribute", "tool", "behavior", "result", "deployment", "noise"]


def _shade(hex_color: str, i: int, n: int) -> str:
    """One member of a kind, distinguished by lightness.

    Two tags of the same kind should read as siblings rather than strangers,
    so members vary within a band around the kind's colour instead of getting
    unrelated hues. Past about four members the shades converge — which is a
    real ceiling, not a palette failure: no set of colours stays separable
    much beyond eight, so identity past that point has to come from position
    and labels rather than hue.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[j:j + 2], 16) / 255 for j in (0, 2, 4))
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    if n > 1:
        ll = max(0.30, min(0.82, ll - 0.20 + (0.42 * i / max(n - 1, 1))))
        ss = max(0.30, ss - 0.10 * (i / max(n - 1, 1)))
    r2, g2, b2 = colorsys.hls_to_rgb(hh, ll, ss)
    return f"rgb({int(r2*255)},{int(g2*255)},{int(b2*255)})"


def kind_shaded_colors(tags: list[str]) -> list[str]:
    """Colour per tag: the kind's hue, shaded by position within its kind.

    Replaces colouring by FAMILY, which needed 17 hues for a distinction the
    reader was not asking about. Kind is six, under the perceptual ceiling,
    and it is the same palette the rosters and the tag footer already use — so
    a green bar here and a green chip there mean the same thing.
    """
    seen: dict[str, int] = {}
    counts: dict[str, int] = {}
    for t in tags:
        k = kind_of(t) or "noise"
        counts[k] = counts.get(k, 0) + 1
    out = []
    for t in tags:
        k = kind_of(t) or "noise"
        i = seen.get(k, 0)
        seen[k] = i + 1
        out.append(_shade(KIND_COLORS.get(k, COLORS["neutral"]), i, counts[k]))
    return out


def _chip(tag: str, kind: str | None = None, evidence: str | None = None):
    """One trait rendered as a tag: kind-coloured, evidence on hover."""
    color = KIND_COLORS.get(kind or "", COLORS["subtext"])
    return html.Span(
        tag,
        title=evidence or "",
        style={
            "display": "inline-block",
            "backgroundColor": f"{color}22",   # 13% tint of the kind colour
            "color": color,
            "border": f"1px solid {color}55",
            "borderRadius": "10px",
            "padding": "1px 7px",
            "marginRight": "4px",
            "marginBottom": "3px",
            "fontSize": "0.7rem",
            "fontWeight": "600",
            # Deliberately NOT nowrap: in the per-kind column layout the
            # columns are narrow and tag names are long ("station-to-station",
            # "high steal attempts"), so nowrap made chips overflow into the
            # neighbouring column instead of wrapping inside their own.
            "lineHeight": "1.5",
            "cursor": "help" if evidence else "default",
        },
    )


def kind_legend(title: str | None = None):
    """The six kinds as swatches — sits above a roster so the colours mean
    something before the reader hits their first chip."""
    items = [
        html.Span([
            html.Span(style={
                "display": "inline-block", "width": "9px", "height": "9px",
                "borderRadius": "2px", "backgroundColor": KIND_COLORS[k],
                "marginRight": "5px", "verticalAlign": "middle",
            }),
            html.Span(k, style={"color": COLORS["subtext"], "fontSize": "0.68rem",
                                "marginRight": "14px", "verticalAlign": "middle"}),
        ]) for k in KIND_ORDER
    ]
    head = ([html.Span("tag kind:", style={
        "color": COLORS["subtext"], "fontSize": "0.66rem",
        "textTransform": "uppercase", "letterSpacing": "0.06em",
        "marginRight": "10px"})] if title is None else [])
    return html.Div(head + items, className="mb-2",
                    style={"lineHeight": "1.6"})


def _comp_tooltip(unit: str, near: list, uq: dict) -> str:
    """The numbers behind the one-word comp strength."""
    top = near[0] if near else {}
    who, sim = top.get("team", "?"), top.get("similarity")
    med, q1, q3 = uq.get("unit_median"), uq.get("unit_q1"), uq.get("unit_q3")
    parts = []
    if sim is not None:
        parts.append(f"Closest comparable team is {who}, at {sim:.2f} similarity "
                     f"(cosine, over per-tag deviations from league average).")
    if med is not None:
        parts.append(f"Across this season's {unit}s the closest comp typically "
                     f"lands at {med:.2f}; below {q1:.2f} counts as loose and "
                     f"above {q3:.2f} as close.")
    parts.append("Each unit is banded against its own spread — offenses and "
                 "rotations do not sit on a common scale.")
    return " ".join(parts)


def _ordinal_short(n: int) -> str:
    """1st / 2nd / 3rd / 4th …"""
    i = int(n)
    if 11 <= (i % 100) <= 13:
        return f"{i}th"
    return f"{i}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th') }"


# Empty-state wording per kind. "league-typical" on its own did not say
# typical in WHAT — it read as clear for `result` and as a shrug for `tool`
# and `deployment`. Each line states the thing that was measured and came back
# ordinary; the hover carries the precise rule. Descriptive, not evaluative:
# these report that no tag cleared the bar, they do not grade the team.
_TYPICAL_TEXT: dict[tuple, tuple] = {
    ("league-typical", "attribute"): (
        "ordinary mix",
        "Handedness and delivery split close to the league's."),
    ("league-typical", "tool"): (
        "no standout tools",
        "Nobody's speed, arm or velocity is far from league norm — in either "
        "direction. The team is neither notably toolsy nor notably lacking."),
    ("league-typical", "behavior"): (
        "no distinctive tendencies",
        "Swing decisions, running and pitch selection all sit near the "
        "league's."),
    ("league-typical", "result"): (
        "league-average outcomes",
        "What the players produced — power, contact, command, defense — lands "
        "near league average."),
    ("league-typical", "deployment"): (
        "conventional usage",
        "The front office used this unit in unremarkable ways: no unusual "
        "leverage, workload or platoon patterns."),
    ("league-typical", "noise"): (
        "ordinary luck",
        "Results tracked the underlying contact; no notable over- or "
        "under-performance."),
    ("not measured", "noise"): (
        "not measured",
        "Luck is computed from a hitter's wOBA against his xwOBA, so it has "
        "no definition for a pitching staff. This is an absence of vocabulary, "
        "not a finding about the team."),
}


def team_identity_card(portrait: dict, fingerprint: dict | None = None,
                       position: dict | None = None) -> html.Div:
    """
    Team Identity — three headline labels (Offense / Rotation / Bullpen)
    synthesized from the per-player archetype distributions, plus a short
    "fit" readout (rotation vs defense, bullpen complement).
    """
    import dash_bootstrap_components as dbc

    identity = portrait.get("team_identity") or {}
    offense  = identity.get("offense") or {}
    rotation = identity.get("rotation") or {}
    bullpen  = identity.get("bullpen") or {}
    fit      = identity.get("fit") or {}

    if not offense and not rotation:
        return html.Div("No team identity data — rebuild this portrait to populate.",
                         className="text-secondary small p-2")

    def _identity_col(title: str, label: str | None, sub: str | None, color: str):
        return dbc.Col(html.Div([
            html.Small(title, className="text-secondary text-uppercase",
                       style={"fontSize": "0.65rem", "letterSpacing": "0.08em"}),
            html.Div(label or "—", style={"color": color, "fontSize": "1.05rem",
                                           "fontWeight": "700", "marginTop": "2px"}),
            html.Div(sub or "", className="text-secondary",
                     style={"fontSize": "0.7rem", "marginTop": "2px"}),
        ]), md=4, xs=12, className="mb-2")

    # `contact_share` / `complete_share` retired with the tags that fed them —
    # they rendered as a permanent "Contact 0%". Power share survives because
    # it is still computed from live tags.
    offense_sub = None
    if offense.get("power_share") is not None:
        offense_sub = f"Power {offense['power_share']:.0%} of PA"

    rotation_sub = None
    density = rotation.get("trait_density") or {}
    if density:
        # Densest traits, e.g. "tunneler 52% · bat-misser 40%". This line is
        # COMPOSITION — what the staff is mostly made of — as against the
        # fingerprint below it, which is deviation and says what is unusual.
        #
        # Attributes and luck are excluded. Density ranks by how common a tag
        # is, and `right-handed pitcher` is the most common thing about almost
        # every rotation ever assembled: it took 292 of 720 subtitle slots,
        # 44% of them going to a kind nobody chose. It is a real fact and it
        # has its own fingerprint row; it just cannot be allowed to describe
        # the staff.
        shown = [(t, v) for t, v in density.items()
                 if kind_of(t) not in ("attribute", "noise")][:2]
        rotation_sub = " · ".join(f"{t} {v:.0%}" for t, v in shown) or None
    elif rotation.get("approach_dist"):
        appr = rotation.get("dominant_approach")
        share = rotation["approach_dist"].get(appr, 0)
        rotation_sub = f"{appr} ({share:.0%} of BF)" if appr else None

    bullpen_label = bullpen.get("out_mechanism")
    bullpen_sub = bullpen.get("leverage_structure")

    # Headline labels were retired (hand-chosen cutoffs assigning a team to a
    # bucket). Position replaces them: who this unit most resembles, and how
    # unusual it is. No thresholds, and it follows the roster.
    def _headline(unit: str, fallback=None):
        pos = position or {}
        near = (pos.get("neighbours") or {}).get(unit) or []
        uq = (pos.get("uniqueness") or {}).get(unit) or {}
        if not near:
            return fallback
        names = ", ".join(x.get("team", "") for x in near[:2])
        # Qualify the comparison rather than answer a different question. The
        # word is banded against this unit's own spread of best-comp
        # similarities; the exact figure and that spread ride on the hover, so
        # the headline stays readable without hiding the number.
        band = uq.get("band")
        if not band:
            return f"Plays like {names}"
        word = {"close": "close comp", "fair": "fair comp",
                "loose": "loose comp"}.get(band, band)
        return html.Span([
            f"Plays like {names} · ",
            html.Span(word, title=_comp_tooltip(unit, near, uq),
                      style={"borderBottom": "1px dotted currentColor",
                             "cursor": "help"}),
        ])

    cols = dbc.Row([
        _identity_col("Offense",  _headline("offense"),  offense_sub,  COLORS["offense"]),
        _identity_col("Rotation", _headline("rotation"), rotation_sub, COLORS["pitching"]),
        _identity_col("Bullpen",  _headline("bullpen", bullpen_label),
                      bullpen_sub, COLORS["roster"]),
    ], className="g-3")

    fit_notes = [v for v in fit.values() if v]
    fit_block = None
    if fit_notes:
        fit_block = html.Div([
            html.Hr(style={"borderColor": COLORS["border"], "margin": "10px 0"}),
            *[html.Div(note, className="text-secondary mb-1",
                       style={"fontSize": "0.75rem"}) for note in fit_notes],
        ])

    # ── Fingerprint — where this team deviates from the league ────────────
    fp_block = None
    if fingerprint:
        unit_meta = [("offense", "Offense", COLORS["offense"]),
                     ("rotation", "Rotation", COLORS["pitching"]),
                     ("bullpen", "Bullpen", COLORS["roster"])]
        lines = []
        for unit, label, color in unit_meta:
            rows = fingerprint.get(unit) or []
            # Legacy shape (a flat ranked list) had no `kind`; render it the
            # old way so a stale league file still displays.
            if rows and "kind" not in (rows[0] or {}):
                rows = [{"kind": None, "state": "defined", "tags": rows}]
            if not rows:
                continue
            lines.append(html.Div(
                label, style={"color": color, "fontWeight": "700",
                              "fontSize": "0.7rem", "textTransform": "uppercase",
                              "letterSpacing": "0.06em",
                              "marginTop": "8px", "marginBottom": "3px"}))
            for row in rows:
                kind = row.get("kind")
                state = row.get("state", "defined")
                tags = row.get("tags") or []
                # The kind label is a fixed column so the six rows line up
                # across teams and units — the card is a skeleton, not a list.
                kind_span = html.Span(
                    (kind or ""),
                    style={"color": COLORS["subtext"], "fontSize": "0.66rem",
                           "textTransform": "uppercase", "letterSpacing": "0.05em",
                           "display": "inline-block", "width": "78px",
                           "verticalAlign": "top"})
                if state != "defined" or not tags:
                    # Dimmed but legible: this is a real finding in this
                    # vocabulary, not an absence, so it should read as a quiet
                    # statement rather than a rendering gap. "league-typical"
                    # alone did not say typical IN WHAT, so each kind says
                    # what was actually checked.
                    label, tip = _TYPICAL_TEXT.get(
                        (state, kind),
                        (state, "No tag of this kind separates this team "
                                "from the league."))
                    body = html.Span(
                        label, title=tip,
                        style={"color": COLORS["subtext"], "fontSize": "0.72rem",
                               "fontStyle": "italic", "opacity": "0.75",
                               "cursor": "help"})
                else:
                    parts = []
                    for i, e in enumerate(tags):
                        dev = e.get("deviation", 0) * 100
                        breadth = (f" ({e['carriers']}/{e['qualifiers']})"
                                   if e.get("qualifiers") else "")
                        if i:
                            parts.append(html.Span(" · ", className="text-secondary",
                                                   style={"fontSize": "0.72rem"}))
                        parts.append(html.Span(
                            e.get("tag", ""),
                            style={"color": "#f9fafb", "fontWeight": "600",
                                   "fontSize": "0.76rem"}))
                        parts.append(html.Span(
                            f" {dev:+.0f}pts{breadth}", className="text-secondary",
                            style={"fontSize": "0.7rem"}))
                    body = html.Span(parts)
                lines.append(html.Div([kind_span, body], className="mb-1"))
        if lines:
            fp_block = html.Div([
                html.Hr(style={"borderColor": COLORS["border"], "margin": "10px 0"}),
                html.Small("Fingerprint — what makes them this team",
                           className="text-secondary text-uppercase d-block mb-2",
                           style={"fontSize": "0.65rem", "letterSpacing": "0.08em"}),
                *lines,
            ])

    children = [cols]
    if fit_block:
        children.append(fit_block)
    if fp_block:
        children.append(fp_block)
    return html.Div(children)


def empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=COLORS["subtext"], size=14),
    )
    fig.update_layout(**_DARK_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Team Comparison charts
# ---------------------------------------------------------------------------

# Team A = blue, Team B = orange (consistent across all comparison charts)
_CMP_COLOR_A = "#60a5fa"
_CMP_COLOR_B = "#f97316"


def _portrait_label(portrait: dict) -> str:
    team   = portrait.get("team", "?")
    season = portrait.get("season", "?")
    return f"{team} {season}"


def compare_radar(portrait_a: dict | None, portrait_b: dict | None) -> go.Figure:
    """
    Overlaid philosophy radar — both teams on the same spider.
    Uses the same dimension order as philosophy_radar().
    """
    if not portrait_a and not portrait_b:
        return empty_figure("Load Team A and Team B to compare")

    # Dimension order matches philosophy_radar
    DIMS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"]
    DIM_LABELS = {
        "A1": "TTO", "A2": "Contact+Pressure", "A3": "Aggressive Early",
        "A4": "Lineup Power", "B1": "Stuff Dominant", "B2": "Command+Contact Mgmt",
        "B3": "Pitch Design", "B4": "Defensive Infra",
        "C1": "bWAR Distribution", "C2": "Roster Continuity",
        "C3": "Youth+Dev", "C4": "Veteran Experience",
    }
    labels = [DIM_LABELS.get(d, d) for d in DIMS]

    fig = go.Figure()

    for portrait, color, suffix in [
        (portrait_a, _CMP_COLOR_A, "a"),
        (portrait_b, _CMP_COLOR_B, "b"),
    ]:
        if not portrait:
            continue
        scores = portrait.get("scores", {})
        vals = []
        for d in DIMS:
            s = scores.get(d, {}).get("score")
            vals.append(float(s) if s is not None else 0.0)
        vals_closed = vals + [vals[0]]
        lbl = _portrait_label(portrait)

        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels + [labels[0]],
            fill="toself",
            name=lbl,
            line=dict(color=color, width=2),
            fillcolor=_hex_to_rgba(color, 0.12),
            hovertemplate="<b>%{theta}</b><br>" + lbl + ": %{r:.0f}<extra></extra>",
        ))

    fig.update_layout(
        **_DARK_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color=COLORS["subtext"]),
                            gridcolor="#374151", linecolor="#374151"),
            angularaxis=dict(tickfont=dict(size=10, color=COLORS["subtext"]), gridcolor="#374151"),
            bgcolor=COLORS["surface"],
        ),
        showlegend=True,
        legend=dict(font=dict(color=COLORS["subtext"]), orientation="h", y=-0.08),
    )
    return fig


def compare_dim_bars(portrait_a: dict | None, portrait_b: dict | None) -> go.Figure:
    """Grouped horizontal bars — philosophy dimension scores for both teams."""
    if not portrait_a and not portrait_b:
        return empty_figure("Load both teams to compare")

    DIM_ORDER = ["A1","A2","A3","A4","B1","B2","B3","B4","C1","C2","C3","C4"]
    DIM_NAMES = {
        "A1": "Three True Outcomes",  "A2": "Contact / Speed",
        "A3": "Aggressive Approach",   "A4": "Lineup Power",
        "B1": "Stuff Dominance",        "B2": "Command & Defense",
        "B3": "Pitch Design",           "B4": "Defensive Infra",
        "C1": "WAR Distribution",       "C2": "Roster Continuity",
        "C3": "Youth & Dev",            "C4": "Veteran Experience",
    }

    dims  = [DIM_NAMES.get(d, d) for d in reversed(DIM_ORDER)]
    codes = list(reversed(DIM_ORDER))

    fig = go.Figure()
    for portrait, color in [(portrait_a, _CMP_COLOR_A), (portrait_b, _CMP_COLOR_B)]:
        if not portrait:
            continue
        scores = portrait.get("scores", {})
        vals   = [float(scores.get(c, {}).get("score") or 0) for c in codes]
        fig.add_trace(go.Bar(
            x=vals, y=dims, orientation="h",
            name=_portrait_label(portrait),
            marker_color=color,
            hovertemplate="<b>%{y}</b><br>Score: %{x:.0f}<extra></extra>",
        ))

    fig.update_layout(**{
        **_DARK_LAYOUT,
        "barmode": "group",
        "margin": dict(l=160, r=20, t=30, b=40),
        "bargap": 0.25, "bargroupgap": 0.08,
        "xaxis": dict(range=[0, 100], title="Score (0–100)", gridcolor="#374151",
                      tickfont=dict(color=COLORS["subtext"]),
                      title_font=dict(color=COLORS["subtext"])),
        "yaxis": dict(tickfont=dict(color=COLORS["text"], size=11)),
        "legend": dict(font=dict(color=COLORS["subtext"]), orientation="h", y=1.04),
    })
    return fig


def compare_batting_bars(portrait_a: dict | None, portrait_b: dict | None) -> go.Figure:
    """Side-by-side team batting profile percentile bars."""
    if not portrait_a and not portrait_b:
        return empty_figure("Load both teams to compare")

    METRICS = [
        ("sprint_speed_pct",  "Sprint Speed"),
        ("Contact_pct_pct",   "Contact%"),
        ("OSwing_pct_pct",    "O-Swing%"),
        ("FPS_pct_pct",       "FPS%"),
        ("SwStr_pct_pct",     "SwStr%"),
        ("Barrel_pct_pct",    "Barrel%"),
        ("xwOBA_pct",         "xwOBA"),
        ("HardHit_pct_pct",   "Hard Hit%"),
        ("BB_pct_pct",        "BB%"),
        ("K_pct_pct",         "K%"),
    ]

    # Batting percentile metrics live in portrait["philosophy_metrics"]
    fig = go.Figure()
    for portrait, color in [(portrait_a, _CMP_COLOR_A), (portrait_b, _CMP_COLOR_B)]:
        if not portrait:
            continue
        pm    = portrait.get("philosophy_metrics", {})
        vals  = [float(pm.get(key) or 0) for key, _ in METRICS]
        names = [label for _, label in METRICS]
        fig.add_trace(go.Bar(
            y=list(reversed(names)), x=list(reversed(vals)), orientation="h",
            name=_portrait_label(portrait),
            marker_color=color,
            hovertemplate="<b>%{y}</b><br>Percentile: %{x:.0f}<extra></extra>",
        ))

    fig.add_vline(x=50, line=dict(color="#6b7280", width=1, dash="dot"))
    fig.update_layout(**{
        **_DARK_LAYOUT,
        "barmode": "group",
        "margin": dict(l=100, r=20, t=30, b=40),
        "bargap": 0.25, "bargroupgap": 0.08,
        "xaxis": dict(range=[0, 100], title="Percentile vs. league", gridcolor="#374151",
                      tickfont=dict(color=COLORS["subtext"]),
                      title_font=dict(color=COLORS["subtext"])),
        "yaxis": dict(tickfont=dict(color=COLORS["text"], size=11)),
        "legend": dict(font=dict(color=COLORS["subtext"]), orientation="h", y=1.04),
    })
    return fig


def compare_archetype_bars(portrait_a: dict | None, portrait_b: dict | None) -> go.Figure:
    """
    Grouped bar chart showing hitter trait density (% of PA) for both teams.
    """
    if not portrait_a and not portrait_b:
        return empty_figure("Load both teams to compare")

    def _density(portrait: dict) -> dict[str, float]:
        hitters = portrait.get("players", {}).get("hitters", [])
        return {tag: share * 100
                for tag, (share, _fam) in _pa_trait_density(hitters).items()}

    dens_a = _density(portrait_a) if portrait_a else {}
    dens_b = _density(portrait_b) if portrait_b else {}

    # Tags worth comparing: union of each team's top tags, ordered by max share
    all_tags = sorted(set(dens_a) | set(dens_b),
                      key=lambda t: -max(dens_a.get(t, 0), dens_b.get(t, 0)))
    tag_order = all_tags[:8]
    if not tag_order:
        return empty_figure("No hitter trait data — rebuild these portraits")

    fig = go.Figure()
    for portrait, dens, bar_color in [
        (portrait_a, dens_a, _CMP_COLOR_A),
        (portrait_b, dens_b, _CMP_COLOR_B),
    ]:
        if not portrait:
            continue
        lbl = _portrait_label(portrait)
        fig.add_trace(go.Bar(
            x=tag_order,
            y=[dens.get(t, 0) for t in tag_order],
            name=lbl,
            marker_color=bar_color,
            hovertemplate="<b>%{x}</b><br>" + lbl + ": %{y:.1f}% of PA<extra></extra>",
        ))

    fig.update_layout(**{
        **_DARK_LAYOUT,
        "barmode": "group",
        "margin": dict(l=50, r=20, t=40, b=60),
        "bargap": 0.3, "bargroupgap": 0.1,
        "yaxis": dict(title="% of team PA", range=[0, 100], gridcolor="#374151",
                      tickfont=dict(color=COLORS["subtext"]),
                      title_font=dict(color=COLORS["subtext"])),
        "xaxis": dict(tickfont=dict(color=COLORS["text"], size=10)),
        "legend": dict(font=dict(color=COLORS["subtext"]), orientation="h", y=1.06),
    })
    return fig


# ---------------------------------------------------------------------------
# Construction vs Results charts
# ---------------------------------------------------------------------------

def construction_vs_results_radar(portrait: dict) -> go.Figure:
    """
    Philosophy radar with projected (dotted/amber) vs actual (solid/blue) overlay.
    """
    proj   = portrait.get("projected", {})
    scores = portrait.get("philosophy", {}).get("scores", {})
    team   = portrait.get("team", "")
    season = portrait.get("season", "")

    DIMS = ["A1","A2","A3","A4","B1","B2","B3","B4","C1","C2","C3","C4"]
    DIM_LABELS = {
        "A1":"TTO","A2":"Contact+Pressure","A3":"Aggressive Early",
        "A4":"Lineup Power","B1":"Stuff Dominant","B2":"Command+Contact Mgmt",
        "B3":"Pitch Design","B4":"Defensive Infra",
        "C1":"bWAR Distribution","C2":"Roster Continuity",
        "C3":"Youth+Dev","C4":"Veteran Experience",
    }
    labels = [DIM_LABELS.get(d, d) for d in DIMS]
    fig = go.Figure()

    # Actual (solid blue)
    actual_vals   = [float(scores.get(d, {}).get("score") or 0) for d in DIMS]
    actual_closed = actual_vals + [actual_vals[0]]
    fig.add_trace(go.Scatterpolar(
        r=actual_closed, theta=labels + [labels[0]],
        fill="toself", name=f"{season} Actual",
        line=dict(color=COLORS["primary"], width=2),
        fillcolor=_hex_to_rgba(COLORS["primary"], 0.12),
        hovertemplate="<b>%{theta}</b><br>Actual: %{r:.0f}<extra></extra>",
    ))

    # Projected (dotted amber)
    proj_metrics = proj.get("philosophy_metrics", {})
    prior        = proj.get("prior_season")
    proj_dims: list[str] = []
    proj_vals: list[float] = []
    if proj_metrics and prior:
        try:
            from philosophy import compute_all_philosophies
            proj_scores = compute_all_philosophies(proj_metrics)
        except Exception:
            proj_scores = {}

        # The projection is built from prior-year hitter-side data only, so
        # offense dims (A1-A4, PA-weighted) and now stuff/command dims
        # (B1 fully, B2 partially — minus the team-aggregate TeamDefense_pct,
        # BF-weighted) typically resolve to a real score. B3/B4 and the
        # roster dims (C1-C4) are team/roster-construction constructs that
        # can't be reconstructed from individual prior-year stats, so they
        # stay unprojected. Plotting `None` as 0 would draw a misleading
        # near-collapsed shape implying "projected to be terrible" rather
        # than "not projectable" — so we only plot dimensions with a score.
        proj_dims   = [d for d in DIMS if proj_scores.get(d, {}).get("score") is not None]
        proj_vals   = [float(proj_scores[d]["score"]) for d in proj_dims]
        proj_labels = [DIM_LABELS.get(d, d) for d in proj_dims]

        if proj_vals:
            proj_closed  = proj_vals + [proj_vals[0]]
            theta_closed = proj_labels + [proj_labels[0]]
            prior_str    = f"⚡ {prior}" if proj.get("prior_season_flag") else str(prior)
            coverage     = proj.get("coverage", 0)
            pitch_cov    = proj.get("pitching_coverage")
            if len(proj_dims) < len(DIMS):
                groups = []
                if any(d.startswith("A") for d in proj_dims):
                    groups.append("offense")
                if any(d.startswith("B") for d in proj_dims):
                    groups.append("stuff/command")
                scope_note = " + ".join(groups) + " dims only" if groups else "partial"
            else:
                scope_note = ""
            name = f"Projected (from {prior_str} · {coverage:.0%} PA"
            if pitch_cov is not None:
                name += f" · {pitch_cov:.0%} BF"
            name += f" · {scope_note})" if scope_note else ")"
            fig.add_trace(go.Scatterpolar(
                r=proj_closed, theta=theta_closed,
                fill="toself",
                name=name,
                line=dict(color="#f59e0b", width=2, dash="dot"),
                fillcolor=_hex_to_rgba("#f59e0b", 0.08),
                hovertemplate="<b>%{theta}</b><br>Projected: %{r:.0f}<extra></extra>",
            ))

    annotations = []
    if proj_metrics and prior and proj_vals and len(proj_dims) < len(DIMS):
        missing = [DIM_LABELS.get(d, d) for d in DIMS if d not in proj_dims]
        annotations.append(dict(
            text=("<b>Why is the projection incomplete?</b> Dimensions like "
                  + ", ".join(missing) +
                  " depend on team-level constructs — bWAR distribution, "
                  "roster tenure/shares, spin-efficiency &amp; tunneling models, "
                  "arsenal diversity, platoon/opener usage, OAA/DRS, park "
                  "factors — that can't be rebuilt from one player's prior "
                  "raw stats without re-running the full team pipeline on a "
                  "hypothetical roster. Showing them as 0 would misleadingly "
                  "imply \"projected to be terrible\" rather than \"not "
                  "projectable\", so they're left off the projected trace."),
            xref="paper", yref="paper",
            x=0.5, y=-0.30, xanchor="center", yanchor="top",
            showarrow=False, align="center",
            font=dict(size=9, color=COLORS["subtext"]),
            width=520,
        ))

    _layout = dict(_DARK_LAYOUT)
    if annotations:
        _layout["margin"] = dict(l=24, r=24, t=40, b=120)
    fig.update_layout(
        **_layout,
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(size=9, color=COLORS["subtext"]),
                            gridcolor="#374151", linecolor="#374151"),
            angularaxis=dict(tickfont=dict(size=10, color=COLORS["subtext"]),
                             gridcolor="#374151"),
            bgcolor=COLORS["surface"],
        ),
        showlegend=True,
        legend=dict(font=dict(color=COLORS["subtext"]), orientation="h", y=-0.14),
        title=dict(text=f"{team} {season} — Construction vs Results",
                   font=dict(size=13, color=COLORS["text"]), x=0.5),
        annotations=annotations,
    )
    return fig


def construction_vs_results_archetypes(portrait: dict) -> go.Figure:
    """Grouped bars: projected vs actual hitter trait density (% of PA)."""
    proj    = portrait.get("projected", {})
    hitters = portrait.get("players", {}).get("hitters", [])
    team    = portrait.get("team", "")
    season  = portrait.get("season", "")

    actual_density = {tag: share * 100
                      for tag, (share, _f) in _pa_trait_density(hitters).items()}
    proj_density   = {tag: share * 100
                      for tag, share in (proj.get("trait_density") or {}).items()}

    all_tags = sorted(set(actual_density) | set(proj_density),
                      key=lambda t: -max(actual_density.get(t, 0), proj_density.get(t, 0)))
    tag_order = all_tags[:8]
    if not tag_order:
        return empty_figure("No trait data — rebuild this portrait")

    prior     = proj.get("prior_season")
    coverage  = proj.get("coverage", 0)
    excl      = proj.get("excluded_pa_pct", 0)
    prior_str = (f"⚡ {prior}" if proj.get("prior_season_flag") else str(prior)) if prior else "N/A"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tag_order, y=[actual_density.get(t, 0) for t in tag_order],
        name=f"{season} Actual",
        marker_color=COLORS["primary"],
        hovertemplate="<b>%{x}</b><br>Actual: %{y:.1f}% of PA<extra></extra>",
    ))
    if prior and proj_density:
        fig.add_trace(go.Bar(
            x=tag_order, y=[proj_density.get(t, 0) for t in tag_order],
            name=f"Projected (from {prior_str}, {coverage:.0%} PA)",
            marker_color="#f59e0b",
            hovertemplate="<b>%{x}</b><br>Projected: %{y:.1f}% of PA<extra></extra>",
        ))

    note = (f"Projection covers {coverage:.0%} of PA · "
            f"{excl:.0%} excluded (rookies / no prior data)") if prior else "No prior-year data"

    fig.update_layout(**{
        **_DARK_LAYOUT,
        "barmode":"group","bargap":0.3,"bargroupgap":0.1,
        "title":dict(text=f"{team} {season}<br><sup>{note}</sup>",
                     font=dict(size=12, color=COLORS["text"]), x=0.5, xanchor="center"),
        "yaxis":dict(title="% of team PA", range=[0,100], gridcolor="#374151",
                     tickfont=dict(color=COLORS["subtext"]),
                     title_font=dict(color=COLORS["subtext"])),
        "xaxis":dict(tickfont=dict(color=COLORS["text"], size=10)),
        "legend":dict(font=dict(color=COLORS["subtext"]), orientation="h", y=1.1),
    })
    return fig


# ---------------------------------------------------------------------------
# Batter L/R split resistance heatmap
# ---------------------------------------------------------------------------

def batter_split_heatmap(portrait: dict) -> go.Figure:
    """
    Heatmap showing each hitter's AVG / OBP / SLG / OPS split between
    LHP and RHP.  Cell colour = red intensity ∝ size of difference.
    Larger gap = brighter red.  Hover shows both sides + gap.
    """
    hitters = portrait.get("players", {}).get("hitters", [])
    if not hitters:
        return empty_figure("No hitter data")

    METRICS = [
        ("avg",  "AVG"),
        ("obp",  "OBP"),
        ("slg",  "SLG"),
        ("ops",  "OPS"),
    ]

    rows_with_splits = [
        h for h in sorted(hitters, key=lambda h: -(h.get("pa") or 0))
        if h.get("splits") and (
            h["splits"].get("vs_lhp") or h["splits"].get("vs_rhp")
        )
    ]
    if not rows_with_splits:
        return empty_figure("No split data — requires Statcast season data")

    col_labels  = [label for _, label in METRICS]
    row_labels  = []
    z_vals      = []
    text_vals   = []
    hover_vals  = []

    for h in rows_with_splits:
        name  = (h.get("name") or "").strip() or f"ID {h.get('player_id','?')}"
        pa    = h.get("pa", 0)
        splits = h.get("splits", {})
        lhp   = splits.get("vs_lhp", {})
        rhp   = splits.get("vs_rhp", {})

        row_labels.append(f"{name} ({pa} PA)")
        row_z, row_txt, row_hover = [], [], []

        for metric, label in METRICS:
            lv = lhp.get(metric)
            rv = rhp.get(metric)

            if lv is None and rv is None:
                row_z.append(float("nan"))
                row_txt.append("")
                row_hover.append(f"{name}<br>{label}: no data")
                continue

            if lv is None or rv is None:
                # Only one side available — show as grey
                row_z.append(float("nan"))
                side = "vs LHP" if lv is not None else "vs RHP"
                val  = lv if lv is not None else rv
                row_txt.append(f"{val:.3f}*")
                row_hover.append(f"{name}<br>{label} {side}: {val:.3f} (one side only)")
                continue

            gap   = abs(rv - lv)        # absolute split gap
            sign  = rv - lv             # positive = better vs RHP
            fmt_lv = f"{lv:.3f}"
            fmt_rv = f"{rv:.3f}"
            fmt_gap = f"Δ {gap:.3f}"
            direction = "vs RHP" if sign >= 0 else "vs LHP"

            row_z.append(gap)           # heatmap intensity = size of gap
            row_txt.append(f"Δ{gap:.3f}")
            row_hover.append(
                f"<b>{name}</b><br>"
                f"{label} vs LHP: {fmt_lv} ({lhp.get('pa',0)} PA)<br>"
                f"{label} vs RHP: {fmt_rv} ({rhp.get('pa',0)} PA)<br>"
                f"{fmt_gap} — better {direction}"
            )

        z_vals.append(row_z)
        text_vals.append(row_txt)
        hover_vals.append(row_hover)

    z_array = np.array([[v for v in row] for row in z_vals], dtype=float)

    fig = go.Figure(go.Heatmap(
        z=z_array,
        x=col_labels,
        y=row_labels,
        text=text_vals,
        texttemplate="%{text}",
        hovertext=hover_vals,
        hoverinfo="text",
        colorscale=[
            [0.0,  "#1f2937"],   # 0 gap → dark background (no split)
            [0.2,  "#7f1d1d"],   # small gap → very muted red
            [0.5,  "#dc2626"],   # moderate gap → bright red
            [1.0,  "#fca5a5"],   # large gap → pale red (washed out at extremes)
        ],
        zmin=0,
        zmax=0.100,             # >0.100 split is essentially the max you'd see
        showscale=True,
        colorbar=dict(
            title=dict(text="Split gap", font=dict(color=COLORS["subtext"], size=10)),
            tickfont=dict(color=COLORS["subtext"], size=9),
            tickvals=[0, 0.025, 0.050, 0.075, 0.100],
            ticktext=["0", ".025", ".050", ".075", "≥.100"],
            len=0.6,
        ),
        textfont=dict(size=9, color="#f9fafb"),
    ))
    fig.update_layout(
        **{**_DARK_LAYOUT, "margin": dict(l=160, r=80, t=70, b=10)},
        title=dict(
            text="Batter Split Resistance — LHP vs RHP (gap size = red intensity)",
            font=dict(size=13, color=COLORS["text"]), x=0.5,
        ),
        xaxis=dict(side="top", tickfont=dict(color=COLORS["text"], size=11)),
        yaxis=dict(tickfont=dict(color=COLORS["text"], size=10), autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# 3-D pitch arsenal trajectory chart
# ---------------------------------------------------------------------------

def _reconstruct_trajectory(
    entry: dict, n_points: int = 30, n_tail: int = 6
) -> tuple[list, list, list, int] | None:
    """
    Reconstruct 3-D flight path using Statcast kinematic parameters.
    Returns (x_path, y_path, z_path, i_cross), or None if physics are invalid,
    where i_cross indexes the sample at the front of the plate.

    The path runs past that front edge to the plate's back tip at y=0. It used
    to stop dead at PLATE_Y, which nothing revealed until the plate itself was
    drawn and every pitch turned out to expire against the leading edge without
    ever crossing the thing. The ball is still working over those 17 inches:
    about 10 ms, in which a four-seamer drops another 1.4 inches and a curve
    3.4 — a fifth of the zone's height.

    The tail is extrapolation; the nine kinematic parameters are fit to the
    flight up to the plate crossing. But it is 10 ms of extrapolation under
    constant acceleration, over which drag and Magnus barely move. Past the
    back tip is not drawn at all — the catcher receives it there, and the
    model knows nothing about a catcher.

    i_cross is returned so the caller can keep the crossing MARKER on
    PLATE_Y, where plate_x/plate_z are measured and where the zone is judged.
    Letting it slide to the back tip would move the measurement point.

    Coordinate system (raw Statcast convention — catcher's/umpire's view,
    i.e. standing behind home plate looking out toward the pitcher):
      x — horizontal: positive = catcher's right / first-base side / LHP arm
          side; negative = catcher's left / third-base side / RHP arm side
          (confirmed empirically: RHP release_x ≈ -1.3 to -1.9 ft,
          LHP release_x ≈ +1.3 to +2.2 ft, consistent across all pitch types)
      y — distance from home plate (0 = plate, ~54 ft = release)
      z — height above ground
    """
    try:
        x0 = entry["release_x"]
        y0 = entry["release_y"]
        z0 = entry["release_z"]
        vx0, vy0, vz0 = entry["vx0"], entry["vy0"], entry["vz0"]
        ax_, ay_, az_ = entry["ax"], entry["ay"], entry["az"]

        # Time to a given distance from the plate, on the approaching root:
        # y0 + vy0·t + ½·ay·t² = y_target
        a_c = 0.5 * ay_
        b_c = vy0

        def _t_at(y_target: float):
            disc = b_c**2 - 4 * a_c * (y0 - y_target)
            if disc < 0:
                return None
            return (-b_c - np.sqrt(disc)) / (2 * a_c)

        t_flight = _t_at(PLATE_Y)            # front of plate — the crossing
        if t_flight is None or t_flight <= 0 or t_flight > 0.65:
            return None

        t = np.linspace(0, t_flight, n_points)
        # Tail across the plate to the back tip. If it will not solve, draw
        # the flight on its own rather than dropping the pitch entirely.
        t_back = _t_at(0.0)
        if t_back is not None and t_back > t_flight:
            t = np.concatenate([t, np.linspace(t_flight, t_back, n_tail + 1)[1:]])

        xs = (x0 + vx0 * t + 0.5 * ax_ * t**2).tolist()
        ys = (y0 + vy0 * t + 0.5 * ay_ * t**2).tolist()
        zs = (z0 + vz0 * t + 0.5 * az_ * t**2).tolist()
        return xs, ys, zs, n_points - 1
    except Exception:
        return None


def arsenal_pitcher_choices(portrait: dict) -> list[tuple[int, str, int]]:
    """(player_id, name, total pitches) for everyone with trajectory data,
    heaviest workload first."""
    by_type = (portrait.get("arsenal_trajectories") or {}).get("by_pitch_type", {})
    totals: dict[int, list] = {}
    for entries in by_type.values():
        for e in entries:
            pid = e.get("player_id")
            if pid is None:
                continue
            row = totals.setdefault(int(pid), [e.get("name") or "?", 0])
            row[1] += int(e.get("pitch_count") or 0)
    return sorted(((pid, n, c) for pid, (n, c) in totals.items()),
                  key=lambda r: -r[2])


def arsenal_pitch_choices(portrait: dict, player_ids: list[int]) -> list[str]:
    """Pitch types actually thrown by the selected pitchers, usage-ordered."""
    by_type = (portrait.get("arsenal_trajectories") or {}).get("by_pitch_type", {})
    wanted = set(player_ids or [])
    tot: dict[str, float] = {}
    for pt, entries in by_type.items():
        for e in entries:
            if int(e.get("player_id") or -1) in wanted:
                tot[pt] = tot.get(pt, 0.0) + float(e.get("usage_pct") or 0)
    return sorted(tot, key=lambda k: -tot[k])


COMMIT_FT = 23.0   # hitter's decision point, feet from the plate
_FASTBALL_CODES = ("FF", "FA", "SI", "FT", "FC")


def _path_at(entry: dict, ys: np.ndarray):
    """
    (x, z) of a pitch at each distance-to-plate in `ys`, in feet.

    Same kinematics as _reconstruct_trajectory, but solved at chosen y values
    rather than evenly in time, so two pitches can be compared at the same
    point in space — which is what the hitter actually sees.
    """
    x0, y0, z0 = entry["release_x"], entry["release_y"], entry["release_z"]
    vx0, vy0, vz0 = entry["vx0"], entry["vy0"], entry["vz0"]
    ax_, ay_, az_ = entry["ax"], entry["ay"], entry["az"]
    a = 0.5 * ay_
    disc = vy0 ** 2 - 4 * a * (y0 - ys)
    t = np.where(disc >= 0, (-vy0 - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.nan)
    t = np.where(t >= 0, t, np.nan)
    return (x0 + vx0 * t + 0.5 * ax_ * t ** 2,
            z0 + vz0 * t + 0.5 * az_ * t ** 2)


def tunnel_separation(a: dict, b: dict, n: int = 140):
    """
    (ys, inches) — how far apart two pitches are at every point of the flight.

    Starts at the later of the two release points, since before that one of
    them does not exist yet. Distance is in the plane the hitter reads, so
    horizontal and vertical separation are combined.
    """
    y_start = min(a["release_y"], b["release_y"])
    ys = np.linspace(y_start, PLATE_Y, n)
    ax_, az_ = _path_at(a, ys)
    bx_, bz_ = _path_at(b, ys)
    return ys, np.sqrt((ax_ - bx_) ** 2 + (az_ - bz_) ** 2) * 12.0


def _reference_pitch(pitches: dict) -> str | None:
    """
    The pitch everything else is measured against — his most-thrown fastball,
    falling back to his most-thrown pitch.

    Tunnelling is not really a property of a pair picked at random: the hitter
    is timing the fastball, and every other pitch either mimics it or does
    not. Anchoring here also turns N² pairs into N-1 readable lines.
    """
    if not pitches:
        return None
    fbs = {k: v for k, v in pitches.items()
           if _PITCH_NAME_TO_CODE.get(k, "") in _FASTBALL_CODES}
    pool = fbs or pitches
    return max(pool, key=lambda k: pool[k].get("usage_pct") or 0)


_TUNNEL_GRID_MAX = 55.0    # ft; beyond the deepest release point in the data
_TUNNEL_GRID_N   = 140
_TUNNEL_MIN_COUNT = 50     # pitches before an arm's shape joins the average
_TUNNEL_LEAGUE_CACHE: dict[int, dict[str, np.ndarray]] = {}


def _tunnel_grid() -> np.ndarray:
    return np.linspace(_TUNNEL_GRID_MAX, PLATE_Y, _TUNNEL_GRID_N)


def tunnel_league_average(season: int) -> dict[str, np.ndarray]:
    """
    The league's mean separation-from-fastball curve for each pitch type.

    Answers the question the single-pitcher view cannot: is this curveball
    hiding behind the fastball longer than a curveball normally does, or does
    every curveball look like that? Built the same way as the pitcher's own
    lines — each arm measured against HIS reference fastball — then averaged
    per pitch type on a common grid.

    One arm, one vote, above a minimum pitch count. Weighting by volume would
    let a handful of high-usage starters set the shape of the league.
    """
    if season in _TUNNEL_LEAGUE_CACHE:
        return _TUNNEL_LEAGUE_CACHE[season]

    grid = _tunnel_grid()
    totals: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    try:
        import json
        pdir = _HERE.parent / "data" / "processed" / "portraits"
        for path in sorted(pdir.glob(f"*_{season}.json")):
            try:
                p = json.loads(path.read_text())
            except Exception:
                continue
            by_type = (p.get("arsenal_trajectories") or {}).get("by_pitch_type") or {}
            arms: dict[int, dict] = {}
            for pt, entries in by_type.items():
                for e in entries:
                    if (e.get("pitch_count") or 0) >= _TUNNEL_MIN_COUNT:
                        arms.setdefault(int(e.get("player_id") or -1), {})[pt] = e
            for pitches in arms.values():
                ref = _reference_pitch(pitches)
                if ref is None:
                    continue
                for name, e in pitches.items():
                    if name == ref:
                        continue
                    try:
                        ys, d = tunnel_separation(pitches[ref], e)
                    except Exception:
                        continue
                    # ys descends; np.interp needs it ascending. Outside the
                    # pair's own range (before the later release) stays NaN.
                    v = np.interp(grid, ys[::-1], d[::-1],
                                  left=np.nan, right=np.nan)
                    ok = np.isfinite(v)
                    if name not in totals:
                        totals[name] = np.zeros(_TUNNEL_GRID_N)
                        counts[name] = np.zeros(_TUNNEL_GRID_N)
                    totals[name][ok] += v[ok]
                    counts[name][ok] += 1
    except Exception:
        pass

    out = {k: np.where(counts[k] > 0, totals[k] / np.maximum(counts[k], 1), np.nan)
           for k in totals}
    _TUNNEL_LEAGUE_CACHE[season] = out
    return out


def tunnel_profile(portrait: dict,
                   player_ids: list[int] | None = None,
                   pitch_types: list[str] | None = None) -> go.Figure:
    """
    How well each pitch mirrors the fastball, from release to the plate.

    The single-number tunnel score in tunneling.py samples one instant and
    calls it deception. What matters is the SHAPE: two pitches that sit on
    top of each other until the hitter has to commit and then split are a
    tunnel; two that separate early are just two pitches. So this draws
    separation continuously and marks the decision point on it, rather than
    reporting the value there and discarding the curve.

    y = separation from the reference fastball, in inches
    x = distance to the plate, release on the left
    """
    arsenal = portrait.get("arsenal_trajectories", {})
    by_type = arsenal.get("by_pitch_type", {})
    if not by_type:
        return empty_figure("No pitch trajectory data available")

    choices = arsenal_pitcher_choices(portrait)
    if not choices:
        return empty_figure("No pitch trajectory data available")
    wanted = list(dict.fromkeys(player_ids or [choices[0][0]]))

    arms: dict[int, dict] = {}
    for pt, entries in by_type.items():
        for e in entries:
            pid = int(e.get("player_id") or -1)
            if pid in wanted and (not pitch_types or pt in pitch_types):
                arms.setdefault(pid, {})[pt] = e
    if not arms:
        return empty_figure("No pitches selected")

    fig = go.Figure()
    # "dot" is reserved for the league-average reference lines below, so a
    # second pitcher cannot be confused for an average.
    dashes = ["solid", "dash", "longdash", "dashdot"]
    shown_types: set[str] = set()
    for i, pid in enumerate([p for p in wanted if p in arms]):
        pitches = arms[pid]
        ref = _reference_pitch(pitches)
        others = {k: v for k, v in pitches.items() if k != ref}
        if ref is None or not others:
            continue
        who = pitches[ref].get("name") or "?"
        multi = len(arms) > 1
        for name, e in sorted(others.items(),
                              key=lambda kv: -(kv[1].get("usage_pct") or 0)):
            ys, d = tunnel_separation(pitches[ref], e)
            at_commit = float(np.interp(COMMIT_FT, ys[::-1], d[::-1]))
            code = _PITCH_NAME_TO_CODE.get(name, "")
            shown_types.add(name)
            fig.add_trace(go.Scatter(
                x=ys, y=d, mode="lines",
                line=dict(color=_PITCH_COLORS.get(code, COLORS["subtext"]),
                          width=2, dash=dashes[i % len(dashes)]),
                name=f"{name} · {who}" if multi else name,
                # Grouped by pitch type so the legend toggles a pitch and its
                # league-average reference together. The average is a property
                # of the pitch type, so with two pitchers on the chart both
                # sliders and the one slider average share the group.
                legendgroup=name,
                hovertemplate=(f"<b>{name}</b> vs {ref}"
                               "<br>%{y:.1f}\" apart at %{x:.0f} ft"
                               f"<br>At the decision point: {at_commit:.1f}\""
                               "<extra></extra>"),
            ))

    if not fig.data:
        return empty_figure("Need at least two pitches to compare")

    # League average for each pitch type on the chart, drawn underneath in the
    # same colour. Without it a curve at 2" reads as remarkable when it may
    # just be what a curveball does.
    league = tunnel_league_average(int(portrait.get("season") or 0))
    grid = _tunnel_grid()
    drew_avg = False
    for name in sorted(shown_types):
        avg = league.get(name)
        if avg is None or not np.isfinite(avg).any():
            continue
        code = _PITCH_NAME_TO_CODE.get(name, "")
        fig.add_trace(go.Scatter(
            x=grid, y=avg, mode="lines",
            line=dict(color=_PITCH_COLORS.get(code, COLORS["subtext"]),
                      width=1, dash="dot"),
            opacity=0.5,
            name=f"{name} — league avg", showlegend=False, legendgroup=name,
            hovertemplate=(f"<b>{name}</b> — league average"
                           "<br>%{y:.1f}\" apart at %{x:.0f} ft<extra></extra>"),
        ))
        drew_avg = True

    # The decision point is the whole reason the curve matters — separation to
    # the left of this line is what the hitter gets to use.
    fig.add_vline(x=COMMIT_FT, line=dict(color="#9ca3af", width=1, dash="dot"))
    # Inside the plot, not above it: the subtitle occupies the space over the
    # axis, and at the decision point every curve is still near the floor, so
    # the top of the panel is free.
    fig.add_annotation(x=COMMIT_FT, yref="paper", y=0.97, yanchor="top",
                       text="decision point", showarrow=False,
                       font=dict(size=9, color="#9ca3af"))

    ref_names = sorted({_reference_pitch(p) for p in arms.values()} - {None})
    fig.update_layout(
        **{**_DARK_LAYOUT, "margin": dict(l=54, r=16, t=78, b=44)},
        title=dict(
            text="Tunnelling — separation from the fastball<br>"
                 f"<sup>vs {', '.join(ref_names)} · lower for longer is a "
                 "better tunnel"
                 + (" · dotted = league average for that pitch" if drew_avg else "")
                 + "</sup><br>"
                 "<sup>everything right of the vertical line is break the "
                 "hitter cannot act on</sup>",
            font=dict(size=13, color=COLORS["text"]), x=0.5,
        ),
        xaxis=dict(
            title="Distance to plate (ft)",
            autorange="reversed",          # release left, plate right
            gridcolor=COLORS["border"], zeroline=False,
            tickfont=dict(color=COLORS["subtext"], size=10),
        ),
        yaxis=dict(
            title="Separation (inches)",
            gridcolor=COLORS["border"], zeroline=False, rangemode="tozero",
            tickfont=dict(color=COLORS["subtext"], size=10),
        ),
        legend=dict(font=dict(color=COLORS["subtext"], size=10),
                    bgcolor="rgba(0,0,0,0)"),
        height=380,
    )
    return fig


def pitch_arsenal_3d(portrait: dict,
                     player_ids: list[int] | None = None,
                     pitch_types: list[str] | None = None) -> go.Figure:
    """
    3-D trajectories for one pitcher's whole arsenal — one curve per pitch.

    It used to draw every pitcher in the organisation who threw the selected
    pitch type: 88 traces on HOU 2025, which is a hairball rather than a
    chart. Inverted, so the unit is a PITCHER and the curves are his pitches.
    That is also the comparison worth making — how a slider leaves the same
    hand as the fastball is the point of the view, and it is invisible when
    twenty-nine pitchers' four-seamers are drawn on top of each other.

    Defaults to the heaviest-workload arm; more can be added from the picker.

    x = horizontal position (ft, catcher's view)
    y = distance from home plate (ft)
    z = height (ft)
    Colour  = pitch type, matching the arsenal column on the roster above
    Width   = usage share within that pitcher's mix
    """
    arsenal = portrait.get("arsenal_trajectories", {})
    by_type = arsenal.get("by_pitch_type", {})
    if not by_type:
        return empty_figure("No pitch trajectory data available")

    choices = arsenal_pitcher_choices(portrait)
    if not choices:
        return empty_figure("No pitch trajectory data available")
    wanted = set(player_ids or [choices[0][0]])

    # Flatten to (pitch_type, entry) for the selected pitchers only.
    selected = [(pt, e) for pt, entries in by_type.items() for e in entries
                if int(e.get("player_id") or -1) in wanted
                and (not pitch_types or pt in pitch_types)]
    if not selected:
        return empty_figure("No pitches selected")

    # Longest label decides whether the pitcher's name is worth repeating.
    multi = len({int(e.get("player_id")) for _, e in selected}) > 1
    names_in_view = list(dict.fromkeys(e.get("name") or "?" for _, e in selected))
    who = " vs ".join(names_in_view) if multi else names_in_view[0]

    fig = go.Figure()
    for pitch_type, p in sorted(selected, key=lambda r: -(r[1].get("usage_pct") or 0)):
        traj = _reconstruct_trajectory(p)
        if traj is None:
            continue
        xs, ys, zs, i_cross = traj

        wr      = p["whiff_rate"]
        usage   = p["usage_pct"]
        # Same palette as the roster's arsenal column, keyed off the long
        # pitch name rather than the code.
        code = _PITCH_NAME_TO_CODE.get(pitch_type, "")
        color = _PITCH_COLORS.get(code, COLORS["subtext"])
        lw      = max(2, min(9, int(usage * 18)))

        label = (
            f"<b>{pitch_type}</b>"
            + (f" — {p['name']}" if multi else "")
            + f"<br>Usage: {usage:.0%}  ({p['pitch_count']} pitches)<br>"
            f"Velo: {p['velo']:.1f} mph<br>"
            f"Whiff: {wr:.1%}<br>"
            + (f"RV/100: {p['run_value_per100']:+.1f}" if p.get("run_value_per100") is not None else "")
        )
        p = {**p, "name": (f"{pitch_type} · {p['name']}" if multi else pitch_type)}

        # Draw trajectory as 3D line — solid up to the plate crossing.
        fig.add_trace(go.Scatter3d(
            x=xs[:i_cross + 1], y=ys[:i_cross + 1], z=zs[:i_cross + 1],
            mode="lines",
            line=dict(color=color, width=lw),
            name=p["name"],
            hovertemplate=label + "<extra></extra>",
            showlegend=True,
        ))

        # The 17 inches over the plate, faded — still the same pitch, but past
        # the plane the kinematics were fit to and past where it was judged.
        if i_cross < len(xs) - 1:
            fig.add_trace(go.Scatter3d(
                x=xs[i_cross:], y=ys[i_cross:], z=zs[i_cross:],
                mode="lines",
                line=dict(color=_hex_to_rgba(color, 0.35), width=lw),
                showlegend=False,
                hoverinfo="skip",
            ))

        # Mark release point
        fig.add_trace(go.Scatter3d(
            x=[xs[0]], y=[ys[0]], z=[zs[0]],
            mode="markers",
            marker=dict(size=4, color=color, symbol="circle"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Mark plate crossing — at i_cross, not at the end of the path, which
        # is now the back tip. This marker is the measured plate_x/plate_z.
        fig.add_trace(go.Scatter3d(
            x=[xs[i_cross]], y=[ys[i_cross]], z=[zs[i_cross]],
            mode="markers",
            marker=dict(size=6, color=color, symbol="square"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Strike zone box at the FRONT OF THE PLATE, at this season's real height.
    #
    # Two things were wrong with it. It was drawn at y=0, but
    # _reconstruct_trajectory solves flight time to y=PLATE_Y (1.4167 ft),
    # which is also where Statcast measures plate_x/plate_z and where the zone
    # is judged — so the zone hung about seventeen inches behind the point
    # every pitch actually ended at, and nothing crossed the plane it was
    # drawn on. And its height was the fixed ABS pair, 3.38/1.59, which is not
    # any season's zone: Statcast sets sz_top/sz_bot per pitch from the
    # batter's stance, and the average moves (3.361/1.594 in 2023,
    # 3.435/1.605 in 2025). Now it is the season's own mean.
    sz_top, sz_bot = strike_zone(int(portrait.get("season") or 0))
    sz_x = [-PLATE_HALF_WIDTH, PLATE_HALF_WIDTH, PLATE_HALF_WIDTH,
            -PLATE_HALF_WIDTH, -PLATE_HALF_WIDTH]
    sz_z = [sz_bot,            sz_bot,           sz_top,
            sz_top,            sz_bot]
    sz_y = [PLATE_Y] * 5
    fig.add_trace(go.Scatter3d(
        x=sz_x, y=sz_y, z=sz_z,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
        name="Strike Zone",
        hovertemplate=(f"<b>Strike zone</b><br>{portrait.get('season', '')} league "
                       f"average<br>Top: {sz_top:.2f} ft<br>Bottom: {sz_bot:.2f} ft"
                       "<br>Width: 17 in (plate)<extra></extra>"),
        showlegend=False,
    ))

    # Home plate, so the zone stands on a real object instead of floating.
    # The two planes are perpendicular and meet along the plate's front edge:
    # the zone is x–z at fixed y=PLATE_Y, the plate is x–y on the ground,
    # running back from that edge to its tip at y=0.
    plate_x, plate_y, plate_z = _home_plate_prism()
    fig.add_trace(go.Mesh3d(
        x=plate_x, y=plate_y, z=plate_z,
        alphahull=0,
        color="#e5e7eb",
        opacity=0.30,
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
    ))
    # Top face outline. The slab alone loses its shape against the dark floor
    # at this camera; the pentagon is the part that says "home plate".
    fig.add_trace(go.Scatter3d(
        x=[v[0] for v in PLATE_VERTS] + [PLATE_VERTS[0][0]],
        y=[v[1] for v in PLATE_VERTS] + [PLATE_VERTS[0][1]],
        z=[PLATE_THICKNESS] * (len(PLATE_VERTS) + 1),
        mode="lines",
        line=dict(color="rgba(255,255,255,0.55)", width=2),
        name="Home plate",
        hovertemplate=("<b>Home plate</b><br>17 in across the front edge<br>"
                       "Front edge at y=1.42 ft<extra></extra>"),
        showlegend=False,
    ))
    # The zone's bottom corners dropped to the plate corners they stand on.
    # Without them the rectangle and the pentagon read as unrelated objects.
    fig.add_trace(go.Scatter3d(
        x=[-PLATE_HALF_WIDTH, -PLATE_HALF_WIDTH, None,
           PLATE_HALF_WIDTH, PLATE_HALF_WIDTH],
        y=[PLATE_Y, PLATE_Y, None, PLATE_Y, PLATE_Y],
        z=[PLATE_THICKNESS, sz_bot, None, PLATE_THICKNESS, sz_bot],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.15)", width=1),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        # Names the pitcher(s), not a pitch — the chart is one arm's arsenal
        # now. Previously this interpolated `pitch_type`, which after the
        # rewrite was whatever the draw loop happened to leave behind.
        title=dict(
            # Two short lines rather than one long one: the card is ~570px
            # wide and a single subtitle carrying both notes ran off both
            # edges and collided with the modebar.
            text=f"Pitch Trajectories — {who}<br>"
                 f"<sup>Catcher's-eye view — LHP release appears right, "
                 f"RHP left</sup><br>"
                 f"<sup>Zone: {portrait.get('season', '')} league average, "
                 f"{sz_bot:.2f}–{sz_top:.2f} ft</sup>",
            font=dict(size=13, color=COLORS["text"]), x=0.5,
        ),
        # Every axis is pinned, and the aspect is fixed in real feet.
        #
        # Only z had a range before; x and y used bare autorange, so they
        # rescaled to whichever pitcher and pitches were selected. The strike
        # zone is drawn at fixed dimensions, so it appeared to change size and
        # shape between selections — the zone was the one thing on the chart
        # that should never move. Without aspectmode Plotly also normalises
        # each axis to its own range independently, which squashed the box
        # regardless.
        scene=dict(
            xaxis=dict(
                title="Horizontal (ft)",
                backgroundcolor=COLORS["surface"],
                gridcolor=COLORS["border"],
                tickfont=dict(color=COLORS["subtext"], size=9),
                # reversed via [max, min] so RHP arm-side stays on the right
                range=[SCENE_X, -SCENE_X],
            ),
            yaxis=dict(
                # Short, because this title is drawn along the receding edge
                # and runs straight into the legend at the full spelling.
                title="To plate (ft)",
                backgroundcolor=COLORS["surface"],
                gridcolor=COLORS["border"],
                tickfont=dict(color=COLORS["subtext"], size=9),
                range=[SCENE_Y, 0],     # release at back, plate at front
            ),
            zaxis=dict(
                title="Height (ft)",
                backgroundcolor=COLORS["surface"],
                gridcolor=COLORS["border"],
                tickfont=dict(color=COLORS["subtext"], size=9),
                range=[0, SCENE_Z],
            ),
            aspectmode="manual",
            aspectratio=dict(x=ASPECT_X, y=ASPECT_Y, z=ASPECT_Z),
            # An actual catcher's-eye view: on the centre line, behind the
            # plate, looking out at the mound.
            #
            # The old eye was (0, -1.8, 0.5) against a y half-extent of 2.0,
            # which put the camera INSIDE the scene box. That is what produced
            # the wide-angle three-quarter view the subtitle kept calling a
            # catcher's-eye view — the trajectories swept across the frame
            # sideways instead of coming at you, and the zone sat off-centre.
            #
            # y is positive because the y axis is drawn reversed (range
            # [SCENE_Y, 0]) to put the plate at the near end, so the plate
            # side of the box is +y in camera space, not -y. Sitting
            # EYE_GAP beyond the near face keeps the whole box in frame; any
            # closer and the axis titles collide with each other.
            camera=dict(
                eye=dict(x=0.0, y=ASPECT_Y + EYE_GAP, z=0.12),
            ),
            bgcolor=COLORS["surface"],
        ),
        legend=dict(
            font=dict(color=COLORS["subtext"], size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        # l/b leave room for the "Height (ft)" and "Horizontal (ft)" titles,
        # which the straight-on camera pushes to the very edge of the scene.
        margin=dict(l=44, r=0, t=68, b=10),
        height=520,
    )
    return fig
