"""
charts.py — Plotly figure builders and Dash component helpers.

Most functions return a plotly Figure; philosophy_breakdown_card returns a
Dash html.Div component tree (for use in collapsible breakdown panels).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import html

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

# ---------------------------------------------------------------------------
# ABS (Automated Ball-Strike) standardized strike zone
# Width : ±0.833 ft  — 17" home plate + ½ ball radius (1.44" dia) each side
# Top   :  3.38 ft  — league-average sz_top (~53.5 % of avg MLB height 6'1")
# Bottom:  1.59 ft  — league-average sz_bot (~26.7 % of avg MLB height 6'1")
# These replace the old round-number fallbacks of 3.5 / 1.5 ft.
# ---------------------------------------------------------------------------
ABS_SZ_TOP    = 3.38   # ft
ABS_SZ_BOT    = 1.59   # ft
ABS_SZ_WIDTH  = 0.833  # ft (half-width from center)


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

    for h in sorted(hitters, key=lambda x: -(x.get("pa") or 0)):
        name = (h.get("name") or "").strip() or f"ID {h.get('player_id', '?')}"
        war  = h.get("war")

        names.append(name)
        poss.append(h.get("home_position") or "—")
        ages.append(str(h.get("age")) if h.get("age") else "—")
        pas.append(str(h.get("pa") or "—"))
        wars.append(f"{war:.1f}" if war is not None else "—")

        # Spectrum: 0 = extreme contact, 100 = extreme power
        spec = h.get("spectrum")
        specs.append(f"{spec:.0f}" if spec is not None else "—")

        tags = [t["tag"] for t in h.get("traits") or []]
        traits_col.append(", ".join(tags) if tags else "—")

    n = len(names)
    row_colors = [COLORS["surface"] if i % 2 == 0 else COLORS["background"] for i in range(n)]

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
            font=dict(color=COLORS["text"], size=11),
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
        traits_col.append(", ".join(shown) if shown else "—")

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
        traits_col.append(", ".join(shown) if shown else "—")

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
    colors = [TRAIT_FAMILY_COLORS.get(density[t][1], COLORS["neutral"]) for t in tags]

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

    fig.update_layout(
        **{**_DARK_LAYOUT, "margin": dict(l=118, r=42, t=26, b=30)},
        barmode="overlay",
        showlegend=bool(baseline),
        legend=dict(orientation="h", x=0, y=1.06,
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


def team_drift_chart(team: str, unit: str,
                     seasons_data: dict[int, dict]) -> go.Figure:
    """
    "Through the Years" — one unit's identity drift across seasons.

    seasons_data — {year: league_identity dict for that season}. For each
    season we compute deviation = team density − league baseline per tag,
    then draw the team's most-defining tags (largest |deviation| anywhere
    in the window) as lines against the league zero-line.
    """
    # Deviation matrix: tag -> {year: (dev_pp, density)}
    dev: dict[str, dict[int, tuple[float, float]]] = {}
    for yr, league in sorted(seasons_data.items()):
        ident = (league.get(team) or {}).get(unit) or {}
        density = ident.get("trait_density") or {}
        base = (league.get("_baselines") or {}).get(unit) or {}
        for tag in set(density) | set(base):
            d = float(density.get(tag, 0) or 0)
            b = float(base.get(tag, 0) or 0)
            if d < 0.02 and b < 0.02:
                continue   # micro-tags: noise, not identity
            dev.setdefault(tag, {})[yr] = ((d - b) * 100, d * 100)

    if not dev:
        return empty_figure("No multi-season data for this team")

    # The team's defining tags: largest |deviation| anywhere, seen ≥3 seasons
    ranked = sorted(
        (t for t in dev if len(dev[t]) >= 3),
        key=lambda t: -max(abs(v[0]) for v in dev[t].values()),
    )[:6]

    years = sorted(seasons_data.keys())
    fig = go.Figure()
    fig.add_hline(y=0, line_color=COLORS["subtext"], line_width=1,
                  line_dash="dot",
                  annotation_text="league average",
                  annotation_font=dict(size=10, color=COLORS["subtext"]),
                  annotation_position="bottom right")

    # Family color per tag — falls back to a rotating neutral-safe palette
    _fallback = ["#60a5fa", "#f59e0b", "#34d399", "#a78bfa", "#f472b6", "#22d3ee"]
    for i, tag in enumerate(ranked):
        fam = None
        # family isn't stored in the density map; color via known tag→family
        color = TRAIT_TAG_COLORS.get(tag) or _fallback[i % len(_fallback)]
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


def team_identity_card(portrait: dict, fingerprint: dict | None = None) -> html.Div:
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

    offense_sub = None
    if offense.get("power_share") is not None or offense.get("trait_density"):
        # Show the raw share mix so the data speaks for itself —
        # complete hitters are their own bucket, not folded into either side.
        parts = [f"Power {offense.get('power_share', 0):.0%}",
                 f"Contact {offense.get('contact_share', 0):.0%}"]
        if offense.get("complete_share"):
            parts.append(f"Complete {offense['complete_share']:.0%}")
        offense_sub = " · ".join(parts) + " of PA"

    rotation_sub = None
    density = rotation.get("trait_density") or {}
    if density:
        # Top trait densities, e.g. "tunneler 52% · bat-misser 40%"
        top = list(density.items())[:2]
        rotation_sub = " · ".join(f"{t} {v:.0%}" for t, v in top)
    elif rotation.get("approach_dist"):
        appr = rotation.get("dominant_approach")
        share = rotation["approach_dist"].get(appr, 0)
        rotation_sub = f"{appr} ({share:.0%} of BF)" if appr else None

    bullpen_label = bullpen.get("out_mechanism")
    bullpen_sub = bullpen.get("leverage_structure")

    cols = dbc.Row([
        _identity_col("Offense",  offense.get("label"),  offense_sub,  COLORS["offense"]),
        _identity_col("Rotation", rotation.get("label"), rotation_sub, COLORS["pitching"]),
        _identity_col("Bullpen",  bullpen_label,         bullpen_sub,  COLORS["roster"]),
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
            for e in (fingerprint.get(unit) or []):
                dev = e.get("deviation", 0) * 100
                breadth = (f" · {e['carriers']}/{e['qualifiers']} regulars"
                           if e.get("qualifiers") else "")
                lines.append(html.Div([
                    html.Span(label, style={"color": color, "fontWeight": "600",
                                            "fontSize": "0.7rem",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "0.05em",
                                            "marginRight": "6px"}),
                    html.Span(e.get("tag", ""), style={"color": "#f9fafb",
                                                       "fontWeight": "600",
                                                       "fontSize": "0.78rem"}),
                    html.Span(f" {dev:+.0f} pts vs league{breadth}",
                              className="text-secondary",
                              style={"fontSize": "0.72rem"}),
                ], className="mb-1"))
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
    entry: dict, n_points: int = 30
) -> tuple[list, list, list] | None:
    """
    Reconstruct 3-D flight path using Statcast kinematic parameters.
    Returns (x_path, y_path, z_path) or None if physics are invalid.

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

        # Solve for time of flight: y0 + vy0·t + ½·ay·t² = 1.4167 (front of plate)
        a_c = 0.5 * ay_
        b_c = vy0
        c_c = y0 - 1.4167
        disc = b_c**2 - 4 * a_c * c_c
        if disc < 0:
            return None
        t_flight = (-b_c - np.sqrt(disc)) / (2 * a_c)
        if t_flight <= 0 or t_flight > 0.65:
            return None

        t = np.linspace(0, t_flight, n_points)
        xs = (x0 + vx0 * t + 0.5 * ax_ * t**2).tolist()
        ys = (y0 + vy0 * t + 0.5 * ay_ * t**2).tolist()
        zs = (z0 + vz0 * t + 0.5 * az_ * t**2).tolist()
        return xs, ys, zs
    except Exception:
        return None


def pitch_arsenal_3d(portrait: dict, pitch_type: str | None = None) -> go.Figure:
    """
    3-D trajectory chart — one curve per pitcher for the selected pitch type.

    x = horizontal position (ft, catcher's view)
    y = distance from home plate (ft)
    z = height (ft)
    Colour ramp = whiff rate (darker = more swing-and-miss)
    Line width  = proportional to usage %
    """
    arsenal = portrait.get("arsenal_trajectories", {})
    by_type = arsenal.get("by_pitch_type", {})
    available = arsenal.get("pitch_types", [])

    if not by_type:
        return empty_figure("No pitch trajectory data available")

    # Default to first pitch type if none selected or invalid
    if not pitch_type or pitch_type not in by_type:
        pitch_type = available[0] if available else None
    if not pitch_type:
        return empty_figure("No pitch types found")

    pitchers = by_type[pitch_type]
    if not pitchers:
        return empty_figure(f"No data for {pitch_type}")

    fig = go.Figure()

    # Whiff rate colour scale: 0 = muted, 1 = bright
    whiff_rates = [p["whiff_rate"] for p in pitchers]
    max_whiff   = max(whiff_rates) if whiff_rates else 0.40
    min_whiff   = min(whiff_rates) if whiff_rates else 0.00

    def _whiff_color(wr: float) -> str:
        """Map whiff rate to a blue-green-yellow colour."""
        t = (wr - min_whiff) / (max_whiff - min_whiff + 0.001)
        # Interpolate: low whiff = steel blue, high whiff = bright yellow
        r = int(30  + t * 225)
        g = int(144 + t * 60)
        b = int(255 - t * 200)
        return f"rgb({r},{g},{b})"

    for p in pitchers:
        traj = _reconstruct_trajectory(p)
        if traj is None:
            continue
        xs, ys, zs = traj

        wr      = p["whiff_rate"]
        usage   = p["usage_pct"]
        color   = _whiff_color(wr)
        lw      = max(2, min(8, int(usage * 16)))   # line width 2–8 px

        label = (
            f"<b>{p['name']}</b> ({p['p_throws']}HP)<br>"
            f"Pitches: {p['pitch_count']}  Usage: {usage:.0%}<br>"
            f"Velo: {p['velo']:.1f} mph<br>"
            f"Whiff: {wr:.1%}<br>"
            + (f"RV/100: {p['run_value_per100']:+.1f}" if p.get("run_value_per100") is not None else "")
        )

        # Draw trajectory as 3D line
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=lw),
            name=p["name"],
            hovertemplate=label + "<extra></extra>",
            showlegend=True,
        ))

        # Mark release point
        fig.add_trace(go.Scatter3d(
            x=[xs[0]], y=[ys[0]], z=[zs[0]],
            mode="markers",
            marker=dict(size=4, color=color, symbol="circle"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Mark plate crossing
        fig.add_trace(go.Scatter3d(
            x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
            mode="markers",
            marker=dict(size=6, color=color, symbol="square"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Strike zone reference box at y=0 (home plate face) — ABS standard dimensions
    sz_x = [-ABS_SZ_WIDTH, ABS_SZ_WIDTH, ABS_SZ_WIDTH, -ABS_SZ_WIDTH, -ABS_SZ_WIDTH]
    sz_z = [ABS_SZ_BOT,    ABS_SZ_BOT,   ABS_SZ_TOP,   ABS_SZ_TOP,    ABS_SZ_BOT]
    sz_y = [0.0] * 5
    fig.add_trace(go.Scatter3d(
        x=sz_x, y=sz_y, z=sz_z,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
        name="Strike Zone",
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        title=dict(
            text=f"Pitch Trajectories — {pitch_type}<br>"
                 f"<sup>Catcher's-eye view (behind home plate, looking out) — "
                 f"LHP arm-side release appears right, RHP appears left</sup>",
            font=dict(size=13, color=COLORS["text"]), x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="Horizontal (ft)",
                backgroundcolor=COLORS["surface"],
                gridcolor=COLORS["border"],
                tickfont=dict(color=COLORS["subtext"], size=9),
                autorange="reversed",   # flip so RHP arm-side renders on the right (pitcher's-eye view)
            ),
            yaxis=dict(
                title="Distance to plate (ft)",
                backgroundcolor=COLORS["surface"],
                gridcolor=COLORS["border"],
                tickfont=dict(color=COLORS["subtext"], size=9),
                autorange="reversed",   # release at back, plate at front
            ),
            zaxis=dict(
                title="Height (ft)",
                backgroundcolor=COLORS["surface"],
                gridcolor=COLORS["border"],
                tickfont=dict(color=COLORS["subtext"], size=9),
                range=[0, 7],
            ),
            camera=dict(
                eye=dict(x=0.0, y=-1.8, z=0.5),   # roughly catcher's POV
            ),
            bgcolor=COLORS["surface"],
        ),
        legend=dict(
            font=dict(color=COLORS["subtext"], size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig
