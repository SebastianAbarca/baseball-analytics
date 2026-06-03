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
    "P5": "#06b6d4",   # cyan — Power Sinker (distinct from P3 green and P4 amber)
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

def hitter_archetype_pie(portrait: dict) -> go.Figure:
    """
    Pie chart of hitter archetype types on the roster.
    """
    hitters = portrait.get("players", {}).get("hitters", [])

    counts: dict[str, int] = {}
    for h in hitters:
        t = h.get("primary", {}).get("type", "Undetermined")
        counts[t] = counts.get(t, 0) + 1

    if not counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No hitter data", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=COLORS["subtext"], size=14),
        )
        fig.update_layout(**_DARK_LAYOUT)
        return fig

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [COLORS.get(l, COLORS["neutral"]) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color=COLORS["background"], width=2)),
        hovertemplate="<b>%{label}</b><br>%{value} players (%{percent})<extra></extra>",
        textfont=dict(size=12, color=COLORS["text"]),
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        showlegend=True,
        legend=dict(font=dict(color=COLORS["subtext"])),
        title=dict(
            text="Hitter Archetype Distribution",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Hitter archetype table
# ---------------------------------------------------------------------------

def hitter_archetype_table(portrait: dict) -> go.Figure:
    """
    Table: one row per hitter with archetype, confidence, modifiers.
    """
    hitters = portrait.get("players", {}).get("hitters", [])

    if not hitters:
        fig = go.Figure(go.Table(
            header=dict(values=["No hitter data"], fill_color=COLORS["surface"]),
            cells=dict(values=[[]], fill_color=COLORS["background"]),
        ))
        fig.update_layout(**_DARK_LAYOUT)
        return fig

    names, ages, pas, wars, types, modifiers_col = [], [], [], [], [], []

    for h in sorted(hitters, key=lambda x: -(x.get("pa") or 0)):
        pri  = h.get("primary", {})
        mod  = h.get("modifiers", {})
        name = (h.get("name") or "").strip() or f"ID {h.get('player_id', '?')}"
        war  = h.get("war")

        names.append(name)
        ages.append(str(h.get("age")) if h.get("age") else "—")
        pas.append(str(h.get("pa") or "—"))
        wars.append(f"{war:.1f}" if war is not None else "—")
        types.append(pri.get("type") or "—")

        # Build modifier string: speed tier + flags
        tags = []
        speed = mod.get("speed")
        if speed in ("Elite", "Fast"):
            tags.append(speed)
        elif speed == "Slow":
            tags.append("Slow")
        if mod.get("aggressive"):
            tags.append("Aggressive")
        if mod.get("free_swinger"):
            tags.append("Free Swinger")
        disrupt = mod.get("disruptiveness", {}).get("modifier")
        if disrupt:
            tags.append(disrupt)
        modifiers_col.append(", ".join(tags) if tags else "—")

    n = len(names)
    row_colors = [COLORS["surface"] if i % 2 == 0 else COLORS["background"] for i in range(n)]

    fig = go.Figure(go.Table(
        columnwidth=[3, 1, 1, 1, 2, 2],
        header=dict(
            values=["<b>Player</b>", "<b>Age</b>", "<b>PA</b>",
                    "<b>bWAR</b>", "<b>Archetype</b>", "<b>Modifiers</b>"],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["subtext"], size=11),
            align=["left", "center", "center", "center", "left", "left"],
            line_color=COLORS["border"],
            height=32,
        ),
        cells=dict(
            values=[names, ages, pas, wars, types, modifiers_col],
            fill_color=[row_colors] * 6,
            font=dict(color=COLORS["text"], size=11),
            align=["left", "center", "center", "center", "left", "left"],
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
    primary_types   = []

    for h in hitters_sorted:
        name    = (h.get("name") or f"ID {h.get('player_id', '?')}").strip()
        pa      = h.get("pa") or 0
        arc     = h.get("primary", {}).get("type", "")
        metrics = h.get("metrics_pct", {})

        row = []
        for _, positives, negatives in _ARCHETYPE_AFFINITIES:
            score = _archetype_affinity_score(metrics, positives, negatives)
            row.append(score)

        # Only include players with at least one affinity score
        if any(v is not None for v in row):
            player_labels.append(f"{name}  ({pa} PA)")
            primary_types.append(arc)
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
            text="Hitter Archetype Affinity Scores  (0–100, historical percentile)",
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
        arc  = h.get("primary", {}).get("type", "?")
        if pid is not None:
            options.append({
                "label": f"{name}  ({arc}, {pa} PA)",
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
        arc  = h.get("primary", {}).get("type", "")

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
        arc  = h.get("primary", {}).get("type", "")
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
    Table: one row per starter with name, age, BF, archetype.
    """
    starters = portrait.get("players", {}).get("starters", [])

    if not starters:
        return empty_figure("No starter data — load a full-season portrait")

    CODE_NAMES = {
        "P1": "Power Ace", "P2": "Craft Strikeout",
        "P3": "GB Craftsman", "P4": "Stuff-to-Contact",
        "P5": "Power Sinker", "P6": "Finesse Control", "U0": "Unclassified",
    }

    names, ages, bfs, wars, archetypes, commands = [], [], [], [], [], []

    for s in sorted(starters, key=lambda x: -(x.get("bf") or 0)):
        pri  = s.get("primary", {})
        code = pri.get("type_code", "U0")
        name = (s.get("name") or "").strip() or f"ID {s.get('player_id', '?')}"
        war  = s.get("war")
        names.append(name)
        ages.append(str(s.get("age")) if s.get("age") else "—")
        bfs.append(str(s.get("bf") or "—"))
        wars.append(f"{war:.1f}" if war is not None else "—")
        archetypes.append(CODE_NAMES.get(code, code) or "—")
        commands.append(s.get("modifiers", {}).get("command") or "—")

    n = len(names)
    row_colors = [COLORS["surface"] if i % 2 == 0 else COLORS["background"] for i in range(n)]

    fig = go.Figure(go.Table(
        columnwidth=[3, 1, 1, 1, 2, 1],
        header=dict(
            values=["<b>Pitcher</b>", "<b>Age</b>", "<b>BF</b>",
                    "<b>bWAR</b>", "<b>Archetype</b>", "<b>Command</b>"],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["subtext"], size=11),
            align=["left", "center", "center", "center", "left", "center"],
            line_color=COLORS["border"],
            height=32,
        ),
        cells=dict(
            values=[names, ages, bfs, wars, archetypes, commands],
            fill_color=[row_colors] * 6,
            font=dict(color=COLORS["text"], size=11),
            align=["left", "center", "center", "center", "left", "center"],
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
    velos, ks, whiffs, gbs, hcs, bars = [], [], [], [], [], []

    for arm in rows:
        m = arm.get("metrics_pct", {})
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
        bars.append(_fmt(m.get("Barrel_allowed_pct")))

    n = len(rows)
    row_colors = [COLORS["surface"] if i % 2 == 0 else COLORS["background"] for i in range(n)]

    fig = go.Figure(go.Table(
        columnwidth=[3, 1, 1, 1, 1, 1, 1, 1, 1],
        header=dict(
            values=[
                "<b>Reliever</b>", "<b>Age</b>", "<b>BF</b>", "<b>bWAR</b>",
                "<b>Velo%</b>", "<b>K%</b>", "<b>Whiff%</b>",
                "<b>GB%</b>", "<b>HC Supp%</b>",
            ],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["subtext"], size=11),
            align=["left", "center", "center", "center",
                   "center", "center", "center", "center", "center"],
            line_color=COLORS["border"],
            height=32,
        ),
        cells=dict(
            values=[names, ages, bfs, wars, velos, ks, whiffs, gbs, hcs],
            fill_color=[row_colors] * 9,
            font=dict(color=COLORS["text"], size=11),
            align=["left", "center", "center", "center",
                   "center", "center", "center", "center", "center"],
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


# ---------------------------------------------------------------------------
# 5c. Bullpen dimension bars (collective profile)
# ---------------------------------------------------------------------------

def bullpen_dimension_bars(portrait: dict) -> go.Figure:
    """
    Horizontal bars showing the team's collective bullpen scores on each
    available dimension vs. the league.
    """
    bp = portrait.get("players", {}).get("bullpen_profile", {})

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
    "AvgTenure_inv_pct":     "Avg MLB Tenure (inv.)",
    "NewPlayerShare_pct":    "Rookies / Sophs Share (≤2 yrs)",
    # C4
    "AvgTenure_pct":         "Avg MLB Tenure",
    "VeteranShare_pct":      "Veteran Share (≥5 yrs)",
}

METRIC_MISSING_REASON: dict[str, str] = {
    # C2 — requires prior-year roster data
    "CoreRetention_pct": "Requires prior-season roster data (not yet implemented)",
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

def empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=COLORS["subtext"], size=14),
    )
    fig.update_layout(**_DARK_LAYOUT)
    return fig
