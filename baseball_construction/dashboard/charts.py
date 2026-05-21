"""
charts.py — Pure Plotly figure builders.

Each function takes a portrait dict (or subset) and returns a plotly Figure.
No Dash imports. No side effects. Easy to test in isolation.
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import plotly.express as px


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
    "Pure Power":           "#f97316",
    "Pure Contact":         "#22c55e",
    "Undetermined":         "#6b7280",
    # Pitcher type colors
    "P1": "#ef4444",
    "P2": "#8b5cf6",
    "P3": "#10b981",
    "P4": "#f59e0b",
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
    "A4": "Power Concentration",
    "B1": "Stuff Dominant",
    "B2": "Command+Contact Mgmt",
    "B3": "Pitch Design",
    "B4": "Defensive Infra",
    "C1": "Star+Support",
    "C2": "Roster Balance",
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
    Horizontal bar chart: one bar per philosophy dimension showing
    top score and primary philosophy label.
    """
    confidences = portrait.get("philosophy", {}).get("confidences", {})
    scores_data  = portrait.get("philosophy", {}).get("scores", {})

    dims   = ["offense", "pitching", "roster"]
    labels = []
    top_scores   = []
    sec_scores   = []
    confs        = []
    primary_names = []
    bar_colors   = []

    for dim in dims:
        conf  = confidences.get(dim, {})
        primary_code  = conf.get("primary")
        secondary_code = conf.get("secondary")

        top_s = conf.get("top_score") or 0.0
        sec_s = conf.get("second_score") or 0.0
        disp  = conf.get("display_confidence") or 0

        pname = scores_data.get(primary_code, {}).get("name", primary_code or "—")

        labels.append(dim.title())
        top_scores.append(top_s)
        sec_scores.append(sec_s)
        confs.append(disp)
        primary_names.append(pname)
        bar_colors.append(COLORS.get(dim, COLORS["primary"]))

    fig = go.Figure()

    # Secondary score (lighter, behind)
    fig.add_trace(go.Bar(
        x=sec_scores,
        y=labels,
        orientation="h",
        name="2nd philosophy",
        marker_color=[_hex_to_rgba(c, 0.33) for c in bar_colors],
        hovertemplate="%{x:.1f} (secondary)<extra></extra>",
    ))

    # Primary score
    fig.add_trace(go.Bar(
        x=top_scores,
        y=labels,
        orientation="h",
        name="Top philosophy",
        marker_color=bar_colors,
        text=[f"{n} ({c}%)" for n, c in zip(primary_names, confs)],
        textposition="inside",
        insidetextanchor="start",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="%{x:.1f} | conf %{text}<extra></extra>",
    ))

    fig.update_layout(
        **_DARK_LAYOUT,
        barmode="overlay",
        xaxis=dict(range=[0, 100], gridcolor=COLORS["border"], title="Score"),
        yaxis=dict(gridcolor=COLORS["border"]),
        legend=dict(
            orientation="h", y=-0.15,
            font=dict(color=COLORS["subtext"], size=10),
        ),
        title=dict(
            text="Primary Philosophy by Dimension",
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

    names, ages, pas, types, spectrums, modifiers_col = [], [], [], [], [], []

    for h in sorted(hitters, key=lambda x: -(x.get("pa") or 0)):
        pri  = h.get("primary", {})
        mods = h.get("modifiers", {})

        names.append(h.get("name") or f"ID {h.get('player_id','?')}")
        ages.append(str(h.get("age")) if h.get("age") else "—")
        pas.append(str(h.get("pa") or "—"))
        types.append(pri.get("type", "—"))
        sp = pri.get("spectrum_score")
        spectrums.append(f"{sp:.1f}" if sp is not None else "—")

        mod_parts = []
        if mods.get("free_swinger"):
            mod_parts.append("FS")
        if mods.get("speed"):
            mod_parts.append("SPD")
        disrupt = mods.get("disruptiveness", {}) or {}
        if disrupt.get("modifier"):
            mod_parts.append(disrupt["modifier"][:3].upper())
        modifiers_col.append(", ".join(mod_parts) or "—")

    fig = go.Figure(go.Table(
        header=dict(
            values=["Player", "Age", "PA", "Archetype", "Spectrum", "Modifiers"],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["subtext"], size=11),
            align="left",
            line_color=COLORS["border"],
        ),
        cells=dict(
            values=[names, ages, pas, types, spectrums, modifiers_col],
            fill_color=COLORS["background"],
            font=dict(color=COLORS["text"], size=11),
            align="left",
            line_color=COLORS["border"],
        ),
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(
            text="Hitter Roster Detail",
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
        "P3": "GB Craftsman", "P4": "Stuff-to-Contact", "U0": "Unclassified",
    }

    names, ages, bfs, archetypes, commands = [], [], [], [], []

    for s in sorted(starters, key=lambda x: -(x.get("bf") or 0)):
        pri = s.get("primary", {})
        code = pri.get("type_code", "U0")
        names.append(s.get("name") or f"ID {s.get('player_id','?')}")
        ages.append(str(s.get("age")) if s.get("age") else "—")
        bfs.append(str(s.get("bf") or "—"))
        archetypes.append(CODE_NAMES.get(code, code))
        commands.append(s.get("modifiers", {}).get("command") or "—")

    fig = go.Figure(go.Table(
        header=dict(
            values=["Pitcher", "Age", "BF", "Archetype", "Command"],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["subtext"], size=11),
            align="left",
            line_color=COLORS["border"],
        ),
        cells=dict(
            values=[names, ages, bfs, archetypes, commands],
            fill_color=COLORS["background"],
            font=dict(color=COLORS["text"], size=11),
            align="left",
            line_color=COLORS["border"],
        ),
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(
            text="Starter Roster Detail",
            font=dict(size=14, color=COLORS["text"]),
            x=0.5,
        ),
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
