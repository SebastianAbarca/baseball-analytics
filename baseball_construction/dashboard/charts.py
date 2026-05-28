"""
charts.py — Plotly figure builders and Dash component helpers.

Most functions return a plotly Figure; philosophy_breakdown_card returns a
Dash html.Div component tree (for use in collapsible breakdown panels).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

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
    "C1": "bWAR Distribution",
    "C3": "Youth+Dev",
    "C4": "Veteran Experience",
}

DIM_CODES = {
    "offense":  ["A1", "A2", "A3", "A4"],
    "pitching": ["B1", "B2", "B3", "B4"],
    "roster":   ["C1", "C3", "C4"],
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
    all_codes = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C3", "C4"]
    dim_color_map = {
        "offense":  COLORS["offense"],
        "pitching": COLORS["pitching"],
        "roster":   COLORS["roster"],
    }

    labels, values, colors, hover = [], [], [], []
    for code in all_codes:
        s = scores_data.get(code, {})
        score = s.get("score")
        name  = s.get("name") or PHIL_LABELS.get(code, code)
        dim   = s.get("dimension", "offense")
        cov   = s.get("coverage", 0)
        if score is None:
            continue
        labels.append(f"{code}")
        values.append(float(score))
        colors.append(dim_color_map.get(dim, COLORS["primary"]))
        hover.append(f"<b>{name}</b><br>Score: {score:.1f}<br>Data coverage: {cov:.0%}")

    if not labels:
        return empty_figure("No philosophy scores available")

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
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
        x=50, y=len(labels) - 0.3, text="Avg",
        showarrow=False, font=dict(color=COLORS["subtext"], size=9),
    )

    layout = {**_DARK_LAYOUT}
    layout["margin"] = dict(l=40, r=40, t=40, b=24)
    fig.update_layout(
        **layout,
        xaxis=dict(range=[0, 115], gridcolor=COLORS["border"],
                   title="Score (0–100, league percentile)"),
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

    names, ages, pas, wars, types, spectrums = [], [], [], [], [], []

    for h in sorted(hitters, key=lambda x: -(x.get("pa") or 0)):
        pri  = h.get("primary", {})
        name = (h.get("name") or "").strip() or f"ID {h.get('player_id', '?')}"
        war  = h.get("war")

        names.append(name)
        ages.append(str(h.get("age")) if h.get("age") else "—")
        pas.append(str(h.get("pa") or "—"))
        wars.append(f"{war:.1f}" if war is not None else "—")
        types.append(pri.get("type") or "—")
        sp = pri.get("spectrum_score")
        spectrums.append(f"{sp:.1f}" if sp is not None else "—")

    fig = go.Figure(go.Table(
        columnwidth=[3, 1, 1, 1, 2, 1],
        header=dict(
            values=["<b>Player</b>", "<b>Age</b>", "<b>PA</b>",
                    "<b>bWAR</b>", "<b>Archetype</b>", "<b>Spectrum</b>"],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["subtext"], size=11),
            align=["left", "center", "center", "center", "left", "center"],
            line_color=COLORS["border"],
            height=32,
        ),
        cells=dict(
            values=[names, ages, pas, wars, types, spectrums],
            fill_color=COLORS["background"],
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
        archetypes.append(CODE_NAMES.get(code, code))
        commands.append(s.get("modifiers", {}).get("command") or "—")

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
            fill_color=COLORS["background"],
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
    "WAR_concentration_pct": "bWAR Concentration",
    "wRCplus_spread_pct":    "wRC+ Spread",
    "ISO_spread_pct":        "ISO Spread",
    "PA_concentration_pct":  "PA Concentration",
    # B1
    "TeamK_pct_pct":         "Team Strikeout Rate",
    "AvgVelo_pct":           "Avg Fastball Velocity",
    "SwStr_pct_pct":         "Swinging Strike Rate",
    "AvgSpinRate_pct":       "Avg Spin Rate",
    # B2
    "BB_pitch_inv_pct":      "Walk Rate (inv.)",
    "Zone_pct_pct":          "Zone Rate",
    "GB_pct_pct":            "Ground Ball Rate",
    "TeamDefense_pct":       "Team Defense (OAA proxy)",
    # B3
    "SpinEfficiency_pct":    "Spin Efficiency",
    "ArsenalDiversity_pct":  "Arsenal Diversity",
    "PlatoonOptimization_pct": "Platoon Optimization",
    "OpenerUsage_pct":       "Opener Usage",
    "CSW_pct_pct":           "Called Strike + Whiff%",
    # B4
    "OAA_pct":               "Outs Above Average",
    "DRS_pct":               "Def. Runs Saved (OAA proxy)",
    "GB_pitch_pct":          "Ground Ball Rate (pitch.)",
    "ParkPitcherFriendly_pct": "Park Pitcher-Friendliness",
    # C1
    "WAR_concentration_pct": "bWAR Concentration",
    "WAR_variance_inv_pct":  "bWAR Variance (inv.)",
    "RosterFloor_pct":       "Roster Floor (bWAR > 0)",
    # C3
    "AvgAge_inv_pct":        "Average Age (inv.)",
    "PreArb_share_pct":      "Pre-Arb Share",
    "Turnover_pct":          "Roster Turnover",
    "Pipeline_pct":          "Prospect Pipeline",
    # C4
    "AvgAge_pct":            "Average Age (veteran proxy)",
    "PostArb_share_pct":     "Post-Arb Share",
    "CoreRetention_pct":     "Core Player Retention",
    "AvgTenure_pct":         "Average Roster Tenure",
}

METRIC_MISSING_REASON: dict[str, str] = {
    # C3
    "PreArb_share_pct":  "Requires service time data (not publicly available)",
    "Pipeline_pct":      "Requires MLB Pipeline prospect rankings (not publicly available)",
    # C4
    "PostArb_share_pct": "Requires service time data (not publicly available)",
    "AvgTenure_pct":     "Requires service time data (not publicly available)",
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

    header_style = {"color": COLORS["subtext"], "fontSize": "0.72rem",
                    "textTransform": "uppercase", "letterSpacing": "0.06em",
                    "fontWeight": "600"}
    return html.Div([
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
        html.Small(
            f"Composite score: {score_str} · Data coverage: {coverage:.0%}",
            className="text-secondary mt-2 d-block",
            style={"fontSize": "0.75rem", "padding": "4px 0 8px 4px"},
        ),
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
