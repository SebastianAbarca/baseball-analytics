"""
layout.py — Bootstrap component builders for the dashboard layout.

All functions return Dash/DBC component trees. No callbacks here.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

CARD_STYLE  = {"backgroundColor": "#1f2937", "border": "1px solid #374151", "borderRadius": "8px"}
BADGE_STYLE = {"fontSize": "0.75rem", "fontWeight": "600", "letterSpacing": "0.05em"}

MODE_COLORS = {
    "historical": "secondary",
    "early":      "warning",
    "current":    "success",
}


# ---------------------------------------------------------------------------
# Navbar
# ---------------------------------------------------------------------------

def navbar() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(
                        src="https://upload.wikimedia.org/wikipedia/commons/a/a6/MLB_Logo.svg",
                        height="28px", className="me-2",
                    )),
                    dbc.Col(dbc.NavbarBrand(
                        "Baseball Construction", className="fw-bold text-white",
                    )),
                ], align="center", className="g-0"),
                href="#", style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            html.Small(
                "Team Construction Analytics · Statcast Era (2015+)",
                className="text-secondary ms-auto d-none d-md-block",
                style={"fontSize": "0.75rem"},
            ),
        ], fluid=True),
        color="#111827",
        dark=True,
        className="mb-3 border-bottom border-secondary",
    )


# ---------------------------------------------------------------------------
# Controls row (team selector + season)
# ---------------------------------------------------------------------------

MLB_TEAMS = [
    "ATH", "ATL", "AZ", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS",
    "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]

SEASONS = list(range(2015, 2027))


def controls_row() -> dbc.Row:
    return dbc.Row([
        dbc.Col([
            html.Label("Team", className="text-secondary small mb-1"),
            dbc.Select(
                id="team-dropdown",
                options=[{"label": t, "value": t} for t in MLB_TEAMS],
                value="HOU",
            ),
        ], md=3, xs=6),
        dbc.Col([
            html.Label("Season", className="text-secondary small mb-1"),
            dbc.Select(
                id="season-dropdown",
                options=[{"label": str(s), "value": s} for s in SEASONS],
                value="2023",
            ),
        ], md=2, xs=6),
        dbc.Col([
            html.Label(" ", className="text-secondary small mb-1 d-block"),
            dbc.Button(
                "Load Portrait", id="load-btn", color="primary", className="w-100",
            ),
        ], md=2, xs=12),
        dbc.Col([
            html.Div(id="status-banner", className="mt-1"),
        ], md=5, xs=12),
    ], className="mb-3 align-items-end")


# ---------------------------------------------------------------------------
# Status / header badges
# ---------------------------------------------------------------------------

def team_header(portrait: dict) -> html.Div:
    """
    Team name + season + temporal mode badge + data coverage bar.
    Called from callback to populate #team-header.
    """
    team   = portrait.get("team", "—")
    season = portrait.get("season", "—")
    temp   = portrait.get("temporal", {})
    mode   = temp.get("mode", "unknown")
    pa     = temp.get("pa", 0)
    games  = temp.get("games", 0)
    cov    = portrait.get("data_coverage", 0.0)

    badge_color = MODE_COLORS.get(mode, "secondary")

    return html.Div([
        html.H4(
            [f"{team}  ", html.Span(str(season), className="text-secondary fw-normal")],
            className="mb-1 text-white",
        ),
        dbc.Row([
            dbc.Col([
                dbc.Badge(
                    mode.upper(), color=badge_color, className="me-2", style=BADGE_STYLE,
                ),
                html.Small(
                    f"{games} games · {pa} PA",
                    className="text-secondary",
                ),
            ], width="auto"),
            dbc.Col([
                html.Small("Data coverage", className="text-secondary me-2"),
                dbc.Progress(
                    value=int(cov * 100), color="primary", striped=False,
                    className="d-inline-block", style={"width": "120px", "height": "8px"},
                ),
                html.Small(f" {cov:.0%}", className="text-secondary ms-1"),
            ], width="auto", className="d-flex align-items-center"),
        ], className="g-2"),
    ])


# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

def _card(title: str, graph_id: str, height: int = 380) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(
            html.Small(title, className="text-secondary fw-semibold text-uppercase",
                       style={"letterSpacing": "0.07em", "fontSize": "0.7rem"}),
            style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
        ),
        dbc.CardBody(
            dcc.Graph(
                id=graph_id,
                config={"displayModeBar": False},
                style={"height": f"{height}px"},
            ),
            style={"padding": "8px"},
        ),
    ], style=CARD_STYLE, className="mb-3")


def overview_tab() -> dbc.Tab:
    return dbc.Tab(label="Overview", tab_id="tab-overview", children=[
        html.Div(id="team-header", className="mb-3 p-3 rounded",
                 style={"backgroundColor": "#1f2937", "border": "1px solid #374151"}),
        dbc.Row([
            dbc.Col(_card("Philosophy Profile (All Dimensions)", "radar-all", height=420), md=6),
            dbc.Col(_card("Primary Philosophy by Dimension", "dim-bars", height=280), md=6),
        ]),
        dbc.Row([
            dbc.Col(_card("Park Factor — Pitcher Friendliness", "park-gauge", height=320), md=4),
            dbc.Col(_card("Team Spin Efficiency", "spin-bar", height=200), md=8),
        ]),
    ])


def offense_tab() -> dbc.Tab:
    return dbc.Tab(label="Offense", tab_id="tab-offense", children=[
        dbc.Row([
            dbc.Col(_card("Offensive Philosophy Radar", "radar-offense", height=380), md=5),
            dbc.Col(_card("Team Batting Metrics (Percentile)", "batting-bars", height=380), md=7),
        ]),
        dbc.Row([
            dbc.Col(_card("Hitter Archetype Distribution", "hitter-pie", height=340), md=5),
            dbc.Col(_card("Hitter Roster Detail", "hitter-table", height=480), md=7),
        ]),
    ])


def pitching_tab() -> dbc.Tab:
    return dbc.Tab(label="Pitching", tab_id="tab-pitching", children=[
        dbc.Row([
            dbc.Col(_card("Pitching Philosophy Radar", "radar-pitching", height=380), md=5),
            dbc.Col(_card("Starter Roster Detail", "starter-bars", height=420), md=7),
        ]),
    ])


def roster_tab() -> dbc.Tab:
    return dbc.Tab(label="Roster Construction", tab_id="tab-roster", children=[
        dbc.Row([
            dbc.Col(_card("Roster Philosophy Radar", "radar-roster", height=380), md=6),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.Small("C-Dimension Data Gaps", className="text-secondary fw-semibold text-uppercase",
                                   style={"fontSize": "0.7rem", "letterSpacing": "0.07em"}),
                        style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
                    ),
                    dbc.CardBody([
                        dbc.Alert(
                            [
                                html.Strong("Roster construction (C1–C4) requires external data sources:"),
                                html.Ul([
                                    html.Li("WAR data — FanGraphs CSV (GAP 1)"),
                                    html.Li("Payroll — Spotrac CSV (GAP 4)"),
                                    html.Li("Prospect pipeline — MLB Pipeline CSV (GAP 5)"),
                                ], className="mt-2 mb-0"),
                            ],
                            color="warning", className="mb-0",
                            style={"fontSize": "0.85rem"},
                        ),
                    ]),
                ], style=CARD_STYLE),
                md=6,
            ),
        ]),
    ])


def tabs_layout() -> dbc.Tabs:
    return dbc.Tabs(
        id="main-tabs",
        active_tab="tab-overview",
        children=[overview_tab(), offense_tab(), pitching_tab(), roster_tab()],
        className="mb-3",
    )


# ---------------------------------------------------------------------------
# Full page layout
# ---------------------------------------------------------------------------

def full_layout() -> html.Div:
    return html.Div([
        navbar(),
        dbc.Container([
            controls_row(),
            dcc.Store(id="portrait-store"),
            dcc.Loading(
                id="global-spinner",
                type="circle",
                color="#1a56db",
                children=tabs_layout(),
            ),
        ], fluid=True),
    ], style={"backgroundColor": "#111827", "minHeight": "100vh", "color": "#f9fafb"})
