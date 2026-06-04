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

# Full names used in philosophy collapse buttons (layout-local copy to avoid import cycle)
_PHIL_FULL: dict[str, str] = {
    "A1": "Three True Outcomes",
    "A2": "Contact / Speed",
    "A3": "Aggressive Approach",
    "A4": "Lineup Power",
    "B1": "Stuff Dominance",
    "B2": "Command & Defense",
    "B3": "Pitch Design",
    "B4": "Defensive Infrastructure",
    "C1": "bWAR Distribution",
    "C2": "Roster Continuity",
    "C3": "Youth and Development",
    "C4": "Veteran Experience",
}

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
_SEASON_LABELS = {s: (f"⚡ {s} (60G)" if s == 2020 else str(s)) for s in SEASONS}


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
                options=[{"label": _SEASON_LABELS[s], "value": s} for s in SEASONS],
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
                *([dbc.Badge(
                    "⚡ 60-GAME SEASON", color="warning", className="me-2",
                    style={**BADGE_STYLE, "fontSize": "0.65rem"},
                )] if int(season) == 2020 else []),
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


def _archetype_guide_card() -> dbc.Card:
    """
    Static reference card — archetype definitions + modifier explanations.
    No callbacks needed; content never changes with portrait load.
    """
    # ── Primary archetypes ────────────────────────────────────────────────────
    ARCHETYPES = [
        {
            "code": "T1", "color": "#6366f1", "name": "Complete Hitter",
            "desc": (
                "Excels across every offensive dimension — on-base ability, power, patience, "
                "and contact quality. The rarest archetype (~2% of regulars), reserved for "
                "players who genuinely do everything well."
            ),
            "gates": [
                "xwOBA ≥ 70th pct — overall expected offensive quality",
                "OBP ≥ 70th pct — elite on-base ability",
                "ISO ≥ 70th pct — real power threat",
                "BB% ≥ 65th pct — draws walks consistently",
                "Barrel% ≥ 65th pct — makes elite contact quality",
                "Contact% ≥ 55th pct — can actually put the ball in play",
            ],
            "examples": "Juan Soto, Freddie Freeman, Yordan Álvarez, Kyle Tucker 2023",
        },
        {
            "code": "T2", "color": "#ef4444", "name": "Three True Outcomes",
            "desc": (
                "Defined by strikeouts, walks, and home runs. Rarely puts the ball in play, "
                "but commands the strike zone and makes pitchers work. High-variance, "
                "high-upside approach that swings the stat line dramatically."
            ),
            "gates": [
                "K% ≥ 55th pct — above-average strikeout rate",
                "BB% ≥ 65th pct — clearly above-average walk rate",
                "ISO ≥ 65th pct — clearly above-average power",
            ],
            "examples": "Joey Gallo, Kyle Schwarber, Aaron Judge, Shohei Ohtani",
        },
        {
            "code": "T3", "color": "#22c55e", "name": "Contact",
            "desc": (
                "Makes contact as the primary offensive weapon. Leans toward putting the "
                "ball in play, reaching base via singles, and applying constant pressure "
                "through volume and bat control. Pairs naturally with speed."
            ),
            "gates": ["Spectrum score < 45 (contact-leaning batted ball profile)"],
            "examples": "Luis Arraez, Alex Bregman, Jeff McNeil, Nico Hoerner",
        },
        {
            "code": "T4", "color": "#06b6d4", "name": "Balanced",
            "desc": (
                "No strong lean toward contact or power — sits in the middle of the spectrum. "
                "Modifiers do the heavy lifting here: look at speed tier, discipline grade, "
                "and baserunning tags to understand this hitter's real identity."
            ),
            "gates": ["Spectrum score 45–55 (no strong directional lean)"],
            "examples": "José Abreu, Gleyber Torres, Ketel Marte (select seasons)",
        },
        {
            "code": "T5", "color": "#f97316", "name": "Power",
            "desc": (
                "Swings for damage. Extra-base hits are the primary contribution — may carry "
                "a higher strikeout rate but produces serious run value when they connect. "
                "Barrel and hard-hit rates are the identity markers here."
            ),
            "gates": ["Spectrum score > 55 (power-leaning batted ball profile)"],
            "examples": "Chas McCormick, Teoscar Hernández, Spencer Torkelson",
        },
    ]

    # ── Modifiers ────────────────────────────────────────────────────────────
    MODIFIERS = [
        {
            "tag": "Elite / Fast / Slow", "color": "#a78bfa",
            "desc": (
                "Sprint speed tier based on raw ft/sec from Statcast. "
                "Elite ≥ 29.0 ft/s (top ~10%) — Fast ≥ 28.0 ft/s (top ~35%) — "
                "Average 26.5–28.0 (no tag) — Slow < 26.5 ft/s (bottom ~25%)."
            ),
            "examples": "Elite: Bobby Witt Jr., Elly De La Cruz · Fast: Tommy Edman, Mauricio Dubón",
        },
        {
            "tag": "Lucky / Unlucky", "color": "#fbbf24",
            "desc": (
                "Compares actual wOBA to expected wOBA (xwOBA) via the luck delta. "
                "Lucky = actual results significantly beat expected contact quality — "
                "potential regression risk. Unlucky = hitting the ball better than "
                "stats show — positive regression candidate."
            ),
            "examples": "Lucky: Acuña 2023, Torres 2023 · Unlucky: Semien 2023, Olson 2023, Carroll 2023",
        },
        {
            "tag": "Plus Contact / Weak Contact", "color": "#34d399",
            "desc": (
                "Batted ball quality tier — composite of Barrel% and Hard Hit% percentile ranks. "
                "Plus Contact = top 30% of Statcast-tracked hitters. "
                "Weak Contact = bottom 30%. Requires sufficient batted ball sample."
            ),
            "examples": "Plus Contact: Yordan Álvarez, Juan Soto · Weak Contact: low-exit-velo slap hitters",
        },
        {
            "tag": "Elite Discipline / Disciplined / Free Swinger", "color": "#60a5fa",
            "desc": (
                "Plate approach grade — weighted composite of walk rate (60%) and "
                "inverted chase rate (40%). Elite Discipline = top 25%, "
                "Disciplined = top 45%, Free Swinger = bottom 35% of the composite."
            ),
            "examples": "Elite Discipline: Juan Soto, Bryce Harper · Free Swinger: Lane Thomas, Jazz Chisholm Jr.",
        },
        {
            "tag": "Table Setter", "color": "#10b981",
            "desc": (
                "Leadoff-type modifier: above-average on-base ability (OBP ≥ 65th pct) + "
                "Fast or Elite speed + limited power output (ISO ≤ 45th pct). "
                "These players live to get on base and create chaos on the basepaths."
            ),
            "examples": "Steven Kwan, Nico Hoerner, Myles Straw, Tommy Edman, Ha-Seong Kim",
        },
        {
            "tag": "Aggressive", "color": "#fb923c",
            "desc": (
                "Attack-early approach — high first-pitch swing rate (FPS ≥ 65th pct) "
                "AND high chase rate (OSwing ≥ 60th pct). These players initiate contact "
                "regardless of count. Good hitters can still carry this tag."
            ),
            "examples": "Kyle Tucker, Freddie Freeman, Bryce Harper, George Springer",
        },
        {
            "tag": "Disruptive / Chaotic", "color": "#f472b6",
            "desc": (
                "Baserunning personality. Disruptive = high-efficiency, frequent base stealers "
                "who create net positive run value (sb% > 72%, positive efficiency score). "
                "Chaotic = frequent but inefficient — high attempt rate, negative run value."
            ),
            "examples": "Disruptive: Acuña 2023, Corbin Carroll · Chaotic: Jeremy Peña 2023",
        },
    ]

    def _arch_item(a: dict) -> dbc.AccordionItem:
        badge = dbc.Badge(a["code"], pill=True,
                          style={"backgroundColor": a["color"], "fontSize": "0.65rem",
                                 "color": "#fff", "fontWeight": "700"})
        return dbc.AccordionItem(
            html.Div([
                html.P(a["desc"], className="text-white mb-2", style={"fontSize": "0.84rem"}),
                html.P("Classification gates:", className="mb-1",
                       style={"fontSize": "0.72rem", "color": "#6b7280", "fontWeight": "600",
                              "textTransform": "uppercase", "letterSpacing": "0.04em"}),
                html.Ul([
                    html.Li(g, style={"fontSize": "0.8rem", "color": "#9ca3af", "marginBottom": "2px"})
                    for g in a["gates"]
                ], className="mb-2 ps-3"),
                html.Small(f"📋  {a['examples']}", className="fst-italic",
                           style={"color": "#6ee7b7", "fontSize": "0.78rem"}),
            ], style={"padding": "4px 2px"}),
            title=html.Span([badge, html.Span(f"  {a['name']}", className="ms-2 fw-semibold",
                                              style={"color": "#f3f4f6"})]),
        )

    def _mod_item(m: dict) -> dbc.AccordionItem:
        badge = dbc.Badge("MOD", pill=True,
                          style={"backgroundColor": m["color"], "fontSize": "0.6rem",
                                 "color": "#fff", "fontWeight": "700"})
        return dbc.AccordionItem(
            html.Div([
                html.P(m["desc"], className="text-white mb-2", style={"fontSize": "0.84rem"}),
                html.Small(f"📋  {m['examples']}", className="fst-italic",
                           style={"color": "#6ee7b7", "fontSize": "0.78rem"}),
            ], style={"padding": "4px 2px"}),
            title=html.Span([badge, html.Span(f"  {m['tag']}", className="ms-2 fw-semibold",
                                              style={"color": "#f3f4f6"})]),
        )

    _acc_style = {
        "backgroundColor": "transparent",
        "--bs-accordion-bg": "#1f2937",
        "--bs-accordion-color": "#f3f4f6",
        "--bs-accordion-border-color": "#374151",
        "--bs-accordion-btn-color": "#f3f4f6",
        "--bs-accordion-btn-bg": "#1a2233",
        "--bs-accordion-active-bg": "#1a2233",
        "--bs-accordion-active-color": "#f3f4f6",
    }

    return dbc.Card([
        dbc.CardHeader(
            html.Span([
                html.Small("Archetype Guide", className="fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em",
                                  "color": "#9ca3af"}),
                html.Small(" — expand any row to see gates, examples, and definitions",
                           style={"fontSize": "0.65rem", "color": "#6b7280"}),
            ]),
            style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
        ),
        dbc.CardBody([
            html.P("Primary Archetypes", className="mb-2",
                   style={"fontSize": "0.7rem", "textTransform": "uppercase",
                          "letterSpacing": "0.06em", "color": "#6b7280", "fontWeight": "600"}),
            dbc.Accordion(
                [_arch_item(a) for a in ARCHETYPES],
                flush=True, always_open=False, style=_acc_style, className="mb-3",
            ),
            html.Hr(style={"borderColor": "#374151", "margin": "10px 0"}),
            html.P("Modifiers", className="mb-2",
                   style={"fontSize": "0.7rem", "textTransform": "uppercase",
                          "letterSpacing": "0.06em", "color": "#6b7280", "fontWeight": "600"}),
            dbc.Accordion(
                [_mod_item(m) for m in MODIFIERS],
                flush=True, always_open=False, style=_acc_style,
            ),
        ], style={"padding": "12px"}),
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
        # ── Construction vs Results ────────────────────────────────────────
        dbc.Row([
            dbc.Col(_card("Construction vs Results — Philosophy", "cvr-radar", height=480), md=6),
            dbc.Col(_card("Construction vs Results — Archetype Mix", "cvr-archetypes", height=380), md=6),
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
        _card("Batted Ball Profile", "spray-heatmap", height=580),
        _card("Hitter Archetype Affinity", "hitter-heatmap", height=420),
        _archetype_guide_card(),
        # ── Player comparison section ───────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(
                html.Small("Player Metrics Comparison", className="text-secondary fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em"}),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody([
                html.Label("Select players to compare", className="text-secondary small mb-1"),
                dcc.Dropdown(
                    id="player-select-dropdown",
                    options=[],
                    value=[],
                    multi=True,
                    placeholder="Select one or more hitters…",
                    style={"backgroundColor": "#1f2937", "color": "#111827"},
                    className="mb-3",
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id="player-radar-chart",
                            config={"displayModeBar": False},
                            style={"height": "420px"},
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id="player-bars-chart",
                            config={"displayModeBar": False},
                            style={"height": "420px"},
                        ),
                        md=6,
                    ),
                ]),
            ], style={"padding": "12px"}),
        ], style=CARD_STYLE, className="mb-3"),
        html.H6("Philosophy Breakdowns", className="text-secondary mt-3 mb-2",
                style={"fontSize": "0.75rem", "letterSpacing": "0.07em",
                       "textTransform": "uppercase"}),
        *[_philosophy_collapse(c) for c in ["A1", "A2", "A3", "A4"]],
    ])


def pitching_tab() -> dbc.Tab:
    return dbc.Tab(label="Pitching", tab_id="tab-pitching", children=[
        dbc.Row([
            dbc.Col(_card("Pitching Philosophy Radar", "radar-pitching", height=380), md=5),
            dbc.Col(_card("Starter Roster Detail", "starter-bars", height=420), md=7),
        ]),
        dbc.Row([
            dbc.Col(_card("Bullpen Collective Profile", "bullpen-dims", height=320), md=5),
            dbc.Col(_card("Bullpen Roster Detail", "bullpen-table", height=420), md=7),
        ], className="mt-3"),
        html.H6("Philosophy Breakdowns", className="text-secondary mt-3 mb-2",
                style={"fontSize": "0.75rem", "letterSpacing": "0.07em",
                       "textTransform": "uppercase"}),
        *[_philosophy_collapse(c) for c in ["B1", "B2", "B3", "B4"]],
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
                                    html.Li("Payroll — Spotrac CSV (GAP 4)"),
                                    html.Li("Service time — not publicly available (GAP 6)"),
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
        html.H6("Philosophy Breakdowns", className="text-secondary mt-3 mb-2",
                style={"fontSize": "0.75rem", "letterSpacing": "0.07em",
                       "textTransform": "uppercase"}),
        *[_philosophy_collapse(c) for c in ["C1", "C3", "C4"]],
    ])


def _philosophy_collapse(code: str) -> html.Div:
    """Collapsible row for one philosophy dimension with click-to-expand breakdown."""
    label = _PHIL_FULL.get(code, code)
    return html.Div([
        dbc.Button(
            [
                html.Span(f"{code} — {label}", className="fw-semibold me-2",
                          style={"fontSize": "0.85rem"}),
                html.Small("▼ show breakdown", className="text-secondary",
                           style={"fontSize": "0.72rem"}),
            ],
            id={"type": "phil-btn", "code": code},
            color="link",
            className="text-white text-start p-2 w-100",
            style={
                "textDecoration": "none",
                "backgroundColor": "#1a2233",
                "border": "1px solid #374151",
                "borderRadius": "6px",
            },
        ),
        dbc.Collapse(
            html.Div(
                id={"type": "phil-collapse-content", "code": code},
                style={
                    "backgroundColor": "#1a2233",
                    "border": "1px solid #374151",
                    "borderTop": "none",
                    "borderRadius": "0 0 6px 6px",
                },
            ),
            id={"type": "phil-collapse", "code": code},
            is_open=False,
        ),
    ], className="mb-2")


def compare_tab() -> dbc.Tab:
    """
    Team A vs Team B comparison tab.
    Each team gets its own selectors + Load button, independent of the main controls.
    """
    def _team_selector(suffix: str, label: str, default_team: str, default_season: str,
                       color: str) -> dbc.Card:
        return dbc.Card([
            dbc.CardHeader(
                html.Small(label, className="fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em",
                                  "color": color}),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Team", className="text-secondary small mb-1"),
                        dbc.Select(
                            id=f"cmp-team-{suffix}",
                            options=[{"label": t, "value": t} for t in MLB_TEAMS],
                            value=default_team,
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Season", className="text-secondary small mb-1"),
                        dbc.Select(
                            id=f"cmp-season-{suffix}",
                            options=[{"label": _SEASON_LABELS[s], "value": s} for s in SEASONS],
                            value=default_season,
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label(" ", className="text-secondary small mb-1 d-block"),
                        dbc.Button("Load", id=f"cmp-btn-{suffix}", color="primary",
                                   className="w-100"),
                    ], md=4),
                ], className="align-items-end"),
                html.Div(id=f"cmp-status-{suffix}", className="mt-2"),
            ], style={"padding": "10px"}),
        ], style=CARD_STYLE, className="mb-0")

    return dbc.Tab(label="Compare", tab_id="tab-compare", children=[
        dbc.Row([
            dbc.Col(_team_selector("a", "Team A", "HOU", "2023", "#60a5fa"), md=6),
            dbc.Col(_team_selector("b", "MIA", "MIA", "2022", "#f97316"), md=6),
        ], className="mb-3"),

        # Overlaid philosophy radar
        dbc.Card([
            dbc.CardHeader(
                html.Small("Philosophy Profile — Overlay", className="text-secondary fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em"}),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody(
                dcc.Graph(id="cmp-radar", config={"displayModeBar": False}, style={"height": "420px"}),
                style={"padding": "8px"},
            ),
        ], style=CARD_STYLE, className="mb-3"),

        # Dimension scores side-by-side
        dbc.Card([
            dbc.CardHeader(
                html.Small("Philosophy Scores by Dimension", className="text-secondary fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em"}),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody(
                dcc.Graph(id="cmp-dim-bars", config={"displayModeBar": False}, style={"height": "400px"}),
                style={"padding": "8px"},
            ),
        ], style=CARD_STYLE, className="mb-3"),

        # Batting metrics + archetype dist side-by-side
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(
                    html.Small("Batting Metrics", className="text-secondary fw-semibold text-uppercase",
                               style={"fontSize": "0.7rem", "letterSpacing": "0.07em"}),
                    style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
                ),
                dbc.CardBody(
                    dcc.Graph(id="cmp-batting", config={"displayModeBar": False}, style={"height": "380px"}),
                    style={"padding": "8px"},
                ),
            ], style=CARD_STYLE), md=7),
            dbc.Col(dbc.Card([
                dbc.CardHeader(
                    html.Small("Hitter Archetype Mix", className="text-secondary fw-semibold text-uppercase",
                               style={"fontSize": "0.7rem", "letterSpacing": "0.07em"}),
                    style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
                ),
                dbc.CardBody(
                    dcc.Graph(id="cmp-archetypes", config={"displayModeBar": False}, style={"height": "380px"}),
                    style={"padding": "8px"},
                ),
            ], style=CARD_STYLE), md=5),
        ]),
    ])


def scout_tab() -> dbc.Tab:
    """
    Player profile search — find players matching archetype + modifier criteria
    across all seeded seasons. Independent of any team portrait.
    """
    ARCHETYPES = ["Any", "Complete Hitter", "Three True Outcomes",
                  "Contact", "Balanced", "Power"]

    MODIFIERS = [
        "Aggressive", "Gap Hitter", "Plus Power", "Table Setter",
        "Elite Discipline", "Disciplined", "Free Swinger",
        "Plus Contact", "Weak Contact",
        "Lucky", "Unlucky",
        "Elite", "Fast", "Slow",
        "Disruptive", "Chaotic",
    ]

    return dbc.Tab(label="Scout", tab_id="tab-scout", children=[
        dbc.Card([
            dbc.CardHeader(
                html.Small("Player Profile Search", className="fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em",
                                  "color": "#9ca3af"}),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Archetype", className="text-secondary small mb-1"),
                        dbc.Select(
                            id="scout-archetype",
                            options=[{"label": a, "value": a} for a in ARCHETYPES],
                            value="Any",
                        ),
                    ], md=2),
                    dbc.Col([
                        html.Label("Season range", className="text-secondary small mb-1"),
                        dbc.Row([
                            dbc.Col(dbc.Select(
                                id="scout-season-min",
                                options=[{"label": _SEASON_LABELS[s], "value": s} for s in SEASONS],
                                value=2021,
                            ), width=6),
                            dbc.Col(dbc.Select(
                                id="scout-season-max",
                                options=[{"label": _SEASON_LABELS[s], "value": s} for s in SEASONS],
                                value=2026,
                            ), width=6),
                        ], className="g-1"),
                    ], md=2),
                    dbc.Col([
                        html.Label("Min PA", className="text-secondary small mb-1"),
                        dbc.Input(id="scout-min-pa", type="number",
                                  value=200, min=50, max=700, step=50,
                                  style={"backgroundColor": "#1f2937",
                                         "color": "#f9fafb", "border": "1px solid #374151"}),
                    ], md=1),
                    dbc.Col([
                        html.Label("Show", className="text-secondary small mb-1"),
                        dbc.Select(
                            id="scout-mode",
                            options=[
                                {"label": "Most recent season", "value": "recent"},
                                {"label": "All seasons",        "value": "all"},
                                {"label": "Best season (xwOBA)", "value": "best"},
                            ],
                            value="recent",
                        ),
                    ], md=2),
                    dbc.Col([
                        html.Label(" ", className="text-secondary small mb-1 d-block"),
                        dbc.Button("Search", id="scout-btn", color="primary",
                                   className="w-100"),
                    ], md=1),
                ], className="mb-3 align-items-end"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Modifiers (must have ALL selected)",
                                   className="text-secondary small mb-1"),
                        dcc.Dropdown(
                            id="scout-modifiers",
                            options=[{"label": m, "value": m} for m in MODIFIERS],
                            value=[],
                            multi=True,
                            placeholder="Any modifiers…",
                            style={"backgroundColor": "#1f2937", "color": "#111827"},
                        ),
                    ]),
                ], className="mb-3"),
                html.Div(id="scout-status", className="mb-2"),
            ], style={"padding": "12px"}),
        ], style=CARD_STYLE, className="mb-3"),

        dbc.Card([
            dbc.CardHeader(
                html.Small("Results", className="fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em",
                                  "color": "#9ca3af"}),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody(
                dcc.Graph(id="scout-results", config={"displayModeBar": False},
                          style={"minHeight": "400px"}),
                style={"padding": "8px"},
            ),
        ], style=CARD_STYLE),
    ])


def tabs_layout() -> dbc.Tabs:
    return dbc.Tabs(
        id="main-tabs",
        active_tab="tab-overview",
        children=[overview_tab(), offense_tab(), pitching_tab(), compare_tab(), scout_tab()],
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
            dcc.Store(id="cmp-store-a"),
            dcc.Store(id="cmp-store-b"),
            dcc.Loading(
                id="global-spinner",
                type="circle",
                color="#1a56db",
                children=tabs_layout(),
            ),
        ], fluid=True),
    ], style={"backgroundColor": "#111827", "minHeight": "100vh", "color": "#f9fafb"})
