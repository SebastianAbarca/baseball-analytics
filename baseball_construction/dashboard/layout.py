"""
layout.py — Bootstrap component builders for the dashboard layout.

All functions return Dash/DBC component trees. No callbacks here.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

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
                value="2025",
            ),
        ], md=2, xs=6),
        dbc.Col([
            html.Div(id="status-banner", className="mt-1"),
        ], md=7, xs=12),
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
    Static reference card — trait glossary, grouped by family.
    No callbacks needed; content never changes with portrait load.
    """
    # ── Hitter trait families ────────────────────────────────────────────────
    ARCHETYPES = [
        {
            "code": "SPEC", "color": "#6366f1", "name": "Power–Contact Spectrum (continuous)",
            "desc": (
                "Every hitter gets a 0–100 spectrum score: 0 = extreme contact "
                "(Arraez), 100 = extreme power (Gallo). It is a ratio of power "
                "production to contact production, shown as a number — not a box. "
                "Tags mark only the tails, with absolute floors so the ratio "
                "can't mislead."
            ),
            "gates": [
                "no tags come off the spectrum — a ratio cannot describe a hitter "
                "who is moderate in both, which is where doubles hitters live",
                "kept as a continuous field for the Scout search band",
            ],
            "examples": "0 = Arraez · 100 = Gallo · doubles hitters sit near the middle",
        },
        {
            "code": "BAT", "color": "#f97316", "name": "Bat traits",
            "desc": "Production shape, measured directly rather than as a ratio.",
            "gates": [
                "power bat — ISO ≥ 85th pct (absorbs the old hard contact tag, which it contained 82% of)",
                "plus power — ISO ≥ 60th pct in a non-power-bat (suppressed by gap hitter)",
                "gap hitter — doubles/triples into the gaps WITHOUT home-run power (XB ≥ 60th + gap tendency ≥ 55th + HR/FB ≤ 70th)",
                "weak contact — Barrel% + HardHit% composite bottom 30%",
            ],
            "examples": "gap hitter now means doubles, not homers",
        },
        {
            "code": "APPR", "color": "#22c55e", "name": "Approach traits",
            "desc": (
                "Swing decisions. Chase rate alone separates patient from free "
                "swinger — the old composite mixed the decision with its "
                "result, which is why free swinger fired on 30% of the league."
            ),
            "gates": [
                "patient — chase ≤ 15th pct · free swinger — chase ≥ 85th pct",
                "zone hunter — swing discrimination (zone-swing rate minus chase "
                "rate) ≥ 85th pct: attacks strikes AND lays off balls",
                "walk machine — BB% ≥ 80th pct (a result, so it stands alone)",
                "aggressive — first-pitch swing ≥ 65th pct AND chase ≥ 60th pct",
            ],
            "examples": ("zone hunter: Seager 2023 (100th), Semien, Judge · "
                         "aggressive: Tucker, Acuña 2023 · walk machine: Soto"),
        },
        {
            "code": "B2B", "color": "#fb923c", "name": "Bat-to-ball traits",
            "desc": (
                "Contact ability, kept separate from approach on purpose. Chase "
                "rate and strikeout rate correlate +0.02 — swing decisions and "
                "bat-to-ball skill are orthogonal, so filing strikeouts under "
                "approach was simply wrong."
            ),
            "gates": [
                "high-K — strikeout rate ≥ 80th pct of the season",
                "rarely strikes out — ≤ 10th pct (min 200 PA; small samples "
                "produce fake 8% strikeout rates)",
            ],
            "examples": "rarely strikes out: Arraez, Kwan · high-K: Gallo",
        },
    ]

    # ── Athleticism / luck + pitcher trait families ──────────────────────────
    MODIFIERS = [
        {
            "tag": "elite speed / fast / station-to-station / high|low steal attempts / high|low steal rate / extra base taker / everyday player / super-utility", "color": "#a78bfa",
            "desc": (
                "Speed is a tool; running is a choice; getting there safely is a "
                "result — so they are separate tags. steal attempts = how often he "
                "goes (top/bottom 15% of attempt rate). steal rate = how often he "
                "makes it (≥80% / ≤65%, min 10 attempts). The old disruptive / "
                "chaotic pair blended the two. extra base taker = top-15% rate of "
                "taking the extra base on hits; runners thrown out are not "
                "recorded, so it measures willingness rather than success. "
                "Speed tiers from raw sprint speed (elite ≥ 29 ft/s, fast ≥ 28). "
                "super-utility = real innings at 4+ positions."
            ),
            "examples": "elite speed: Witt Jr., Peña",
        },
        {
            "tag": "lucky / unlucky", "color": "#fbbf24",
            "desc": (
                "wOBA vs xwOBA gap, season decile tails only. lucky = results "
                "outran the contact quality (regression risk); unlucky = contact "
                "deserved better (positive regression candidate)."
            ),
            "examples": "lucky: Altuve 2023 · unlucky: Albies 2023",
        },
        {
            "tag": "Hitter: batted-ball family", "color": "#ec4899",
            "desc": (
                "Where the ball goes off the bat. pull-heavy / oppo bat "
                "(directional BIP rate ≥ 85th pct), air-ball bat / ground-ball "
                "bat (batter GB% bottom/top decile)."
            ),
            "examples": "pull-heavy: Maldonado 2023 · ground-ball bat: Peña 2023",
        },
        {
            "tag": "Hitter: platoon family", "color": "#14b8a6",
            "desc": (
                "Handedness structure from statcast splits (min 50 PA each side). "
                "switch hitter (≥15% of PAs from each side), platoon liability "
                "(wOBA vs same-hand ≥ .060 below vs opposite — exploitable), "
                "reverse split (better vs same-hand by ≥ .040 — matchup-proof "
                "backwards)."
            ),
            "examples": "platoon liability: Peña 2023 · reverse split: Bregman 2023 (crushed RHP, .215 vs LHP)",
        },
        {
            "tag": "Pitcher: role family", "color": "#84cc16",
            "desc": (
                "Usage shape vs the season's starter/reliever pools. workhorse "
                "(starter BF ≥ 85th pct), short-outing starter (BF/start ≤ 15th "
                "pct, non-workhorse), heavy usage (reliever appearances ≥ 85th), "
                "multi-inning reliever (BF/appearance ≥ 85th), swingman (3+ "
                "starts and 5+ relief outings)."
            ),
            "examples": "workhorse: Framber, Verlander 2023 · swingman: Urquidy 2023",
        },
        {
            "tag": "Pitcher: platoon family", "color": "#f43f5e",
            "desc": (
                "Splits + deployment (min 50 BF each side). platoon-vulnerable "
                "(opp-hand hitters ≥ .060 better), reverse split (own-hand "
                "hitters do more damage), platoon specialist (≥ 60% of BF vs "
                "same-hand — sheltered/specialist usage)."
            ),
            "examples": "platoon-vulnerable: Javier 2023 · reverse split: Verlander 2023 · platoon specialist: Graveman 2023",
        },
        {
            "tag": "Pitcher: mechanics family", "color": "#8b5cf6",
            "desc": (
                "Where the ball comes from. submarine (< 10° arm angle), sidearm "
                "(10–25°), over-the-top (≥ 55°), deep/short extension (top/bottom "
                "decile of release extension), wide release (cross-fire angle). "
                "A normal three-quarters slot carries no tag. Also tempo: "
                "quick pitcher / slow pitcher (fastest/slowest 15% between "
                "pitches, bases empty)."
            ),
            "examples": "submarine: Ryan Thompson · sidearm: Cimber, Nola · over-the-top: Kershaw, Verlander",
        },
        {
            "tag": "Pitcher: sequencing family", "color": "#06b6d4",
            "desc": (
                "How the arsenal is deployed. unpredictable / patterned (pitch-to-pitch "
                "transition entropy, arsenal-size adjusted), count-shifter / steady mix "
                "(how much the mix moves between ahead and behind counts), "
                "first-pitch attacker (F-strike ≥ 90th pct)."
            ),
            "examples": "unpredictable: Cole, Kershaw 2023 · patterned: Gausman · count-shifter: Framber",
        },
        {
            "tag": "Pitcher: deception family", "color": "#f59e0b",
            "desc": (
                "How the stuff plays beyond its raw quality. tunneler (league tunnel "
                "score ≥ 75th pct), invisible ball (whiff rate above what velocity "
                "predicts, ≥ 90th pct residual), high spin efficiency. The old "
                "gyro-heavy tag retired — 79% of the pitchers carrying it also "
                "threw a slider as their out pitch, so it mostly restated the "
                "pitch."
            ),
            "examples": "tunneler: Javier, Pressly 2023 · invisible ball: Luis García, Abreu 2023",
        },
        {
            "tag": "Pitcher: arsenal + outcome families", "color": "#3b82f6",
            "desc": (
                "Arsenal: sinker-baller / ride four-seam / cutter-primary, the out "
                "pitch, deep arsenal / two-pitch (starters only), velo tiers. Outcome: bat-misser / "
                "pitch-to-contact, ground-baller / fly-ball prone, contact suppressor, "
                "elite/plus command, walk prone, controls runners (fewest mid-PA "
                "runner advances allowed — steals, WP and PB included), and the "
                "ace badge (dominance composite ≥ 70 with 2+ elite traits)."
            ),
            "examples": "sinker-baller + ground-baller: Webb, Framber · ace: Pressly, Abreu, Maton 2023",
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
        badge = dbc.Badge("TAG", pill=True,
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
                html.Small("Trait Guide", className="fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em",
                                  "color": "#9ca3af"}),
                html.Small(" — expand any row to see gates, examples, and definitions",
                           style={"fontSize": "0.65rem", "color": "#6b7280"}),
            ]),
            style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
        ),
        dbc.CardBody([
            html.P("Hitter Traits & Spectrum", className="mb-2",
                   style={"fontSize": "0.7rem", "textTransform": "uppercase",
                          "letterSpacing": "0.06em", "color": "#6b7280", "fontWeight": "600"}),
            dbc.Accordion(
                [_arch_item(a) for a in ARCHETYPES],
                flush=True, always_open=False, style=_acc_style, className="mb-3",
            ),
            html.Hr(style={"borderColor": "#374151", "margin": "10px 0"}),
            html.P("Athleticism, Luck & Pitcher Trait Families", className="mb-2",
                   style={"fontSize": "0.7rem", "textTransform": "uppercase",
                          "letterSpacing": "0.06em", "color": "#6b7280", "fontWeight": "600"}),
            dbc.Accordion(
                [_mod_item(m) for m in MODIFIERS],
                flush=True, always_open=False, style=_acc_style,
            ),
        ], style={"padding": "12px"}),
    ], style=CARD_STYLE, className="mb-3")


def identity_tab() -> dbc.Tab:
    """Landing tab — the team's identity, fingerprint first."""
    return dbc.Tab(label="Identity", tab_id="tab-identity", children=[
        dbc.Card([
            dbc.CardHeader(
                html.Small("Who this team is",
                           className="fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em",
                                  "color": "#9ca3af"}),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody(
                html.Div(id="team-identity-card"),
                style={"padding": "14px"},
            ),
        ], style=CARD_STYLE, className="mb-3"),
        _card("The lineup — share of plate appearances", "hitter-pie", height=520),
        _card("The rotation — share of batters faced", "rotation-trait-density", height=520),
        _card("The bullpen — share of batters faced", "bullpen-trait-density", height=520),
    ])


def years_tab() -> dbc.Tab:
    """Identity drift — the team's defining traits vs league, 2015–2026."""
    return dbc.Tab(label="Through the Years", tab_id="tab-years", children=[
        html.P(
            "Each line is one of this franchise's defining traits — how far above "
            "or below the league it sat, season by season. The dotted line is the "
            "league average; identity is departure from it.",
            className="text-secondary", style={"fontSize": "0.85rem"},
        ),
        _card("The lineup through the years", "drift-offense", height=430),
        _card("The rotation through the years", "drift-rotation", height=430),
        _card("The bullpen through the years", "drift-bullpen", height=430),
    ])


def players_tab() -> dbc.Tab:
    """The people behind the identity — rosters with trait tags."""
    return dbc.Tab(label="The Players", tab_id="tab-players", children=[
        _card("The lineup", "hitter-table", height=520),
        dbc.Row([
            dbc.Col(_card("The rotation", "starter-bars", height=420), md=6),
            dbc.Col(_card("The bullpen", "bullpen-table", height=420), md=6),
        ]),
        dbc.Card([
            dbc.CardHeader(
                html.Small("Compare hitters head-to-head", className="text-secondary fw-semibold text-uppercase",
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
        _archetype_guide_card(),
    ])


def deep_tab() -> dbc.Tab:
    """Analyst layer — philosophy scores, physics charts, raw percentiles."""
    return dbc.Tab(label="Deep Data", tab_id="tab-deep", children=[
        dbc.Row([
            dbc.Col(_card("Philosophy Profile (All Dimensions)", "radar-all", height=420), md=6),
            dbc.Col(_card("Primary Philosophy by Dimension", "dim-bars", height=280), md=6),
        ]),
        dbc.Row([
            dbc.Col(_card("Park Factor — Pitcher Friendliness", "park-gauge", height=320), md=4),
            dbc.Col(_card("Team Spin Efficiency", "spin-bar", height=200), md=8),
        ]),
        dbc.Card([
            dbc.CardHeader(
                html.Small("Team Split Resistance — LHP vs RHP",
                           className="fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em",
                                  "color": "#9ca3af"}),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody(
                html.Div(id="team-split-card"),
                style={"padding": "12px"},
            ),
        ], style=CARD_STYLE, className="mb-3"),
        dbc.Row([
            dbc.Col(_card("Construction vs Results — Philosophy", "cvr-radar", height=480), md=6),
            dbc.Col(_card("Construction vs Results — Archetype Mix", "cvr-archetypes", height=380), md=6),
        ]),
        dbc.Row([
            dbc.Col(_card("Offensive Philosophy Radar", "radar-offense", height=380), md=5),
            dbc.Col(_card("Team Batting Metrics (Percentile)", "batting-bars", height=380), md=7),
        ]),
        _card("Batted Ball Profile", "spray-heatmap", height=580),
        _card("Hitter Skill Affinity", "hitter-heatmap", height=500),
        _card("Batter Split Resistance — LHP vs RHP", "split-heatmap", height=520),
        dbc.Row([
            dbc.Col(_card("Pitching Philosophy Radar", "radar-pitching", height=380), md=6),
            dbc.Col(_card("Bullpen Collective Profile", "bullpen-dims", height=380), md=6),
        ]),
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(
                        html.Small("Pitch Arsenal — 3D Trajectories",
                                   className="text-secondary fw-semibold text-uppercase",
                                   style={"fontSize": "0.7rem", "letterSpacing": "0.07em"}),
                        width="auto", className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="arsenal-pitch-type-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                            placeholder="Load a portrait to see pitch types…",
                            style={"backgroundColor": "#1f2937", "color": "#111827",
                                   "minWidth": "220px"},
                        ),
                        width="auto",
                    ),
                ], className="g-2 align-items-center"),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="arsenal-3d-chart",
                    config={"displayModeBar": True},
                    style={"height": "520px"},
                ),
                style={"padding": "8px"},
            ),
        ], style=CARD_STYLE, className="mb-3 mt-3"),
        dbc.Row([
            dbc.Col(_card("Roster Philosophy Radar", "radar-roster", height=380), md=6),
            dbc.Col(_card("Roster Control", "roster-control-chart", height=180), md=6),
        ]),
        dbc.Row([
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
                                html.Strong("Some C-dimension metrics require external sources:"),
                                html.Ul([
                                    html.Li("Payroll — Spotrac CSV (GAP 4)"),
                                    html.Li("Prospect pipeline — MLB Pipeline CSV (GAP 5)"),
                                ], className="mt-2 mb-0"),
                            ],
                            color="warning", className="mb-0",
                            style={"fontSize": "0.85rem"},
                        ),
                    ]),
                ], style=CARD_STYLE),
                md=12,
            ),
        ], className="mt-3"),
        html.H6("Philosophy Breakdowns", className="text-secondary mt-3 mb-2",
                style={"fontSize": "0.75rem", "letterSpacing": "0.07em",
                       "textTransform": "uppercase"}),
        *[_philosophy_collapse(c) for c in
          ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"]],
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

        # ── League Identity Board ──────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(
                html.Small("League Identity Board", className="text-secondary fw-semibold text-uppercase",
                           style={"fontSize": "0.7rem", "letterSpacing": "0.07em"}),
                style={"backgroundColor": "#1a2233", "borderBottom": "1px solid #374151"},
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Season", className="text-secondary small mb-1"),
                        dbc.Select(
                            id="league-board-season",
                            options=[{"label": _SEASON_LABELS[s], "value": s} for s in SEASONS],
                            value=2023,
                        ),
                    ], md=2),
                ], className="mb-2"),
                html.Div(id="league-board-table"),
                html.Small(
                    "Click a column header to sort. Generated by scripts/build_league_identity.py — "
                    "run it to refresh or add a season.",
                    className="text-secondary",
                    style={"fontSize": "0.65rem"},
                ),
            ], style={"padding": "10px"}),
        ], style=CARD_STYLE, className="mb-3 mt-3"),
    ])


def scout_tab() -> dbc.Tab:
    """
    Player profile search — find players matching trait-tag criteria across
    all seeded seasons. Independent of any team portrait.
    """
    # Spectrum band replaces the old archetype box filter
    SPECTRUM_BANDS = [
        ("Any spectrum",        "Any"),
        ("Power side (≥ 60)",   "power"),
        ("Balanced (40–60)",    "balanced"),
        ("Contact side (≤ 40)", "contact"),
    ]

    # Hitter trait tags (hitter_traits.py) — search requires ALL selected
    TRAIT_TAGS = [
        "power bat", "plus power", "gap hitter", "weak contact",
        "high-K", "rarely strikes out",
        "patient", "free swinger", "zone hunter", "walk machine", "aggressive",
        "pull-heavy", "oppo bat", "air-ball bat", "ground-ball bat",
        "elite speed", "fast", "station-to-station",
        "high steal attempts", "low steal attempts",
        "high steal rate", "low steal rate", "extra base taker",
        "elite defender", "plus defender", "defensive liability",
        "cannon arm", "super-utility",
        "elite framer", "poor framer", "good blocker", "bad blocker", "quick pop",
        "left-handed hitter", "right-handed hitter", "switch hitter",
        "platoon liability", "reverse split",
        "everyday player", "lucky", "unlucky",
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
                        html.Label("Spectrum", className="text-secondary small mb-1"),
                        dbc.Select(
                            id="scout-archetype",
                            options=[{"label": l, "value": v} for l, v in SPECTRUM_BANDS],
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
                        html.Label("Traits (must have ALL selected)",
                                   className="text-secondary small mb-1"),
                        dcc.Dropdown(
                            id="scout-modifiers",
                            options=[{"label": t, "value": t} for t in TRAIT_TAGS],
                            value=[],
                            multi=True,
                            placeholder="Any traits…",
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
        active_tab="tab-identity",
        children=[identity_tab(), years_tab(), players_tab(), compare_tab(), scout_tab(), deep_tab()],
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
            html.Div(id="team-header", className="mb-3 p-3 rounded",
                     style={"backgroundColor": "#1f2937", "border": "1px solid #374151"}),
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
