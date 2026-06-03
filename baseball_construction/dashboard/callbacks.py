"""
callbacks.py — Dash callback definitions.

Pattern:
  1. (team, season, button) → portrait-store   [builds portrait JSON]
  2. portrait-store → each chart               [pure figure builders]

All heavy computation happens in callback 1.
Callback 2+ are lightweight — they just call charts.py functions.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, no_update, MATCH, ctx, clientside_callback

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

from team_portrait import build_team_portrait
from ingest import pull_statcast_season
import charts
from layout import team_header

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        return super().default(obj)


def _serialize(portrait: dict) -> str:
    return json.dumps(portrait, cls=_NumpyEncoder, default=str)


def _deserialize(data: str | None) -> dict | None:
    if not data:
        return None
    return json.loads(data)


# ---------------------------------------------------------------------------
# Callback 1 — Build portrait on button click
# ---------------------------------------------------------------------------

@callback(
    Output("portrait-store", "data"),
    Output("status-banner",  "children"),
    Input("load-btn",        "n_clicks"),
    State("team-dropdown",   "value"),
    State("season-dropdown", "value"),
    running=[
        (Output("load-btn", "disabled"), True, False),
        (Output("status-banner", "children"),
         __import__("dash_bootstrap_components").Alert(
             [
                 __import__("dash").html.Span(
                     className="spinner-border spinner-border-sm me-2",
                     style={"width": "14px", "height": "14px"},
                     **{"role": "status"},
                 ),
                 "Building portrait… this takes 10–30 seconds",
             ],
             color="primary", className="py-1 mb-0 d-flex align-items-center",
         ),
         no_update),
    ],
    prevent_initial_call=True,
)
def build_portrait(n_clicks, team: str, season: int):
    import dash_bootstrap_components as dbc
    from dash import html

    if not team or not season:
        return no_update, dbc.Alert("Select a team and season.", color="warning", className="py-1")

    try:
        statcast = pull_statcast_season(int(season))
        portrait = build_team_portrait(team, int(season), statcast=statcast)
        serialized = _serialize(portrait)
        mode = portrait.get("temporal", {}).get("mode", "—")
        cov  = portrait.get("data_coverage", 0.0)
        banner = dbc.Alert(
            [
                html.Strong(f"{team} {season}"),
                f" loaded · mode={mode} · coverage={cov:.0%}",
            ],
            color="success", className="py-1 mb-0",
        )
        return serialized, banner

    except Exception as exc:
        log.exception("build_portrait failed: %s", exc)
        return no_update, dbc.Alert(f"Error: {exc}", color="danger", className="py-1")


# ---------------------------------------------------------------------------
# Callback 2 — Team header
# ---------------------------------------------------------------------------

@callback(
    Output("team-header", "children"),
    Input("portrait-store", "data"),
)
def update_team_header(data):
    portrait = _deserialize(data)
    if not portrait:
        return "Select a team and click Load Portrait."
    return team_header(portrait)


# ---------------------------------------------------------------------------
# Callback 3 — Radar charts (3 separate callbacks, one per radar)
# ---------------------------------------------------------------------------

@callback(Output("radar-all",      "figure"), Input("portrait-store", "data"))
def radar_all(data):
    p = _deserialize(data)
    return charts.philosophy_radar(p, "all") if p else charts.empty_figure("No portrait loaded")


@callback(Output("radar-offense",  "figure"), Input("portrait-store", "data"))
def radar_offense(data):
    p = _deserialize(data)
    return charts.philosophy_radar(p, "offense") if p else charts.empty_figure()


@callback(Output("radar-pitching", "figure"), Input("portrait-store", "data"))
def radar_pitching(data):
    p = _deserialize(data)
    return charts.philosophy_radar(p, "pitching") if p else charts.empty_figure()


@callback(Output("radar-roster",   "figure"), Input("portrait-store", "data"))
def radar_roster(data):
    p = _deserialize(data)
    return charts.philosophy_radar(p, "roster") if p else charts.empty_figure()


# ---------------------------------------------------------------------------
# Callback 4 — Dimension confidence bars
# ---------------------------------------------------------------------------

@callback(Output("dim-bars", "figure"), Input("portrait-store", "data"))
def dim_bars(data):
    p = _deserialize(data)
    return charts.dimension_confidence_bars(p) if p else charts.empty_figure()


# ---------------------------------------------------------------------------
# Callback 5 — Park factor gauge
# ---------------------------------------------------------------------------

@callback(Output("park-gauge", "figure"), Input("portrait-store", "data"))
def park_gauge(data):
    p = _deserialize(data)
    return charts.park_factor_gauge(p) if p else charts.empty_figure()


# ---------------------------------------------------------------------------
# Callback 6 — Spin efficiency
# ---------------------------------------------------------------------------

@callback(Output("spin-bar", "figure"), Input("portrait-store", "data"))
def spin_bar(data):
    p = _deserialize(data)
    return charts.spin_efficiency_bar(p) if p else charts.empty_figure()


# ---------------------------------------------------------------------------
# Callback 7 — Batting metrics
# ---------------------------------------------------------------------------

@callback(Output("batting-bars", "figure"), Input("portrait-store", "data"))
def batting_bars(data):
    p = _deserialize(data)
    return charts.team_batting_bars(p) if p else charts.empty_figure()


# ---------------------------------------------------------------------------
# Callback 8 — Hitter archetype pie
# ---------------------------------------------------------------------------

@callback(Output("hitter-pie", "figure"), Input("portrait-store", "data"))
def hitter_pie(data):
    p = _deserialize(data)
    return charts.hitter_archetype_pie(p) if p else charts.empty_figure()


# ---------------------------------------------------------------------------
# Callback 9 — Hitter archetype table
# ---------------------------------------------------------------------------

@callback(Output("hitter-table", "figure"), Input("portrait-store", "data"))
def hitter_table(data):
    p = _deserialize(data)
    return charts.hitter_archetype_table(p) if p else charts.empty_figure()


# ---------------------------------------------------------------------------
# Callback 10 — Starter archetype bars
# ---------------------------------------------------------------------------

@callback(Output("starter-bars", "figure"), Input("portrait-store", "data"))
def starter_bars(data):
    p = _deserialize(data)
    return charts.starter_archetype_bars(p) if p else charts.empty_figure()


# ---------------------------------------------------------------------------
# Callback 10a — Bullpen charts
# ---------------------------------------------------------------------------

@callback(Output("bullpen-dims", "figure"), Input("portrait-store", "data"))
def bullpen_dims(data):
    p = _deserialize(data)
    return charts.bullpen_dimension_bars(p) if p else charts.empty_figure()


@callback(Output("bullpen-table", "figure"), Input("portrait-store", "data"))
def bullpen_table(data):
    p = _deserialize(data)
    return charts.bullpen_detail_table(p) if p else charts.empty_figure()


# ---------------------------------------------------------------------------
# Callback 9b — Hitter archetype affinity heatmap
# ---------------------------------------------------------------------------

@callback(Output("hitter-heatmap", "figure"), Input("portrait-store", "data"))
def hitter_heatmap(data):
    p = _deserialize(data)
    return charts.hitter_archetype_heatmap(p) if p else charts.empty_figure("Load a portrait to see hitter affinities")


# ---------------------------------------------------------------------------
# Callback 10b — Populate player dropdown options from portrait
# ---------------------------------------------------------------------------

@callback(
    Output("player-select-dropdown", "options"),
    Output("player-select-dropdown", "value"),
    Input("portrait-store", "data"),
)
def populate_player_dropdown(data):
    portrait = _deserialize(data)
    if not portrait:
        return [], []
    options = charts.player_options_from_portrait(portrait)
    # Default: pre-select top-3 by PA
    default = [o["value"] for o in options[:3]]
    return options, default


# ---------------------------------------------------------------------------
# Callback 10c — Render player comparison charts from dropdown selection
# ---------------------------------------------------------------------------

@callback(
    Output("player-radar-chart", "figure"),
    Output("player-bars-chart",  "figure"),
    Input("player-select-dropdown", "value"),
    State("portrait-store", "data"),
)
def update_player_comparison(selected_ids, data):
    portrait = _deserialize(data)
    if not portrait or not selected_ids:
        empty = charts.empty_figure("Select players from the dropdown above")
        return empty, empty
    radar = charts.player_metrics_radar(portrait, selected_ids)
    bars  = charts.player_metrics_bars(portrait, selected_ids)
    return radar, bars


# ---------------------------------------------------------------------------
# Callback 11 — Toggle philosophy collapse open/closed
# ---------------------------------------------------------------------------

@callback(
    Output({"type": "phil-collapse", "code": MATCH}, "is_open"),
    Input({"type": "phil-btn", "code": MATCH}, "n_clicks"),
    State({"type": "phil-collapse", "code": MATCH}, "is_open"),
    prevent_initial_call=True,
)
def toggle_philosophy_collapse(n_clicks, is_open):
    return not is_open


# ---------------------------------------------------------------------------
# Callback 12 — Populate breakdown content when a collapse opens
# ---------------------------------------------------------------------------

@callback(
    Output({"type": "phil-collapse-content", "code": MATCH}, "children"),
    Input({"type": "phil-collapse", "code": MATCH}, "is_open"),
    State("portrait-store", "data"),
    State({"type": "phil-collapse-content", "code": MATCH}, "id"),
    prevent_initial_call=True,
)
def populate_philosophy_breakdown(is_open, portrait_json, component_id):
    if not is_open or not portrait_json:
        return no_update
    portrait = _deserialize(portrait_json)
    if not portrait:
        return no_update
    code = component_id["code"]
    return charts.philosophy_breakdown_card(portrait, code)


# ---------------------------------------------------------------------------
# Clientside callback — force Plotly repaint after portrait loads
# Fixes a GPU-compositing bug where Plotly Table text doesn't paint on the
# first render after transitioning from an empty figure to real data.
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(data, activeTab) {
        // Fire multiple repaints to catch Plotly tables before and after
        // they finish rendering into previously-hidden tab containers.
        // Use Plotly.redraw() for table traces (resize() doesn't repaint text).
        function repaint() {
            window.dispatchEvent(new Event('resize'));
            var graphs = document.querySelectorAll('.js-plotly-plot');
            graphs.forEach(function(g) {
                try {
                    var d = g._fullData || g.data;
                    if (d && d[0] && d[0].type === 'table') {
                        Plotly.redraw(g);
                    } else {
                        Plotly.Plots.resize(g);
                    }
                } catch(e) {}
            });
        }
        setTimeout(repaint, 200);
        setTimeout(repaint, 700);
        return window.dash_clientside.no_update;
    }
    """,
    Output("status-banner", "id"),   # dummy stable output (id never changes)
    Input("portrait-store", "data"),
    Input("main-tabs", "active_tab"),
)


# ---------------------------------------------------------------------------
# Comparison Callbacks — Team A and Team B portrait builders + chart renderers
# ---------------------------------------------------------------------------

def _build_cmp_portrait(team: str, season: int):
    """Shared portrait builder for comparison tab (no running spinner)."""
    import dash_bootstrap_components as dbc
    from dash import html
    try:
        statcast = pull_statcast_season(int(season))
        portrait = build_team_portrait(team, int(season), statcast=statcast)
        serialized = _serialize(portrait)
        cov = portrait.get("data_coverage", 0.0)
        banner = dbc.Alert(
            [html.Strong(f"{team} {season}"), f" loaded · {cov:.0%} coverage"],
            color="success", className="py-1 mb-0",
        )
        return serialized, banner
    except Exception as exc:
        log.exception("compare portrait failed: %s", exc)
        return no_update, dbc.Alert(f"Error: {exc}", color="danger", className="py-1")


@callback(
    Output("cmp-store-a",    "data"),
    Output("cmp-status-a",   "children"),
    Input("cmp-btn-a",       "n_clicks"),
    State("cmp-team-a",      "value"),
    State("cmp-season-a",    "value"),
    running=[
        (Output("cmp-btn-a", "disabled"), True, False),
        (Output("cmp-status-a", "children"),
         __import__("dash_bootstrap_components").Alert(
             "Loading Team A…", color="primary", className="py-1 mb-0"),
         no_update),
    ],
    prevent_initial_call=True,
)
def build_cmp_portrait_a(n_clicks, team, season):
    if not team or not season:
        return no_update, no_update
    return _build_cmp_portrait(team, int(season))


@callback(
    Output("cmp-store-b",    "data"),
    Output("cmp-status-b",   "children"),
    Input("cmp-btn-b",       "n_clicks"),
    State("cmp-team-b",      "value"),
    State("cmp-season-b",    "value"),
    running=[
        (Output("cmp-btn-b", "disabled"), True, False),
        (Output("cmp-status-b", "children"),
         __import__("dash_bootstrap_components").Alert(
             "Loading Team B…", color="warning", className="py-1 mb-0"),
         no_update),
    ],
    prevent_initial_call=True,
)
def build_cmp_portrait_b(n_clicks, team, season):
    if not team or not season:
        return no_update, no_update
    return _build_cmp_portrait(team, int(season))


@callback(
    Output("cmp-radar",      "figure"),
    Output("cmp-dim-bars",   "figure"),
    Output("cmp-batting",    "figure"),
    Output("cmp-archetypes", "figure"),
    Input("cmp-store-a",     "data"),
    Input("cmp-store-b",     "data"),
)
def update_compare_charts(data_a, data_b):
    pa = _deserialize(data_a)
    pb = _deserialize(data_b)
    return (
        charts.compare_radar(pa, pb),
        charts.compare_dim_bars(pa, pb),
        charts.compare_batting_bars(pa, pb),
        charts.compare_archetype_bars(pa, pb),
    )
