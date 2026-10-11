"""
callbacks.py — Dash callback definitions.

Pattern:
  1. (team, season) → portrait-store   [a "TEAM|SEASON" key, resolved by store]
  2. portrait-store → each chart       [pure figure builders]

Nothing here builds or writes anything. Every read goes through store.py,
which reads the published tables in Supabase with the anon key; portraits are
built and published by scripts/refresh.py on the build machine.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from dash import (Input, Output, State, callback, no_update, MATCH, ctx,
                  clientside_callback, html)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

import charts
import store
from dash.exceptions import PreventUpdate
from layout import team_header

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The portrait store holds a KEY, not a portrait
# ---------------------------------------------------------------------------
#
# It used to hold the serialised portrait — 286 KB of JSON. Dash sends a
# dcc.Store's contents back to the server for every callback that reads it, and
# 23 callbacks read this one, so a single team switch uploaded 5.41 MB from the
# browser. The store holds "TEAM|SEASON" — about a dozen bytes — and the
# portrait stays server-side, cached by store.get_portrait.

def _store_key(team: str, season) -> str:
    return f"{team}|{season}"


def _deserialize(data):
    """Resolve whatever the portrait store holds into a portrait dict."""
    if not data or not isinstance(data, str) or "|" not in data:
        return None
    team, _, season = data.partition("|")
    try:
        portrait, _status = store.get_portrait(team, int(season))
    except Exception as exc:
        log.warning("Portrait resolve failed for %s: %s", data, exc)
        return None
    return portrait


# ---------------------------------------------------------------------------
# Callback 1 — Resolve the selected portrait
# ---------------------------------------------------------------------------

@callback(
    Output("portrait-store", "data"),
    Output("status-banner",  "children"),
    Input("team-dropdown",   "value"),
    Input("season-dropdown", "value"),
)
def build_portrait(team: str, season: int):
    import dash_bootstrap_components as dbc
    from dash import html

    if not team or not season:
        return no_update, dbc.Alert("Select a team and season.", color="warning", className="py-1")

    t0 = time.perf_counter()
    try:
        portrait, status = store.get_portrait(team, int(season))
    except Exception as exc:
        log.exception("portrait read failed for %s %s", team, season)
        return no_update, dbc.Alert(
            [html.Strong(f"{team} {season}"),
             f" could not be loaded — the database did not answer ({exc})."],
            color="danger", className="py-1 mb-0")

    if status == store.MISSING:
        return no_update, dbc.Alert(
            [html.Strong(f"{team} {season}"), " has not been published yet."],
            color="warning", className="py-1 mb-0")
    if status == store.OUTDATED:
        return no_update, dbc.Alert(
            [html.Strong(f"{team} {season}"),
             " was built by an older version of the tag logic and is being "
             "replaced. It will appear after the next publish."],
            color="warning", className="py-1 mb-0")

    mode = portrait.get("temporal", {}).get("mode", "—")
    cov = portrait.get("data_coverage", 0.0)
    log.info("Portrait %s %s served in %.2fs", team, season, time.perf_counter() - t0)
    return _store_key(team, season), dbc.Alert(
        [html.Strong(f"{team} {season}"), f" loaded · mode={mode} · coverage={cov:.0%}"],
        color="success", className="py-1 mb-0")


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

# ---------------------------------------------------------------------------
# Callback 4 — Dimension confidence bars
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Callback 5 — Park factor gauge
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Callback 5b — Roster control chart
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Callback 6 — Spin efficiency
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Callback 7 — Batting metrics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Callback 8 — Hitter archetype pie
# ---------------------------------------------------------------------------

def _league_identity_file(season: int) -> dict:
    """One season's league identity, {} if that season is unpublished."""
    return store.get_league(season, "identity")


def _drift_seasons() -> dict:
    """Every league identity file, keyed by season."""
    out = {}
    for yr in range(2015, 2027):
        d = _league_identity_file(yr)
        if d:
            out[yr] = d
    return out


# How many tags to pre-select. Enough to show a shape, few enough to read;
# the reader adds or removes from there.
_DRIFT_DEFAULT_N = 6


@callback(
    Output({"type": "drift-tags", "unit": MATCH}, "options"),
    Output({"type": "drift-tags", "unit": MATCH}, "value"),
    Input("team-dropdown", "value"),
    Input({"type": "drift-kind", "unit": MATCH}, "value"),
    Input({"type": "drift-tags", "unit": MATCH}, "value"),
    State({"type": "drift-tags", "unit": MATCH}, "id"),
)
def drift_tag_options(team, kind, selected, comp_id):
    """
    Repopulate the tag checklist, and hold the selection to DRIFT_MAX_TAGS.

    Options are ranked by the largest deviation the tag ever reached, so the
    order itself says which tags defined this franchise — the ranking the old
    chart applied silently.

    Once the cap is reached the remaining options are disabled rather than
    silently ignored: the limit exists because more than eight lines stop
    being separable by colour, and a control that quietly drops a pick would
    be worse than one that says it is full.
    """
    unit = (comp_id or {}).get("unit", "offense")
    if not team:
        return [], []
    choices = charts.drift_tag_choices(team, unit, _drift_seasons(), kind)
    available = [t for t, _ in choices]

    trig = ctx.triggered_id
    picked_tags = (isinstance(trig, dict) and trig.get("type") == "drift-tags")
    if picked_tags and selected is not None:
        value = [t for t in selected if t in available][:charts.DRIFT_MAX_TAGS]
    else:
        # Team or kind changed — reset to the most-defining handful.
        value = available[:_DRIFT_DEFAULT_N]

    full = len(value) >= charts.DRIFT_MAX_TAGS
    # The swatch in the label ties each option to the line it will draw, so
    # the picker and the chart share one identity for a tag.
    slot = {t: i for i, t in enumerate(value)}
    options = []
    for t in available:
        colour = (charts.DRIFT_LINE_COLORS[slot[t] % len(charts.DRIFT_LINE_COLORS)]
                  if t in slot else None)
        label = (html.Span([
            html.Span(style={"display": "inline-block", "width": "8px",
                             "height": "8px", "borderRadius": "2px",
                             "backgroundColor": colour, "marginRight": "6px"}),
            t,
        ]) if colour else t)
        options.append({"label": label, "value": t,
                        "disabled": full and t not in slot})
    return options, value


@callback(
    Output({"type": "drift-chart", "unit": MATCH}, "figure"),
    Input("team-dropdown", "value"),
    Input({"type": "drift-tags", "unit": MATCH}, "value"),
    State({"type": "drift-chart", "unit": MATCH}, "id"),
)
def drift_chart(team, tags, comp_id):
    unit = (comp_id or {}).get("unit", "offense")
    if not team:
        return charts.empty_figure("Select a team")
    return charts.team_drift_chart(team, unit, _drift_seasons(), tags=tags)


def _league_baselines(season) -> dict:
    """League mean trait densities per unit, from the season's league identity."""
    return _league_identity_file(season).get("_baselines", {})

@callback(Output("hitter-pie", "figure"), Input("portrait-store", "data"))
def hitter_pie(data):
    p = _deserialize(data)
    if not p:
        return charts.empty_figure()
    base = _league_baselines(p.get("season")).get("offense")
    return charts.hitter_trait_density(p, baseline=base)


# ---------------------------------------------------------------------------
# Callback 9 — Hitter archetype table
# ---------------------------------------------------------------------------

@callback(Output("hitter-plane", "figure"), Input("portrait-store", "data"))
def hitter_plane(data):
    p = _deserialize(data)
    return charts.hitter_plane(p) if p else charts.empty_figure()


@callback(
    Output({"type": "roster-body", "unit": MATCH}, "children"),
    Output({"type": "roster-pageinfo", "unit": MATCH}, "children"),
    Input("portrait-store", "data"),
    Input({"type": "roster-prev", "unit": MATCH}, "n_clicks"),
    Input({"type": "roster-next", "unit": MATCH}, "n_clicks"),
    State({"type": "roster-body", "unit": MATCH}, "id"),
)
def roster_body(data, prev_clicks, next_clicks, comp_id):
    """
    One callback for all three rosters via MATCH on the unit.

    Page is derived from the click counts rather than held in a Store: with
    only forward/back controls, next − prev IS the page, and roster_table
    clamps it to the valid range. One less piece of state to keep in sync
    with the portrait changing underneath it.
    """
    p = _deserialize(data)
    unit = (comp_id or {}).get("unit", "hitters")
    if not p:
        return "Load a portrait to see the roster.", ""
    page = (next_clicks or 0) - (prev_clicks or 0)
    table, page, n_pages = charts.roster_table(p, unit, page)
    info = f"{page + 1} / {n_pages}" if n_pages > 1 else ""
    return table, info


# ---------------------------------------------------------------------------
# Callback 10a — Bullpen charts
# ---------------------------------------------------------------------------
# The starter and bullpen Plotly tables were replaced by the paginated
# per-kind roster above (one MATCH callback drives all three units), so their
# callbacks are gone. charts.starter_archetype_bars / bullpen_detail_table
# remain for now and are unused.

@callback(Output("bullpen-trait-density", "figure"), Input("portrait-store", "data"))
def bullpen_trait_density_chart(data):
    p = _deserialize(data)
    if not p:
        return charts.empty_figure()
    base = _league_baselines(p.get("season")).get("bullpen")
    return charts.bullpen_trait_density(p, baseline=base)


@callback(Output("rotation-trait-density", "figure"), Input("portrait-store", "data"))
def rotation_trait_density_chart(data):
    p = _deserialize(data)
    if not p:
        return charts.empty_figure()
    base = _league_baselines(p.get("season")).get("rotation")
    return charts.rotation_trait_density(p, baseline=base)


# ---------------------------------------------------------------------------
# Callback 9b — Hitter skill affinity heatmap
# ---------------------------------------------------------------------------

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
# Compare — player-seasons side by side
# ---------------------------------------------------------------------------
#
# Replaces the team-vs-team callbacks. Two of those fed charts that read
# portrait["scores"], a key that has been nested under `philosophy` since the
# schema changed, so cmp-radar and cmp-dim-bars had been drawing flat zero for
# every team — 13 and 12 values, all 0.0 — for as long as the nesting.

@callback(
    Output("cmp-players",  "options"),
    Input("cmp-players",   "search_value"),
    Input("cmp-side",      "value"),
    State("cmp-players",   "value"),
)
def cmp_player_options(search, side, chosen):
    """
    Search from inside the dropdown, resolved server-side.

    The index holds 18,664 player-seasons; handing them all to the dropdown
    would ship megabytes for the browser to filter. Driving it from
    `search_value` means the box the user types in is the box that searches —
    this used to be fed by a separate input beside it, so the dropdown's own
    search filtered an empty list until you had typed somewhere else.

    Already-chosen player-seasons are appended to every result. Dash renders a
    selected value using the options list, so a selection whose option has been
    replaced by the next search would show as a bare key or vanish.
    """
    opts = charts.compare_player_options(search, side or "H")
    have = {o["value"] for o in opts}
    for row in charts.compare_rows(chosen or []):
        k = charts.compare_key(row)
        if k not in have:
            opts.append({"label": f"{row['n']} · {row['t']} {row['s']} · "
                                  f"{row.get('v') or 0:,} {'PA' if row['d']=='H' else 'BF'}",
                         "value": k})
    return opts


@callback(
    Output("cmp-summary", "children"),
    Output("cmp-matrix",  "children"),
    Output("cmp-metrics", "figure"),
    Output("cmp-status",  "children"),
    Input("cmp-players",  "value"),
)
def cmp_render(keys):
    import dash_bootstrap_components as dbc

    keys = (keys or [])[:charts.COMPARE_MAX]
    rows = charts.compare_rows(keys)
    if not rows:
        return (html.Div(), charts.compare_tag_matrix([]),
                charts.empty_figure("Search for player-seasons to compare"), "")

    sides = {r.get("d") for r in rows}
    status = dbc.Alert(
        [html.Strong(f"{len(rows)} player-season{'s' if len(rows) != 1 else ''}"),
         " · " + " vs ".join(f"{r['n']} {r['t']} {r['s']}" for r in rows)]
        + ([html.Br(), html.Small(
            "Mixing hitters and pitchers — they share no percentile axes, so "
            "the chart below cannot be drawn.", className="text-warning")]
           if len(sides) > 1 else []),
        color="warning" if len(sides) > 1 else "success", className="py-1 mb-0")

    return (charts.compare_summary(rows),
            charts.compare_tag_matrix(rows),
            charts.compare_metrics_bars(rows),
            status)


# ---------------------------------------------------------------------------
# Scout Tab — profile search across seasons
# ---------------------------------------------------------------------------

@callback(
    Output("scout-tags", "options"),
    Input("scout-side",  "value"),
    Input("scout-kinds", "value"),
)
def scout_tag_choices(side, kinds):
    """
    Narrow the tag picker to the side, and to the kinds if any are chosen.

    Tags are side-scoped by construction — 40 hitter-only, 53 pitcher-only,
    one shared — so offering all 94 regardless of side would make most of any
    list dead options.
    """
    return charts.scout_tag_options(side=side or "H", kinds=kinds or None)


@callback(
    Output("scout-store",  "data"),
    Output("scout-status", "children"),
    Output("scout-page",   "data", allow_duplicate=True),
    Input("scout-btn",     "n_clicks"),
    State("scout-side",    "value"),
    State("scout-kinds",   "value"),
    State("scout-tags",    "value"),
    State("scout-match",   "value"),
    State("scout-season-min", "value"),
    State("scout-season-max", "value"),
    State("scout-min-vol", "value"),
    State("scout-sort",    "value"),
    State("scout-flags",   "value"),
    prevent_initial_call=True,
)
def scout_search(n_clicks, side, kinds, tags, match, s_min, s_max, min_vol, sort, flags):
    """
    Run the search in the database (scout_search). No re-derivation.
    """
    import dash_bootstrap_components as dbc
    from dash import html

    if not n_clicks:
        return no_update, "", 0

    rows, total = charts.scout_query(
        tags=tags or None, kinds=kinds or None, side=side or "H",
        match_all=(match != "any"),
        season_min=int(s_min) if s_min else None,
        season_max=int(s_max) if s_max else None,
        min_vol=int(min_vol or 0),
        exclude_limited="lim" in (flags or []),
        sort=sort or "extreme",
    )

    what = (" + " if match != "any" else " / ").join(tags) if tags else \
           (", ".join(kinds) if kinds else "any tag")
    status = dbc.Alert(
        [html.Strong(f"{total:,} player-seasons"),
         f" · {what} · {'hitters' if (side or 'H') == 'H' else 'pitchers'}"
         f" · {s_min}–{s_max}"]
        + ([html.Br(), html.Small(f"showing the first {len(rows):,}")]
           if total > len(rows) else []),
        color="success" if rows else "secondary", className="py-1 mb-0",
    )
    # The database already returns slim rows — only what the table renders.
    return rows, status, 0


@callback(
    Output("scout-page", "data"),
    Input("scout-prev",  "n_clicks"),
    Input("scout-next",  "n_clicks"),
    State("scout-page",  "data"),
    State("scout-store", "data"),
    prevent_initial_call=True,
)
def scout_paginate(prev_c, next_c, page, rows):
    page = int(page or 0)
    n = len(rows or [])
    last = max(0, (n + charts.SCOUT_PAGE_SIZE - 1) // charts.SCOUT_PAGE_SIZE - 1)
    if ctx.triggered_id == "scout-prev":
        return max(0, page - 1)
    if ctx.triggered_id == "scout-next":
        return min(last, page + 1)
    return page


@callback(
    Output("comp-player", "options"),
    Input("comp-player",  "search_value"),
    Input("scout-side",   "value"),
    State("comp-player",  "value"),
)
def comp_player_options(search, side, chosen):
    """Searches from inside the dropdown — see cmp_player_options."""
    opts = charts.compare_player_options(search, side or "H")
    have = {o["value"] for o in opts}
    for row in charts.compare_rows([chosen] if chosen else []):
        k = charts.compare_key(row)
        if k not in have:
            opts.append({"label": f"{row['n']} · {row['t']} {row['s']} · "
                                  f"{row.get('v') or 0:,} {'PA' if row['d']=='H' else 'BF'}",
                         "value": k})
    return opts


@callback(
    Output("comp-require", "options"),
    Input("scout-side",    "value"),
)
def comp_require_options(side):
    """Attribute tags — the ones held out of the vector — offered as a filter."""
    return charts.attribute_options(side or "H")


@callback(
    Output("comp-results", "children"),
    Input("comp-player",   "value"),
    Input("comp-min-vol",  "value"),
    Input("comp-require",  "value"),
)
def comp_render(key, min_vol, require):
    """
    Nearest player-seasons by tag profile — the League Identity Board's
    mechanism one level down, computed per request rather than stored.
    """
    if not key:
        return charts.comp_results(None, [])
    target, comps = charts.player_comps(key, min_vol=int(min_vol or 0),
                                        require=require or None)
    return charts.comp_results(target, comps)


@callback(
    Output("scout-results", "children"),
    Input("scout-store",    "data"),
    Input("scout-page",     "data"),
)
def scout_render(rows, page):
    if not rows:
        from dash import html
        return html.Div("Pick a side and one or more tags, then Search.",
                        className="text-secondary small p-3")
    return charts.scout_results(rows, int(page or 0))



# ---------------------------------------------------------------------------
# Construction vs Results callbacks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Team split resistance card (Overview tab)
# ---------------------------------------------------------------------------

@callback(Output("team-split-card", "children"), Input("portrait-store", "data"))
def team_split_card(data):
    p = _deserialize(data)
    if not p:
        return "Load a portrait to see team split resistance."
    return charts.team_split_card(p)


@callback(Output("team-identity-card", "children"), Input("portrait-store", "data"))
def team_identity_card(data):
    p = _deserialize(data)
    if not p:
        return "Load a portrait to see team identity."
    fingerprint = position = None
    try:
        league = _league_identity_file(p.get("season"))
        if league:
            entry = league.get(p.get("team")) or {}
            fingerprint = entry.get("fingerprint")
            # Where the team sits relative to everyone else — this is what
            # replaced the headline labels, and it had never been wired in, so
            # the card rendered a bare "—" where the label used to be.
            position = {"neighbours": entry.get("neighbours") or {},
                        "uniqueness": entry.get("uniqueness") or {}}
    except Exception:
        pass
    return charts.team_identity_card(p, fingerprint=fingerprint, position=position)


# ---------------------------------------------------------------------------
# Batter split resistance heatmap
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Arsenal 3D trajectory — populate dropdown + render chart
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Arsenal panel — two round trips, not four
# ---------------------------------------------------------------------------
#
# This was four callbacks in a chain: the portrait chose a pitcher, the
# pitcher's value fetched his pitch list, and the pitch list drew the 3D chart
# and then the tunnel chart. Each arrow was a full browser-server round trip
# waiting on the one before, so loading a team took until 960 ms to finish
# drawing the arsenal while no single callback took longer than 82 ms. The
# cost was latency, not work.
#
# Both option lists come from the portrait alone — the pitch list only needs
# to know which pitcher was picked, and the same callback is the thing picking
# him — so they collapse into one. The two figures take identical inputs, so
# they collapse into the other.

@callback(
    Output("arsenal-pitcher-dropdown", "options"),
    Output("arsenal-pitcher-dropdown", "value"),
    Output("arsenal-pitch-filter",     "options"),
    Output("arsenal-pitch-filter",     "value"),
    Input("portrait-store", "data"),
)
def populate_arsenal(data):
    """
    Pitcher list and pitch list together.

    Defaults to the heaviest-workload arm with all of his pitches on. The
    pitch list used to wait for the pitcher dropdown's value to travel back to
    the server before it could be built, which is a round trip to learn
    something this callback already knew.
    """
    p = _deserialize(data)
    if not p:
        return [], [], [], []
    choices = charts.arsenal_pitcher_choices(p)
    if not choices:
        return [], [], [], []
    options = [{"label": f"{n}  ({c:,} pitches)", "value": pid}
               for pid, n, c in choices]
    default = [choices[0][0]]
    pitches = charts.arsenal_pitch_choices(p, default)
    return options, default, [{"label": pt, "value": pt} for pt in pitches], pitches


@callback(
    Output("arsenal-pitch-filter", "options", allow_duplicate=True),
    Output("arsenal-pitch-filter", "value",   allow_duplicate=True),
    Input("arsenal-pitcher-dropdown", "value"),
    State("portrait-store", "data"),
    State("arsenal-pitch-filter", "options"),
    prevent_initial_call=True,
)
def arsenal_pitches_for_pitcher(player_ids, data, current_options):
    """
    Re-derive the pitch list when the USER changes pitcher.

    Fires on any change to the pitcher value, including the one
    populate_arsenal makes when a new portrait loads — Dash cannot tell a
    programmatic set from a click. Left alone it re-sets a pitch list that is
    already correct, and that write re-triggers both figures, so every team
    switch drew the arsenal twice: once at 682 ms and again at 917 ms.
    #
    So compare against what the filter already offers and stand down when it
    matches. A real pitcher change yields a different pitch list and proceeds;
    populate_arsenal's set yields an identical one and stops here.
    """
    p = _deserialize(data)
    if not p or not player_ids:
        raise PreventUpdate
    pitches = charts.arsenal_pitch_choices(p, player_ids)
    if [o.get("value") for o in (current_options or [])] == list(pitches):
        raise PreventUpdate
    return [{"label": pt, "value": pt} for pt in pitches], pitches


@callback(
    Output("arsenal-3d-chart",      "figure"),
    Output("tunnel-profile-chart",  "figure"),
    Input("arsenal-pitcher-dropdown", "value"),
    Input("arsenal-pitch-filter",     "value"),
    State("portrait-store", "data"),
)
def arsenal_figures(player_ids, pitch_types, data):
    """Both arsenal views in one trip — same inputs, same portrait, same work."""
    p = _deserialize(data)
    if not p:
        empty = charts.empty_figure("Load a portrait to see pitch trajectories")
        return empty, charts.empty_figure("Load a portrait to see tunnelling")
    return (charts.pitch_arsenal_3d(p, player_ids, pitch_types),
            charts.tunnel_profile(p, player_ids, pitch_types))


# ---------------------------------------------------------------------------
# League Identity Board
# ---------------------------------------------------------------------------

@callback(
    Output("league-board-table", "children"),
    Input("league-board-season", "value"),
)
def league_identity_board(season):
    if not season:
        return "Select a season."

    try:
        league = _league_identity_file(season)
    except Exception as exc:
        return f"Failed to load league identity data: {exc}"

    if not league:
        return dbc_alert_no_data(season)

    rows = []
    for team, identity in sorted(league.items()):
        if team.startswith("_"):
            continue
        offense  = identity.get("offense") or {}
        rotation = identity.get("rotation") or {}
        bullpen  = identity.get("bullpen") or {}
        fit      = identity.get("fit") or {}
        rot_density = rotation.get("trait_density") or {}
        top_traits = " · ".join(f"{t} {v:.0%}"
                                for t, v in list(rot_density.items())[:3]) or "—"
        fp = identity.get("fingerprint") or {}
        fp_items = []
        for unit, prefix in [("offense", "O"), ("rotation", "R"), ("bullpen", "B")]:
            for row in (fp.get(unit) or []):
                # Fingerprints are per-kind rows carrying a `tags` list; older
                # league files stored a flat list of entries.
                entries = row.get("tags") if isinstance(row, dict) and "tags" in row \
                    else [row]
                for e in entries or []:
                    if "deviation" not in e:
                        continue
                    fp_items.append((abs(e["deviation"]),
                                     f"{prefix}: {e['tag']} {e['deviation']*100:+.0f}"))
        fp_items.sort(key=lambda x: -x[0])
        fp_str = " · ".join(t for _, t in fp_items[:3]) or "league-typical"

        # Headline labels retired — a team is described by where it sits
        # relative to the league, not by which bucket a cutoff assigned it to.
        nb = (identity.get("neighbours") or {}).get("offense") or []
        uq = (identity.get("uniqueness") or {}).get("offense") or {}
        rows.append({
            "Team":              team,
            "Fingerprint":       fp_str,
            "Plays Like":        ", ".join(x["team"] for x in nb[:2]) or "—",
            # How strong the "Plays Like" comparison actually is — similarity
            # to the closest comparable offense. A sortable number suits a
            # table better than the card's one-word band. The old 0–100
            # "distinctiveness" was a min-max rescale of near-zero mean
            # similarities and rated teams with near-twins as highly distinct.
            "Distinctiveness":   uq.get("closest"),
            "Power Share":       (f"{offense['power_share']:.0%}"
                                  if offense.get("power_share") is not None else "—"),
            "Top Rotation Traits": top_traits,
            "Bullpen Mechanism": bullpen.get("out_mechanism") or "—",
            "Fit Notes":         " ".join(v for v in fit.values() if v) or "—",
        })

    columns = [
        {"name": "Team",              "id": "Team"},
        {"name": "Fingerprint",       "id": "Fingerprint"},
        {"name": "Plays Like",        "id": "Plays Like"},
        {"name": "Comp Strength",     "id": "Distinctiveness"},
        {"name": "Power Share",       "id": "Power Share"},
        {"name": "Top Rotation Traits", "id": "Top Rotation Traits"},
        {"name": "Bullpen Mechanism", "id": "Bullpen Mechanism"},
        {"name": "Fit Notes",         "id": "Fit Notes"},
    ]

    from dash import dash_table
    return dash_table.DataTable(
        data=rows,
        columns=columns,
        sort_action="native",
        filter_action="native",
        page_size=30,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1a2233", "color": "#9ca3af",
            "fontWeight": "600", "fontSize": "0.7rem",
            "textTransform": "uppercase", "letterSpacing": "0.05em",
            "border": "1px solid #374151",
        },
        style_cell={
            "backgroundColor": "#1f2937", "color": "#f9fafb",
            "fontSize": "0.8rem", "border": "1px solid #374151",
            "padding": "6px 10px", "textAlign": "left",
        },
        style_data_conditional=[
            {"if": {"column_id": "Fit Notes"}, "maxWidth": "420px",
             "whiteSpace": "normal", "height": "auto"},
        ],
    )


def dbc_alert_no_data(season):
    import dash_bootstrap_components as dbc
    return dbc.Alert(
        f"No league identity data for {season} has been published. On the "
        f"build machine: build_league_identity.py {season}, then publish.py.",
        color="warning", className="py-2 mb-0",
    )
