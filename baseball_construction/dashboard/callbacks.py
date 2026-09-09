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
from dash import (Input, Output, State, callback, no_update, MATCH, ctx,
                  clientside_callback, html)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

from team_portrait import (build_team_portrait, PORTRAIT_SCHEMA_VERSION,
                           portrait_is_current)
from ingest import pull_statcast_season
import charts
from layout import team_header

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Portrait disk cache
# ---------------------------------------------------------------------------

import time as _time
from datetime import datetime as _datetime

_PORTRAIT_CACHE = _HERE.parent / "data" / "processed" / "portraits"
_PORTRAIT_CACHE.mkdir(parents=True, exist_ok=True)

# Read-only mode: serve portraits, never build one.
#
# On a cache miss the dashboard otherwise loads a full season of Statcast
# (~1.2 GB resident for 712k pitches) and builds the portrait itself, then
# writes it. Two things wrong with that off a build machine. It needs the
# whole ~977 MB of raw pitch data on the web tier, when the artefact a reader
# actually consumes is a 239 KB JSON. And it makes the dashboard a WRITER —
# which is how a portrait built against half-regenerated pools got saved
# mid-rebuild and then looked like a valid cache hit forever.
#
# Set BASEBALL_READ_ONLY=1 in any deployment. Portraits then come from disk
# or Supabase Storage, and a genuine miss reports itself instead of silently
# spending a gigabyte to paper over it. scripts/refresh.py is the only writer.
import os as _os

READ_ONLY = _os.environ.get("BASEBALL_READ_ONLY", "").strip().lower() in (
    "1", "true", "yes", "on")
if READ_ONLY:
    log.info("BASEBALL_READ_ONLY set — portraits will be served, never built")


def _read_only_banner(team: str, season) -> "object":
    import dash_bootstrap_components as dbc
    from dash import html
    return dbc.Alert(
        [html.Strong(f"{team} {season}"),
         " is not in the portrait cache or storage, and this instance is "
         "read-only. Build it with scripts/refresh.py."],
        color="warning", className="py-1 mb-0",
    )


def _cache_path(team: str, season: int) -> Path:
    return _PORTRAIT_CACHE / f"{team}_{season}.json"


def _cache_valid(path: Path, season: int) -> bool:
    """Historical seasons: cache indefinitely. Current season: 24h TTL."""
    if not path.exists():
        return False
    current_year = _datetime.now().year
    if int(season) < current_year:
        return True
    return (_time.time() - path.stat().st_mtime) < 86400


def _schema_current(portrait: dict | None) -> bool:
    """
    True if a stored portrait is safe to serve — current schema AND built
    against the current pools and gates. See team_portrait.build_fingerprint;
    the schema alone cannot tell a portrait built before a data repair from
    one built after, which is how stale portraits reached readers.
    """
    return portrait_is_current(portrait)


def _storage_download(team: str, season: int) -> str | None:
    """
    Try to fetch portrait JSON from Supabase Storage 'portraits' bucket.
    Returns the JSON string if found, None otherwise.

    Used in production where the local filesystem is ephemeral — portraits
    are pre-generated and uploaded to Supabase Storage so deploys don't
    require building from scratch.
    """
    try:
        from database import get_client
        filename = f"{team}_{season}.json"
        data = get_client().storage.from_("portraits").download(filename)
        if data:
            text = data.decode("utf-8") if isinstance(data, bytes) else data
            if not _schema_current(json.loads(text)):
                log.info(
                    "Stale portrait in Supabase Storage (schema mismatch): %s %s — ignoring",
                    team, season,
                )
                return None
            # Write to local cache so subsequent requests hit disk
            cp = _cache_path(team, int(season))
            cp.write_text(text)
            log.info("Portrait downloaded from Supabase Storage: %s %s", team, season)
            return text
    except Exception as exc:
        log.debug("Storage download miss for %s %s: %s", team, season, exc)
    return None


def _storage_upload(team: str, season: int, json_text: str) -> None:
    """Upload portrait JSON to Supabase Storage (best-effort, non-blocking)."""
    try:
        from database import get_client
        filename = f"{team}_{season}.json"
        data = json_text.encode("utf-8")
        client = get_client()
        # Upsert: overwrite if exists
        try:
            client.storage.from_("portraits").remove([filename])
        except Exception:
            pass
        client.storage.from_("portraits").upload(
            filename, data,
            file_options={"content-type": "application/json"},
        )
        log.info("Portrait uploaded to Supabase Storage: %s %s", team, season)
    except Exception as exc:
        log.warning("Storage upload failed for %s %s: %s", team, season, exc)


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
    Input("team-dropdown",   "value"),
    Input("season-dropdown", "value"),
    running=[
        (Output("team-dropdown", "disabled"), True, False),
        (Output("season-dropdown", "disabled"), True, False),
        # Targets status-spinner, NOT status-banner: this callback returns the
        # banner itself, and aiming `running` at the same Output forced
        # no_update as the off-value, which Dash renders as a literal
        # {_dash_no_update} object child. Empty string clears the spinner
        # cleanly and leaves the banner to the return value.
        (Output("status-spinner", "children"),
         __import__("dash_bootstrap_components").Alert(
             [
                 __import__("dash").html.Span(
                     className="spinner-border spinner-border-sm me-2",
                     style={"width": "14px", "height": "14px"},
                     **{"role": "status"},
                 ),
                 "Loading team… a first-time build takes 10–30 seconds",
             ],
             color="primary", className="py-1 mb-0 d-flex align-items-center",
         ),
         ""),
    ],
)
def build_portrait(team: str, season: int):
    import dash_bootstrap_components as dbc
    from dash import html

    if not team or not season:
        return no_update, dbc.Alert("Select a team and season.", color="warning", className="py-1")

    # ── Check portrait cache first ────────────────────────────────────────
    cp = _cache_path(team, int(season))

    # 1. Local disk cache (fast — sub-millisecond)
    if _cache_valid(cp, int(season)):
        try:
            t0 = _time.perf_counter()
            cached_json = cp.read_text()
            cached = json.loads(cached_json)
            if _schema_current(cached):
                elapsed = _time.perf_counter() - t0
                mode = cached.get("temporal", {}).get("mode", "—")
                cov  = cached.get("data_coverage", 0.0)
                banner = dbc.Alert(
                    [html.Strong(f"{team} {season}"),
                     f" loaded · mode={mode} · coverage={cov:.0%} · ⚡ cached ({elapsed:.2f}s)"],
                    color="success", className="py-1 mb-0",
                )
                log.info("Portrait cache hit (disk): %s %s (%.2fs)", team, season, elapsed)
                return cached_json, banner
            else:
                log.info("Stale disk-cached portrait (schema mismatch): %s %s — rebuilding", team, season)
        except Exception as exc:
            log.warning("Portrait disk cache read failed: %s", exc)

    # 2. Supabase Storage (production fallback — ~1-2s download)
    storage_json = _storage_download(team, int(season))
    if storage_json:
        try:
            cached = json.loads(storage_json)
            mode = cached.get("temporal", {}).get("mode", "—")
            cov  = cached.get("data_coverage", 0.0)
            banner = dbc.Alert(
                [html.Strong(f"{team} {season}"),
                 f" loaded · mode={mode} · coverage={cov:.0%} · ☁️ from storage"],
                color="success", className="py-1 mb-0",
            )
            return storage_json, banner
        except Exception as exc:
            log.warning("Storage portrait parse failed: %s", exc)

    if READ_ONLY:
        log.warning("Portrait miss for %s %s and READ_ONLY is set — not building",
                    team, season)
        return no_update, _read_only_banner(team, season)

    try:
        statcast = pull_statcast_season(int(season))
        portrait = build_team_portrait(team, int(season), statcast=statcast)
        serialized = _serialize(portrait)
        # Save to local disk cache
        try:
            cp.write_text(serialized)
        except Exception as exc:
            log.warning("Portrait disk cache write failed: %s", exc)
        # Also upload to Supabase Storage (async-ish — non-blocking on failure)
        _storage_upload(team, int(season), serialized)
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

_LEAGUE_FILE_CACHE: dict[int, dict] = {}

def _league_identity_file(season: int) -> dict:
    if season in _LEAGUE_FILE_CACHE:
        return _LEAGUE_FILE_CACHE[season]
    try:
        path = _LEAGUE_IDENTITY_DIR / f"league_identity_{season}.json"
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        data = {}
    _LEAGUE_FILE_CACHE[season] = data
    return data


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


_BASELINE_CACHE: dict[int, dict] = {}

def _league_baselines(season) -> dict:
    """League mean trait densities per unit, from league_identity_{season}.json."""
    try:
        season = int(season)
    except (TypeError, ValueError):
        return {}
    if season in _BASELINE_CACHE:
        return _BASELINE_CACHE[season]
    try:
        path = _LEAGUE_IDENTITY_DIR / f"league_identity_{season}.json"
        base = json.loads(path.read_text()).get("_baselines", {}) if path.exists() else {}
    except Exception:
        base = {}
    _BASELINE_CACHE[season] = base
    return base


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
    Output("cmp-players", "options"),
    Input("cmp-search",   "value"),
    Input("cmp-side",     "value"),
)
def cmp_player_options(query, side):
    """
    Resolve the name search server-side.

    The index holds 18,664 player-seasons; handing them all to the dropdown
    would ship megabytes of options for the browser to filter.
    """
    return charts.compare_player_options(query, side or "H")


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
    Run the search against the portrait index. No database, no re-derivation.
    """
    import dash_bootstrap_components as dbc
    from dash import html

    if not n_clicks:
        return no_update, "", 0

    rows = charts.scout_query(
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
    note = []
    if charts.scout_index_stale():
        note = [html.Br(), html.Small(
            "The index was built from a different portrait build — "
            "re-run scripts/refresh.py.", className="text-warning")]

    status = dbc.Alert(
        [html.Strong(f"{len(rows):,} player-seasons"),
         f" · {what} · {'hitters' if (side or 'H') == 'H' else 'pitchers'}"
         f" · {s_min}–{s_max}", *note],
        color="success" if rows else "secondary", className="py-1 mb-0",
    )
    # Stored trimmed: the full row carries every tag and metric, and only the
    # matched ones are rendered.
    slim = [{k: r[k] for k in ("n", "t", "s", "a", "v", "w", "l", "matched", "score")}
            for r in rows[:2000]]
    return slim, status, 0


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
        path = _LEAGUE_IDENTITY_DIR / f"league_identity_{p.get('season')}.json"
        if path.exists():
            league = json.loads(path.read_text())
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

@callback(
    Output("arsenal-pitcher-dropdown", "options"),
    Output("arsenal-pitcher-dropdown", "value"),
    Input("portrait-store", "data"),
)
def populate_arsenal_pitchers(data):
    """One pitcher selected by default — the heaviest workload on the staff."""
    p = _deserialize(data)
    if not p:
        return [], []
    choices = charts.arsenal_pitcher_choices(p)
    options = [{"label": f"{n}  ({c:,} pitches)", "value": pid}
               for pid, n, c in choices]
    return options, ([choices[0][0]] if choices else [])


@callback(
    Output("arsenal-pitch-filter", "options"),
    Output("arsenal-pitch-filter", "value"),
    Input("arsenal-pitcher-dropdown", "value"),
    State("portrait-store", "data"),
)
def populate_arsenal_pitches(player_ids, data):
    """Pitch filter follows the pitcher choice — all his pitches on by default."""
    p = _deserialize(data)
    if not p or not player_ids:
        return [], []
    pitches = charts.arsenal_pitch_choices(p, player_ids)
    return [{"label": pt, "value": pt} for pt in pitches], pitches


@callback(
    Output("arsenal-3d-chart", "figure"),
    Input("arsenal-pitcher-dropdown", "value"),
    Input("arsenal-pitch-filter", "value"),
    State("portrait-store", "data"),
)
def arsenal_3d(player_ids, pitch_types, data):
    p = _deserialize(data)
    if not p:
        return charts.empty_figure("Load a portrait to see pitch trajectories")
    return charts.pitch_arsenal_3d(p, player_ids, pitch_types)


@callback(
    Output("tunnel-profile-chart", "figure"),
    Input("arsenal-pitcher-dropdown", "value"),
    Input("arsenal-pitch-filter", "value"),
    State("portrait-store", "data"),
)
def tunnel_profile(player_ids, pitch_types, data):
    p = _deserialize(data)
    if not p:
        return charts.empty_figure("Load a portrait to see tunnelling")
    return charts.tunnel_profile(p, player_ids, pitch_types)


# ---------------------------------------------------------------------------
# League Identity Board
# ---------------------------------------------------------------------------

_LEAGUE_IDENTITY_DIR = _HERE.parent / "data" / "processed"


@callback(
    Output("league-board-table", "children"),
    Input("league-board-season", "value"),
)
def league_identity_board(season):
    if not season:
        return "Select a season."

    path = _LEAGUE_IDENTITY_DIR / f"league_identity_{season}.json"
    if not path.exists():
        return dbc_alert_no_data(season)

    try:
        league = json.loads(path.read_text())
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
        f"No league identity data for {season}. Run "
        f"'python3 scripts/build_league_identity.py {season}' to generate it.",
        color="warning", className="py-2 mb-0",
    )
