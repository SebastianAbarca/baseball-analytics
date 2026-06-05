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
# Portrait disk cache
# ---------------------------------------------------------------------------

import time as _time
from datetime import datetime as _datetime

_PORTRAIT_CACHE = _HERE.parent / "data" / "processed" / "portraits"
_PORTRAIT_CACHE.mkdir(parents=True, exist_ok=True)


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
            # Write to local cache so subsequent requests hit disk
            cp = _cache_path(team, int(season))
            cp.write_text(text)
            log.info("Portrait downloaded from Supabase Storage: %s %d", team, season)
            return text
    except Exception as exc:
        log.debug("Storage download miss for %s %d: %s", team, season, exc)
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
        log.info("Portrait uploaded to Supabase Storage: %s %d", team, season)
    except Exception as exc:
        log.warning("Storage upload failed for %s %d: %s", team, season, exc)


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

    # ── Check portrait cache first ────────────────────────────────────────
    cp = _cache_path(team, int(season))

    # 1. Local disk cache (fast — sub-millisecond)
    if _cache_valid(cp, int(season)):
        try:
            t0 = _time.perf_counter()
            cached_json = cp.read_text()
            cached = json.loads(cached_json)
            elapsed = _time.perf_counter() - t0
            mode = cached.get("temporal", {}).get("mode", "—")
            cov  = cached.get("data_coverage", 0.0)
            banner = dbc.Alert(
                [html.Strong(f"{team} {season}"),
                 f" loaded · mode={mode} · coverage={cov:.0%} · ⚡ cached ({elapsed:.2f}s)"],
                color="success", className="py-1 mb-0",
            )
            log.info("Portrait cache hit (disk): %s %d (%.2fs)", team, season, elapsed)
            return cached_json, banner
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
    return charts.hitter_vs_archetype_heatmap(p) if p else charts.empty_figure("Load a portrait to see hitter benchmarks")


@callback(Output("spray-heatmap", "figure"), Input("portrait-store", "data"))
def spray_heatmap(data):
    p = _deserialize(data)
    return charts.team_spray_heatmap(p) if p else charts.empty_figure("Load a portrait to see batted ball profile")


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
    """Shared portrait builder for comparison tab — checks disk cache first."""
    import dash_bootstrap_components as dbc
    from dash import html

    cp = _cache_path(team, int(season))

    if _cache_valid(cp, int(season)):
        try:
            cached_json = cp.read_text()
            cached = json.loads(cached_json)
            cov  = cached.get("data_coverage", 0.0)
            banner = dbc.Alert(
                [html.Strong(f"{team} {season}"), f" loaded · {cov:.0%} coverage · ⚡ cached"],
                color="success", className="py-1 mb-0",
            )
            return cached_json, banner
        except Exception as exc:
            log.warning("Compare portrait disk cache read failed: %s", exc)

    try:
        statcast = pull_statcast_season(int(season))
        portrait = build_team_portrait(team, int(season), statcast=statcast)
        serialized = _serialize(portrait)
        try:
            cp.write_text(serialized)
        except Exception as exc:
            log.warning("Compare portrait disk cache write failed: %s", exc)
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


# ---------------------------------------------------------------------------
# Scout Tab — profile search across seasons
# ---------------------------------------------------------------------------

@callback(
    Output("scout-results", "figure"),
    Output("scout-status",  "children"),
    Input("scout-btn",      "n_clicks"),
    State("scout-archetype",   "value"),
    State("scout-modifiers",   "value"),
    State("scout-season-min",  "value"),
    State("scout-season-max",  "value"),
    State("scout-min-pa",      "value"),
    State("scout-mode",        "value"),
    prevent_initial_call=True,
)
def scout_search(n_clicks, archetype, modifiers, season_min, season_max, min_pa, mode):
    import dash_bootstrap_components as dbc
    import pandas as pd
    import numpy as np
    from ingest import normalize_percentile, pull_fg_batting
    from database import query_batting
    from hitter_archetypes import build_hitter_profile

    if not n_clicks:
        return charts.empty_figure("Set filters and click Search"), ""

    season_min = int(season_min or 2021)
    season_max = int(season_max or 2026)
    min_pa     = int(min_pa or 200)
    modifiers  = modifiers or []

    NORM_SPECS = [
        ("iso",        "ISO_pct",        False),
        ("obp",        "OBP_pct",        False),
        ("bb_rate",    "BB_pct_pct",     False),
        ("k_rate",     "K_pct_raw_pct",  True),
        ("barrel_pct", "Barrel_pct_pct", False),
        ("avg",        "AVG_pct",        False),
        ("contact_pct","Contact_pct_pct",False),
        ("xwoba",      "xwOBA_pct",      False),
    ]

    all_profiles = []

    for season in range(season_min, season_max + 1):
        try:
            df = query_batting(season)
            if df is None or df.empty:
                continue
            df = df[df["pa"] >= min_pa].copy()
            if df.empty:
                continue

            for raw_col, pct_col, invert in NORM_SPECS:
                if raw_col in df.columns:
                    df[pct_col] = normalize_percentile(
                        pd.to_numeric(df[raw_col], errors="coerce"), invert=invert)

            if "k_rate" in df.columns:
                df["K_pct_raw_pct_noninv"] = normalize_percentile(
                    pd.to_numeric(df["k_rate"], errors="coerce"), invert=False)
            if "obp" in df.columns and "iso" in df.columns:
                df["OBP_ISO_gap"] = (pd.to_numeric(df["obp"], errors="coerce") -
                                     pd.to_numeric(df["iso"], errors="coerce"))
                df["OBP_ISO_gap_pct"] = normalize_percentile(df["OBP_ISO_gap"])

            # LuckDelta
            try:
                fg = pull_fg_batting(season)
                if "est_woba_minus_woba_diff" in fg.columns and "key_mlbam" in fg.columns:
                    fg["_luck"] = pd.to_numeric(fg["est_woba_minus_woba_diff"], errors="coerce")
                    fv = fg[fg["_luck"].notna()].copy()
                    if len(fv) > 1:
                        fv["LuckDelta_pct"] = normalize_percentile(fv["_luck"]).values
                        lm = dict(zip(pd.to_numeric(fv["key_mlbam"], errors="coerce"),
                                      fv["LuckDelta_pct"]))
                        df["LuckDelta_pct"] = pd.to_numeric(
                            df["key_mlbam"], errors="coerce").map(lm)
            except Exception:
                pass

            # Attempt rate pct
            df["_tob"] = (pd.to_numeric(df["obp"], errors="coerce").fillna(0) *
                          pd.to_numeric(df["pa"], errors="coerce").fillna(1))
            df["_att"] = (pd.to_numeric(df["sb"], errors="coerce").fillna(0) +
                          pd.to_numeric(df["cs"], errors="coerce").fillna(0))
            df["_att_rate"] = df["_att"] / df["_tob"].clip(lower=1)
            has_att = df["_att"] >= 2
            df["_att_rate_pct"] = np.nan
            if has_att.sum() > 1:
                df.loc[has_att, "_att_rate_pct"] = normalize_percentile(
                    df.loc[has_att, "_att_rate"]).values

            pct_cols = [c for c in df.columns if c.endswith("_pct") and not c.startswith("_")]

            for _, row in df.iterrows():
                metrics = {}
                for col in pct_cols:
                    val = row.get(col)
                    if val is not None and not pd.isna(val):
                        fval = float(val)
                        metrics[col.removesuffix("_pct")] = fval
                        metrics[col] = fval

                k_ni = row.get("K_pct_raw_pct_noninv")
                if k_ni is not None and not pd.isna(float(k_ni)):
                    metrics["K_pct_raw"] = float(k_ni)
                if "BB_pct" in metrics:
                    metrics["BB_inv_pct"] = 100.0 - metrics["BB_pct"]

                k_raw = row.get("k_rate")
                if k_raw is not None and not pd.isna(k_raw):
                    metrics["k_rate_raw"] = float(k_raw)

                sprint = row.get("sprint_speed")
                sprint_raw = float(sprint) if sprint is not None and not pd.isna(sprint) else None
                att_pct_v = row.get("_att_rate_pct")
                att_pct_v = float(att_pct_v) if att_pct_v is not None and not pd.isna(float(att_pct_v)) else None

                p = build_hitter_profile(
                    player_id=int(row.get("key_mlbam", 0) or 0),
                    metrics=metrics,
                    sprint_speed_raw=sprint_raw,
                    sb=int(row.get("sb", 0) or 0),
                    cs=int(row.get("cs", 0) or 0),
                    opportunities=int(float(row.get("obp") or 0) * float(row.get("pa") or 1)),
                    attempt_rate_pct=att_pct_v,
                )
                all_profiles.append({
                    "name":    row.get("name", "?"),
                    "season":  season,
                    "pa":      int(row.get("pa", 0) or 0),
                    "avg":     row.get("avg"),
                    "obp":     row.get("obp"),
                    "slg":     row.get("slg"),
                    "iso":     row.get("iso"),
                    "k_rate":  row.get("k_rate"),
                    "bb_rate": row.get("bb_rate"),
                    "xwoba":   row.get("xwoba"),
                    "war":     row.get("war"),
                    "primary_type": p["primary"]["type"],
                    "mods":    p["modifiers"],
                    "sprint_raw": sprint_raw,
                })
        except Exception as exc:
            log.warning("Scout season %d failed: %s", season, exc)

    if not all_profiles:
        return (charts.empty_figure("No data for selected seasons"),
                dbc.Alert("No players found.", color="warning", className="py-1"))

    # ── Filter by archetype ───────────────────────────────────────────────────
    if archetype and archetype != "Any":
        all_profiles = [p for p in all_profiles if p["primary_type"] == archetype]

    # ── Filter by modifiers (player must have ALL selected) ──────────────────
    def _has_modifier(p, mod):
        mods = p["mods"]
        m = mod.lower().replace(" ", "_")
        # Speed tiers
        if mod in ("Elite", "Fast", "Slow"):
            return mods.get("speed") == mod
        # Plate discipline
        if mod in ("Elite Discipline", "Disciplined", "Free Swinger"):
            return mods.get("plate_discipline") == mod
        # Contact quality
        if mod in ("Plus Contact", "Weak Contact"):
            return mods.get("contact_quality") == mod
        # Lucky/Unlucky
        if mod in ("Lucky", "Unlucky"):
            return mods.get("lucky_unlucky") == mod
        # Disruptive/Chaotic
        if mod in ("Disruptive", "Chaotic"):
            return mods.get("disruptiveness", {}).get("modifier") == mod
        # Boolean modifiers
        return bool(mods.get(m.replace(" ", "_"), False))

    for mod in modifiers:
        all_profiles = [p for p in all_profiles if _has_modifier(p, mod)]

    if not all_profiles:
        return (charts.empty_figure("No players match the selected criteria"),
                dbc.Alert(f"0 results — try relaxing the filters.", color="warning", className="py-1"))

    # ── Apply mode ────────────────────────────────────────────────────────────
    if mode == "recent":
        seen = {}
        for p in sorted(all_profiles, key=lambda x: x["season"], reverse=True):
            if p["name"] not in seen:
                seen[p["name"]] = p
        all_profiles = list(seen.values())
    elif mode == "best":
        seen = {}
        for p in all_profiles:
            xw = float(p["xwoba"] or 0)
            if p["name"] not in seen or xw > float(seen[p["name"]]["xwoba"] or 0):
                seen[p["name"]] = p
        all_profiles = list(seen.values())

    all_profiles.sort(key=lambda x: -(float(x.get("war") or 0)))

    # ── Build results table ──────────────────────────────────────────────────
    def _fmt(v, fmt=".3f"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{float(v):{fmt}}"

    def _mods_str(p):
        mods = p["mods"]
        tags = []
        spd = mods.get("speed")
        if spd in ("Elite", "Fast", "Slow"):  tags.append(spd)
        lu = mods.get("lucky_unlucky")
        if lu:                                 tags.append(lu)
        cq = mods.get("contact_quality")
        if cq:                                 tags.append(cq)
        pd_ = mods.get("plate_discipline")
        if pd_:                                tags.append(pd_)
        if mods.get("table_setter"):           tags.append("Table Setter")
        if mods.get("plus_power"):             tags.append("Plus Power")
        if mods.get("gap_hitter"):             tags.append("Gap Hitter")
        if mods.get("aggressive"):             tags.append("Aggressive")
        dis = mods.get("disruptiveness", {}).get("modifier")
        if dis:                                tags.append(dis)
        return ", ".join(tags) or "—"

    n = len(all_profiles)
    COLORS_TBL = {"surface": "#1f2937", "background": "#111827",
                  "text": "#f9fafb", "subtext": "#9ca3af", "border": "#374151"}
    row_colors = [COLORS_TBL["surface"] if i % 2 == 0 else COLORS_TBL["background"]
                  for i in range(n)]

    import plotly.graph_objects as go
    fig = go.Figure(go.Table(
        columnwidth=[3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2],
        header=dict(
            values=["<b>Player</b>", "<b>Yr</b>", "<b>PA</b>", "<b>WAR</b>",
                    "<b>Archetype</b>", "<b>AVG</b>", "<b>OBP</b>",
                    "<b>ISO</b>", "<b>K%</b>", "<b>BB%</b>", "<b>Modifiers</b>"],
            fill_color=COLORS_TBL["surface"],
            font=dict(color=COLORS_TBL["subtext"], size=11),
            align=["left","center","center","center","left","center",
                   "center","center","center","center","left"],
            line_color=COLORS_TBL["border"], height=32,
        ),
        cells=dict(
            values=[
                [p["name"] for p in all_profiles],
                [p["season"] for p in all_profiles],
                [p["pa"] for p in all_profiles],
                [_fmt(p.get("war"), ".1f") for p in all_profiles],
                [p["primary_type"] for p in all_profiles],
                [_fmt(p.get("avg")) for p in all_profiles],
                [_fmt(p.get("obp")) for p in all_profiles],
                [_fmt(p.get("iso")) for p in all_profiles],
                [_fmt(p.get("k_rate"), ".1%") for p in all_profiles],
                [_fmt(p.get("bb_rate"), ".1%") for p in all_profiles],
                [_mods_str(p) for p in all_profiles],
            ],
            fill_color=[row_colors] * 11,
            font=dict(color=COLORS_TBL["text"], size=11),
            align=["left","center","center","center","left","center",
                   "center","center","center","center","left"],
            line_color=COLORS_TBL["border"], height=27,
        ),
    ))
    fig.update_layout(
        paper_bgcolor=COLORS_TBL["background"],
        plot_bgcolor=COLORS_TBL["background"],
        margin=dict(l=0, r=0, t=10, b=0),
    )

    status = dbc.Alert(
        f"{n} player{'s' if n != 1 else ''} matched · "
        f"archetype={archetype} · modifiers={modifiers or 'any'} · "
        f"seasons {season_min}–{season_max} · min PA={min_pa}",
        color="success", className="py-1 mb-0",
    )
    return fig, status


# ---------------------------------------------------------------------------
# Construction vs Results callbacks
# ---------------------------------------------------------------------------

@callback(Output("cvr-radar",      "figure"), Input("portrait-store", "data"))
def cvr_radar(data):
    p = _deserialize(data)
    return charts.construction_vs_results_radar(p) if p else charts.empty_figure("Load a portrait to see Construction vs Results")


@callback(Output("cvr-archetypes", "figure"), Input("portrait-store", "data"))
def cvr_archetypes(data):
    p = _deserialize(data)
    return charts.construction_vs_results_archetypes(p) if p else charts.empty_figure()
