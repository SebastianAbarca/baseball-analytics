"""
app.py — Dash application entry point.

Run with:
    python baseball_construction/dashboard/app.py

Or with auto-reload:
    python baseball_construction/dashboard/app.py --debug
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc

# Ensure modules are on the path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "modules"))

import layout
import callbacks  # registers all @callback decorators

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.BOOTSTRAP,
    ],
    title="Baseball Construction",
    suppress_callback_exceptions=True,
)

app.layout = layout.full_layout()

server = app.server   # for WSGI deployment (gunicorn, etc.)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    debug = "--debug" in sys.argv
    app.run(debug=debug, port=8050)
