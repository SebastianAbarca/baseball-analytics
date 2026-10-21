#!/usr/bin/env python3
"""
pull_mls.py — the published Major League Service lists, one per season.

A public Google Sheet ("MLS 2007-2025.xls") holds MLB's service-time list as
of each January, 2006 through 2025: every player under reserve that day with
his career service in years.days (172 days = one year). Its provenance is not
known, so it is checked against the roster ledger (service_time.py) rather
than trusted: for 2009, a player's increase between lists matched his roster
days exactly for 73% of players on both lists and within 3 days for 95%.

Saved to data/raw/mls/mls_thru_{season}.csv (gitignored — third-party data)
with columns player, pos, mls (the sheet's text, e.g. "5.143").

    python scripts/pull_mls.py
"""
from __future__ import annotations

import html
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("pull_mls")
SHEET = "1m9ap5cOX3j4ZYnmceOZ0oK8GtLg5YGesNsxMZb6GFIs"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "mls"

# gid of each "MLS thru {season}" tab
TABS = {2025: 1468452604, 2024: 490383299, 2023: 1536365959, 2022: 1868431725,
        2021: 2132495346, 2020: 1219230069, 2019: 619724678, 2018: 1210153842,
        2017: 173041157, 2016: 2044635785, 2015: 788177855, 2014: 638554007,
        2013: 470212433, 2012: 1257000605, 2011: 1219122113, 2010: 1734723822,
        2009: 435379795, 2008: 1567859412, 2007: 1788471332, 2006: 721433265}

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _cells(page: str) -> list[list[str]]:
    rows = []
    for row in _ROW.findall(page):
        cells = [html.unescape(_TAG.sub("", c)).strip() for c in _CELL.findall(row)]
        if any(cells):
            rows.append(cells)
    return rows


def pull_tab(season: int) -> pd.DataFrame:
    """One season's list. Columns vary by year — Player | [Club |] Pos | MLS —
    so they are found by header, not position."""
    url = (f"https://docs.google.com/spreadsheets/d/{SHEET}/pubhtml/sheet"
           f"?headers=false&gid={TABS[season]}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = _cells(r.text)
    h = next(i for i, row in enumerate(rows) if row and row[0] == "Player")
    head = rows[h]
    pos_col = next(i for i, c in enumerate(head) if c.lower().startswith("pos"))
    mls_col = next(i for i, c in enumerate(head) if c.upper().startswith("MLS"))
    out = [{"player": row[0], "pos": row[pos_col] if pos_col < len(row) else "",
            "mls": row[mls_col] if mls_col < len(row) else ""}
           for row in rows[h + 1:] if row and row[0]]
    return pd.DataFrame(out)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    OUT.mkdir(parents=True, exist_ok=True)
    for season in sorted(TABS):
        df = pull_tab(season)
        df.to_csv(OUT / f"mls_thru_{season}.csv", index=False)
        log.info("%d: %d players", season, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
