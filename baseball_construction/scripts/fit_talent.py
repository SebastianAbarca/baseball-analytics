#!/usr/bin/env python3
"""
fit_talent.py — choose each talent spec's decay and regression (modules/talent.py).

For every spec, every (decay d, regression k) on a grid predicts each
player's NEXT season from the estimate at this one; the pair with the best
correlation on 2016-21 targets is reported with its score on 2022-26, which
the fit never saw, beside season N alone (d = 0, k = 0) and the unweighted
career (d = 1). Fitted over every player with a season-N reading; the
regulars column re-scores the same choice on players with 60% of a typical
season in N, where tags mostly live.

Prints a table; the chosen values are copied into talent.HITTER_SPECS /
PITCHER_SPECS by hand, so a refit is a reviewed change.

    python scripts/fit_talent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))
import talent  # noqa: E402

TRAIN, TEST = range(2016, 2022), range(2022, 2027)
D_GRID = (0.0, 0.3, 0.5, 0.7, 0.85, 1.0)
K_GRID = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def actual(counts: pd.DataFrame, side: str) -> pd.DataFrame:
    """Each (player, season)'s own rate per spec, and its denominator."""
    c = counts[counts.side == side].set_index(["player", "season"])
    out = {}
    for s in talent.SPECS[side]:
        with np.errstate(all="ignore"):
            r = c[s.num] / c[s.den]
        out[s.stem] = 1 - r if s.complement else r
        out[f"{s.stem}__den"] = c[s.den]
    return pd.DataFrame(out)


def score(pred: pd.Series, act: pd.DataFrame, spec, seasons, regulars=False) -> float:
    xs, ys = [], []
    for n1 in seasons:
        try:
            p = pred.xs(n1 - 1, level="season")
            a = act.xs(n1, level="season")
            prior = act.xs(n1 - 1, level="season")[f"{spec.stem}__den"]
        except KeyError:
            continue
        j = p.index.intersection(a.index)
        ok = (a.loc[j, f"{spec.stem}__den"] >= 0.6 * spec.typical) & p.loc[j].notna()
        if regulars:
            ok &= prior.reindex(j).fillna(0) >= 0.6 * spec.typical
        j = j[ok.values]
        x, y = p.loc[j].values, a.loc[j, spec.stem].values
        xs.append(x - x.mean()); ys.append(y - y.mean())   # season environment out
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    counts = talent.load_counts(range(2015, 2027))
    for side in ("H", "P"):
        act = actual(counts, side)
        grid = {(d, k): talent.estimate_panel(counts, side, d=d, k=k)
                for d in D_GRID for k in K_GRID}
        rows = []
        for s in talent.SPECS[side]:
            fit = {dk: score(g[s.stem], act, s, TRAIN) for dk, g in grid.items()}
            d, k = max(fit, key=fit.get)
            best = grid[(d, k)][s.stem]
            rows.append({
                "stem": s.stem, "d": d, "k": k,
                "season": score(grid[(0.0, 0.0)][s.stem], act, s, TEST),
                "career": score(grid[(1.0, k)][s.stem], act, s, TEST),
                "talent": score(best, act, s, TEST),
                "regulars: season": score(grid[(0.0, 0.0)][s.stem], act, s, TEST, True),
                "regulars: talent": score(best, act, s, TEST, True),
                "current d,k": f"{s.decay},{s.k}",
            })
        print(f"\n{'HITTERS' if side == 'H' else 'PITCHERS'} (r with next season, 2022-26 targets)")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
