"""
battery_base.py — What the pitcher x catcher data can and cannot support.

This is the FOUNDATION for battery attribution, not the attribution itself.
It exists to answer one question before any credit is assigned: on a
shadow-zone taken pitch, how much of the called-strike outcome is actually
SEPARABLE between the catcher and the pitcher?

Separation is only possible where the same pitcher throws to different
catchers. If every pitcher had one catcher all season, the two effects would
be perfectly confounded and no model could tell them apart. So the base is the
crossover structure, plus an honest test of whether a fitted decomposition
survives contact with held-out data.

HEADLINE FINDING (2023)
-----------------------
An UNREGULARISED two-way fit is almost entirely noise: split-half reliability
0.18 against 0.72 for the plain raw metric. It also produces a WIDER spread of
catcher effects than the raw numbers, which reads like a discovery and is in
fact the fit absorbing sampling error. Anyone building battery attribution
will hit this first and should not believe it.

With heavy ridge shrinkage the decomposition becomes real but modest: +0.039
split-half reliability over raw (SD 0.017 across 8 random splits, 0 sign
flips — small, but it never once reversed).

The reason the gain is small is itself the useful result. The
opportunity-weighted correlation between catcher and pitcher effects is small
and positive (+0.08 at ridge 250, +0.19 at 1000): good framers caught
marginally easier-to-frame staffs, so the raw number mildly flatters them, but
nothing close to the catcher's number really belonging to his pitchers.

That is the answer to "what belongs to whom" for framing: nearly all of it
already belongs to the catcher, and the machinery to prove it buys about four
points of reliability.

USAGE
-----
    python battery_base.py            # 2023
    python battery_base.py 2024
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

from pitch_aggregates import _FR_TOP, _FR_BOT, _FR_W, _SHADOW_BAND  # noqa: E402

MIN_PAIR = 25        # opportunities before a battery pair is usable
MIN_CATCHER = 500    # matches compute_catcher_framing's qualification bar
RIDGE = 1000         # shrinkage; see _reliability_table for why it is this big


def _frameable(season: int) -> pd.DataFrame:
    """Taken pitches in the shadow band around the zone edge."""
    raw = _HERE.parent / "data" / "raw" / f"statcast_{season}.parquet"
    sc = pd.read_parquet(raw, columns=[
        "fielder_2", "pitcher", "description", "plate_x", "plate_z",
        "sz_top", "sz_bot"])
    sc = sc[sc["description"].isin(["called_strike", "ball"])]
    px = pd.to_numeric(sc["plate_x"], errors="coerce")
    pz = pd.to_numeric(sc["plate_z"], errors="coerce")
    top = pd.to_numeric(sc["sz_top"], errors="coerce").fillna(_FR_TOP)
    bot = pd.to_numeric(sc["sz_bot"], errors="coerce").fillna(_FR_BOT)
    outside = pd.concat([(px.abs() - _FR_W), pz - top, bot - pz], axis=1).max(axis=1)
    sc = sc[(outside.abs() <= _SHADOW_BAND) & px.notna() & pz.notna()].copy()
    sc["cs"] = (sc["description"] == "called_strike").astype(float)
    sc = sc.dropna(subset=["fielder_2", "pitcher"])
    sc["fielder_2"] = sc["fielder_2"].astype(int)
    sc["pitcher"] = sc["pitcher"].astype(int)
    return sc


def two_way_effects(df: pd.DataFrame, ridge: float = RIDGE, iters: int = 40):
    """
    Catcher and pitcher effects, each estimated holding the other fixed and
    iterated to convergence. `ridge` shrinks an effect toward zero in inverse
    proportion to how much data supports it — without it the fit is noise
    (see the module docstring).
    """
    league = df["cs"].mean()
    w = (df.groupby(["fielder_2", "pitcher"])["cs"]
           .agg(["size", "mean"]).reset_index()
           .rename(columns={"size": "n", "mean": "rate"}))
    cat = pd.Series(0.0, index=np.sort(df.fielder_2.unique()))
    pit = pd.Series(0.0, index=np.sort(df.pitcher.unique()))
    for _ in range(iters):
        r = w["rate"] - league - w["pitcher"].map(pit).values
        g = w.assign(x=r * w.n).groupby("fielder_2").agg(s=("x", "sum"), n=("n", "sum"))
        cat = (g.s / (g.n + ridge)).reindex(cat.index).fillna(0.0)
        r2 = w["rate"] - league - w["fielder_2"].map(cat).values
        g2 = w.assign(x=r2 * w.n).groupby("pitcher").agg(s=("x", "sum"), n=("n", "sum"))
        pit = (g2.s / (g2.n + ridge)).reindex(pit.index).fillna(0.0)
    return cat, pit, league


def _raw_effect(df: pd.DataFrame) -> pd.Series:
    league = df["cs"].mean()
    return df.groupby("fielder_2")["cs"].mean() - league


def _structure(sc: pd.DataFrame) -> None:
    print("=" * 74)
    print(f"BATTERY BASE — shadow-zone taken pitches")
    print("=" * 74)
    print(f"frameable pitches : {len(sc):,}")
    print(f"league CS rate    : {sc.cs.mean():.3f}")
    print(f"catchers          : {sc.fielder_2.nunique():,}")
    print(f"pitchers          : {sc.pitcher.nunique():,}")

    pairs = (sc.groupby(["fielder_2", "pitcher"])["cs"]
               .agg(["size", "mean"]).reset_index()
               .rename(columns={"size": "n", "mean": "cs_rate"}))
    usable = pairs[pairs.n >= MIN_PAIR]
    print(f"battery pairs     : {len(pairs):,}  "
          f"({len(usable):,} with >= {MIN_PAIR} opps, covering "
          f"{100*usable.n.sum()/pairs.n.sum():.0f}% of opportunities)")

    print("\n-- CROSSOVER (what makes separation possible at all) --")
    pc = usable.groupby("pitcher").size()
    cp = usable.groupby("fielder_2").size()
    print(f"  catchers per pitcher : median {pc.median():.0f}, "
          f"but {(pc==1).sum()} of {len(pc)} pitchers ({100*(pc==1).mean():.0f}%) "
          f"threw to only one")
    print(f"  pitchers per catcher : median {cp.median():.0f}, max {cp.max()}")

    parent: dict = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in usable.itertuples():
        a, b = find(("C", r.fielder_2)), find(("P", r.pitcher))
        if a != b:
            parent[a] = b
    comps: dict = {}
    for node in parent:
        comps.setdefault(find(node), []).append(node)
    big = max(comps.values(), key=len)
    n_c = sum(1 for x in big if x[0] == "C")
    print(f"  graph components     : {len(comps)}; the largest holds {n_c} "
          f"catchers ({100*n_c/len(cp):.0f}%)")
    print("  => every catcher in that component is comparable to every other")
    print("     through a chain of shared pitchers, so the design is identified")


def _reliability_table(sc: pd.DataFrame, seeds: int = 8) -> None:
    """The test that decides whether any of this is trustworthy."""
    qual = sc.groupby("fielder_2").size()
    qual = qual[qual >= MIN_CATCHER].index

    print("\n" + "-" * 74)
    print("IS THE DECOMPOSITION SIGNAL OR OVERFIT?")
    print("-" * 74)
    print("Split each catcher's pitches at random, fit both halves separately,")
    print("correlate. Real skill replicates across halves; fitted noise does not.\n")
    print(f"{'ridge':>7}{'raw r':>10}{'adjusted r':>14}{'adj SD (pp)':>14}")
    print("-" * 46)
    rng = np.random.default_rng(17)
    sc = sc.assign(half=rng.integers(0, 2, len(sc)))
    for lam in (0, 100, 250, 500, 1000):
        a, b = sc[sc.half == 0], sc[sc.half == 1]
        ca, _, _ = two_way_effects(a, ridge=lam)
        cb, _, _ = two_way_effects(b, ridge=lam)
        ca, cb = ca.reindex(qual).dropna(), cb.reindex(qual).dropna()
        common = ca.index.intersection(cb.index)
        r_adj = np.corrcoef(ca[common], cb[common])[0, 1]
        ra, rb = _raw_effect(a).reindex(common), _raw_effect(b).reindex(common)
        r_raw = np.corrcoef(ra, rb)[0, 1]
        note = "  <-- unregularised fit is NOISE" if lam == 0 else ""
        print(f"{lam:>7}{r_raw:>10.3f}{r_adj:>14.3f}"
              f"{ca[common].std()*100:>14.2f}{note}")

    print(f"\nStability of the gain at ridge={RIDGE}, across {seeds} random splits:")
    diffs = []
    for seed in range(seeds):
        r = np.random.default_rng(seed)
        s = sc.assign(half=r.integers(0, 2, len(sc)))
        a, b = s[s.half == 0], s[s.half == 1]
        ca, _, _ = two_way_effects(a); cb, _, _ = two_way_effects(b)
        ca, cb = ca.reindex(qual).dropna(), cb.reindex(qual).dropna()
        common = ca.index.intersection(cb.index)
        diffs.append(np.corrcoef(ca[common], cb[common])[0, 1]
                     - np.corrcoef(_raw_effect(a).reindex(common),
                                   _raw_effect(b).reindex(common))[0, 1])
    d = np.array(diffs)
    print(f"  mean {d.mean():+.3f}   SD {d.std():.3f}   "
          f"range [{d.min():+.3f}, {d.max():+.3f}]   "
          f"sign flips {int((d<0).sum())}/{len(d)}")


def _confound(sc: pd.DataFrame) -> None:
    cat, pit, _ = two_way_effects(sc)
    w = (sc.groupby(["fielder_2", "pitcher"])["cs"].agg(["size"])
           .reset_index().rename(columns={"size": "n"}))
    w["ce"], w["pe"] = w.fielder_2.map(cat), w.pitcher.map(pit)
    num = np.cov(w.ce, w.pe, aweights=w.n)[0, 1]
    den = np.sqrt(np.cov(w.ce, aweights=w.n) * np.cov(w.pe, aweights=w.n))
    print("\n" + "-" * 74)
    print("HOW BIG IS THE CONFOUND?")
    print("-" * 74)
    print(f"opportunity-weighted corr(catcher effect, pitcher effect) = "
          f"{num/den:+.3f}  (ridge={RIDGE})")
    print("Estimate is shrinkage-sensitive: +0.08 at ridge 250, +0.19 at 1000.")
    print("Either way it is small and POSITIVE — good framers caught slightly")
    print("easier-to-frame staffs, so raw framing mildly flatters them. It is")
    print("nowhere near the level where the catcher number is really the")
    print("pitcher's, which is why adjustment buys so little.")


def main(season: int) -> None:
    sc = _frameable(season)
    print(f"\nSEASON {season}")
    _structure(sc)
    _reliability_table(sc)
    _confound(sc)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2023)
