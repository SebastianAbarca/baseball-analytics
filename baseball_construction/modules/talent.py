"""
talent.py — what a player IS as of a season, not only what one season showed.

A season is a sample. For the stats that settle fast (K%, chase, whiff) one
season pins a hitter down; for the slow ones (BABIP-driven rates, HR/FB, ISO,
walks, contact allowed) it mostly measures luck, and a player with 120 PA
gets no usable reading at all. A plain career total fixes the sample and
breaks everything that changes: velocity fades, pitchers add pitches,
hitters rebuild swings. So the estimate here is between the two:

    est(N) = sum_s d^(N-s) * (num_s - lg_s * den_s)
             ------------------------------------------   + lg_N
                  sum_s d^(N-s) * den_s  +  k

over the player's seasons s <= N — never a later one, so a 2019 portrait
knows only what was known in 2019. Each season counts relative to its own
league (K% rose two points over the era), older seasons weigh d per year
less, and k (in the stat's own denominator) pulls a thin sample toward the
league. d and k are per metric, fitted by scripts/fit_talent.py: each season
predicts the next, fitted on 2016-21 targets, scored on 2022-26.

Measured (fit_talent.py, 2026-11-01; r with season N+1, 2022-26 targets),
against season N alone: every spec gains. Least where one season already
settles the stat (regulars: K% .82 -> .84, whiff .88 -> .89), most where it
does not (ISO .54 -> .63, HR/FB .57 -> .66, pull .62 -> .73, pitcher xwOBA
allowed .43 -> .52, hard-hit allowed .34 -> .48) and for thin samples. An
unweighted career total is WORSE than one season for pitcher stuff (K% .62
-> .60, whiff .68 -> .66; velo .94 -> .89 in the prototype): pitchers change,
which is why their stuff decays at 0.3 a season. Velocity is not here — the
season is its best estimate. Platoon splits are not here either: even career
data repeats them at r <= .20 (the tags were retired in #55/#56).

The counts come straight from Statcast, one CSV per season
(talent_counts_{s}.csv in data/processed). Finished seasons are frozen in
data/era like the fg roots, so the nightly runner — which has only the
season in progress — computes identical estimates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TALENT_METHOD_VERSION = 1

# ---------------------------------------------------------------------------
# Counts — one row per (side, player) per season
# ---------------------------------------------------------------------------

_SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
          "foul_bunt", "missed_bunt", "bunt_foul_tip"}
_WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
_K = {"strikeout", "strikeout_double_play"}
_NOT_AB = {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play",
           "sac_bunt_double_play", "catcher_interf", "truncated_pa"}
_COLS = ["game_type", "batter", "pitcher", "stand", "events", "description", "zone",
         "bb_type", "launch_speed", "launch_speed_angle", "estimated_woba_using_speedangle",
         "woba_value", "woba_denom", "iso_value", "hc_x", "hc_y", "pitch_number"]

# Every count a spec below reads. `pa` doubles as the volume the cohorts gate on.
COUNTS = ("pa", "k", "bb", "pitch", "swing", "whiff", "inz", "outz", "oswing", "zswing",
          "fp", "fp_swing", "csw", "bip", "gb", "fb", "xb", "hr", "bbe", "hard", "barrel",
          "ev_sum", "wden", "xw_sum", "ab", "iso_sum", "spray", "pull", "oppo")


def season_counts(statcast: pd.DataFrame) -> pd.DataFrame:
    """Per-player counts for one season of pitches, as hitter (side H, by
    batter) and as pitcher (side P, by pitcher). Regular season only."""
    d = statcast[[c for c in _COLS if c in statcast.columns]]
    if "game_type" in d.columns:
        d = d[d["game_type"] == "R"]
    d = d.reset_index(drop=True)
    desc, ev = d["description"], d["events"]
    swing, whiff = desc.isin(_SWING), desc.isin(_WHIFF)
    inz, outz = d["zone"].between(1, 9), d["zone"].between(11, 14)
    bbe = d["launch_speed"].notna() & (desc == "hit_into_play")
    pa = ev.notna() & (ev != "truncated_pa")
    wd = d["woba_denom"].fillna(0)
    xw = np.where(bbe, d["estimated_woba_using_speedangle"].fillna(0), d["woba_value"].fillna(0))
    # Spray angle from the hit coordinates; pulled = more than 15 degrees to
    # the hitter's own side of centre.
    ang = np.degrees(np.arctan2(d["hc_x"] - 125.42, 198.27 - d["hc_y"]))
    spray = bbe & d["hc_x"].notna()
    right, left = d["stand"] == "R", d["stand"] == "L"
    f = pd.DataFrame({
        "pa": pa, "k": ev.isin(_K), "bb": ev == "walk", "pitch": True,
        "swing": swing, "whiff": whiff, "inz": inz, "outz": outz,
        "oswing": swing & outz, "zswing": swing & inz,
        "fp": d["pitch_number"] == 1, "fp_swing": swing & (d["pitch_number"] == 1),
        "csw": whiff | (desc == "called_strike"),
        "bip": d["bb_type"].notna(), "gb": d["bb_type"] == "ground_ball",
        "fb": d["bb_type"] == "fly_ball", "xb": ev.isin({"double", "triple"}),
        "hr": ev == "home_run", "bbe": bbe, "hard": bbe & (d["launch_speed"] >= 95),
        "barrel": bbe & (d["launch_speed_angle"] == 6),
        "ev_sum": np.where(bbe, d["launch_speed"], 0.0),
        "wden": wd, "xw_sum": np.where(wd > 0, xw, 0.0),
        "ab": pa & ~ev.isin(_NOT_AB), "iso_sum": np.where(pa, d["iso_value"].fillna(0), 0.0),
        "spray": spray,
        "pull": spray & ((right & (ang < -15)) | (left & (ang > 15))),
        "oppo": spray & ((right & (ang > 15)) | (left & (ang < -15))),
    }).astype(float)
    out = []
    for side, who in (("H", "batter"), ("P", "pitcher")):
        g = f.groupby(d[who].values).sum()
        g.index.name = "player"
        g = g.reset_index()
        g.insert(0, "side", side)
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res["player"] = res["player"].astype(int)
    return res.round(3)


def counts_path(season: int) -> Path:
    from ingest import PROCESSED_DIR
    return PROCESSED_DIR / f"talent_counts_{season}.csv"


def build_counts(season: int, statcast: Optional[pd.DataFrame] = None,
                 force: bool = False) -> pd.DataFrame:
    """Write (or reuse) one season's counts. refresh.stage_league calls this
    whenever the season's Statcast moves, BEFORE the portraits stage: the
    file is a fingerprinted pool root, and creating it mid-build would move
    the fingerprint under a running rebuild."""
    path = counts_path(season)
    if path.exists() and not force:
        return pd.read_csv(path)
    if statcast is None:
        from ingest import pull_statcast_season
        statcast = pull_statcast_season(season, top_up=False)
    c = season_counts(statcast)
    path.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(path, index=False)
    log.info("talent counts %d: %d hitters, %d pitchers", season,
             (c.side == "H").sum(), (c.side == "P").sum())
    return c


def load_counts(seasons) -> pd.DataFrame:
    """Every season's counts that exists, with a `season` column."""
    frames = []
    for s in sorted(seasons):
        p = counts_path(s)
        if p.exists():
            frames.append(pd.read_csv(p).assign(season=int(s)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    stem: str          # the contract stem: talent percentile is `{stem}_pct`
    num: str
    den: str
    decay: float       # weight per season back (0 = this season only)
    k: float           # regression toward the league, in seasons of `typical`
    typical: float     # a regular's season of the denominator
    complement: bool = False   # the stat is 1 - num/den (contact = 1 - whiff)
    season_rank: bool = False  # also ranked within the season (a hitter tag gate)


# Fitted by scripts/fit_talent.py (2026-11-01): decay and k chosen on 2016-21
# targets over every player with a season-N reading, scored on 2022-26.
HITTER_SPECS: tuple[Spec, ...] = (
    Spec("K_pct_raw", "k", "pa", 0.7, 0.125, 500, season_rank=True),
    Spec("K_pct", "k", "pa", 0.7, 0.125, 500),
    Spec("BB_pct", "bb", "pa", 0.85, 0.25, 500, season_rank=True),
    Spec("OSwing_pct", "oswing", "outz", 0.5, 0.125, 1000, season_rank=True),
    Spec("ZSwing_pct", "zswing", "inz", 0.3, 0.125, 900),
    Spec("FPS_pct", "fp_swing", "fp", 0.3, 0.125, 500, season_rank=True),
    Spec("Whiff_pct", "whiff", "swing", 0.5, 0.125, 1000),
    Spec("Contact_pct", "whiff", "swing", 0.5, 0.125, 1000, complement=True),
    Spec("BatGB_pct", "gb", "bip", 0.5, 0.25, 350, season_rank=True),
    Spec("Pull_pct", "pull", "spray", 0.5, 0.5, 350),
    Spec("Oppo_pct", "oppo", "spray", 0.7, 0.25, 350),
    Spec("HardHit_pct", "hard", "bbe", 0.7, 0.25, 350, season_rank=True),
    Spec("Barrel_pct", "barrel", "bbe", 0.7, 0.125, 350, season_rank=True),
    Spec("EV_avg", "ev_sum", "bbe", 0.7, 0.125, 350),
    Spec("ISO", "iso_sum", "ab", 0.7, 1.0, 450, season_rank=True),
    Spec("HR_FB_pct", "hr", "fb", 0.85, 0.25, 120),
    Spec("XB_pct", "xb", "bip", 0.7, 4.0, 350),
    Spec("xwOBA", "xw_sum", "wden", 0.7, 0.5, 500),
)
PITCHER_SPECS: tuple[Spec, ...] = (
    Spec("K_pct_pitch", "k", "pa", 0.3, 0.5, 600),
    Spec("BB_pct_pitch", "bb", "pa", 0.85, 1.0, 600),
    Spec("Whiff_pct_pitch", "whiff", "swing", 0.3, 0.125, 1100),
    Spec("CSW_pct", "csw", "pitch", 0.3, 0.25, 2400),
    Spec("Zone_pct", "inz", "pitch", 0.7, 0.25, 2400),
    Spec("GB_pct_pitch", "gb", "bip", 0.7, 0.125, 450),
    Spec("HardHit_allowed", "hard", "bbe", 0.7, 1.0, 450),
    Spec("Barrel_allowed", "barrel", "bbe", 0.85, 4.0, 450),
    Spec("xwOBA_allowed", "xw_sum", "wden", 0.5, 4.0, 600),
)
SPECS = {"H": HITTER_SPECS, "P": PITCHER_SPECS}

# Swing discrimination (zone swing minus chase) is differenced from the two
# estimates, not ranked from a count: the hitter `zone hunter` gate.
DERIVED_H = {"SwingDiscrim": ("ZSwing_pct", "OSwing_pct")}

# Cohort volume floors — the era cohorts' (metric_contracts.COHORTS).
MIN_VOLUME = {"H": 100, "P": 30}
# The effective sample a talent estimate rests on: decayed PA (hitters) or BF
# (pitchers), at the K% spec's decay.
N_EFF_DECAY = 0.5


def estimate_panel(counts: pd.DataFrame, side: str, d=None, k=None) -> pd.DataFrame:
    """Talent estimates for every (player, season) in `counts` of one side,
    each from that season and the ones before it only. Columns: the spec
    stems (absolute rates in that season's league), `volume` (that season's
    PA/BF) and `n_eff`. `d`/`k` override every spec's (fitting only)."""
    c = counts[counts["side"] == side]
    if c.empty:
        return pd.DataFrame()
    seasons = sorted(c["season"].unique())
    idx = pd.MultiIndex.from_frame(c[["player", "season"]])
    out = pd.DataFrame(index=idx)

    def pivot(col):
        return c.pivot(index="player", columns="season", values=col).reindex(
            columns=seasons).fillna(0.0)

    played = pivot("pa")
    players = played.index

    def decayed(M, dk):
        """(players x seasons) -> sum over s <= N of dk^(N-s) * M[:, s]."""
        A, R = M.values, np.zeros_like(M.values)
        for j, N in enumerate(seasons):
            w = np.array([(dk ** (N - s) if dk > 0 else float(s == N)) if s <= N else 0.0
                          for s in seasons])
            R[:, j] = A @ w
        return pd.DataFrame(R, index=players, columns=seasons)

    def stack(M):
        return M.stack().reindex(idx).values

    for spec in SPECS[side]:
        num, den = pivot(spec.num), pivot(spec.den)
        lg = num.sum() / den.sum().replace(0, np.nan)
        rel = num - den * lg
        dk = spec.decay if d is None else d
        kk = (spec.k if k is None else k) * spec.typical
        with np.errstate(all="ignore"):
            est = decayed(rel, dk) / (decayed(den, dk) + kk) + lg
        est = est.where(decayed(den, dk) > 0)
        vals = stack(est)
        out[spec.stem] = 1.0 - vals if spec.complement else vals
    if side == "H":
        for stem, (a, b) in DERIVED_H.items():
            out[stem] = out[a] - out[b]
    out["volume"] = stack(played)
    out["n_eff"] = stack(decayed(played, N_EFF_DECAY))
    return out


@lru_cache(maxsize=4)
def _panels(seasons: tuple) -> dict:
    counts = load_counts(seasons)
    if counts.empty:
        return {}
    return {side: estimate_panel(counts, side) for side in ("H", "P")}


def _rank(pool: np.ndarray, v: float) -> float:
    """Percentile of v in pool, scipy 'mean' kind (team_portrait._pool_rank)."""
    lo = np.searchsorted(pool, v, side="left")
    hi = np.searchsorted(pool, v, side="right")
    return (lo + hi) / 2.0 / len(pool) * 100.0


def player_talent(season: int, side: str) -> dict[int, dict]:
    """{player_id: {"talent": {...}, "talent_season": {...}, "n_eff": float}}
    for every player with counts in `season`.

    `talent` ranks each estimate against the era cohort — every player-season
    of the era pool with the cohort's volume, each at its OWN estimate — the
    talent twin of the era `metrics`. `talent_season` (hitters, the specs a
    season gate reads) ranks within this season's cohort, the twin of
    `season_metrics`. Both carry the raw estimates beside the percentiles
    (`{stem}_est`), so V27 can check every rank points the way it says.
    Direction follows the era twin's contract."""
    import metric_contracts as mc
    from ingest import era_pool_seasons
    pool_seasons = tuple(era_pool_seasons(season))
    panel = _panels(tuple(sorted(set(pool_seasons) | set(range(min(pool_seasons), season + 1)))))
    P = panel.get(side)
    if P is None or P.empty:
        return {}
    in_pool = P.index.get_level_values("season").isin(pool_seasons) & (P["volume"] >= MIN_VOLUME[side])
    this = P.xs(season, level="season") if season in P.index.get_level_values("season") else None
    if this is None:
        return {}
    in_season = this["volume"] >= MIN_VOLUME[side]
    scope = f"{side}.metrics"
    stems = [s.stem for s in SPECS[side]] + (list(DERIVED_H) if side == "H" else [])
    season_stems = {s.stem for s in SPECS[side] if s.season_rank} | (set(DERIVED_H) if side == "H" else set())
    out: dict[int, dict] = {int(pid): {"talent": {}, "talent_season": {},
                                       "n_eff": float(round(this.at[pid, "n_eff"], 1))}
                            for pid in this.index}
    for stem in stems:
        key = f"{stem}_pct"
        twin = mc.lookup(scope, key) or mc.lookup(f"{side}.season", key)
        inv = bool(twin and twin.inverted)
        era_pool = np.sort(P.loc[in_pool, stem].dropna().values)
        szn_pool = np.sort(this.loc[in_season, stem].dropna().values)
        for pid, v in this[stem].items():
            if v != v:
                continue
            o = out[int(pid)]
            if stem in DERIVED_H:
                pass
            elif len(era_pool):
                r = _rank(era_pool, v)
                o["talent"][key] = round(100.0 - r if inv else r, 2)
                o["talent"][f"{stem}_est"] = round(float(v), 5)
            if side == "H" and stem in season_stems and len(szn_pool):
                r = _rank(szn_pool, v)
                o["talent_season"][key] = round(100.0 - r if inv else r, 2)
                o["talent_season"][f"{stem}_est"] = round(float(v), 5)
    return out
