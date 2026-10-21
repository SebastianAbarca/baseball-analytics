"""
service_time.py — Major League service, from two independent sources.

  1. The published MLS lists (scripts/pull_mls.py): MLB's service list as of
     each January, 2006-2025 — every player under reserve that day with his
     career service in years.days. Names and positions only, no IDs.
  2. The roster ledger (scripts/pull_rosters.py): every club's 40-man roster
     on every regular-season day, 2009 on. A day on the active roster or the
     MLB injured list is a service day; 172 make a year.

Service through a season = that season's list where the player is on it;
otherwise his last earlier list plus his ledger days since. The current season
(no list until January) is the last list plus the ledger. Before 2006 nothing
is needed: the 2006 list already carries each career's total.

It replaces database.query_estimated_service_time, which assumed a player
spent every day from his debut in the majors and subtracted injured-list days
— which earn service — so it was wrong in both directions.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MLS_DIR = ROOT / "data" / "raw" / "mls"
ROSTER_DIR = ROOT / "data" / "raw" / "rosters"
REFERENCE_DIR = ROOT / "data" / "reference"
POSITIONS_CACHE = ROOT / "data" / "processed" / "mlb_primary_positions.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

DAYS_PER_YEAR = 172
# 40-man statuses that earn service: active, MLB injured list, bereavement,
# paternity, family medical emergency — and suspended: the official lists
# credit suspended players in full (A-Rod's 2014, Mejia's and Santana's 2015),
# where the roster archive drops them. Not: optioned (RM), restricted, etc.
SERVICE_STATUSES = frozenset({"A", "D7", "D10", "D15", "D60", "BRV", "PL", "FME", "SU"})
# A short season scaled to a full one. 2020's 67 days were prorated to the
# normal 186-day season (capped at 172): fitted on the 270 players the 2020
# list credits with a partial year, round((days + 1) x 186/67) has a median
# error of 0 and 91% within 10 days; 172/67 left 89% more than 3 days short.
# Full-time players get exactly 1.000 either way (Pujols 19.000 -> 20.000).
SEASON_CREDIT = {2020: (186 / 67, 1)}       # (factor, days added before scaling)
FIRST_LEDGER_SEASON = 2009

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b")


# ---------------------------------------------------------------------------
# The MLS lists
# ---------------------------------------------------------------------------

def mls_days(text) -> Optional[int]:
    """'5.143' -> 5*172 + 143 service days."""
    if not isinstance(text, str) or not text.strip():
        return None
    years, _, days = text.strip().partition(".")
    try:
        return int(years) * DAYS_PER_YEAR + (int(days.ljust(3, "0")) if days else 0)
    except ValueError:
        return None


def is_pitcher_pos(pos) -> Optional[bool]:
    """The list's position: 'rhp-s', 'lhp', 'p', 'sp' are pitchers; a list of
    fielding positions is not; 'dh-rhp' style two-way entries count as both
    (None)."""
    if not isinstance(pos, str) or not pos.strip():
        return None
    parts = pos.lower().split("-")
    isp = lambda p: p in {"p", "sp", "rp"} or p.endswith("hp")
    if isp(parts[0]):
        # after a pitcher code, 's' / 'c' are roles (starter, closer) — but
        # 'c' leading a list ('c-1b') is a catcher
        others = [p for p in parts[1:] if p not in {"s", "c"} and not isp(p)]
        return True if not others else None
    return None if any(isp(p) for p in parts) else False


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = _SUFFIX.sub(" ", s.replace(".", " "))
    return re.sub(r"[^a-z]", "", s)


def load_mls() -> pd.DataFrame:
    """season, player, first, last, pos, pitcher, days — every list row."""
    frames = []
    for f in sorted(MLS_DIR.glob("mls_thru_*.csv")):
        season = int(re.search(r"(\d{4})", f.name).group(1))
        df = pd.read_csv(f, dtype=str).assign(season=season)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no MLS lists in {MLS_DIR}; run scripts/pull_mls.py")
    d = pd.concat(frames, ignore_index=True)
    split = d["player"].str.split(",", n=1, expand=True)
    d["last"], d["first"] = split[0].str.strip(), split[1].fillna("").str.strip()
    d["k_last"], d["k_first"] = d["last"].map(norm_name), d["first"].map(norm_name)
    d["pitcher"] = d["pos"].map(is_pitcher_pos)
    d["days"] = d["mls"].map(mls_days)
    return d


# ---------------------------------------------------------------------------
# Matching list names to MLBAM ids
# ---------------------------------------------------------------------------

def _chadwick_people() -> pd.DataFrame:
    cols = ["key_mlbam", "name_first", "name_last", "name_given", "name_nick",
            "mlb_played_first", "mlb_played_last"]
    frames = [pd.read_csv(f, usecols=cols, low_memory=False)
              for f in sorted(REFERENCE_DIR.glob("people-*.csv"))]
    p = pd.concat(frames, ignore_index=True).dropna(subset=["key_mlbam", "mlb_played_first"])
    p = p[p["mlb_played_last"] >= 2000].copy()
    p["key_mlbam"] = p["key_mlbam"].astype(int)
    p["k_last"] = p["name_last"].map(norm_name)
    p["k_first"] = p["name_first"].map(norm_name)
    p["k_nick"] = p["name_nick"].fillna("").map(norm_name)
    return p


def primary_positions(ids: list[int]) -> dict[int, str]:
    """MLB's primary position abbreviation per id ('P', 'C', 'SS', 'TWP', ...),
    cached — positions do not change for the purpose of telling two
    same-named players apart."""
    import requests
    have = (pd.read_csv(POSITIONS_CACHE) if POSITIONS_CACHE.exists()
            else pd.DataFrame(columns=["key_mlbam", "pos"]))
    known = dict(zip(have["key_mlbam"].astype(int), have["pos"]))
    todo = [i for i in ids if i not in known]
    for i in range(0, len(todo), 200):
        batch = ",".join(str(x) for x in todo[i:i + 200])
        r = requests.get("https://statsapi.mlb.com/api/v1/people",
                         params={"personIds": batch, "fields": "people,id,primaryPosition,abbreviation"},
                         timeout=60)
        r.raise_for_status()
        for person in r.json().get("people", []):
            known[int(person["id"])] = (person.get("primaryPosition") or {}).get("abbreviation", "")
    if todo:
        POSITIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sorted(known.items()), columns=["key_mlbam", "pos"]).to_csv(POSITIONS_CACHE, index=False)
    return {i: known.get(i, "") for i in ids}


def match_mls_ids(mls: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    One row per list row with key_mlbam (or none) and how it was decided:

      name      exactly one MLB player with that name active by that season
      position  several; one whose primary position matches the list's
      roster    several; one on a 40-man roster that season (ledger, 2009+)
      chain     several; the one the same name resolved to in another season
      ambiguous several remain
      unmatched no MLB player with that name
    """
    mls = load_mls() if mls is None else mls
    people = _chadwick_people()
    pos = primary_positions(sorted(people["key_mlbam"].unique().tolist()))
    people["mlb_pos"] = people["key_mlbam"].map(pos)
    people["mlb_pitcher"] = people["mlb_pos"].map(lambda p: None if p == "TWP" else p == "P")

    by_name: dict[tuple, list[int]] = {}
    for col in ("k_first", "k_nick"):
        for (last, first), ids in people.groupby(["k_last", col])["key_mlbam"]:
            if first:
                by_name.setdefault((last, first), []).extend(ids.tolist())
    info = people.drop_duplicates("key_mlbam").set_index("key_mlbam")
    on_roster = _roster_presence()

    out = []
    for r in mls.itertuples():
        cands = sorted(set(by_name.get((r.k_last, r.k_first), [])))
        name_cands = [c for c in cands if info.at[c, "mlb_played_first"] <= r.season
                      <= info.at[c, "mlb_played_last"] + 3]
        # active around that season (the list is as of the following January;
        # a player can sit out injured or unsigned). This is what separates
        # Vladimir Guerrero Jr. from his father once "Jr." is normalized away.
        cands = [c for c in cands if info.at[c, "mlb_played_first"] <= r.season
                 <= info.at[c, "mlb_played_last"] + 3]
        method = "name"
        if len(cands) > 1 and r.pitcher is not None:
            pos_ok = [c for c in cands if info.at[c, "mlb_pitcher"] in (r.pitcher, None)]
            if pos_ok and len(pos_ok) < len(cands):
                cands, method = pos_ok, "position"
        if len(cands) > 1 and r.season >= FIRST_LEDGER_SEASON:
            ros = [c for c in cands if (r.season, c) in on_roster]
            if len(ros) == 1:
                cands, method = ros, "roster"
        key = cands[0] if len(cands) == 1 else None
        method = method if len(cands) == 1 else ("unmatched" if not cands else "ambiguous")
        out.append({"season": r.season, "player": r.player, "pos": r.pos, "days": r.days,
                    "pitcher": r.pitcher, "key_mlbam": key, "method": method,
                    "candidates": cands, "name_key": (r.k_last, r.k_first),
                    "all_cands": tuple(name_cands)})
    res = pd.DataFrame(out)

    # chain: a still-ambiguous name resolves to the id the same name+position
    # resolved to in any other season
    solved = (res.dropna(subset=["key_mlbam"]).groupby(["player", "pos"])["key_mlbam"]
              .agg(lambda s: set(s)))
    for i in res.index[res["method"] == "ambiguous"]:
        ids = solved.get((res.at[i, "player"], res.at[i, "pos"]))
        if ids and len(ids) == 1 and next(iter(ids)) in res.at[i, "candidates"]:
            res.at[i, "key_mlbam"], res.at[i, "method"] = next(iter(ids)), "chain"
    _fallback(res, mls, people, info)
    _resolve_by_continuity(res, info, on_roster)
    _drop_chain_breaks(res)
    res.loc[(res["method"] == "unmatched") & (res["days"].fillna(0) == 0), "method"] = "no service"
    res["key_mlbam"] = res["key_mlbam"].astype("Int64")
    return res


# The list sometimes gives a legal name where Chadwick has the name a player
# goes by (and the reverse), plus the odd typo. These are tried only for rows
# the exact name did not resolve, and only among players active that season.
_DIMINUTIVE = {
    "mike": "michael", "tom": "thomas", "jon": "jonathan", "dave": "david",
    "bob": "robert", "bobby": "robert", "rob": "robert", "ron": "ronald",
    "ronnie": "ronald", "matt": "matthew", "chris": "christopher", "jim": "james",
    "jimmy": "james", "joe": "joseph", "joey": "joseph", "nick": "nicholas",
    "dan": "daniel", "danny": "daniel", "tony": "anthony", "andy": "andrew",
    "drew": "andrew", "alex": "alexander", "zach": "zachary", "zack": "zachary",
    "jake": "jacob", "will": "william", "bill": "william", "billy": "william",
    "ben": "benjamin", "sam": "samuel", "greg": "gregory", "jeff": "jeffrey",
    "ken": "kenneth", "kenny": "kenneth", "steve": "steven", "pat": "patrick",
    "rick": "richard", "ricky": "richard", "dick": "richard", "ed": "edward",
    "eddie": "edward", "fred": "frederick", "freddy": "frederick", "josh": "joshua",
    "tim": "timothy", "tommy": "thomas", "jerry": "gerald", "larry": "lawrence",
    "charlie": "charles", "chuck": "charles", "johnny": "john", "jack": "john",
    "kike": "enrique", "manny": "manuel", "rafa": "rafael", "vinny": "vincent",
    "nate": "nathan", "phil": "philip", "al": "alberto", "benji": "bengie",
    "dee": "devaris", "cj": "carl", "aj": "anthony",
}
# One group per canonical name: Tom, Tommy and Thomas all try each other.
_NAME_GROUP: dict[str, set[str]] = {}
for _short, _full in _DIMINUTIVE.items():
    _g = _NAME_GROUP.setdefault(_full, {_full})
    _g.add(_short)
    _NAME_GROUP[_short] = _g

# Players MLB lists under a name Chadwick does not carry at all.
_ALIAS = {("carmona", "fausto"): ("hernandez", "roberto"),
          ("nunez", "leo"): ("oviedo", "juancarlos"),
          ("gordon", "dee"): ("strangegordon", "dee"),
          ("sanchez", "yolmercarlos"): ("sanchez", "yolmer"),
          ("heredia", "robertohernandez"): ("hernandez", "roberto"),
          ("palmiero", "orlando"): ("palmeiro", "orlando")}


def _resolve_by_continuity(res: pd.DataFrame, info, on_roster: set) -> None:
    """
    Same-named players, decided by whether their service CHAINS. A player's
    total can only grow, by at most 172 days a season, and can never exceed
    172 x his seasons since his first MLB year. Matching rows one at a time
    gave two Chad Tracys one id, so the "same" player lost 1,032 days in a
    year; and two Nick Greens, so one gained 690.

    For every group of list rows sharing the same set of 2+ name candidates,
    assign rows season by season to the candidate whose running total they
    continue, using position and 40-man presence (2009+) as further evidence.
    A season with no unique best assignment leaves its rows ambiguous.
    """
    from itertools import permutations
    groups: dict[tuple, list[int]] = {}
    for i, c in res["all_cands"].items():
        if len(c) >= 2:
            groups.setdefault((res.at[i, "name_key"], c), []).append(i)
    for (_, cands), idx in groups.items():
        last: dict[int, tuple[int, int]] = {}
        rows = res.loc[idx].sort_values(["season", "days"])
        for season, srows in rows.groupby("season"):
            srows = srows[srows["days"].notna()]
            if srows.empty:
                continue

            def score(i, c) -> Optional[int]:
                days, want = int(res.at[i, "days"]), res.at[i, "pitcher"]
                first = int(info.at[c, "mlb_played_first"])
                if not first <= season <= info.at[c, "mlb_played_last"] + 3:
                    return None
                if days > DAYS_PER_YEAR * (season - first + 1):
                    return None
                mp = info.at[c, "mlb_pitcher"]
                if want is not None and mp is not None and mp != want:
                    return None
                if c in last:
                    s0, d0 = last[c]
                    if not 0 <= days - d0 <= DAYS_PER_YEAR * (season - s0):
                        return None
                return 1 + (c in last) + ((season, c) in on_roster)

            best, best_score, tie = None, -1, False
            for perm in permutations(cands, len(srows)):
                sc = [score(i, c) for i, c in zip(srows.index, perm)]
                if any(x is None for x in sc):
                    continue
                total = sum(sc)
                if total > best_score:
                    best, best_score, tie = perm, total, False
                elif total == best_score:
                    tie = True
            for k, i in enumerate(srows.index):
                if best is not None and not tie:
                    c = best[k]
                    res.at[i, "key_mlbam"], res.at[i, "method"] = c, "continuity"
                    last[c] = (int(season), int(res.at[i, "days"]))
                else:
                    res.at[i, "key_mlbam"], res.at[i, "method"] = None, "ambiguous"
                    res.at[i, "candidates"] = list(cands)


# MLB corrects a list after the fact now and then: 1-3% of players gain a few
# days more than a season holds between consecutive lists (Lester 2007: +175).
# Beyond that slack a break is a wrong match, not a correction.
CHAIN_SLACK = 18


def _drop_chain_breaks(res: pd.DataFrame) -> None:
    """
    For each id, keep the LONGEST run of its list rows that chains — every
    total at least the last, and grown by no more than 172 days a season plus
    CHAIN_SLACK — and unmatch the rest as "chain break". Those belong to a
    same-named player Chadwick does not separate (two Chad Tracys: 1,032
    days, then 0). Longest rather than first-come, so one early wrong row
    cannot throw out a correct career. At most one row per season survives.
    """
    m = res[res["key_mlbam"].notna() & res["days"].notna()]
    for _, g in m.groupby("key_mlbam"):
        rows = sorted(g.index, key=lambda i: (res.at[i, "season"], res.at[i, "days"]))
        if len(rows) < 2:
            continue
        seas = [int(res.at[i, "season"]) for i in rows]
        days = [int(res.at[i, "days"]) for i in rows]

        def ok(j, k):           # row k can follow row j
            gap = seas[k] - seas[j]
            return gap >= 1 and 0 <= days[k] - days[j] <= DAYS_PER_YEAR * gap + CHAIN_SLACK

        best = [1] * len(rows)
        back = [-1] * len(rows)
        for k in range(len(rows)):
            for j in range(k):
                if ok(j, k) and best[j] + 1 > best[k]:
                    best[k], back[k] = best[j] + 1, j
        k = max(range(len(rows)), key=lambda x: (best[x], -x))
        keep = set()
        while k != -1:
            keep.add(rows[k])
            k = back[k]
        for i in rows:
            if i not in keep:
                res.at[i, "key_mlbam"], res.at[i, "method"] = None, "chain break"


def _edit1(a: str, b: str) -> bool:
    """Within one insertion, deletion or substitution (typos: Padillia/Padilla)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def _name_variants(player: str) -> list[tuple[str, str]]:
    """(last, first) keys to try: the name as listed, without a middle initial,
    and any name given in parentheses ('Upton, Melvin Jr. (B.J.)',
    'Casilla, Santiago (Garcia, Jairo)')."""
    base = re.sub(r"\(.*?\)", "", player)
    paren = re.findall(r"\((.*?)\)", player)
    last, _, first = base.partition(",")
    first = re.sub(r"\b[A-Za-z]\.?\s*$", "", first.strip()).strip() or first.strip()
    out = [(last, first)]
    for p in paren:
        if "," in p:
            l2, _, f2 = p.partition(",")
            out.append((l2, f2))
        else:
            out.append((last, p))
    keys = [(norm_name(l), norm_name(f)) for l, f in out if norm_name(l)]
    return keys + [_ALIAS[k] for k in keys if k in _ALIAS]


def _fallback(res: pd.DataFrame, mls: pd.DataFrame, people: pd.DataFrame, info) -> None:
    """Resolve unmatched rows with real service: middle initials, parentheses,
    given names, diminutives, then one-letter typos in either name — each
    among players active that season (first MLB year <= season <= last + 3)
    and position-compatible. Only a single surviving candidate is accepted."""
    people = people.assign(k_given=people["name_given"].fillna("").map(
        lambda g: norm_name(str(g).split(" ")[0])))
    by_last = {k: g for k, g in people.groupby("k_last")}
    pitcher = dict(zip(mls.index, mls["pitcher"]))

    def active(c, season):
        return info.at[c, "mlb_played_first"] <= season <= info.at[c, "mlb_played_last"] + 3

    def pos_ok(c, want):
        mp = info.at[c, "mlb_pitcher"]
        return want is None or mp is None or mp == want

    for i in res.index[(res["method"] == "unmatched") & (res["days"].fillna(0) > 0)]:
        season, want = int(res.at[i, "season"]), pitcher.get(i)
        found, how = set(), None
        for last, first in _name_variants(res.at[i, "player"]):
            firsts = {first} | _NAME_GROUP.get(first, set())
            if len(first) > 4:      # 'johnryan' -> 'john', a two-word first name
                firsts |= {first[:4]}
            pool = by_last.get(last)
            if pool is not None:
                hit = pool[pool["k_first"].isin(firsts) | pool["k_given"].isin(firsts)
                           | pool["k_nick"].isin(firsts)]
                found = {c for c in hit["key_mlbam"] if active(c, season) and pos_ok(c, want)}
                how = "variant"
            if not found:       # one-letter typo in the last name, then the first
                for l2, pool2 in by_last.items():
                    if l2[:1] == last[:1] and _edit1(l2, last):
                        hit = pool2[pool2["k_first"].isin(firsts) | pool2["k_given"].isin(firsts)]
                        found |= {c for c in hit["key_mlbam"] if active(c, season) and pos_ok(c, want)}
                if not found and pool is not None:
                    hit = pool[pool["k_first"].map(lambda f: any(_edit1(f, x) for x in firsts))]
                    found = {c for c in hit["key_mlbam"] if active(c, season) and pos_ok(c, want)}
                how = "typo"
            if found:
                break
        if len(found) == 1:
            res.at[i, "key_mlbam"], res.at[i, "method"] = found.pop(), how
        elif found:
            res.at[i, "method"], res.at[i, "candidates"] = "ambiguous", sorted(found)


# ---------------------------------------------------------------------------
# The roster ledger
# ---------------------------------------------------------------------------

def _roster_presence() -> set[tuple[int, int]]:
    """(season, player_id) for everyone on any 40-man roster in a season."""
    out: set[tuple[int, int]] = set()
    for f in sorted(ROSTER_DIR.glob("rosters_*.parquet")):
        if ".partial" in f.name:
            continue
        df = pd.read_parquet(f, columns=["season", "player_id"]).drop_duplicates()
        out |= set(zip(df["season"].astype(int), df["player_id"].astype(int)))
    return out


def ledger_days(season: int) -> Optional[pd.Series]:
    """Service days per player in one season from the roster ledger: distinct
    days on a service status, capped at 172. None without that season's file."""
    f = ROSTER_DIR / f"rosters_{season}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f, columns=["date", "player_id", "status"])
    svc = df[df["status"].isin(SERVICE_STATUSES)][["date", "player_id"]]
    # MLB's archive occasionally drops a player from a roster (Figgins, LAA
    # 2009, on no roster all season). A day he played in a game is a day he
    # was on the active roster: from 2015, Statcast's appearances fill it.
    raw = ROOT / "data" / "raw" / f"statcast_{season}.parquet"
    if raw.exists():
        sc = pd.read_parquet(raw, columns=["game_date", "batter", "pitcher"])
        played = pd.concat([sc[["game_date", "batter"]].set_axis(["date", "player_id"], axis=1),
                            sc[["game_date", "pitcher"]].set_axis(["date", "player_id"], axis=1)])
        played["date"] = pd.to_datetime(played["date"])
        svc = pd.concat([svc, played.drop_duplicates()])
    days = svc.drop_duplicates().groupby("player_id")["date"].nunique()
    factor, plus = SEASON_CREDIT.get(season, (1.0, 0))
    days = ((days + plus) * factor).round().astype(int)
    return days.clip(upper=DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Service through a season
# ---------------------------------------------------------------------------

def service_days_through(season: int, matched: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Career service days through the end of `season`, per key_mlbam: the
    season's list where the player is on it; else his last earlier list plus
    ledger days for each season since. Players on no list by then are counted
    from the ledger alone (their careers began 2009 or later). A season the
    ledger lacks adds nothing — logged, not guessed.
    """
    matched = match_mls_ids() if matched is None else matched
    m = matched.dropna(subset=["key_mlbam", "days"])
    m = m[m["season"] <= season].sort_values("season")
    last = m.groupby("key_mlbam").last()[["season", "days"]]
    ledgers = {s: ledger_days(s) for s in range(FIRST_LEDGER_SEASON, season + 1)}
    missing = [s for s, l in ledgers.items() if l is None]
    if missing:
        log.warning("service time: no roster ledger for %s — those seasons add nothing", missing)
    ids = set(last.index)
    for l in ledgers.values():
        if l is not None:
            ids |= set(l.index.astype(int))
    out = {}
    for pid in ids:
        if pid in last.index:
            base_season, total = int(last.at[pid, "season"]), int(last.at[pid, "days"])
        else:
            base_season, total = FIRST_LEDGER_SEASON - 1, 0
        for s in range(base_season + 1, season + 1):
            l = ledgers.get(s)
            if l is not None:
                total += int(l.get(pid, 0))
        out[int(pid)] = total
    return pd.Series(out, name="service_days")


# ---------------------------------------------------------------------------
# Frozen totals — what the nightly runner reads
# ---------------------------------------------------------------------------
#
# The lists are third-party data kept out of git, and the ledger is 18
# seasons of rosters: neither is on the GitHub runner. So each finished
# season's result is frozen like the other era roots — service_through_{s}.csv,
# one row per player, committed to data/era by freeze_era.py — and the runner
# builds the season in progress as last season's frozen totals plus this
# season's ledger, which it pulls a day at a time.

def frozen_path(season: int) -> Path:
    return PROCESSED_DIR / f"service_through_{season}.csv"


_MATCHED: Optional[pd.DataFrame] = None
_THROUGH_MEMO: dict[int, pd.Series] = {}


def _finished(season: int) -> bool:
    from ingest import season_state
    finished, _ = season_state()
    return season in finished


def service_through(season: int) -> pd.Series:
    """
    Career service days through `season`, per key_mlbam.

    Where the lists are present (the build machine) it is computed from
    lists + ledger, and a finished season is written to frozen_path for
    freeze_era. Without them (the runner) a finished season reads its frozen
    file, and the season in progress is the previous season's frozen totals
    plus this season's ledger days.
    """
    global _MATCHED
    if season in _THROUGH_MEMO:
        return _THROUGH_MEMO[season]
    path = frozen_path(season)
    if any(MLS_DIR.glob("mls_thru_*.csv")):
        if _MATCHED is None:
            _MATCHED = match_mls_ids()
        out = service_days_through(season, _MATCHED)
        if _finished(season):
            text = out.rename_axis("key_mlbam").reset_index().sort_values("key_mlbam").to_csv(index=False)
            if not path.exists() or path.read_text() != text:
                path.write_text(text)
    elif path.exists():
        f = pd.read_csv(path)
        out = pd.Series(f["service_days"].to_numpy(), index=f["key_mlbam"].astype(int), name="service_days")
    else:
        prev = frozen_path(season - 1)
        if not prev.exists():
            raise FileNotFoundError(f"no service lists, no {path.name}, no {prev.name}: "
                                    "run scripts/pull_mls.py, or freeze_era.py where they are")
        f = pd.read_csv(prev)
        base = pd.Series(f["service_days"].to_numpy(), index=f["key_mlbam"].astype(int))
        led = ledger_days(season)
        led = pd.Series(dtype=int) if led is None else led
        out = base.add(led, fill_value=0).astype(int).rename("service_days")
    _THROUGH_MEMO[season] = out
    return out


def fingerprint_inputs(season: int) -> list[Path]:
    """The files a season's service time rests on, for the build fingerprint —
    the same bytes on the build machine and the runner."""
    if _finished(season):
        service_through(season)               # writes the frozen file where lists exist
        return [frozen_path(season)]
    service_through(season - 1)
    return [frozen_path(season - 1), ROSTER_DIR / f"rosters_{season}.parquet"]


_YEARS_MEMO: dict[int, dict[int, float]] = {}


def service_years_days(player_ids, season: int) -> dict[int, float]:
    """
    {key_mlbam: service through `season` as years.days} — 5.143 is 5 years
    and 143 days, the MLB convention, so the CBA thresholds read directly:
    < 3.000 pre-arbitration, 3.000-5.171 arbitration, >= 6.000 free agency.
    Players with no service source are left out. Memoised per season.
    """
    if season not in _YEARS_MEMO:
        days = service_through(season)
        _YEARS_MEMO[season] = {pid: d // DAYS_PER_YEAR + (d % DAYS_PER_YEAR) / 1000
                               for pid, d in days.items()}
    memo = _YEARS_MEMO[season]
    return {int(p): memo[int(p)] for p in player_ids if int(p) in memo}


def service_years(days) -> int:
    """Whole service years: 2.171 is 2 — the decimal is days, not tenths."""
    return int(days) // DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Reconciliation report
# ---------------------------------------------------------------------------

def reconcile(matched: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    For every season with both lists and a ledger: each matched player's
    increase from the previous list against his ledger days that season.
    Only players on BOTH consecutive lists — a player missing from a list
    (an unsigned free agent that January) has no increase to compare.
    """
    matched = match_mls_ids() if matched is None else matched
    m = matched.dropna(subset=["key_mlbam", "days"]).drop_duplicates(["season", "key_mlbam"])
    wide = m.pivot(index="key_mlbam", columns="season", values="days")
    names = m.sort_values("season").groupby("key_mlbam")["player"].last()
    rows = []
    for s in wide.columns:
        if (s - 1) not in wide.columns:
            continue
        l = ledger_days(s)
        if l is None:
            continue
        both = wide[[s - 1, s]].dropna()
        for pid, (a, b) in both.iterrows():
            rows.append({"season": s, "key_mlbam": int(pid), "player": names.get(pid),
                         "mls_increase": int(b - a), "ledger_days": int(l.get(pid, 0))})
    r = pd.DataFrame(rows)
    if len(r):
        r["diff"] = r["ledger_days"] - r["mls_increase"]
    return r
