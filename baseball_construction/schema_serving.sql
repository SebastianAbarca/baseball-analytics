-- Baseball Analytics — serving schema
-- Run in: Supabase Dashboard → SQL Editor → New query. Safe to re-run.
--
-- What the dashboard reads, and nothing else. scripts/publish.py is the only
-- writer and uses the service key, which bypasses Row Level Security; the
-- dashboard uses the anon key, which RLS limits to SELECT on these tables and
-- EXECUTE on the functions below. So the serving path is read-only by
-- construction, not by a flag in the app.
--
-- Supabase serves every table here as REST (/rest/v1/<table>) and every
-- function as RPC (/rest/v1/rpc/<function>). That is the API.
--
-- JSON vs JSONB. Anything published as-built is JSON, not JSONB: JSONB
-- re-sorts object keys (shortest first), and the dashboard reads meaning
-- into key order — `trait_density` is written densest-first and the identity
-- card shows its first two entries. Stored as JSONB, HOU 2025's rotation read
-- "swingman 30% · patterned 24%" instead of "deep arsenal 61% · contact
-- suppressor 53%". JSON keeps the text exactly as published. Only columns the
-- SQL functions below query with JSONB operators (tag_detail, vec) are JSONB,
-- and neither carries meaning in key order.

-- Functions are dropped first because their return type changed from JSONB to
-- JSON, which CREATE OR REPLACE cannot do. They hold no data.
DROP FUNCTION IF EXISTS scout_comps(TEXT, INT, INT, TEXT[], BOOLEAN);
DROP FUNCTION IF EXISTS scout_rows(TEXT[]);
DROP FUNCTION IF EXISTS scout_name_search(TEXT, TEXT, INT);
DROP FUNCTION IF EXISTS scout_search(TEXT[], TEXT[], TEXT, BOOLEAN, INT, INT, INT, BOOLEAN, TEXT[], TEXT, INT, INT);
DROP FUNCTION IF EXISTS scout_row(scout_players);

-- ── Portraits ──────────────────────────────────────────────────────────────
-- One row per team-season: the whole portrait, as built.
-- gate_fingerprint lets a reader refuse a portrait cut by different tag logic
-- without needing the build machine's pool files (see team_portrait.py).
CREATE TABLE IF NOT EXISTS portraits (
    team              TEXT        NOT NULL,
    season            INT         NOT NULL,
    schema_version    INT         NOT NULL,
    build_fingerprint TEXT,
    gate_fingerprint  TEXT,
    content_hash      TEXT        NOT NULL,
    built_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    data              JSON        NOT NULL,
    PRIMARY KEY (team, season)
);

-- ── League seasons ─────────────────────────────────────────────────────────
-- Season-wide artefacts the dashboard used to read from local files:
--   identity        league_identity_{season}.json (fingerprints, neighbours,
--                   baselines)
--   tunnel_average  league mean separation curve per pitch type, computed at
--                   publish time instead of reading 30 portraits per request
--   pitch_mix       {pitcher_id: [{pitch, usage, velo, whiff}, ...]}
CREATE TABLE IF NOT EXISTS league_seasons (
    season            INT         PRIMARY KEY,
    build_fingerprint TEXT,
    gate_fingerprint  TEXT,
    content_hash      TEXT        NOT NULL,
    built_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    identity          JSON,
    tunnel_average    JSON,
    pitch_mix         JSON
);

-- ── Scout ──────────────────────────────────────────────────────────────────
-- Tag catalogue: one row per tag. `dir` = 'low' when a LOW percentile is the
-- trait (`patient` is chase rate at the 5th pct); derived at publish time.
CREATE TABLE IF NOT EXISTS scout_tags (
    tag     TEXT PRIMARY KEY,
    kind    TEXT,
    family  TEXT,
    side    TEXT,     -- 'H', 'P' or 'B' (both)
    n       INT,
    dir     TEXT      -- 'high', 'low' or NULL (categorical)
);

-- One row per player-season-group, flattened from the portraits.
-- `ord` is the row's position in the flattening order (portrait files sorted
-- by name, then hitters / starters / bullpen_arms). Every sort breaks ties on
-- it, which reproduces the stable-sort order the in-memory index had.
-- `vec` is the comp vector (each metric's era percentile, centred and
-- domain-weighted: team_similarity.player_vector), `vec_norm` its L2 norm —
-- both computed at publish time by the same Python that defined them.
CREATE TABLE IF NOT EXISTS scout_players (
    season      INT     NOT NULL,
    team        TEXT    NOT NULL,
    player_id   BIGINT  NOT NULL,
    grp         TEXT    NOT NULL,   -- hitters / starters / bullpen_arms
    ord         INT     NOT NULL,
    side        TEXT    NOT NULL,   -- 'H' or 'P'
    name        TEXT,
    age         INT,
    volume      INT,                -- PA for hitters, BF for pitchers
    war         REAL,
    limited     BOOLEAN NOT NULL DEFAULT false,
    position    TEXT,
    hand        TEXT,
    populations TEXT[]  NOT NULL DEFAULT '{}',
    metrics     JSON    NOT NULL DEFAULT '{}',
    tags        TEXT[]  NOT NULL DEFAULT '{}',
    tag_detail  JSONB   NOT NULL DEFAULT '[]',   -- [[tag, pct, evidence], ...]
    vec         JSONB   NOT NULL DEFAULT '{}',
    vec_norm    DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (season, team, player_id, grp)
);
-- Installs created before the JSON change: convert in place. A no-op when the
-- column is already JSON. The rows still need re-publishing (publish.py
-- --force), since JSONB had already discarded their key order.
ALTER TABLE portraits      ALTER COLUMN data           TYPE JSON USING data::json;
ALTER TABLE league_seasons ALTER COLUMN identity       TYPE JSON USING identity::json;
ALTER TABLE league_seasons ALTER COLUMN tunnel_average TYPE JSON USING tunnel_average::json;
ALTER TABLE league_seasons ALTER COLUMN pitch_mix      TYPE JSON USING pitch_mix::json;
ALTER TABLE scout_players  ALTER COLUMN metrics DROP DEFAULT;
ALTER TABLE scout_players  ALTER COLUMN metrics        TYPE JSON USING metrics::json;
ALTER TABLE scout_players  ALTER COLUMN metrics SET DEFAULT '{}';

CREATE INDEX IF NOT EXISTS scout_players_tags_gin ON scout_players USING GIN (tags);
CREATE INDEX IF NOT EXISTS scout_players_side_season ON scout_players (side, season);
CREATE INDEX IF NOT EXISTS scout_players_ord ON scout_players (ord);
CREATE INDEX IF NOT EXISTS scout_players_key ON scout_players (player_id, team, season);

-- ── Access: read-only for the public key ───────────────────────────────────
ALTER TABLE portraits      ENABLE ROW LEVEL SECURITY;
ALTER TABLE league_seasons ENABLE ROW LEVEL SECURITY;
ALTER TABLE scout_tags     ENABLE ROW LEVEL SECURITY;
ALTER TABLE scout_players  ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "public read" ON portraits;
DROP POLICY IF EXISTS "public read" ON league_seasons;
DROP POLICY IF EXISTS "public read" ON scout_tags;
DROP POLICY IF EXISTS "public read" ON scout_players;
CREATE POLICY "public read" ON portraits      FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "public read" ON league_seasons FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "public read" ON scout_tags     FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "public read" ON scout_players  FOR SELECT TO anon, authenticated USING (true);

-- Belt and braces: no write privilege at all, so a future permissive policy
-- cannot open writes by accident.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE
    ON portraits, league_seasons, scout_tags, scout_players
    FROM anon, authenticated;
GRANT SELECT ON portraits, league_seasons, scout_tags, scout_players
    TO anon, authenticated;

-- ── API functions ──────────────────────────────────────────────────────────
-- Rows come back in the short-key shape the dashboard renderers already use
-- (i, n, t, s, d, g, a, v, w, l, p, h, pop, tg, m), so swapping the source
-- did not mean rewriting the tables and charts that draw them.

CREATE OR REPLACE FUNCTION scout_row(p scout_players, p_sim DOUBLE PRECISION DEFAULT NULL)
RETURNS JSON LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE WHEN p_sim IS NULL THEN json_build_object(
        'i', p.player_id, 'n', p.name, 't', p.team, 's', p.season,
        'd', p.side, 'g', p.grp, 'a', p.age, 'v', p.volume,
        'w', p.war, 'l', p.limited, 'p', p.position, 'h', p.hand,
        'pop', to_json(p.populations), 'tg', p.tag_detail, 'm', p.metrics)
    ELSE json_build_object(
        'i', p.player_id, 'n', p.name, 't', p.team, 's', p.season,
        'd', p.side, 'g', p.grp, 'a', p.age, 'v', p.volume,
        'w', p.war, 'l', p.limited, 'p', p.position, 'h', p.hand,
        'pop', to_json(p.populations), 'tg', p.tag_detail, 'm', p.metrics,
        'sim', p_sim) END
$$;

-- Direction-normalised strength of one [tag, pct, evidence] triple, 0..100,
-- or NULL when the tag has no percentile. Mirrors charts.tag_strength.
CREATE OR REPLACE FUNCTION scout_strength(triple JSONB)
RETURNS DOUBLE PRECISION LANGUAGE sql STABLE AS $$
    SELECT CASE
        WHEN jsonb_typeof(triple -> 1) <> 'number' THEN NULL
        WHEN (SELECT dir FROM scout_tags WHERE tag = triple ->> 0) = 'low'
            THEN 100.0 - (triple ->> 1)::DOUBLE PRECISION
        ELSE (triple ->> 1)::DOUBLE PRECISION
    END
$$;

-- Scout search. Mirrors charts.scout_query: same filters, same five sorts,
-- same tie order. Returns slim rows (what the results table renders) plus
-- `total`, the full match count, on every row.
CREATE OR REPLACE FUNCTION scout_search(
    p_tags            TEXT[]  DEFAULT NULL,
    p_kinds           TEXT[]  DEFAULT NULL,
    p_side            TEXT    DEFAULT 'H',
    p_match_all       BOOLEAN DEFAULT true,
    p_season_min      INT     DEFAULT NULL,
    p_season_max      INT     DEFAULT NULL,
    p_min_vol         INT     DEFAULT 0,
    p_exclude_limited BOOLEAN DEFAULT true,
    p_teams           TEXT[]  DEFAULT NULL,
    p_sort            TEXT    DEFAULT 'extreme',
    p_limit           INT     DEFAULT 1000,
    p_offset          INT     DEFAULT 0
)
RETURNS SETOF JSON LANGUAGE sql STABLE SECURITY INVOKER AS $$
WITH prm AS (
    SELECT
        COALESCE(cardinality(p_tags), 0)  > 0 AS by_tag,
        COALESCE(cardinality(p_kinds), 0) > 0 AS by_kind,
        CASE WHEN p_sort IN ('extreme','war','volume','recent','name')
             THEN p_sort ELSE 'extreme' END AS srt
),
base AS (
    SELECT p.* FROM scout_players p, prm
    WHERE (p_side IS NULL OR p_side = '' OR p.side = p_side)
      AND (NOT COALESCE(p_exclude_limited, false) OR NOT p.limited)
      AND (p_season_min IS NULL OR p_season_min = 0 OR p.season >= p_season_min)
      AND (p_season_max IS NULL OR p_season_max = 0 OR p.season <= p_season_max)
      AND (COALESCE(p_min_vol, 0) = 0 OR COALESCE(p.volume, 0) >= p_min_vol)
      AND (COALESCE(cardinality(p_teams), 0) = 0 OR p.team = ANY (p_teams))
      AND (NOT prm.by_tag
           OR (p_match_all AND p.tags @> p_tags)
           OR (NOT p_match_all AND p.tags && p_tags))
),
matched AS (
    SELECT b.*,
        (SELECT COALESCE(jsonb_agg(d.e ORDER BY d.o), '[]'::jsonb)
           FROM jsonb_array_elements(b.tag_detail) WITH ORDINALITY d(e, o)
          WHERE CASE
                  WHEN prm.by_tag  THEN d.e ->> 0 = ANY (p_tags)
                  WHEN prm.by_kind THEN (SELECT kind FROM scout_tags
                                          WHERE tag = d.e ->> 0) = ANY (p_kinds)
                  ELSE true
                END) AS mt
    FROM base b, prm
),
scored AS (
    SELECT m.*,
        COALESCE((SELECT avg(scout_strength(e)) FROM jsonb_array_elements(m.mt) e), 0.0) AS score
    FROM matched m, prm
    WHERE prm.by_tag OR NOT prm.by_kind OR jsonb_array_length(m.mt) > 0
)
SELECT json_build_object(
        'i', s.player_id, 'n', s.name, 't', s.team, 's', s.season, 'd', s.side,
        'a', s.age, 'v', s.volume, 'w', s.war, 'l', s.limited,
        'matched', s.mt, 'score', s.score, 'total', count(*) OVER ())
FROM scored s, prm
ORDER BY
    CASE WHEN prm.srt = 'extreme' THEN s.score END DESC NULLS LAST,
    CASE WHEN prm.srt = 'extreme' THEN COALESCE(s.volume, 0) END DESC NULLS LAST,
    -- Python's `w or -99` treats a WAR of exactly 0.0 as missing too.
    CASE WHEN prm.srt = 'war' THEN
        CASE WHEN s.war IS NULL OR s.war = 0 THEN -99 ELSE s.war END END DESC NULLS LAST,
    CASE WHEN prm.srt = 'volume' THEN COALESCE(s.volume, 0) END DESC NULLS LAST,
    CASE WHEN prm.srt = 'recent' THEN s.season END DESC NULLS LAST,
    CASE WHEN prm.srt = 'recent' THEN COALESCE(s.volume, 0) END DESC NULLS LAST,
    CASE WHEN prm.srt = 'name' THEN COALESCE(s.name, '') END COLLATE "C" ASC NULLS LAST,
    s.ord ASC
LIMIT p_limit OFFSET p_offset
$$;

-- Name search for the Compare picker. Mirrors charts.compare_player_options.
CREATE OR REPLACE FUNCTION scout_name_search(
    p_query TEXT, p_side TEXT DEFAULT 'H', p_limit INT DEFAULT 40)
RETURNS SETOF JSON LANGUAGE sql STABLE SECURITY INVOKER AS $$
    SELECT json_build_object('i', player_id, 'n', name, 't', team, 's', season,
                              'd', side, 'v', volume)
      FROM scout_players
     WHERE length(trim(COALESCE(p_query, ''))) >= 2
       AND (p_side IS NULL OR p_side = '' OR side = p_side)
       AND strpos(lower(COALESCE(name, '')), lower(trim(p_query))) > 0
     ORDER BY COALESCE(volume, 0) DESC, ord
     LIMIT p_limit
$$;

-- Full rows for "player_id|team|season|side" keys, in the order given.
-- The side is part of the key: without it a player listed as both hitter and
-- pitcher for one team-season (348 such keys, mostly pitchers who batted)
-- resolved to whichever row flattened later, so picking the hitter could
-- open the pitcher. A legacy three-part key still resolves, as before.
CREATE OR REPLACE FUNCTION scout_rows(p_keys TEXT[])
RETURNS SETOF JSON LANGUAGE sql STABLE SECURITY INVOKER AS $$
    SELECT DISTINCT ON (k.pos) scout_row(p)
      FROM unnest(p_keys) WITH ORDINALITY k(key, pos)
      JOIN scout_players p
        ON k.key IN (p.player_id::text || '|' || p.team || '|' || p.season::text || '|' || p.side,
                     p.player_id::text || '|' || p.team || '|' || p.season::text)
     ORDER BY k.pos, ((k.key LIKE '%|%|%|%') AND p.side = split_part(k.key, '|', 4)) DESC,
              p.ord DESC
$$;

-- Nearest player-seasons by their measurements. Mirrors charts.player_comps:
-- cosine similarity over the precomputed comp vectors, same side, volume
-- floor, optional required tags (applied to candidates only).
-- The target's whole player-season is excluded, not just its team row: a
-- traded player's other stint carries the same season-wide tags and came
-- back as a near-perfect comp of himself. Candidates are one per
-- player-season too — a traded comp's stints would otherwise fill two slots
-- with one season. Other seasons of the same player stay (the face check).
CREATE OR REPLACE FUNCTION scout_comps(
    p_key        TEXT,
    p_limit      INT     DEFAULT 8,
    p_min_vol    INT     DEFAULT 300,
    p_require    TEXT[]  DEFAULT NULL,
    p_same_season BOOLEAN DEFAULT false
)
RETURNS SETOF JSON LANGUAGE sql STABLE SECURITY INVOKER AS $$
WITH tgt AS (
    SELECT * FROM scout_players
     WHERE p_key IN (player_id::text || '|' || team || '|' || season::text || '|' || side,
                     player_id::text || '|' || team || '|' || season::text)
     ORDER BY ((p_key LIKE '%|%|%|%') AND side = split_part(p_key, '|', 4)) DESC, ord DESC
     LIMIT 1
),
cand AS (
    -- `p AS rw` keeps the whole row as a scout_players value for scout_row;
    -- `p.*` plus a sim column would no longer be one.
    SELECT p AS rw, p.ord, p.player_id, p.season,
        (SELECT COALESCE(sum((t.vec ->> k)::DOUBLE PRECISION * (p.vec ->> k)::DOUBLE PRECISION), 0)
           FROM jsonb_object_keys(t.vec) k WHERE p.vec ? k)
        / NULLIF(t.vec_norm * p.vec_norm, 0) AS sim
      FROM scout_players p, tgt t
     WHERE p.side = t.side
       AND NOT (p.player_id = t.player_id AND p.season = t.season)
       AND COALESCE(p.volume, 0) >= COALESCE(p_min_vol, 0)
       AND (COALESCE(cardinality(p_require), 0) = 0 OR p.tags @> p_require)
       AND (NOT p_same_season OR p.season = t.season)
       AND t.vec <> '{}'::jsonb
),
best AS (
    -- One row per player-season: its most similar stint.
    SELECT DISTINCT ON (c.player_id, c.season) c.rw, c.ord, c.sim
      FROM cand c
     WHERE c.sim > 0
     ORDER BY c.player_id, c.season, c.sim DESC, c.ord
)
SELECT scout_row(b.rw, b.sim)
  FROM best b
 ORDER BY b.sim DESC, b.ord
 LIMIT p_limit
$$;

GRANT EXECUTE ON FUNCTION scout_row(scout_players, DOUBLE PRECISION), scout_strength(JSONB),
    scout_search(TEXT[], TEXT[], TEXT, BOOLEAN, INT, INT, INT, BOOLEAN, TEXT[], TEXT, INT, INT),
    scout_name_search(TEXT, TEXT, INT), scout_rows(TEXT[]),
    scout_comps(TEXT, INT, INT, TEXT[], BOOLEAN)
    TO anon, authenticated;
