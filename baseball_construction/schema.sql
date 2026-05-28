-- Baseball Analytics — Supabase schema
-- Run this in: Supabase Dashboard → SQL Editor → New query
-- If tables already exist, drop them first:
--   DROP TABLE IF EXISTS player_batting, player_pitching, team_philosophy, park_factors;

-- ── Player batting aggregates ──────────────────────────────────────────────
-- One row per player per season (aggregate across all teams they played for).
-- Team assignment for portrait filtering comes from Statcast, not this table.
CREATE TABLE IF NOT EXISTS player_batting (
    key_mlbam       BIGINT,
    season          INT,
    name            TEXT,
    age             INT,
    pa              INT,
    avg             REAL,
    obp             REAL,
    slg             REAL,
    iso             REAL,
    k_rate          REAL,
    bb_rate         REAL,
    sb              INT,
    cs              INT,
    xwoba           REAL,
    xba             REAL,
    hard_hit_pct    REAL,
    barrel_pct      REAL,
    whiff_pct       REAL,
    chase_pct       REAL,
    sprint_speed    REAL,
    exit_velocity   REAL,
    join_source     TEXT,
    -- Statcast pitch aggregates + BRef WAR
    gb_pct          REAL,
    fps_pct         REAL,
    pitches_per_pa  REAL,
    war             REAL,
    zone_swing_pct  REAL,
    contact_pct     REAL,
    PRIMARY KEY (key_mlbam, season)
);

-- ── Player pitching aggregates ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_pitching (
    key_mlbam       BIGINT,
    season          INT,
    name            TEXT,
    age             INT,
    gs              INT,
    bf              INT,
    k_rate          REAL,
    bb_rate         REAL,
    xwoba_allowed   REAL,
    hard_hit_pct    REAL,
    barrel_pct      REAL,
    whiff_pct       REAL,
    fb_velocity     REAL,
    join_source     TEXT,
    -- Statcast pitch aggregates + BRef WAR/FIP
    gb_pct           REAL,
    zone_pct         REAL,
    csw_pct          REAL,
    pitches_per_bf   REAL,
    war              REAL,
    fip              REAL,
    avg_spin_rate    REAL,
    arsenal_diversity REAL,
    PRIMARY KEY (key_mlbam, season)
);

-- ── Pre-computed team philosophy scores ────────────────────────────────────
CREATE TABLE IF NOT EXISTS team_philosophy (
    team          TEXT,
    season        INT,
    a1_score      REAL,  a1_coverage  REAL,
    a2_score      REAL,  a2_coverage  REAL,
    a3_score      REAL,  a3_coverage  REAL,
    a4_score      REAL,  a4_coverage  REAL,
    b1_score      REAL,  b1_coverage  REAL,
    b2_score      REAL,  b2_coverage  REAL,
    b3_score      REAL,  b3_coverage  REAL,
    b4_score      REAL,  b4_coverage  REAL,
    c1_score      REAL,  c1_coverage  REAL,
    c2_score      REAL,  c2_coverage  REAL,
    c3_score      REAL,  c3_coverage  REAL,
    c4_score      REAL,  c4_coverage  REAL,
    data_coverage REAL,
    PRIMARY KEY (team, season)
);

-- ── Park factors ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS park_factors (
    team              TEXT,
    season            INT,
    pitcher_friendly  REAL,
    PRIMARY KEY (team, season)
);
