-- Apex Coach AI — Phase 2 athlete history schema
-- Run this once in your Supabase project's SQL editor.

create extension if not exists "pgcrypto";

create table if not exists athletes (
    id uuid primary key default gen_random_uuid(),
    name text not null unique,
    created_at timestamptz not null default now()
);

create table if not exists sessions (
    id uuid primary key default gen_random_uuid(),
    athlete_id uuid not null references athletes(id) on delete cascade,
    session_date timestamptz not null default now(),
    video_filename text,
    camera_mode text,
    fps numeric,

    -- biomechanical_metrics payload from orchestrator, stored as-is
    metrics jsonb not null,

    -- speed_estimation payload (phase durations + optional arm speed)
    phase_durations jsonb,
    release_arm_speed_kmh numeric,       -- null if not calibrated for that session
    speed_status text,                   -- "success" | "not_calibrated" | "error"

    created_at timestamptz not null default now()
);

create index if not exists idx_sessions_athlete_id on sessions(athlete_id);
create index if not exists idx_sessions_date on sessions(session_date desc);
