-- Apex Coach AI — Ball Tracking (Phase 2) schema
-- NOT YET APPLIED to the live project — review before running.
-- Entirely new tables, isolated from athletes/sessions — no foreign keys
-- into the existing biomechanics schema, so this can be dropped or
-- redesigned freely without touching any live coaching data.

create extension if not exists "pgcrypto";

-- One row per test video run through the classical/ML detector.
-- raw_candidates stores the per-frame detector output as-is for later
-- offline analysis — this is detector OUTPUT, not verified ground truth.
create table if not exists ball_tracking_runs (
    id uuid primary key default gen_random_uuid(),
    source_video_filename text not null,
    camera_setup_label text,             -- free text, e.g. "behind bowler, stumps in frame"
    detector_name text not null,         -- e.g. "classical_mog2_v1"
    fps numeric,
    frame_width integer,
    frame_height integer,
    total_frames integer,
    frames_with_candidates integer,
    raw_candidates jsonb,                -- {frame_idx: [{x_px,y_px,radius_px,...}, ...]}
    created_at timestamptz not null default now()
);

-- Human-confirmed ground truth: a coach/reviewer marks the TRUE ball
-- position at specific frames of a specific video. This is the actual
-- training/validation data — every label recorded here is a real,
-- verified fact, never inferred or auto-filled.
create table if not exists ball_tracking_labels (
    id uuid primary key default gen_random_uuid(),
    source_video_filename text not null,
    frame_index integer not null,
    ball_x_px numeric,                   -- null if ball not visible/identifiable this frame
    ball_y_px numeric,
    labeled_by text,                     -- who confirmed this label
    notes text,
    created_at timestamptz not null default now(),
    unique (source_video_filename, frame_index)
);

create index if not exists idx_ball_tracking_runs_video on ball_tracking_runs(source_video_filename);
create index if not exists idx_ball_tracking_labels_video on ball_tracking_labels(source_video_filename);
