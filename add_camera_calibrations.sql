-- Apex Coach AI — persisted camera calibrations
--
-- calibration.py's own docstring says calibration is meant to be done
-- ONCE per fixed camera position and reused for every video from that
-- spot — but until now it only ever lived in st.session_state, which is
-- wiped the moment the browser tab closes or the server sleeps. A coach
-- was redoing the same click-two-points setup far more often than the
-- design ever intended. This table lets a coach save a calibration under
-- a name (e.g. "Home nets — evening camera spot") and reload it instantly
-- next time, instead of re-uploading a reference video and re-clicking.
--
-- unique(coach_user_id, setup_label) means saving again under the same
-- label updates it in place (re-calibrating a setup that moved) rather
-- than piling up duplicates.

create table if not exists camera_calibrations (
    id uuid primary key default gen_random_uuid(),
    coach_user_id uuid not null,
    setup_label text not null,
    calibration jsonb not null,
    frame_width_px integer not null,
    created_at timestamptz not null default now(),
    unique (coach_user_id, setup_label)
);

create index if not exists idx_camera_calibrations_coach_user_id on camera_calibrations(coach_user_id);
