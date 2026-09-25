-- ball_tracking/add_pitch_calibration.sql
--
-- Adds per-clip pitch (stump-line) calibration storage to
-- ball_tracking_runs, so ball_tracking/label_tool.py can capture the
-- same 4 ground-level stump corners pitch_calibration.py's
-- build_ground_homography() already knows how to turn into a real
-- ground-plane mapping (near/far stumps, 22.12m apart) -- previously
-- this had to be measured by hand with a grid-overlay screenshot,
-- separately from the normal per-clip labeling workflow, for every
-- clip that needed it. Coach can now click the 4 stump corners once
-- per clip in the same tool used for ball positions.
--
-- Nullable, no default: a clip with no calibration recorded yet is
-- simply NULL here, never a fabricated set of coordinates.
--
-- Run this once in Supabase's SQL editor (same as every other add_*.sql
-- file in this repo -- not applied automatically by any code).

alter table ball_tracking_runs
    add column if not exists pitch_calibration jsonb;
