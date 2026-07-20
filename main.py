import cv2
import pandas as pd
import numpy as np
import os
import urllib.request

# ============================================================
# REVERTED to a simple baseline after 9 commits of identity-tracking
# heuristics (multi-person warm-up/lock, ankle-visibility gating, body-
# proportion checks, movement-buffered lock-on, multi-candidate position+
# appearance matching) repeatedly failed on real footage — each fix solved
# one edge case while introducing or leaving another. Confirmed against
# user testing: the pre-heuristics version (this one) tracked reliably.
#
# KEPT from that whole effort: MediaPipe's VIDEO running mode with
# detect_for_video() + strictly increasing timestamps. This is a real,
# verified fix (confirmed against MediaPipe's actual API) — the earlier
# baseline ran in IMAGE mode, analyzing every frame independently with
# zero temporal continuity, which is a genuine bug, not a heuristic guess.
# VIDEO mode lets MediaPipe's own internal tracker do the continuity work,
# which is a fundamentally better signal than any of the custom heuristics
# attempted afterward.
#
# ALSO KEPT: brief interpolation across short gaps (<=5 frames) for
# genuine momentary tracking dropout (net occlusion, motion blur) — this
# is separate from and was not the cause of the identity-switching bugs.
#
# If identity confusion (skeleton switching to a different real person)
# resurfaces, the correct next step is NOT another automatic heuristic —
# repeated attempts at that have not held up. The credible next step is a
# one-time manual seed (coach clicks the bowler in a reference frame),
# discussed and explicitly deferred for now.
# ============================================================

LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "LEFT_MOUTH_OUTER", "RIGHT_MOUTH_OUTER", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


# Torso landmarks (nose, both shoulders, both hips) used to match a
# candidate person across frames — stable and central regardless of how
# the arms/legs are swinging, unlike e.g. the wrists.
_TORSO_INDICES = [0, 11, 12, 23, 24]


def _centroid_xy(landmarks_list):
    pts = [(landmarks_list[i].x, landmarks_list[i].y) for i in _TORSO_INDICES]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def extract_video_landmarks(video_path: str, output_csv_path: str,
                             seed_point: tuple = None,
                             seed_frame_index: int = 0) -> dict:
    """
    Headless Perception Layer. Processes every frame sequentially without
    destructive cropping to ensure zero data loss.

    seed_point: (x_px, y_px) pixel coordinates a coach clicked directly on
    the bowler in a reference frame (seed_frame_index), from the SAME
    video. When given, MediaPipe detects multiple candidate people per
    frame instead of just one, and this function tracks whichever
    candidate stays closest, frame to frame, to the person last
    confirmed — walking forward and backward out from the seed frame,
    anchored by the coach's explicit click. This is deliberately NOT
    another "guess who's the bowler" heuristic (see the comment block
    below on why those failed repeatedly) — the identity is given
    explicitly by a human; the only thing this function decides is
    "which detected person, this frame, is closest to where they were a
    moment ago."

    When seed_point is None (default), behaves exactly as before:
    single-person detection, the one candidate MediaPipe returns is used
    every frame — zero behavior change for any existing caller.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error_message": f"Input video file not found: {video_path}"}

    model_dir = "models"
    model_path = os.path.join(model_dir, "pose_landmarker_full.task")
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        try:
            urllib.request.urlretrieve(model_url, model_path)
        except Exception as e:
            return {"status": "error", "error_message": f"Failed to download model file: {str(e)}"}

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        return {"status": "error", "error_message": "MediaPipe Tasks API framework binding is missing."}

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    cap.release()

    columns = ["frame"]
    for name in LANDMARK_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    def run_detection_pass(num_poses):
        base_options_local = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options_local,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
            num_poses=num_poses,
            # LOWERED from 0.5: a bowler still distant/small early in the
            # run-up often doesn't clear a 0.5 confidence threshold, so the
            # skeleton doesn't appear until he's closer/larger in frame later
            # in the clip. This is a detection-confidence issue, NOT an
            # identity-switching issue (confirmed: no other person in frame).
            # Lower threshold trades a little more sensitivity to background
            # false positives for earlier detection of a genuine, distant
            # bowler — acceptable here since there's no second person to be
            # confused with.
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.4
        )
        landmarker_local = vision.PoseLandmarker.create_from_options(options)
        cap_local = cv2.VideoCapture(video_path)
        ms_per_frame_local = 1000.0 / fps
        candidates = []
        idx = 0
        last_ts = -1
        while True:
            success, frame = cap_local.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts = int(round(idx * ms_per_frame_local))
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            detection_result = landmarker_local.detect_for_video(mp_image_frame, ts)
            candidates.append(list(detection_result.pose_landmarks) if detection_result.pose_landmarks else [])
            idx += 1
        cap_local.release()
        landmarker_local.close()
        return candidates

    # ALWAYS run the fast, reliable single-person pass — multi-pose
    # detection (needed to disambiguate between several people) measurably
    # degrades MediaPipe's own per-frame reliability, verified on real
    # footage: the exact same person, at the exact same critical moment,
    # went from fully tracked (single-pose) to completely missing
    # (multi-pose) for several frames. When a seed is given, this result
    # is used as a preferred data SOURCE, not authoritative on its own —
    # see the merge logic below for why.
    single_pass_candidates = run_detection_pass(1)
    total_frames = len(single_pass_candidates)
    single_pose_chosen = [cands[0] if cands else None for cands in single_pass_candidates]

    if seed_point is None:
        chosen_landmarks = single_pose_chosen
    else:
        # A seed was given, so ALWAYS compute the seeded multi-pose track
        # too — not just when has_implausible_jump flags a risk. Verified
        # on real footage that jump-detection has a blind spot: a
        # STATIONARY false-positive (MediaPipe matching a person-like
        # pattern in trees/background clutter, not a real person at all)
        # never "jumps" anywhere, so it slipped through undetected as if
        # it were the real, correctly-tracked bowler for dozens of frames.
        #
        # The seeded track is then used as a VALIDATOR, not the direct
        # data source: for each frame, single-pose's landmarks are used
        # (better per-frame quality, confirmed elsewhere this session)
        # ONLY where they agree with where the seeded track says the
        # tracked person actually is. Where they disagree — single-pose
        # has wandered onto something else — the seeded value is used
        # instead (or honest NaN if the seeded track has no confident
        # match there either). This gets single-pose's better quality on
        # the frames it can be trusted, without ever silently trusting a
        # phantom detection just because it happened to be smooth.
        frame_candidates = run_detection_pass(3)
        # A frame where only ONE candidate is detected still needs a
        # plausibility check, not just an automatic accept — otherwise a
        # different, more-consistently-detected bystander (e.g. a coach
        # standing close to the camera) becomes "the closest available
        # candidate" by default whenever the real tracked person isn't
        # detected that frame, and tracking silently snaps onto them.
        # Verified on real footage: without this cap, seeded tracking
        # still drifted onto a coach who was the sole detected person in
        # most frames while the actual (smaller, more distant) bowler was
        # only sporadically detected. The cap scales with how long it's
        # been since a real match, since the true person could genuinely
        # have moved further by the time detection resumes after a gap.
        MAX_DIST_PER_SECOND = 0.6

        def pick_closest(cands, anchor_xy, max_dist):
            best, best_dist = None, None
            for cand in cands:
                cx, cy = _centroid_xy(cand)
                dist = ((cx - anchor_xy[0]) ** 2 + (cy - anchor_xy[1]) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best, best_dist = cand, dist
            if best is None or best_dist > max_dist:
                return None
            return best

        seed_idx = max(0, min(seed_frame_index, total_frames - 1))
        seed_xy = (seed_point[0] / frame_width, seed_point[1] / frame_height)
        seeded_chosen = [None] * total_frames

        # The seed match needs its own, more generous tolerance — it's
        # answering "which PERSON is this" (a click anywhere on their
        # body vs. a computed torso centroid, which can genuinely differ
        # by a lot: a head-click vs. a centroid averaging head+shoulders+
        # hips), not "did this person move plausibly since last frame"
        # like the tighter per-frame tolerance below. Still much smaller
        # than the distance between two DIFFERENT people in a real scene.
        SEED_MATCH_TOLERANCE = 0.2
        chosen = pick_closest(frame_candidates[seed_idx], seed_xy, SEED_MATCH_TOLERANCE)
        seeded_chosen[seed_idx] = chosen
        seed_anchor = _centroid_xy(chosen) if chosen is not None else seed_xy

        # Tolerance grows with elapsed gap (a real person could genuinely
        # move further before detection resumes), but is capped rather
        # than left to grow indefinitely.
        MAX_DIST_CAP = 0.25

        # BUG FIX: the comment above (and the one that used to be here)
        # described "after roughly half a second with no confirmed match,
        # a re-match is refused entirely" — but the code only ever capped
        # the DISTANCE tolerance, never the GAP LENGTH itself, so it kept
        # trying to reacquire no matter how long the gap ran. Verified on
        # real footage: a 116-frame (~3.9s) gap with zero confirmed
        # matches, then reacquisition locked onto a coach standing near
        # the pitch instead of the actual bowler — the coach happened to
        # be a confidently-detected, roughly-stationary person within
        # reach once the capped tolerance had been sitting at its ceiling
        # for that long. A real bowler could have moved a long way in
        # 3.9 seconds; that's exactly why a stale anchor position is no
        # longer a trustworthy reference for "who's nearby" after this
        # long — the failure mode isn't a wrong DISTANCE calculation, it's
        # that distance-from-a-3.9-second-old-position stops meaning
        # anything. Past this ceiling, stop attempting to reacquire by
        # position at all; those frames — and everything after, until a
        # fresh seed — stay honestly untracked (N/A downstream) rather
        # than confidently locking onto whoever's closest by chance.
        MAX_GAP_FRAMES = max(3, int(round(fps * 0.5)))

        anchor = seed_anchor
        frames_since_confirmed = 1
        for i in range(seed_idx + 1, total_frames):
            if frames_since_confirmed > MAX_GAP_FRAMES:
                seeded_chosen[i] = None
                frames_since_confirmed += 1
                continue
            max_dist = min((MAX_DIST_PER_SECOND / fps) * frames_since_confirmed, MAX_DIST_CAP)
            chosen = pick_closest(frame_candidates[i], anchor, max_dist) if frame_candidates[i] else None
            seeded_chosen[i] = chosen
            if chosen is not None:
                anchor = _centroid_xy(chosen)
                frames_since_confirmed = 1
            else:
                frames_since_confirmed += 1

        anchor = seed_anchor
        frames_since_confirmed = 1
        for i in range(seed_idx - 1, -1, -1):
            if frames_since_confirmed > MAX_GAP_FRAMES:
                seeded_chosen[i] = None
                frames_since_confirmed += 1
                continue
            max_dist = min((MAX_DIST_PER_SECOND / fps) * frames_since_confirmed, MAX_DIST_CAP)
            chosen = pick_closest(frame_candidates[i], anchor, max_dist) if frame_candidates[i] else None
            seeded_chosen[i] = chosen
            if chosen is not None:
                anchor = _centroid_xy(chosen)
                frames_since_confirmed = 1
            else:
                frames_since_confirmed += 1

        # MERGE: prefer single-pose's landmarks (better per-frame
        # completeness/quality) on any frame where they're validated by
        # the seeded track's independently-verified position; otherwise
        # trust the seeded track instead (or NaN if neither can validate
        # a position there). The seeded track doesn't need data at the
        # EXACT same frame to validate single-pose — a brief gap in the
        # (less reliable per-frame) seeded track shouldn't force
        # discarding perfectly good single-pose data next to it, so the
        # NEAREST seeded reference within a short time window is used
        # instead. Verified this matters: requiring an exact-frame match
        # was discarding genuinely correct single-pose tracking right at
        # a fast, blurry release moment where the seeded track happened
        # to have a brief dropout of its own.
        AGREEMENT_TOLERANCE = 0.15
        sd_points = [(i, _centroid_xy(seeded_chosen[i])) for i in range(total_frames) if seeded_chosen[i] is not None]
        MAX_VALIDATION_GAP = max(3, int(round(fps * 0.5)))

        def nearest_sd_centroid(i):
            best, best_gap = None, None
            for j, c in sd_points:
                gap = abs(j - i)
                if gap <= MAX_VALIDATION_GAP and (best_gap is None or gap < best_gap):
                    best, best_gap = c, gap
            return best

        chosen_landmarks = [None] * total_frames
        for i in range(total_frames):
            sp = single_pose_chosen[i]
            sd = seeded_chosen[i]
            reference = _centroid_xy(sd) if sd is not None else nearest_sd_centroid(i)
            if sp is not None and reference is not None:
                spx, spy = _centroid_xy(sp)
                if ((spx - reference[0]) ** 2 + (spy - reference[1]) ** 2) ** 0.5 <= AGREEMENT_TOLERANCE:
                    chosen_landmarks[i] = sp
                    continue
            chosen_landmarks[i] = sd

    dataset_rows = []
    for i in range(total_frames):
        row = [i]
        landmarks_list = chosen_landmarks[i]
        if landmarks_list:
            for landmark in landmarks_list:
                # MediaPipe scores its own confidence in each point
                # (visibility). Previously this was discarded — every
                # point was plotted regardless of confidence, including a
                # stray/uncertain detection before the bowler has properly
                # entered a tight frame. Treating a low-confidence point as
                # missing (same as full occlusion) instead of plotting it
                # lets the existing gap-fill interpolation and outlier
                # filter below bridge across it, instead of the skeleton
                # visibly snapping from an uncertain position once a
                # confident detection appears.
                visibility = landmark.visibility if landmark.visibility is not None else 1.0
                if visibility >= 0.5:
                    row.extend([landmark.x, landmark.y, landmark.z])
                else:
                    row.extend([np.nan, np.nan, np.nan])
        else:
            row.extend([np.nan] * (33 * 3))
        dataset_rows.append(row)

    output_df = pd.DataFrame(dataset_rows, columns=columns)
    landmark_cols = [c for c in output_df.columns if c != "frame"]

    # OUTLIER REJECTION (Hampel filter): a plain moving average (below)
    # blends a single bad frame into its neighbors instead of removing it,
    # which is why the skeleton still looked "loose"/spiky during fast,
    # motion-blurred phases even after widening the averaging window. This
    # flags any frame where a landmark jumps further from its own local
    # neighborhood than is statistically normal for THAT landmark's recent
    # motion, and treats it as a bad detection (filled in like a genuine
    # occlusion gap) before smoothing. It compares each point to the local
    # median absolute deviation rather than a fixed distance/speed number,
    # so it self-adjusts to each landmark's own motion instead of being
    # tuned to one video's camera distance or delivery speed. This is
    # NOT part of "who to track" logic — it only cleans the already-selected
    # trajectory, so it carries none of the identity-switching risk.
    # A landmark's x/y/z describe ONE physical point and must be thrown
    # out together. Checking each coordinate independently (as an earlier
    # version of this did) could flag only x (or only y) as an outlier
    # while leaving the other coordinate untouched — the interpolated x
    # then no longer corresponds to the real y, producing a landmark that
    # snaps to a spatially incoherent position. Verified on real footage:
    # this produced a visible disconnected limb line jumping away from
    # the body. Now flags outliers per coordinate first, then unions the
    # flags across x/y/z before nulling, so a landmark is only ever kept
    # or dropped as a whole point.
    HAMPEL_WINDOW = 5
    HAMPEL_N_SIGMAS = 3
    per_col_outlier = {}
    for col in landmark_cols:
        series = output_df[col]
        rolling_median = series.rolling(window=HAMPEL_WINDOW, center=True, min_periods=1).median()
        abs_dev = (series - rolling_median).abs()
        mad = abs_dev.rolling(window=HAMPEL_WINDOW, center=True, min_periods=1).median()
        threshold = HAMPEL_N_SIGMAS * 1.4826 * mad
        per_col_outlier[col] = (mad > 0) & (abs_dev > threshold)

    landmark_bases = sorted({c.rsplit("_", 1)[0] for c in landmark_cols})
    for base in landmark_bases:
        coord_cols = [c for c in (f"{base}_x", f"{base}_y", f"{base}_z") if c in per_col_outlier]
        combined_outlier = per_col_outlier[coord_cols[0]]
        for c in coord_cols[1:]:
            combined_outlier = combined_outlier | per_col_outlier[c]
        for c in coord_cols:
            output_df.loc[combined_outlier, c] = np.nan

    # Brief gap-fill for genuine short occlusion (net, motion blur) and the
    # outliers just flagged above — NOT related to the identity-switching
    # bugs, kept separately since it was a real, narrow improvement on its
    # own. Limit is time-based (not a fixed frame count) so it means the
    # same ~0.1s across a 30fps and a 120fps video.
    #
    # Kept intentionally tight: verified on real footage that a limb
    # genuinely undetected for longer than this (e.g. an arm occluded by
    # the body during running) would otherwise get silently patched into
    # a fabricated, frozen position once it reappears — which then gets
    # drawn as if it were a real, current position (a stray disconnected
    # limb line), and can also fool event-detection into treating that
    # frozen value as a genuine peak. Anything longer than this now stays
    # real NaN, which the existing drawing/event-detection code already
    # skips gracefully instead of trusting a guess.
    # limit_area="inside" matters as much as the limit itself: without it,
    # pandas pads a one-sided-reachable gap from whichever edge IS within
    # reach even when the other side has no real data at all for a long
    # stretch — verified this is what was still producing a flat/frozen
    # value a few frames deep into a genuinely long tracking gap even
    # after tightening the limit. "inside" only fills a gap that has real
    # data bracketing it on BOTH sides.
    gap_fill_limit = max(1, int(round(fps * 0.1)))
    output_df[landmark_cols] = output_df[landmark_cols].interpolate(
        method="linear", limit=gap_fill_limit, limit_direction="both", limit_area="inside"
    )

    # LIGHT SMOOTHING: was present in an earlier working version, removed
    # by accident when the identity-tracking heuristics (a completely
    # separate, unrelated feature) were reverted from this same file.
    # This is intentionally NOT part of "who to track" logic — it only
    # smooths the already-selected trajectory, so it carries none of the
    # identity-switching risk from last night's heuristics. Now runs on
    # outlier-cleaned data (above), so it's smoothing real motion instead
    # of also having to absorb occasional bad-frame spikes.
    output_df[landmark_cols] = output_df[landmark_cols].rolling(
        window=5, center=True, min_periods=1
    ).mean()

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)

    return {
        "status": "success",
        "total_frames_processed": total_frames,
        "fps": fps,
        "output_file": output_csv_path
    }


if __name__ == "__main__":
    print("=== STARTING KINEMATIC EXTRACTION STATE ===")
    extraction_state = extract_video_landmarks("input/input_video.mp4", "output/landmarks.csv")
    print(extraction_state)
