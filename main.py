import collections
import cv2
import pandas as pd
import numpy as np
import os
import urllib.request

import monitoring

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

# Shoulders+hips ONLY (no nose) — used for the APPEARANCE crop, a
# deliberately different, tighter region than _TORSO_INDICES above (which
# stays as-is for position/centroid matching — see _centroid_xy). See
# _compute_appearance_histogram's docstring for why.
_CLOTHING_BBOX_INDICES = [11, 12, 23, 24]

# Module-level (2026-09-13, promoted out of _walk_from_seed) so
# streamlit_app.py's click-time "did this seed click actually find a real
# person" check (see render_bowler_seed_ui) uses the EXACT same radius
# the real walk matches against — a UI-side copy of this number could
# silently drift from the real one and start giving false reassurance.
SEED_MATCH_TOLERANCE = 0.08


def _centroid_xy(landmarks_list):
    pts = [(landmarks_list[i].x, landmarks_list[i].y) for i in _TORSO_INDICES]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _bbox_from_landmarks(landmarks, width, height, indices=_TORSO_INDICES, margin=0.04, shrink=0.0):
    """
    shrink (2026-09-13, real fix — see _compute_appearance_histogram's
    docstring): pulls the box in toward its own center by this fraction
    on each side AFTER margin is applied — 0.3 keeps the central 70%.
    A rectangular box drawn around a handful of sparse landmarks always
    has background in its corners (a person's real silhouette isn't a
    rectangle), and that background gets worse the wider the margin —
    shrinking back toward center trims exactly that contamination without
    needing a per-pixel body mask.
    """
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    min_x, max_x = max(0.0, min(xs) - margin), min(1.0, max(xs) + margin)
    min_y, max_y = max(0.0, min(ys) - margin), min(1.0, max(ys) + margin)
    if shrink > 0.0:
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        half_w = (max_x - min_x) / 2 * (1.0 - shrink)
        half_h = (max_y - min_y) / 2 * (1.0 - shrink)
        min_x, max_x = cx - half_w, cx + half_w
        min_y, max_y = cy - half_h, cy + half_h
    return int(min_x * width), int(min_y * height), int(max_x * width), int(max_y * height)


def _compute_appearance_histogram(frame_bgr, landmarks):
    """
    HSV color-histogram "appearance fingerprint" of the torso region
    around a candidate — clothing/skin-tone signature, tolerant of the
    exact pose/frame lighting. Verified directly on real footage
    (2026-08-10): mean similarity to a seed crop was 0.71 for the true
    bowler's own later frames vs. 0.25 for a different real person
    (batsman) visible earlier in the same clip — a real, usable signal,
    not assumed.

    An earlier attempt at this (commit b8ff3cd, reverted the next day)
    used cv2.HISTCMP_CORRELATION, which does not exist in this OpenCV
    version (it's HISTCMP_CORREL) — that bug was silently swallowed by
    a broad try/except, so the appearance signal never actually ran and
    could not have been responsible for that revert. No broad except
    here for exactly that reason — a real failure should surface in
    testing, not vanish into a default score.

    CROP TIGHTENED (2026-09-13, real bug — see project memory on the
    bowler-identity seed fix): confirmed on a real clip that this
    histogram wasn't discriminative enough between two actual different
    people post-release — cross-person similarity averaged 0.79, barely
    below either person's own 0.93-0.96 self-similarity, letting the
    identity walk drift onto the wrong one once a real gap grew the
    position radius wide enough to reach them. Traced to background
    contamination: the OLD box used nose+shoulders+hips with a 0.04
    margin — the nose pulls the top edge up into the neck/lower-face
    (skin tone, not clothing, and often similar across different real
    people), and the margin plus the box's own rectangular corners
    (a person's silhouette isn't a rectangle) both add background pixels
    that dilute the signal. Measured directly (not guessed) against the
    same real clip: dropping the nose (shoulders+hips only), removing the
    margin, and shrinking the box to its central 70% took cross-person
    similarity from 0.79 down to 0.005, while each person's own self-
    similarity stayed a real, usable 0.38-0.42 — a clean, verified
    separation instead of a marginal one. Position matching (_centroid_xy,
    _TORSO_INDICES) is deliberately UNCHANGED — the nose is still a good,
    stable point for tracking WHERE someone is; it's specifically a poor
    ingredient for WHAT they look like.
    """
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _bbox_from_landmarks(
        landmarks, width, height, indices=_CLOTHING_BBOX_INDICES, margin=0.0, shrink=0.3)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _hist_similarity(hist_a, hist_b) -> float:
    if hist_a is None or hist_b is None:
        return 0.0
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _seed_appearance_majority_ok(seed_hists: list) -> list:
    """
    REAL BUG FOUND (2026-09-12, coach-reported and confirmed on real
    footage): the seed-frame match itself (see _walk_from_seed below) had
    ZERO appearance verification — the very first click-to-person match
    is pure "nearest detected candidate within SEED_MATCH_TOLERANCE," no
    different from picking the wrong person entirely if they happen to be
    the closer of two candidates to an imprecise click. Confirmed
    directly: a coach's seed click near the bowler still matched the
    batter instead, and the walk then built a confident, internally-
    consistent appearance profile of the WRONG person for that whole zone.

    This is the credible second layer of defense the coach's own workflow
    already provides for free: the SAME bowler is confirmed at multiple
    (typically 4) separate seed points. If most of those seeds' matched
    appearances agree with each other, any one seed whose match looks
    nothing like that agreement is very likely a wrong-person match, not
    a coincidentally different-looking moment of the same real person
    (lighting/pose varies frame to frame, but not usually enough to erase
    all resemblance to your OWN other confirmed sightings of yourself).

    Deliberately conservative — matches this file's own header warning
    that heuristics here have repeatedly looked right and then broken a
    different real clip: with fewer than 3 real (non-None) histograms
    there's nothing meaningful to vote with, so this returns all-True
    (no override) rather than guess from 1-2 data points. The majority
    baseline is computed from the TOP-scoring seeds only (never lowered
    by including the outlier's own poor score), so one bad seed can never
    drag the bar down far enough to excuse itself.

    Returns a list of booleans, same length/order as seed_hists — True
    for a seed whose match should be trusted, False for a clear
    appearance outlier relative to the others.
    """
    real_idx = [i for i, h in enumerate(seed_hists) if h is not None]
    if len(real_idx) < 3:
        return [True] * len(seed_hists)

    pairwise = {}
    for a in range(len(real_idx)):
        for b in range(a + 1, len(real_idx)):
            i, j = real_idx[a], real_idx[b]
            pairwise[(i, j)] = _hist_similarity(seed_hists[i], seed_hists[j])

    avg_sim = {}
    for i in real_idx:
        others = [pairwise[(min(i, j), max(i, j))] for j in real_idx if j != i]
        avg_sim[i] = sum(others) / len(others)

    sorted_scores = sorted(avg_sim.values(), reverse=True)
    majority_count = max(2, (len(real_idx) * 2) // 3)  # e.g. 3 of 4, 2 of 3
    majority_baseline = sum(sorted_scores[:majority_count]) / majority_count

    # ABSOLUTE gap, not a ratio (BUG FOUND via this file's own test suite
    # before shipping: cv2.HISTCMP_CORREL can be negative for two
    # genuinely different appearances, e.g. -0.3 vs -0.6 — multiplying a
    # NEGATIVE baseline by a ratio makes the threshold LESS negative, the
    # opposite of "more lenient," which flagged every seed as an outlier
    # in a case where none of them actually stood out from the others).
    # An absolute gap in HISTCMP_CORREL's bounded [-1, 1] range behaves
    # the same regardless of sign. 0.25 is roughly half the real same-
    # person-vs-different-person gap this project already measured on
    # real footage (_compute_appearance_histogram's docstring: 0.71 for
    # the true bowler's own later frames vs. 0.25 for a different real
    # person, a ~0.46 real gap) — a deliberately conservative half of
    # that margin, not the full gap, so normal appearance noise between
    # a genuine seed's own frames doesn't get over-flagged.
    OUTLIER_GAP = 0.25
    ok = [True] * len(seed_hists)
    for i in real_idx:
        if majority_baseline - avg_sim[i] > OUTLIER_GAP:
            ok[i] = False
    return ok


def _walk_from_seed(seed_idx, seed_xy, frame_candidates, frame_hists, fps, lo_bound, hi_bound,
                     trust_seed_match: bool = True, prior_profile: list = None):
    """
    Anchors at seed_idx (matching seed_xy within SEED_MATCH_TOLERANCE),
    then walks forward to hi_bound and backward to lo_bound using
    distance-growth capped at MAX_DIST_CAP, with a hard MAX_GAP_FRAMES
    ceiling on how long a gap can run before giving up on that
    direction (see extract_video_landmarks for why this ceiling exists
    — a long-unconfirmed anchor position stops being a trustworthy
    reference for "who's nearby").

    APPEARANCE GATE: position alone is the known weak point here — the
    longer a gap runs, the more the acceptance radius grows (up to
    MAX_DIST_CAP), which is exactly when a different, merely-nearby
    person becomes acceptable on position alone. Once we have enough
    confirmed history to know what this specific person actually looks
    like (APPEARANCE_MIN_PROFILE frames), every candidate must also
    clear an appearance-similarity floor — calibrated to THIS clip's own
    footage (a fraction of how similar this person's own confirmed
    frames are to each other), not a fixed number tuned to one video,
    since lighting/distance/camera quality varies a lot across real
    coach-submitted clips. The floor is low right after a fresh match
    and escalates the longer the current gap runs, since that's exactly
    when position is least trustworthy. It is never fully skipped,
    including for fresh matches — an earlier version of this skipped the
    check whenever the gap since the last successful match was small,
    which real-footage testing caught as a real bug: a CHAIN of
    individually-tiny, individually-plausible position steps kept
    resetting that gap back to 1 on every step, so cumulative drift onto
    a completely different (tiny, spurious) detection never looked like
    "a long gap" to the old check at all.

    Only touches frames within [lo_bound, hi_bound] — this is what lets
    multiple seeds coexist: each one only walks within its own assigned
    zone of the clip (split at the midpoint to its neighboring seeds),
    so seeds never fight over which one "wins" a given frame.

    trust_seed_match (2026-09-12, see _seed_appearance_majority_ok above):
    False means skip the seed-frame match entirely and anchor on the raw
    click coordinate with no starting candidate/profile — used by
    extract_video_landmarks when this seed's own match was flagged as an
    appearance outlier against the coach's OTHER confirmed seeds for the
    same clip, so a likely-wrong match is discarded instead of confidently
    seeding a wrong appearance profile across this seed's whole zone.

    prior_profile (2026-09-12, real gap found tracing an actual coach-
    reported failure): when this seed's OWN exact-frame match fails
    (nobody detected close enough to the click — a real, correct outcome
    when the target is momentarily too small/distant/occluded to detect,
    not a bug by itself), the walk used to start with a completely EMPTY
    appearance profile, so pick_closest fails open on position alone
    (APPEARANCE_MIN_PROFILE frames' grace period) for the first few
    matches in this zone — exactly long enough for a wrong-but-nearby
    person to get "confirmed" 3 times and become the trusted profile
    before the real target is ever seen. Confirmed directly on a real
    clip: the bowler wasn't detected at all near an early seed click (too
    small/distant that instant), and the walk locked onto the batter
    instead for the whole zone, appearance-consistent with ONLY itself.
    prior_profile lets the caller pass in real appearance evidence from
    the coach's OTHER confirmed seeds (which matched something) as a
    warm start — the appearance gate is then live from the very first
    frame of this zone's walk, checked against what the target actually
    looks like elsewhere in the SAME clip, instead of blindly trusting
    whichever candidate happens to be nearest first.

    Returns ({frame_idx: chosen_landmarks_or_None} for every frame in
    [lo_bound, hi_bound], chosen_hist_at_the_seed_frame_or_None) — the
    histogram is returned so the caller can cross-check it against other
    seeds before committing to this walk (see extract_video_landmarks).
    """
    # TIGHTENED (2026-09-12, real bug — see _seed_appearance_majority_ok
    # above): 0.2 (20% of frame) was generous enough that an imprecise
    # click on the true target could still land closer to a completely
    # different nearby person than to the intended one, with NO
    # appearance check at all to catch it (the profile is empty at this
    # exact point). A tighter radius fails SAFE, not wrong: worst case,
    # a very imprecise click matches nobody and this zone falls back to
    # position-only tracking from the raw click (the existing, already-
    # tested "no seed candidate" path) — never a confident lock onto the
    # wrong person the way a loose radius could. (Now a module-level
    # constant — see its own definition above.)
    MAX_DIST_PER_SECOND = 0.6
    MAX_DIST_CAP = 0.25
    MAX_GAP_FRAMES = max(3, int(round(fps * 0.5)))
    APPEARANCE_PROFILE_LEN = 8
    APPEARANCE_MIN_PROFILE = 3

    def pick_closest(cands, hists, anchor_xy, max_dist, profile, gap_frames):
        in_range = []
        for cand, hist in zip(cands, hists):
            cx, cy = _centroid_xy(cand)
            dist = ((cx - anchor_xy[0]) ** 2 + (cy - anchor_xy[1]) ** 2) ** 0.5
            if dist <= max_dist:
                in_range.append((dist, cand, hist))
        if not in_range:
            return None, None
        in_range.sort(key=lambda t: t[0])

        if len(profile) < APPEARANCE_MIN_PROFILE:
            _, best_cand, best_hist = in_range[0]
            return best_cand, best_hist

        # BUG FOUND during real-footage validation (2026-08-10): gating
        # this on gap_frames <= APPEARANCE_GRACE_FRAMES let a CHAIN of
        # individually-tiny, individually-plausible position steps skip
        # the appearance check forever, because each successful match
        # resets gap_frames back to 1 — cumulative drift across many
        # small steps never looked like "a long gap" to this check even
        # once it had walked onto a completely different, tiny/spurious
        # detection. Confirmed directly: 3 frames right at the real
        # batsman/bowler boundary leaked this way, each with a near-zero
        # shoulder width (0.005-0.016 vs ~0.045 for the real bowler) that
        # never got appearance-checked because every step "just confirmed
        # a moment ago". Fix: the appearance floor is now always live
        # once the profile has enough data, not skipped for fresh
        # matches — kept LOW at gap_frames==1 (0.2x self-similarity) so
        # normal noisy-but-correct frames still pass, and escalates for
        # longer gaps exactly as before.
        profile_list = list(profile)
        pairwise = [
            _hist_similarity(profile_list[i], profile_list[j])
            for i in range(len(profile_list)) for j in range(i + 1, len(profile_list))
        ]
        self_similarity = sum(pairwise) / len(pairwise) if pairwise else 0.5
        gap_fraction = min(1.0, gap_frames / MAX_GAP_FRAMES)
        required_sim = self_similarity * (0.2 + 0.6 * gap_fraction)

        for dist, cand, hist in in_range:
            sim = max((_hist_similarity(hist, p) for p in profile_list), default=0.0)
            if sim >= required_sim:
                return cand, hist
        return None, None

    result = {}
    seed_cands = frame_candidates[seed_idx]
    seed_hists = frame_hists[seed_idx]
    chosen, chosen_hist = pick_closest(seed_cands, seed_hists, seed_xy, SEED_MATCH_TOLERANCE,
                                        collections.deque(), 0) if (trust_seed_match and seed_cands) else (None, None)
    result[seed_idx] = chosen
    seed_anchor = _centroid_xy(chosen) if chosen is not None else seed_xy

    def walk_direction(frame_range):
        anchor = seed_anchor
        frames_since_confirmed = 1
        profile = collections.deque(maxlen=APPEARANCE_PROFILE_LEN)
        if chosen_hist is not None:
            profile.append(chosen_hist)
        elif prior_profile:
            # See prior_profile's docstring above — a real appearance
            # reference from the coach's OTHER confirmed seeds, used only
            # when THIS seed's own exact-frame match found nothing to
            # seed the profile with directly. Real chosen_hist (above)
            # always takes priority when it exists.
            profile.extend(prior_profile[:APPEARANCE_PROFILE_LEN])
        for i in frame_range:
            if frames_since_confirmed > MAX_GAP_FRAMES:
                result[i] = None
                frames_since_confirmed += 1
                continue
            max_dist = min((MAX_DIST_PER_SECOND / fps) * frames_since_confirmed, MAX_DIST_CAP)
            picked, picked_hist = pick_closest(
                frame_candidates[i], frame_hists[i], anchor, max_dist, profile, frames_since_confirmed
            ) if frame_candidates[i] else (None, None)
            result[i] = picked
            if picked is not None:
                anchor = _centroid_xy(picked)
                frames_since_confirmed = 1
                if picked_hist is not None:
                    profile.append(picked_hist)
            else:
                frames_since_confirmed += 1

    walk_direction(range(seed_idx + 1, hi_bound + 1))
    walk_direction(range(seed_idx - 1, lo_bound - 1, -1))

    return result, chosen_hist


def smooth_without_resurrecting_gaps(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Rolling-mean smoothing that never fills in a position that was
    genuinely missing beforehand.

    BUG FOUND during a broader audit: a plain
    rolling(window, center=True, min_periods=1).mean() will happily
    average whatever non-NaN values fall in its window even when only
    1-2 of them are real — which silently RESURRECTS frames an earlier
    interpolate(..., limit_area="inside") step correctly left as genuine
    NaN (a real tracking gap longer than its fill limit, deliberately
    left unpatched so downstream code treats it as missing, not guessed).
    Verified directly: a 21-frame real gap came out of interpolate()
    correctly NaN from a couple frames in, but the rolling mean alone
    filled those frames with fabricated values derived from just 1-2
    real neighbors — extending the "fabricated data" zone ~2 frames
    deeper into every long gap than intended, with no error or flag
    anywhere downstream that would catch it.

    Smoothing must only ever refine data that's actually present, never
    manufacture data that isn't — this remembers which positions were
    NaN before smoothing and forces them back to NaN after, regardless
    of what the rolling mean computed for them.
    """
    pre_smoothing_nan_mask = df.isna()
    smoothed = df.rolling(window=window, center=True, min_periods=1).mean()
    return smoothed.where(~pre_smoothing_nan_mask)


def extract_video_landmarks(video_path: str, output_csv_path: str,
                             seed_point: tuple = None,
                             seed_frame_index: int = 0,
                             extra_seeds: list = None) -> dict:
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
        # BUG FIX (2026-09-13, robustness audit): urlretrieve used to write
        # straight to model_path — a network drop mid-transfer, or a
        # redirect to an HTML error page saved as if it were the model,
        # left a PARTIAL/CORRUPT file there with no validation. Every
        # subsequent run's os.path.exists(model_path) check above was then
        # true, so it never re-downloaded and crashed uncaught deeper in
        # this function (PoseLandmarker.create_from_options) on every
        # single call until someone manually deleted the file. Now
        # downloads to a temp path first, validates a minimum real size
        # (the genuine model is ~9.4MB; a truncated/error-page download is
        # nowhere close), and only moves it into place once confirmed —
        # cleaning up the temp file on any failure so the next run gets a
        # clean retry instead of a permanently poisoned cache.
        MIN_MODEL_SIZE_BYTES = 1_000_000
        tmp_model_path = model_path + ".part"
        try:
            urllib.request.urlretrieve(model_url, tmp_model_path)
            if not os.path.exists(tmp_model_path) or os.path.getsize(tmp_model_path) < MIN_MODEL_SIZE_BYTES:
                raise IOError(
                    f"Downloaded model file is too small to be genuine "
                    f"({os.path.getsize(tmp_model_path) if os.path.exists(tmp_model_path) else 0} bytes)."
                )
            os.replace(tmp_model_path, model_path)
        except Exception as e:
            monitoring.capture(e)
            if os.path.exists(tmp_model_path):
                try:
                    os.remove(tmp_model_path)
                except OSError:
                    pass
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

    def run_detection_pass(num_poses, compute_appearance=False):
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
        hists = []
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
            cands = list(detection_result.pose_landmarks) if detection_result.pose_landmarks else []
            candidates.append(cands)
            # Computed here (not in a later pass) so the appearance
            # fingerprint uses the exact same frame pixels the candidate
            # was detected from, at no extra video-decode cost — the
            # frame is already in memory for this iteration either way.
            if compute_appearance and cands:
                hists.append([_compute_appearance_histogram(frame, cand) for cand in cands])
            else:
                hists.append([None] * len(cands))
            idx += 1
        cap_local.release()
        landmarker_local.close()
        return candidates, hists

    # ALWAYS run the fast, reliable single-person pass — multi-pose
    # detection (needed to disambiguate between several people) measurably
    # degrades MediaPipe's own per-frame reliability, verified on real
    # footage: the exact same person, at the exact same critical moment,
    # went from fully tracked (single-pose) to completely missing
    # (multi-pose) for several frames. When a seed is given, this result
    # is used as a preferred data SOURCE, not authoritative on its own —
    # see the merge logic below for why.
    # BUG FIX (2026-09-13, robustness audit): the model file can exist and
    # pass the download-size check above yet still not be a genuinely
    # valid model (disk corruption, manual tampering, or any failure mode
    # the size check doesn't catch) — vision.PoseLandmarker.create_from_
    # options raises a native error in that case, previously uncaught here
    # and crashing every single call until someone manually deleted the
    # file. Caught at the first real use, with the poisoned file removed
    # so the NEXT call gets a clean re-download instead of repeating the
    # same crash forever.
    try:
        single_pass_candidates, _ = run_detection_pass(1)
    except Exception as e:
        monitoring.capture(e)
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except OSError:
                pass
        return {"status": "error", "error_message": f"Pose detection model failed to load: {str(e)}"}
    total_frames = len(single_pass_candidates)
    # BUG FIX (2026-09-13, robustness audit): a corrupted/truncated/0-byte
    # upload, or a genuinely unreadable codec (most likely to reach here
    # specifically when ffmpeg isn't on PATH — see
    # orchestrator.compress_video_file's own comment on that fallback),
    # used to fall all the way through this function reporting
    # "status": "success" with an empty landmarks dataframe — no video
    # was actually ever readable, but nothing said so. Downstream code
    # (e.g. camera_angle_detection.estimate_camera_angle's own df.iloc[0])
    # assumes AT LEAST ONE real frame exists and crashes on this silently-
    # empty result instead. Caught here, at the source, with a clear
    # message instead of an unrelated-looking crash several calls later.
    if total_frames == 0:
        return {
            "status": "error",
            "error_message": (
                "Could not read any frames from this video. The file may be corrupted, "
                "empty, or in a format this app can't decode — try re-exporting or "
                "re-uploading it."
            ),
        }
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
        frame_candidates, frame_hists = run_detection_pass(3, compute_appearance=True)
        # A frame where only ONE candidate is detected still needs a
        # plausibility check, not just an automatic accept — otherwise a
        # different, more-consistently-detected bystander (e.g. a coach
        # standing close to the camera) becomes "the closest available
        # candidate" by default whenever the real tracked person isn't
        # detected that frame, and tracking silently snaps onto them.
        # See _walk_from_seed for the distance/gap caps that guard
        # against this — verified on real footage this still isn't
        # airtight for a VERY long tracking loss (several seconds) in a
        # scene with a confidently-detected bystander nearby, which is
        # exactly what MULTIPLE seeds (below) are for: instead of asking
        # one heuristic to survive an arbitrarily long gap, let the coach
        # re-confirm identity at a second point later in the clip, and
        # each seed only has to survive the (much shorter) gap to its
        # nearest neighboring seed.
        #
        # MULTI-SEED: seed_point/seed_frame_index is always the primary
        # seed; extra_seeds (optional) adds more (point, frame_index)
        # anchors anywhere else in the clip. Seeds are sorted by frame
        # index and each one only walks within its own zone — split at
        # the midpoint to its neighboring seeds — so seeds never
        # conflict over who "owns" a given frame, and a single seed
        # never has to carry the whole clip if the coach has re-confirmed
        # identity partway through.
        seeds = [(seed_frame_index, seed_point)]
        if extra_seeds:
            seeds.extend(extra_seeds)
        seeds = sorted(
            ((max(0, min(int(idx), total_frames - 1)), pt) for idx, pt in seeds),
            key=lambda s: s[0]
        )

        # PASS 1: match + walk every seed normally, but keep each seed's
        # own chosen_hist around instead of committing to seeded_chosen
        # yet — needed for the cross-seed appearance check below.
        seed_zones = []
        for k, (s_idx, s_pt) in enumerate(seeds):
            lo_bound = 0 if k == 0 else (seeds[k - 1][0] + s_idx) // 2 + 1
            hi_bound = (total_frames - 1) if k == len(seeds) - 1 else (s_idx + seeds[k + 1][0]) // 2
            s_xy = (s_pt[0] / frame_width, s_pt[1] / frame_height)
            walk_result, chosen_hist = _walk_from_seed(s_idx, s_xy, frame_candidates, frame_hists, fps, lo_bound, hi_bound)
            seed_zones.append({"s_idx": s_idx, "s_xy": s_xy, "lo": lo_bound, "hi": hi_bound,
                                "walk_result": walk_result, "hist": chosen_hist})

        # CROSS-SEED APPEARANCE CHECK (2026-09-12, real bug — see
        # _seed_appearance_majority_ok's own docstring): the coach
        # confirms the SAME bowler at every seed, so their matched
        # appearances should mostly agree with each other. Any seed whose
        # match is a clear appearance outlier likely locked onto the
        # wrong person at that exact click (see SEED_MATCH_TOLERANCE's
        # own note above) — redo ONLY that seed's walk with its own
        # (likely wrong) seed-frame match discarded, falling back to
        # position-only tracking from the raw click instead of
        # confidently propagating a wrong appearance profile across its
        # whole zone.
        majority_ok = _seed_appearance_majority_ok([z["hist"] for z in seed_zones])

        # WARM-START PROFILE (2026-09-12, real gap found tracing an actual
        # coach-reported failure — see _walk_from_seed's prior_profile
        # docstring): a seed whose own exact-frame match found NOBODY
        # (the target was momentarily too small/distant/occluded — a
        # real, correct outcome, not itself a bug) used to hand its zone's
        # walk a completely empty appearance profile, so the walk fails
        # open on position alone for its first few matches — long enough
        # for a wrong-but-nearby person to become the "confirmed" profile
        # before the real target is ever seen in that zone. The coach's
        # OTHER trustworthy seed matches already tell us what this person
        # actually looks like elsewhere in the SAME clip — reusing that
        # here means the appearance gate is live from frame 1 of a zone
        # like this, instead of starting from zero evidence every time.
        reference_profile = [z["hist"] for k, z in enumerate(seed_zones)
                              if z["hist"] is not None and majority_ok[k]]

        for k, z in enumerate(seed_zones):
            if not majority_ok[k]:
                # A real match that disagrees with the coach's OTHER
                # confirmed seeds — discard it (trust_seed_match=False)
                # but still give the walk the same warm-start reference
                # the other zones get, rather than falling all the way
                # back to blind position-only tracking.
                z["walk_result"], _ = _walk_from_seed(
                    z["s_idx"], z["s_xy"], frame_candidates, frame_hists, fps, z["lo"], z["hi"],
                    trust_seed_match=False, prior_profile=reference_profile or None,
                )
            elif z["hist"] is None and reference_profile:
                z["walk_result"], _ = _walk_from_seed(
                    z["s_idx"], z["s_xy"], frame_candidates, frame_hists, fps, z["lo"], z["hi"],
                    prior_profile=reference_profile,
                )

        seeded_chosen = [None] * total_frames
        for z in seed_zones:
            for i, v in z["walk_result"].items():
                seeded_chosen[i] = v

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
    #
    # BUG FOUND during a broader audit, fixed in smooth_without_resurrecting_gaps
    # below: a plain rolling(..., min_periods=1).mean() silently RESURRECTS
    # frames the interpolate step above correctly left as genuine NaN (a
    # real gap longer than gap_fill_limit, deliberately NOT patched) — see
    # that function's docstring for the verified real-footage detail.
    output_df[landmark_cols] = smooth_without_resurrecting_gaps(output_df[landmark_cols])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)

    return {
        "status": "success",
        "total_frames_processed": total_frames,
        "fps": fps,
        "output_file": output_csv_path
    }


def extract_raw_landmarks_window(video_path: str, fps: float, landmark_names: list,
                                  start_idx: int, end_idx: int, num_poses: int = 3) -> dict:
    """
    Re-extracts RAW (unsmoothed, un-outlier-filtered) normalized (0-1)
    positions for specific named landmarks, directly from the source
    video, for frames [start_idx, end_idx] inclusive.

    WHY THIS EXISTS (2026-08-07): speed_estimation._extract_raw_wrist_window
    already proved and fixed this exact problem for wrist velocity — the
    saved landmarks CSV has been through Hampel-filter outlier rejection
    AND a 5-frame rolling-mean smoothing pass, correct for a stable-
    looking skeleton and reliable event timing, but that same smoothing
    dilutes a landmark's TRUE position at a sharp, brief moment like ball
    release exactly the same way it dilutes peak velocity. That fix was
    only ever applied to the wrist for the speed estimate — release_height
    (ankle/nose/knee/hip) and head_stability (nose/shoulders across a
    whole window) kept reading the smoothed CSV, unnecessarily carrying
    the same diluted-position problem the wrist fix already solved
    elsewhere. This generalizes that proven pattern to any set of named
    landmarks over any window, instead of one hardcoded landmark.

    Processes every frame from 0 up to end_idx (not just the window) to
    preserve VIDEO mode's real temporal continuity — matches the proven
    wrist-window pattern exactly; detection needs earlier frames to have
    "warmed up" properly.

    MULTI-CANDIDATE (2026-09-15, real coach-reported bug, second root
    cause found after the identity-consistency gate in orchestrator.py):
    this used to run num_poses=1 and blindly take MediaPipe's own single
    top-ranked candidate. In a multi-person scene, MediaPipe's own
    confidence ranking frequently favors a static, unblurred bystander
    over the actual moving/blurred subject (this app's real footage:
    the bowler is legitimately blurred and harder to score confidently
    during the fastest part of the action, exactly when a stationary
    bystander scores cleanly) — so index [0] silently returning "whoever
    scored highest" is itself part of the bug, independent of anything
    downstream. Now requests num_poses=3 (matching the seeded walk's own
    convention — main.py's run_detection_pass(3, ...)) and returns EVERY
    detected candidate per frame, unranked and unfiltered — selecting
    which candidate is actually the tracked subject is an identity
    question, not a confidence question, and is left entirely to the
    caller (see orchestrator._select_identity_consistent_candidate),
    which validates candidates against the coach's own already-confirmed
    identity rather than trusting MediaPipe's internal scoring.

    Returns {frame_index: [{landmark_name: (x_norm, y_norm, visibility)}, ...]}
    — a list of every detected candidate's landmarks for that frame (never
    just one), in whatever order MediaPipe returned them (not meaningful,
    not to be relied on). A frame with no confident detection at all is
    simply absent, never a fabricated position. Callers must fall back to
    the existing smoothed-CSV values when a needed frame/landmark isn't
    present here (e.g. real occlusion, or the video ended before end_idx).
    """
    import os
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = os.path.join("models", "pose_landmarker_full.task")
    landmark_indices = {name: LANDMARK_NAMES.index(name) for name in landmark_names}

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
        num_poses=num_poses,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.4,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    ms_per_frame = 1000.0 / fps

    positions = {}
    idx = 0
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok or idx > end_idx:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts = int(round(idx * ms_per_frame))
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        res = landmarker.detect_for_video(img, ts)
        if idx >= start_idx and res.pose_landmarks:
            frame_candidates = []
            for candidate_landmarks in res.pose_landmarks:
                frame_landmarks = {}
                for name, lm_idx in landmark_indices.items():
                    lm = candidate_landmarks[lm_idx]
                    if lm.visibility is None or lm.visibility >= 0.5:
                        frame_landmarks[name] = (lm.x, lm.y, lm.visibility)
                if frame_landmarks:
                    frame_candidates.append(frame_landmarks)
            if frame_candidates:
                positions[idx] = frame_candidates
        idx += 1
    cap.release()
    landmarker.close()
    return positions


if __name__ == "__main__":
    print("=== STARTING KINEMATIC EXTRACTION STATE ===")
    extraction_state = extract_video_landmarks("input/input_video.mp4", "output/landmarks.csv")
    print(extraction_state)
