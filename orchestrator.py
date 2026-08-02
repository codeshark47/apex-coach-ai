import os
import subprocess
import pandas as pd
import numpy as np
import cv2

import monitoring
from main import extract_video_landmarks
from kinematics import (
    calculate_knee_bracing,
    calculate_trunk_lean,
    calculate_head_stability
)
import camera_angle_detection as cad

# Shared with calculate_release_height_ratio_safe's own "too small to divide
# by reliably" floor below — kept as one named constant so the search that
# LOOKS for a usable reference frame and the check that later REJECTS an
# unusable one can never silently drift apart.
MIN_BODY_HEIGHT_SPAN = 0.05


def _nearest_complete_row(df: pd.DataFrame, frame_idx: int, required_cols: list, max_search: int = 10):
    """
    Returns the row at frame_idx if all required_cols are real there;
    otherwise searches outward (closest frame first) up to max_search
    frames for the nearest one where they are. Verified directly on real
    footage: a reference frame chosen for being part of a stable, grounded
    stretch (e.g. front-foot-plant) can still have a brief single-landmark
    tracking dropout at that EXACT frame (here: NOSE_y missing while both
    ankles were present), even though the body position barely changes
    across the handful of neighboring frames in that same stable window —
    using one of those instead is a safe substitute specifically because
    the reference frame's whole purpose is representing "a grounded,
    stable position," not that exact instant. Returns None if nothing
    within range is complete (a genuinely longer dropout, not brief noise).
    """
    rows = df[df["frame"] == frame_idx]
    if not rows.empty and not any(pd.isna(rows.iloc[0].get(c)) for c in required_cols):
        return rows.iloc[0]
    for offset in range(1, max_search + 1):
        for candidate in (frame_idx - offset, frame_idx + offset):
            rows = df[df["frame"] == candidate]
            if not rows.empty and not any(pd.isna(rows.iloc[0].get(c)) for c in required_cols):
                return rows.iloc[0]
    return None


def _find_grounded_reference_near(df: pd.DataFrame, frame_idx: int, bowling_arm: str,
                                   max_search: int = 90):
    """
    Searches outward from frame_idx (closest frame first) for the nearest
    frame where the lead ankle is genuinely grounded — below both the
    knee and hip in the frame, same plausibility check already used
    inside calculate_release_height_ratio_safe — to use as the "body
    height" reference for that function, instead of relying on a
    separately-timed front-foot-plant detection.

    WHY THIS EXISTS: verified directly on real footage (a leaping
    delivery filmed rear-view) that front-foot-plant TIMING detection
    can fail outright for bowlers who are airborne through release —
    there's no clean "foot stops moving" moment to find at all, so
    whatever frame that detector picks can be an ordinary mid-run-up
    running stride, which is not a grounded reference. But we don't
    actually need to know WHEN he planted — we only need ANY nearby
    frame where he's plausibly standing upright, to measure his real
    body height. That's a much easier, more robust question, especially
    once frame_idx itself is a coach-CONFIRMED release frame rather
    than an auto-detected guess: searching close to a verified anchor
    is far more trustworthy than searching near an unverified one.

    Returns a pd.Series (the row) or None if nothing within range
    qualifies (a genuinely unusual clip, not a bug in the search).

    BUG FIX: "grounded" used to mean ONLY "ankle sits below knee and hip in
    the frame" — correct ordering, but not checked against NOSE_y at all.
    Verified directly on real footage (a rear-view delivery at fast,
    blurred release) that a frame can pass that ordering check while the
    tracked ankle is still wrong: nose/hip/knee/ankle all landed within a
    ~0.05-normalized band of each other — a physically implausible,
    compressed body span — because the ankle landmark itself was
    inaccurate that frame, not because the bowler was crouched or
    mid-stride. That frame was accepted as "grounded" on the first check
    and returned immediately, so the search never looked further outward
    for a frame where the ankle was actually trustworthy — the resulting
    body_height then failed calculate_release_height_ratio_safe's own
    MIN_BODY_HEIGHT_SPAN floor anyway, just one step too late to keep
    searching. Now checked here too, so the search keeps going until it
    finds a frame that's both correctly ordered AND has a plausible span.
    """
    lead_side = "LEFT" if bowling_arm == "right" else "RIGHT"
    required = ["NOSE_y", f"{lead_side}_ANKLE_y", f"{lead_side}_KNEE_y", f"{lead_side}_HIP_y"]

    def _is_grounded(row) -> bool:
        if any(pd.isna(row.get(c)) for c in required):
            return False
        ankle_y = float(row[f"{lead_side}_ANKLE_y"])
        if ankle_y < float(row[f"{lead_side}_KNEE_y"]) or ankle_y < float(row[f"{lead_side}_HIP_y"]):
            return False
        return abs(ankle_y - float(row["NOSE_y"])) >= MIN_BODY_HEIGHT_SPAN

    rows = df[df["frame"] == frame_idx]
    if not rows.empty and _is_grounded(rows.iloc[0]):
        return rows.iloc[0]
    for offset in range(1, max_search + 1):
        for candidate in (frame_idx - offset, frame_idx + offset):
            rows = df[df["frame"] == candidate]
            if not rows.empty and _is_grounded(rows.iloc[0]):
                return rows.iloc[0]
    return None


def detect_bowling_arm(df: pd.DataFrame) -> str:
    """
    Auto-detects which arm is the bowling arm by comparing vertical
    range of motion of each wrist across the whole clip. The bowling arm
    swings through a dramatically larger vertical arc during delivery
    than the non-bowling arm, which stays comparatively still. Returns
    'right' or 'left'. Defaults to 'right' (the original hardcoded
    assumption) if landmarks are unavailable.
    """
    try:
        r_wrist_y = df["RIGHT_WRIST_y"].dropna()
        l_wrist_y = df["LEFT_WRIST_y"].dropna()
        r_range = float(r_wrist_y.max() - r_wrist_y.min()) if len(r_wrist_y) > 1 else 0.0
        l_range = float(l_wrist_y.max() - l_wrist_y.min()) if len(l_wrist_y) > 1 else 0.0
        return "left" if l_range > r_range else "right"
    except Exception:
        return "right"


def detect_delivery_events(df: pd.DataFrame, fps: int, bowling_arm: str = "right",
                            camera_angle: str = None) -> dict:
    """
    Robust physical milestone detection using velocity windows.
    bowling_arm: 'right' or 'left' — determines which wrist is used for
    release detection and which ankle is front/back foot. The lead (front)
    foot is always the ankle OPPOSITE the bowling arm.

    camera_angle: optional, from camera_angle_detection.estimate_camera_angle
    ("side_on" | "front_or_rear" | "uncertain" | "unavailable" | None). Only
    used to gate the elbow-plausibility check below — see that comment for
    why. Defaults to None (behaves as side_on) so existing callers that
    don't pass this are unaffected.
    """
    total_frames = len(df)

    if total_frames < 10:
        return {
            "BFC": 0,
            "FFC": int(total_frames * 0.4),
            "BR": int(total_frames * 0.8)
        }

    bowl_side = "RIGHT" if bowling_arm == "right" else "LEFT"
    lead_side = "LEFT" if bowling_arm == "right" else "RIGHT"

    wrist_y = df[f"{bowl_side}_WRIST_y"].interpolate(method="linear").bfill().ffill().values
    lead_ankle_y = df[f"{lead_side}_ANKLE_y"].interpolate(method="linear").bfill().ffill().values
    back_ankle_y = df[f"{bowl_side}_ANKLE_y"].interpolate(method="linear").bfill().ffill().values

    # FRONT-FOOT PLANT (FFC), found independently of the wrist: the start
    # of the LAST sustained stretch where the lead ankle stops moving and
    # stays grounded. A normal running stride touches down and lifts again
    # within a couple of frames; the actual delivery-stride plant stays
    # down through release, so it's the only point in the clip where the
    # ankle holds still for a sustained duration. This matters because
    # searching for release (the wrist's highest point) across the WHOLE
    # clip can lock onto an earlier arm-raise during the bowler's
    # gather/jump instead of the real release — constraining the release
    # search to start only after this plant fixes that.
    # Tried a percentile-based threshold (relative to this clip's own
    # distribution) instead of a fixed fraction of total range, on the
    # theory it would self-normalize better across videos. Verified
    # against real footage: it backfired — it became so strict that only
    # an artificial dead-flat stretch survived (see the real-detection
    # guard below), collapsing FFC to frame ~2. Reverted to the looser
    # range-based floor, which verified correctly on real footage once
    # combined with that guard.
    plateau_window = max(2, int(round(fps * 0.12)))
    ankle_range = float(np.nanmax(lead_ankle_y) - np.nanmin(lead_ankle_y))
    stability_floor = max(ankle_range * 0.04, 1e-6)
    rolling_std = pd.Series(lead_ankle_y).rolling(
        window=plateau_window, center=False, min_periods=plateau_window
    ).std().values
    is_stable = rolling_std < stability_floor

    # A frame can only count as "stable" if it — and the rest of its
    # plateau_window — had a REAL detection before any gap-filling. Before
    # the bowler has entered the frame, every point is missing and gets
    # backfilled to one repeated constant (zero variance by construction),
    # which otherwise looks MORE "stable" than genuine stillness and can
    # get mistaken for the plant. Verified: this is what actually excludes
    # the empty-frame stretch, not the threshold formula.
    had_real_detection = (~df[f"{lead_side}_ANKLE_y"].isna()).values
    window_all_real = pd.Series(had_real_detection).rolling(
        window=plateau_window, center=False, min_periods=plateau_window
    ).min().astype(bool).values
    is_stable = is_stable & window_all_real
    # Group is_stable into contiguous runs, not just "the last True index".
    # Verified directly against real footage (Abu Bakar clip, 120 frames):
    # any raw video that keeps rolling after the delivery — which is normal,
    # nobody stops recording mid-follow-through — contains a bowler running/
    # decelerating down the pitch afterward. That motion routinely produces
    # its own brief, coincidental 3-frame stretch where the ankle position
    # barely changes (a natural mid-stride pause), which satisfies the exact
    # same rolling-std stability test as a genuine plant. Because "take the
    # LAST stable stretch" has no concept of run length, that spurious blip
    # — sometimes just one 3-frame window wide — was winning over the real
    # plant simply for coming later in the clip, dragging FFC (and BR,
    # searched forward from FFC) to the wrong end of the video entirely.
    # A genuine plant holds for the whole grounded-contact duration of the
    # delivery stride (~150ms+, confirmed 10 consecutive frames at 29fps on
    # the real test clip) — long enough to distinguish from a one-window
    # coincidence. Requiring a minimum sustained run length before a stretch
    # is even eligible to be picked fixes this without reintroducing the
    # percentile-threshold problems already ruled out above.
    MIN_PLANT_FRAMES = max(2 * plateau_window, int(round(fps * 0.15)))
    stable_runs = []
    run_start = None
    for i, s in enumerate(is_stable):
        if s and run_start is None:
            run_start = i
        elif not s and run_start is not None:
            stable_runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        stable_runs.append((run_start, len(is_stable) - 1))

    # SPAN check, not just local variance: a SLOW, CONTINUOUS drift (e.g.
    # decelerating after follow-through, never actually planted) can still
    # pass the per-window rolling-std check above — each individual
    # plateau_window is nearly flat frame-to-frame even while the value
    # cumulatively slides a long way over 20-30 frames. A genuine grounded
    # plant holds its OVERALL position, not just its frame-to-frame delta.
    RUN_SPAN_MAX_FRACTION = 0.15
    span_floor = max(ankle_range * RUN_SPAN_MAX_FRACTION, 1e-6)

    def _run_span(r):
        seg = lead_ankle_y[r[0]:r[1] + 1]
        return float(np.nanmax(seg) - np.nanmin(seg))

    qualifying_runs = [r for r in stable_runs
                       if (r[1] - r[0] + 1) >= MIN_PLANT_FRAMES and _run_span(r) <= span_floor]

    if qualifying_runs or stable_runs:
        # Prefer the last run that's a genuine sustained hold; only fall
        # back to a short/noisy one if literally nothing else was found,
        # so a short but real clip doesn't lose the signal entirely.
        run_start, _ = (qualifying_runs or stable_runs)[-1]
        ffc_idx = max(0, run_start - plateau_window + 1)
    else:
        # No clear plateau found (short/noisy clip) — fall back to the
        # single-frame peak, restricted to the back half of the clip to
        # avoid an early running stride.
        half = int(total_frames * 0.5)
        ffc_idx = half + int(np.argmax(lead_ankle_y[half:]))

    # BALL RELEASE (BR): the bowling wrist's highest point, searched ONLY
    # in a realistic window after front-foot plant — never before it,
    # since release always follows the plant.
    #
    # WIDENED from 0.4s: verified on real rear-view footage where the
    # front-ankle "sustained stillness" plant-detector locked onto an
    # intermediate running stride instead of the true final plant (a
    # multi-stride run-up can show several strides that briefly look
    # "stable" the same way a genuine plant does) — FFC landed ~1 full
    # second early. The true release was still a single, clean, entirely
    # unambiguous global minimum in the wrist trajectory (no competing
    # arm-raise anywhere else in the clip), it just fell outside the old
    # narrow window. A wider window tolerates FFC being somewhat off
    # without reintroducing the original bug this window exists to
    # prevent — that bug was an arm-raise during the gather/jump, which
    # happens BEFORE BFC/FFC chronologically, so it's structurally
    # unreachable by widening a window that only searches FORWARD from FFC.
    br_search_window = max(2, int(round(fps * 1.2)))
    br_search_end = min(total_frames, ffc_idx + br_search_window)
    br_slice = wrist_y[ffc_idx:br_search_end]

    # Prefer a REAL (non-gap-filled) detection for the peak. Verified on
    # real footage: the fastest part of the arm swing (right at release)
    # is exactly where MediaPipe is most likely to briefly lose the wrist
    # to motion blur, which then gets forward-filled to a frozen, stale
    # value — that stale flat value can look like an unbeatable "peak" to
    # a simple argmin and win the search even though it's not a real,
    # current position. Only fall back to the filled data if the whole
    # window has no real detection at all.
    wrist_had_real = (~df[f"{bowl_side}_WRIST_y"].isna()).values

    # ANATOMICAL PLAUSIBILITY GATE: a bowling arm stays substantially
    # extended through the delivery swing (a sharply bent elbow mid-swing
    # is literally the legal-delivery threshold in cricket law, not a real
    # technique variant). Verified directly on real footage: the exact
    # frames where the raw wrist position jitters wildly frame-to-frame
    # (motion blur right at release) also show anatomically-impossible
    # elbow flexion collapsing to ~45-60 degrees — a second, independent
    # signal confirming the wrist landmark itself was lost/confused there,
    # catching cases the gap-filled/interpolated check above misses
    # (MediaPipe reported these frames with normal-looking confidence, so
    # they were never flagged as missing — just wrong). This gate cannot
    # by itself explain a cross-environment frame discrepancy on identical
    # code, but it removes a real, demonstrated source of an unreliable
    # peak candidate from the search regardless of which frame the
    # decoder-level noise happens to land on.
    sx = df[f"{bowl_side}_SHOULDER_x"].values
    sy = df[f"{bowl_side}_SHOULDER_y"].values
    ex = df[f"{bowl_side}_ELBOW_x"].values
    ey = df[f"{bowl_side}_ELBOW_y"].values
    wx = df[f"{bowl_side}_WRIST_x"].values
    wy = df[f"{bowl_side}_WRIST_y"].values
    se_x, se_y = sx - ex, sy - ey
    we_x, we_y = wx - ex, wy - ey
    elbow_norm = np.hypot(se_x, se_y) * np.hypot(we_x, we_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        elbow_cos = np.clip((se_x * we_x + se_y * we_y) / elbow_norm, -1, 1)
    elbow_angle_deg = np.degrees(np.arccos(elbow_cos))
    # CAMERA-ANGLE GATE on the check above: the 2D angle computed here only
    # faithfully reflects real elbow flexion when the arm swings ACROSS the
    # image plane (side-on filming). Verified directly on real rear-view
    # footage: the same genuinely-tracked, correctly-extending arm produced
    # a wildly oscillating 2D angle (4° -> 174° -> 23° -> 177° within one
    # continuous swing) purely from viewing-angle projection, not real
    # elbow motion — because in front-on/rear-on footage the arm moves
    # mostly toward/away from the camera, which this 2D-only formula can't
    # distinguish from bending. That doesn't just weaken the signal, it can
    # actively exclude genuine release-window frames, so it's only applied
    # when we have positive geometric evidence this is a side-on shot;
    # "uncertain"/"unavailable"/None keep the previous (side-on) behavior
    # rather than guessing.
    ELBOW_MIN_PLAUSIBLE_DEG = 90.0
    elbow_plausible = np.nan_to_num(elbow_angle_deg, nan=0.0) >= ELBOW_MIN_PLAUSIBLE_DEG
    if camera_angle != "front_or_rear":
        wrist_had_real = wrist_had_real & elbow_plausible

    real_mask_slice = wrist_had_real[ffc_idx:br_search_end]

    # RELEASE-DETECTION CONFIDENCE: an honest, decode-independent signal
    # for how much of the search window had usable wrist data (real
    # detection AND anatomically plausible), regardless of which exact
    # frame the peak search lands on. Verified directly against real
    # footage with genuine motion blur at release: different video
    # decoders (different OpenCV builds, or the same clip re-encoded)
    # can each land on a DIFFERENT release frame for the same delivery,
    # because the underlying wrist signal is genuinely ambiguous at the
    # pixel level in that window — not because of a bug in this search.
    # No amount of tuning the search logic itself closes that gap, since
    # it's a property of the source footage (motion blur), not the
    # algorithm. What the app CAN do honestly is detect and disclose it,
    # rather than presenting a specific frame number with false
    # confidence. Below BR_CONFIDENCE_FLOOR of the window being usable,
    # flag low confidence so the UI can warn that release-frame-dependent
    # numbers (release height, speed) may be off by a few frames here.
    br_plausible_fraction = float(real_mask_slice.mean()) if len(real_mask_slice) > 0 else 0.0
    BR_CONFIDENCE_FLOOR = 0.6
    br_confidence = "high" if br_plausible_fraction >= BR_CONFIDENCE_FLOOR else "low"

    # STRICTER extension gate for identifying release CANDIDATES
    # specifically — separate from the looser 90-degree "plausible
    # tracking" gate above, which stays as-is for the confidence number.
    # Verified directly on real side-on footage: a bowler's gather/
    # back-lift arm-raise before the final delivery stride can itself
    # reach a bent-but-plausible elbow angle (measured ~118 degrees) with
    # a LARGER raw wrist-height rise than the true release swing itself —
    # the arm doesn't have as far left to travel to reach extension once
    # already partly raised — so amplitude alone (even after widening the
    # window and preferring the earliest significant rise) still picked
    # the gather motion over the real release. The true release
    # consistently showed near-full extension (~168-177 degrees on the
    # same clip) where the gather did not (~62-118 degrees). Gating
    # candidate peaks on this stricter bar rules out the gather motion
    # regardless of how large its raw amplitude is. Same camera-angle
    # caveat as the looser gate above: only trustworthy for side-on
    # footage, so front/rear stays on the original (unstricter) mask.
    RELEASE_ELBOW_MIN_DEG = 150.0
    if camera_angle != "front_or_rear":
        release_ready_slice = np.nan_to_num(elbow_angle_deg, nan=0.0)[ffc_idx:br_search_end] >= RELEASE_ELBOW_MIN_DEG
        # SUSTAINED extension only, not a single-frame crossing: verified
        # directly on real footage that normal running-arm-swing motion
        # during the run-up can briefly (a few frames) swing through an
        # angle that also happens to clear this bar, purely by chance timing
        # — not because the bowler is anywhere near release. That brief
        # false crossing seeded the running-baseline early, making the
        # REAL, sustained extension later in the window look like a smaller
        # secondary rise instead of the main event. A genuine release-arm
        # extension holds for several consecutive frames, not one.
        MIN_EXTENSION_FRAMES = max(6, int(round(fps * 0.2)))
        # Fill brief gaps before applying the duration filter: verified on
        # real footage that the true release itself can show a short 2-3
        # frame dip in this same angle right around/after the fastest part
        # of the swing (motion blur), which otherwise splits one genuine,
        # long extension into two pieces each too short to qualify alone.
        # A short tolerance bridges real blur without rescuing the
        # genuinely brief, unrelated running-arm-swing crossings seen
        # elsewhere (those gaps run much longer than this tolerance).
        GAP_FILL_TOLERANCE = 3
        filled = release_ready_slice.copy()
        gap_start = None
        for i, s in enumerate(filled):
            if not s and gap_start is None:
                gap_start = i
            elif s and gap_start is not None:
                if gap_start > 0 and (i - gap_start) <= GAP_FILL_TOLERANCE:
                    filled[gap_start:i] = True
                gap_start = None
        release_ready_slice = filled

        release_ready = np.zeros_like(release_ready_slice, dtype=bool)
        run_start_r = None
        for i, s in enumerate(release_ready_slice):
            if s and run_start_r is None:
                run_start_r = i
            elif not s and run_start_r is not None:
                if i - run_start_r >= MIN_EXTENSION_FRAMES:
                    release_ready[run_start_r:i] = True
                run_start_r = None
        if run_start_r is not None and len(release_ready_slice) - run_start_r >= MIN_EXTENSION_FRAMES:
            release_ready[run_start_r:] = True

        release_candidate_mask = real_mask_slice & release_ready
        if not release_candidate_mask.any():
            # No frame in the window meets the stricter bar (e.g. camera
            # angle/action style genuinely doesn't show a clean extension) —
            # fall back to the looser mask rather than finding nothing.
            release_candidate_mask = real_mask_slice
    else:
        release_candidate_mask = real_mask_slice

    # PROMINENCE, not just the single lowest point: a real release swing
    # rises substantially from its own recent baseline (arm coming up
    # from a low, resting position). Verified on real footage this
    # matters once the window above was widened to tolerate FFC timing
    # error — the wider window could otherwise catch a shallow,
    # insignificant dip later on where tracking had briefly drifted onto
    # a bystander doing something unrelated (no real arm-raise dynamics,
    # just a flat, low-amplitude trace with a technically-lower point).
    # For each frame, baseline = the highest real wrist_y (lowest arm
    # position) seen so far since the window started; prominence = how
    # far the current point has risen above that running baseline. The
    # frame with the GREATEST prominence is the real swing-up, not
    # necessarily the frame with the single lowest absolute value.
    if len(br_slice) > 0 and release_candidate_mask.any():
        real_slice = np.where(release_candidate_mask, br_slice, np.nan)
        running_baseline = np.full(len(real_slice), np.nan)
        current_max = -np.inf
        for i, v in enumerate(real_slice):
            if not np.isnan(v):
                current_max = max(current_max, v)
            if current_max > -np.inf:
                running_baseline[i] = current_max
        prominence = running_baseline - real_slice
        if np.any(~np.isnan(prominence)):
            # EARLIEST significant swing, not necessarily the single
            # highest-prominence one anywhere in the window. Verified
            # directly against real footage (a rear-view leaping delivery):
            # after the true release, the arm's own follow-through/recovery
            # motion can produce a SECOND rise-from-baseline that happens to
            # score marginally higher raw prominence than the first, genuine
            # release swing (measured on real data: 0.0367 vs 0.0303 — only
            # ~18% apart) purely because of incidental timing, not because
            # it's a more real swing. Taking the global max picked that
            # later, wrong swing. A real delivery has exactly one release
            # event; anything after it is follow-through/recovery, so the
            # first swing that's clearly a real rise (not noise) — not
            # whichever one wins by a few percent — is the physically
            # correct one to trust.
            global_max_prominence = float(np.nanmax(prominence))
            SIGNIFICANCE_FRACTION = 0.5
            sig_threshold = SIGNIFICANCE_FRACTION * global_max_prominence
            peak_relative_idx = None
            n = len(prominence)
            for i in range(n):
                p = prominence[i]
                if np.isnan(p) or p < sig_threshold:
                    continue
                prev_p = prominence[i - 1] if i > 0 and not np.isnan(prominence[i - 1]) else -np.inf
                next_p = prominence[i + 1] if i < n - 1 and not np.isnan(prominence[i + 1]) else -np.inf
                if p >= prev_p and p >= next_p:
                    peak_relative_idx = i
                    break
            if peak_relative_idx is None:
                peak_relative_idx = int(np.nanargmax(prominence))
            # ONSET, not the deepest point of the swing: verified on real
            # footage (a frame showing the actual ball still at the
            # fingertips) that true release happens when the arm FIRST
            # reaches near-full extension — the wrist continues rising
            # slightly further afterward under its own follow-through
            # momentum even though the ball has already left the hand, so
            # the geometric deepest point can land several frames after
            # the real release. Take the EARLIEST frame (proportional
            # threshold, so it scales with each delivery's own swing
            # size) that already reached most of the peak's own
            # prominence — scanning forward from the window start rather
            # than walking backward from the peak, since the swing can
            # briefly dip back down mid-rise (a real "double-dip" in the
            # wrist trajectory, not tracking noise) before its final
            # deepest point, which would stop a backward walk short of
            # the true, earlier onset.
            peak_prominence = prominence[peak_relative_idx]
            ONSET_FRACTION = 0.85
            threshold = ONSET_FRACTION * peak_prominence
            onset_idx = peak_relative_idx
            for i in range(0, peak_relative_idx + 1):
                if not np.isnan(prominence[i]) and prominence[i] >= threshold:
                    onset_idx = i
                    break
            br_idx = ffc_idx + onset_idx
        else:
            br_idx = ffc_idx + int(np.argmin(br_slice))
    elif len(br_slice) > 0:
        br_idx = ffc_idx + int(np.argmin(br_slice))
    else:
        br_idx = min(ffc_idx + 1, total_frames - 1)

    bfc_lookback = max(0, ffc_idx - int(fps * 0.5))
    bfc_window = back_ankle_y[bfc_lookback:ffc_idx]

    if len(bfc_window) > 0:
        bfc_idx = bfc_lookback + int(np.argmax(bfc_window))
    else:
        bfc_idx = max(0, ffc_idx - int(fps * 0.2))

    if bfc_idx == ffc_idx:
        ffc_idx += 1
    if ffc_idx == br_idx:
        br_idx += 1

    return {
        "BFC": int(max(1, bfc_idx)),
        "FFC": int(max(2, ffc_idx)),
        "BR": int(min(br_idx, total_frames - 1)),
        "BR_confidence": br_confidence,
        "BR_plausible_fraction": round(br_plausible_fraction, 2),
    }


# ALIAS: dual_camera_orchestrator.py imports this function under the name
# "embedded_detect_events" (likely written against an earlier/different
# naming convention that never matched this file). Rather than renaming
# detect_delivery_events itself — which risks breaking anything else that
# may depend on the current name — both names now point to the same
# function. Zero behavior change, fixes the ImportError from dual camera mode.
embedded_detect_events = detect_delivery_events


def calculate_hip_shoulder_separation(df: pd.DataFrame, ffc_frame: int) -> dict:
    """
    Measures rotational separation between hip and shoulder planes at FFC.
    Uses arctan2 method — correct for rear-view and side-view footage.

    BUG FIX (was): the old code took abs(shoulder_angle - hip_angle) directly.
    Since each angle individually is in (-180, 180], their raw difference can
    range up to 360 degrees. The old "if >90: separation = 180-separation"
    fold assumed its input was already safely in [0, 180] — whenever the two
    angles straddled the +/-180 boundary (e.g. shoulder_angle=178,
    hip_angle=-178.74, raw abs diff=356.74), that fold produced a NEGATIVE
    nonsense value (180-356.74 = -176.74) instead of the small real
    separation the wraparound actually represented (3.26 degrees here).
    Fix: wrap the angle difference into (-180, 180] BEFORE folding.
    Verified against known non-wraparound cases to produce identical
    results to the old formula, and against the exact failing case to now
    produce a physically plausible value.
    """
    try:
        row = df[df["frame"] == ffc_frame].iloc[0]

        shoulder_angle = np.degrees(np.arctan2(
            row["LEFT_SHOULDER_y"] - row["RIGHT_SHOULDER_y"],
            row["LEFT_SHOULDER_x"] - row["RIGHT_SHOULDER_x"]
        ))
        hip_angle = np.degrees(np.arctan2(
            row["LEFT_HIP_y"] - row["RIGHT_HIP_y"],
            row["LEFT_HIP_x"] - row["RIGHT_HIP_x"]
        ))

        # BUG FIX: a NaN landmark (tracking dropout at the FFC frame)
        # propagated silently through arctan2 without raising. Every
        # tier comparison below is a "separation >= X" check, and NaN
        # compares False against everything in Python — so a NaN result
        # fell through to the final `else` and was confidently labeled
        # "Blocked rotation" (a real coaching claim) instead of being
        # flagged as a tracking failure. Verified against a real session
        # where this path fired: status came back "success" with a NaN
        # degrees value that _sanitize_for_json later silently turned
        # into null right before saving, hiding the real cause.
        if np.isnan(shoulder_angle) or np.isnan(hip_angle):
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        raw_diff = shoulder_angle - hip_angle
        wrapped_diff = (raw_diff + 180) % 360 - 180  # safely in (-180, 180]
        separation = abs(wrapped_diff)                # safely in [0, 180]

        # Hip/shoulder lines are undirected axes, not directed vectors, so a
        # separation beyond 90 degrees represents the same physical twist as
        # its complement — fold into [0, 90].
        if separation > 90:
            separation = 180 - separation

        separation = round(separation, 2)

        if separation >= 25.0:
            tier = "Optimal stretch"
        elif separation >= 15.0:
            tier = "Moderate separation"
        else:
            tier = "Blocked rotation"

        return {"degrees": separation, "tier": tier, "status": "success"}

    except Exception as e:
        monitoring.capture(e)
        return {
            "degrees": None,
            "tier": "Calculation error",
            "status": "error",
            "error_message": str(e)
        }


def calculate_release_height_ratio_safe(br_row: pd.Series, bowling_arm: str = "right",
                                         reference_row: pd.Series = None,
                                         wrist_override_norm: tuple = None) -> dict:
    """
    Calculates release height leverage ratio with expanded real-world tolerances.
    Prevents N/A dropouts on high-arm actions or varied camera distances.
    bowling_arm: 'right' or 'left' — determines which wrist measures release.

    reference_row: optional row (typically the FFC/front-foot-plant frame)
    used for the ankle+head "body height" measurement instead of br_row.
    Verified directly on real footage: bowlers with a leaping/jumping
    release are legitimately airborne at the BR frame itself, so their
    ankle sits well off the ground at that exact instant — using that
    frame for "body height" measures a compressed, mid-air span that has
    nothing to do with their real standing height, inflating the ratio
    past the physical-plausibility bound below and producing a false N/A
    on an otherwise perfectly well-tracked delivery. FFC is a physically
    grounded reference (that's what front-foot-CONTACT means) regardless
    of whether this bowler's release style is grounded or airborne.
    Defaults to None (uses br_row, prior behavior) for backward
    compatibility with any caller not yet passing it.

    wrist_override_norm: optional (x, y) normalized (0-1) coach-confirmed
    wrist/ball position, overriding the tracked landmark entirely. Verified
    directly on real footage: MediaPipe can systematically under-track how
    far the hand extends during a fast, motion-blurred release swing — not
    a single-frame glitch catchable by an outlier filter, but a sustained
    mistracking across the whole swing (shoulder-to-wrist distance grows
    smoothly frame to frame right through the bad reading, so there's no
    anomalous jump to detect automatically). Only a human directly marking
    the real ball position on the frame reliably fixes this, same reasoning
    as the mandatory BFC/FFC/BR frame confirmation elsewhere. Only the y
    (height) component is actually used, but both are accepted since the
    coach clicks a single point in the UI.
    """
    try:
        bowl_side = "RIGHT" if bowling_arm == "right" else "LEFT"
        if wrist_override_norm is not None:
            y_wrist = wrist_override_norm[1]
        else:
            y_wrist = br_row.get(f"{bowl_side}_WRIST_y")
        height_row = reference_row if reference_row is not None else br_row
        y_head = height_row.get("NOSE_y")

        # Use the FRONT/LEAD ankle (opposite the bowling arm) — that's the
        # foot planted on the ground at release, same "lead_side" convention
        # used elsewhere (detect_delivery_events, calculate_knee_bracing).
        # Previously this just preferred whichever ankle was labeled "LEFT"
        # regardless of which leg that actually was — for a left-arm
        # bowler the left ankle is the TRAILING leg, which normally lifts
        # during follow-through, so checking it against the "ankle should
        # be grounded" rule below was flagging normal motion as an error.
        lead_side = "LEFT" if bowling_arm == "right" else "RIGHT"
        trail_side = "RIGHT" if lead_side == "LEFT" else "LEFT"
        y_ankle_lead = height_row.get(f"{lead_side}_ANKLE_y")
        y_ankle_trail = height_row.get(f"{trail_side}_ANKLE_y")
        if y_ankle_lead is not None and not pd.isna(y_ankle_lead):
            y_ankle, ankle_side = y_ankle_lead, lead_side
        elif y_ankle_trail is not None and not pd.isna(y_ankle_trail):
            y_ankle, ankle_side = y_ankle_trail, trail_side
        else:
            y_ankle, ankle_side = None, None

        if any(v is None or pd.isna(v) for v in [y_wrist, y_head, y_ankle]):
            return {
                "ratio": None,
                "classification": "Landmark missing",
                "status": "error",
                "error_message": "One or more landmarks missing on BR frame."
            }

        body_height = abs(float(y_ankle) - float(y_head))

        debug_raw = {
            "y_wrist": round(float(y_wrist), 4),
            "y_head": round(float(y_head), 4),
            "y_ankle": round(float(y_ankle), 4),
            "body_height": round(float(body_height), 4),
            "bowl_side_used": bowl_side,
            "ankle_side_used": ankle_side,
        }

        # Numerical-stability floor only — a near-zero denominator blows the
        # ratio up regardless of whether tracking is good. This is NOT a
        # mistracking signal by itself: a video's head-to-ankle span in the
        # frame depends heavily on camera distance, so a flat cutoff here
        # (previously 0.35) rejects legitimately well-tracked videos just
        # because they were filmed wider or further away.
        if body_height < MIN_BODY_HEIGHT_SPAN:
            return {
                "ratio": None,
                "classification": "Body height too small",
                "status": "error",
                "error_message": (
                    f"Body height span ({round(body_height, 3)}) too small to "
                    f"divide by reliably."
                ),
                "debug_raw": debug_raw
            }

        # Real mistracking check: at ball release the front/plant foot is on
        # the ground, so the ankle landmark should sit below both the knee
        # and hip in the frame (larger y = lower in image coordinates). This
        # holds regardless of camera distance/framing, unlike a raw span
        # cutoff, so it catches an actual mistracked ankle without punishing
        # videos filmed wider or further away.
        y_knee = height_row.get(f"{ankle_side}_KNEE_y")
        y_hip = height_row.get(f"{ankle_side}_HIP_y")
        if y_knee is not None and not pd.isna(y_knee) and y_hip is not None and not pd.isna(y_hip):
            if float(y_ankle) < float(y_knee) or float(y_ankle) < float(y_hip):
                return {
                    "ratio": None,
                    "classification": "Ankle landmark implausible",
                    "status": "error",
                    "error_message": (
                        f"{ankle_side} ankle is not below the {ankle_side.lower()} "
                        f"knee/hip on the BR frame — landmark is likely "
                        f"mistracked rather than the reading being real."
                    ),
                    "debug_raw": {**debug_raw,
                                  "y_knee": round(float(y_knee), 4),
                                  "y_hip": round(float(y_hip), 4)}
                }

        ratio = round(abs(float(y_ankle) - float(y_wrist)) / body_height, 4)

        # BUG FIX: this ceiling exists to catch a MISTRACKED wrist (the
        # tracker's guess is implausible, so don't trust it) — but it was
        # applying even when wrist_override_norm is set, i.e. even when a
        # COACH directly clicked the real ball/hand position on the frame.
        # Verified directly on real footage: a coach-confirmed point on a
        # genuinely leaping, high-reach delivery produced ratio 1.49 and
        # was silently discarded as "Measurement error" — rejecting a
        # human's direct observation on the theory that it must be a
        # tracking error, when there was no tracker involved at all for
        # this value. A human-confirmed point is ground truth (same
        # reasoning as the mandatory BFC/FFC/BR confirmation elsewhere) —
        # this ceiling now only applies to the AUTOMATIC, unconfirmed
        # reading, where "this is probably a tracking glitch" is actually
        # a reasonable inference.
        if wrist_override_norm is None and (ratio > 1.30 or ratio < 0.30):
            return {
                "ratio": None,
                "classification": "Measurement error — verify camera angle",
                "status": "error",
                "error_message": f"Ratio {ratio} outside physical bounds.",
                "debug_raw": debug_raw
            }

        if ratio >= 0.85:
            classification = "High-Release Leverage"
        elif ratio >= 0.75:
            classification = "Standard Mid-Arm Release"
        else:
            classification = "Low-Sling Action"

        return {"ratio": ratio, "classification": classification, "status": "success", "debug_raw": debug_raw}

    except Exception as e:
        monitoring.capture(e)
        return {
            "ratio": None,
            "classification": "Calculation error",
            "status": "error",
            "error_message": str(e)
        }


def _draw_rounded_rect(img, pt1, pt2, color, radius):
    """Filled rounded rectangle — cv2 has no native support for this."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(img, (cx, cy), radius, color, -1)


def _draw_panel(frame, pt1, pt2, fill_color=(15, 15, 15), radius=14,
                 fill_alpha=0.55, shadow_offset=6, shadow_alpha=0.35):
    """
    Rounded panel with a soft drop shadow, alpha-blended onto frame in
    place. Replaces the old flat sharp-cornered rectangle, which read as
    a basic "programmer UI" overlay rather than a broadcast graphic.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]

    shadow = frame.copy()
    sx1, sy1 = x1 + shadow_offset, y1 + shadow_offset
    sx2, sy2 = min(x2 + shadow_offset, w - 1), min(y2 + shadow_offset, h - 1)
    _draw_rounded_rect(shadow, (sx1, sy1), (sx2, sy2), (0, 0, 0), radius)
    frame[:] = cv2.addWeighted(shadow, shadow_alpha, frame, 1 - shadow_alpha, 0)

    fill = frame.copy()
    _draw_rounded_rect(fill, (x1, y1), (x2, y2), fill_color, radius)
    frame[:] = cv2.addWeighted(fill, fill_alpha, frame, 1 - fill_alpha, 0)


def generate_fail_safe_video(video_path: str, output_path: str,
                              df: pd.DataFrame, events: dict,
                              slow_motion_factor: float = 4.0,
                              bowling_arm: str = "right",
                              camera_angle: str = "side_on"):
    """
    Thin delegating wrapper — the actual rendering logic now lives in
    video_overlay.render_annotated_video (moved out into its own module
    so visual-design code has a clear home separate from the biomechanics
    calculations in this file). Kept here, with the exact same name and
    signature existing callers already use (dual_camera_orchestrator.py,
    run_complete_bowling_analysis below), so nothing else needed to change
    to pick this up. camera_angle is new (defaults to the prior implicit
    behavior, side_on) — existing callers that don't pass it are
    unaffected; see video_overlay.py for what it actually changes (an
    on-video angle indicator, nothing that touches the skeleton/metrics).
    """
    import video_overlay
    return video_overlay.render_annotated_video(
        video_path, output_path, df, events,
        slow_motion_factor=slow_motion_factor,
        bowling_arm=bowling_arm,
        camera_angle=camera_angle,
    )


def save_uploaded_video_capped(uploaded_file, dest_path: str, max_width: int = 1920, max_height: int = 1080) -> None:
    """
    Writes an uploaded video to dest_path, downscaled to fit within
    max_width x max_height (preserving aspect ratio, never upscaling a
    smaller source) and re-encoded to H.264 via ffmpeg.

    BUG FOUND from a real device test: a native 4K (2160x3840) HEVC
    recording straight off a phone camera crashed the app during
    Execute Analysis. Every clip tested before that had gone through
    WhatsApp first, which re-compresses to well under 1080p — this
    pipeline had never actually been asked to decode/process a full-
    resolution native recording. MediaPipe, OpenCV, and ffmpeg all pay a
    per-frame cost proportional to pixel count (4K is ~4x the pixels of
    1080p, ~16x of 720p), and nothing capped that anywhere. Downscaling
    once, upfront, fixes every downstream step at once instead of
    patching each one individually.

    Falls back to writing the original file untouched only if ffmpeg isn't
    installed at all (a deployment issue, not a per-video one). If ffmpeg
    IS installed but fails or times out on this specific file, that means
    the file itself is too demanding to safely hand to the rest of the
    pipeline — raises RuntimeError instead of silently proceeding with an
    even-more-demanding original (see the 2026-08-02 note below).
    """
    import shutil
    import tempfile

    # BUG FOUND (2026-08-02, same iPhone 17 Pro Max incident as above): the
    # server log for that crash showed the app going silent with NO warning
    # printed — but every failure path below prints one before giving up.
    # That means the crash most likely happened even earlier than any of
    # this: reading the raw upload fully into memory (the next few lines)
    # runs BEFORE ffmpeg is ever invoked. Newer Pro-line phones can produce
    # native files large enough (very high bitrate, or stereoscopic
    # "spatial video" on 15/16/17 Pro) to exhaust the server's memory just
    # buffering the raw bytes, before the compressor gets a chance to help
    # at all. Reject grossly oversized uploads outright, using the size
    # Streamlit already reports without needing to read the file first.
    MAX_UPLOAD_BYTES = 300 * 1024 * 1024  # 300MB
    upload_size = getattr(uploaded_file, "size", None)
    if upload_size is not None and upload_size > MAX_UPLOAD_BYTES:
        raise RuntimeError(
            f"This video file is {upload_size / (1024 * 1024):.0f}MB, too large for this "
            "server to safely handle. Try recording at a standard (non-Pro/non-HDR/"
            "non-spatial) quality setting, or trim/compress the clip, then re-upload."
        )

    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)
    raw_fd, raw_path = tempfile.mkstemp(
        suffix=os.path.splitext(uploaded_file.name)[1] or ".mp4", dir=dest_dir
    )
    os.close(raw_fd)
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print(
            "WARNING: ffmpeg not found on PATH. Uploaded video will be used at "
            "its original resolution, which may be slow or memory-heavy for "
            "large 4K+ recordings. Install ffmpeg and ensure it's on PATH to fix this."
        )
        os.replace(raw_path, dest_path)
        return

    cmd = [
        ffmpeg_bin, "-y", "-i", raw_path,
        "-vf", f"scale={max_width}:{max_height}:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "fast",
        "-an", dest_path,
    ]

    # BUG FOUND (2026-08-02): an iPhone 17 Pro native recording crashed the
    # whole shared Streamlit Cloud process (server died outright — "Oh no",
    # no Python traceback in the logs, just silence — not something caught
    # here). The fallback below used to treat ANY downscale failure the
    # same as "ffmpeg isn't installed" and proceed with the untouched
    # original file. That's backwards when ffmpeg fails or times out WHILE
    # actively trying to decode this specific file: it means the source is
    # too demanding for this decode step (resolution, bitrate, HDR/10-bit,
    # framerate — newer phones keep raising this bar), and handing the
    # even-more-demanding original to the rest of the pipeline (MediaPipe,
    # OpenCV) just delays the same crash by a few steps, for every other
    # coach sharing this same free-tier process. Only "ffmpeg missing from
    # PATH entirely" (a deployment issue, not a per-video one) still falls
    # back to the original — an actual failure/timeout on this file now
    # raises a clear, catchable error instead, so Streamlit shows a normal
    # red error box and the server survives instead of dying outright.
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            startupinfo=startupinfo, timeout=180,
        )
    except subprocess.TimeoutExpired:
        os.remove(raw_path)
        raise RuntimeError(
            "This video took too long to process (over 3 minutes) — it's likely "
            "too high-resolution or high-bitrate for this server. Try trimming the "
            "clip or recording at a standard (non-Pro/non-HDR) quality setting, "
            "then re-upload."
        )
    except Exception as e:
        monitoring.capture(e)
        os.remove(raw_path)
        raise RuntimeError(
            f"This video couldn't be processed due to an unexpected error ({e}). "
            "Try trimming the clip or recording at a standard (non-Pro/non-HDR) "
            "quality setting, then re-upload."
        )

    if result.returncode != 0 or not os.path.exists(dest_path):
        stderr_tail = result.stderr.decode(errors='ignore')[:300]
        os.remove(raw_path)
        raise RuntimeError(
            "This video couldn't be processed — it may be too high-resolution, "
            "high-bitrate, or long for this server. Try trimming the clip or "
            "recording at a standard (non-Pro/non-HDR) quality setting, then "
            f"re-upload. (ffmpeg exit {result.returncode}: {stderr_tail})"
        )
    os.remove(raw_path)


def transcode_to_h264(input_path: str) -> str:
    """
    Transcodes mp4v video to H264 for browser playback using ffmpeg.

    FIX (was): the old code hardcoded a Windows-only path
    (r"C:\\ffmpeg\\bin\\ffmpeg.exe") for os.name == "nt", and just the bare
    "ffmpeg" command otherwise — with no check that either actually exists.
    If ffmpeg wasn't at that exact path (or not on PATH on Linux), the
    subprocess call would fail, the exception was silently swallowed, and
    the function returned the original untranscoded mp4v video with NO
    warning anywhere — which may not play back correctly in a browser.

    Fix: use shutil.which() to actually locate ffmpeg on PATH, on any OS.
    If it's genuinely not installed/found, that's surfaced with a clear
    log line instead of failing silently.
    """
    import shutil

    base, ext = os.path.splitext(input_path)
    web_safe_path = f"{base}_h264{ext}"

    if os.path.exists(web_safe_path):
        try:
            os.remove(web_safe_path)
        except OSError:
            pass

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print(
            "WARNING: ffmpeg not found on PATH. Video will be served in its "
            "original codec, which may not play back correctly in all browsers. "
            "Install ffmpeg and ensure it's on PATH to fix this."
        )
        return input_path

    cmd = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "medium",
        "-profile:v", "baseline", "-level", "3.0",
        "-an", web_safe_path
    ]

    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo
        )

        if result.returncode == 0 and os.path.exists(web_safe_path):
            return web_safe_path
        else:
            print(
                f"WARNING: ffmpeg transcode failed (exit code {result.returncode}): "
                f"{result.stderr.decode(errors='ignore')[:300]}"
            )
    except Exception as e:
        monitoring.capture(e)
        print(f"WARNING: ffmpeg transcode raised an exception: {e}")

    return input_path


def landmarks_csv_path(camera_mode: str, output_dir: str = "output") -> str:
    """
    Single source of truth for where each pipeline's per-frame landmark
    CSV lives, so downstream readers can't drift out of sync with what
    each orchestrator actually writes.

    BUG FIX (found during a broader audit): streamlit_app.py's Speed
    Estimation / Run-Up Analysis section hardcoded "landmarks.csv" for
    BOTH Single and Dual Camera results, but dual_camera_orchestrator.py
    has always written "landmarks_side.csv"/"landmarks_rear.csv" instead
    — it never writes "landmarks.csv" at all. In Dual Camera mode this
    meant Speed/Run-Up either silently vanished entirely (the file never
    existed) or, worse, silently read a STALE "landmarks.csv" left over
    from an earlier Single Camera run in the same running app instance —
    computing Dual Camera's speed/run-up numbers from a completely
    different, unrelated delivery's tracking data with no warning at all.
    """
    filename = "landmarks_side.csv" if camera_mode == "Dual Camera" else "landmarks.csv"
    return os.path.join(output_dir, filename)


def extract_and_detect_events(video_path: str,
                               output_dir: str = "output",
                               bowling_arm_override: str = None,
                               seed_point: tuple = None,
                               seed_frame_index: int = 0,
                               extra_seeds: list = None,
                               camera_angle_override: str = None) -> dict:
    """
    STAGE 1+2 split out on its own: landmark extraction, bowling-arm
    detection, camera-angle estimate, and event detection — everything
    needed to show the coach a confirmable camera-angle guess BEFORE
    paying for the more expensive metrics+video-rendering stage below.

    Split out specifically so the UI can ask "is this side-on/front/rear?"
    right after this (relatively fast) stage instead of after the full
    pipeline (including ffmpeg transcoding) has already run once with a
    guess. See run_complete_bowling_analysis, which calls this internally
    and continues with the rest for callers that don't need the two-step
    UI flow (dual-camera, CLI use, etc).

    Returns {"status": "success", "df": DataFrame, "csv_path": str,
    "fps": float, "bowling_arm": str, "events": dict,
    "angle_estimate": AngleEstimate} or {"status": "failed", ...}.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "landmarks.csv")

    extraction = extract_video_landmarks(video_path, csv_path,
                                          seed_point=seed_point,
                                          seed_frame_index=seed_frame_index,
                                          extra_seeds=extra_seeds)

    if extraction["status"] == "error":
        return {
            "status": "failed",
            "stage": "perception",
            "message": extraction["error_message"]
        }

    df = pd.read_csv(csv_path)
    fps = extraction["fps"]

    if bowling_arm_override in ("left", "right"):
        bowling_arm = bowling_arm_override
    else:
        bowling_arm = detect_bowling_arm(df)

    # Camera angle feeds the elbow-plausibility gate inside event detection
    # (see detect_delivery_events).
    #
    # SAMPLED across several frames, not a single midpoint: verified
    # directly on real footage that one arbitrary frame (e.g. the clip's
    # exact midpoint, likely still mid-run-up) can show a misleadingly
    # rotated torso — a genuinely side-on camera setup got classified as
    # "front_or_rear" because that one frame's body pose, not the camera
    # position, happened to have a wide projected shoulder width. Sampling
    # several frames across the back half of the clip (closer to the
    # actual delivery, away from early run-up where he's also smaller/
    # more distant) and taking the majority vote is robust to any single
    # frame's incidental pose.
    angle_estimate = None
    if camera_angle_override in ("side_on", "front_or_rear", "uncertain"):
        camera_angle = camera_angle_override
    else:
        cap_probe = cv2.VideoCapture(video_path)
        probe_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        probe_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        cap_probe.release()
        sample_fracs = (0.5, 0.6, 0.7, 0.8, 0.9)
        votes = {}
        ratios = []
        for frac in sample_fracs:
            est = cad.estimate_camera_angle(df, int(len(df) * frac), probe_w, probe_h)
            if est.angle not in ("unavailable",):
                votes[est.angle] = votes.get(est.angle, 0) + 1
            if est.ratio is not None:
                ratios.append(est.ratio)
        if votes:
            camera_angle = max(votes, key=votes.get)
            angle_estimate = cad.AngleEstimate(
                camera_angle, (ratios[len(ratios) // 2] if ratios else None),
                f"Auto-detected from {len(votes)} frame(s) sampled through the delivery half of the clip."
            )
        else:
            camera_angle = "uncertain"
            angle_estimate = cad.AngleEstimate("uncertain", None, "Could not confidently sample any frame.")

    events = detect_delivery_events(df, fps, bowling_arm=bowling_arm,
                                     camera_angle=camera_angle)

    return {
        "status": "success",
        "df": df,
        "csv_path": csv_path,
        "fps": fps,
        "bowling_arm": bowling_arm,
        "camera_angle": camera_angle,
        "angle_estimate": angle_estimate,
        "events": events,
    }


def run_complete_bowling_analysis(video_path: str,
                                   output_dir: str = "output",
                                   bowling_arm_override: str = None,
                                   seed_point: tuple = None,
                                   seed_frame_index: int = 0,
                                   extra_seeds: list = None,
                                   camera_angle_override: str = None,
                                   precomputed: dict = None) -> dict:
    """
    Core orchestration loop.
    Extracts landmarks, detects events, calculates all 5 biomechanical
    metrics, generates annotated video, and returns unified payload.

    seed_point/seed_frame_index: optional coach click identifying the
    bowler in a reference frame, passed straight through to
    extract_video_landmarks — see that function's docstring.

    extra_seeds: optional list of (frame_index, point) pairs — lets a
    coach re-confirm the bowler's identity at additional points later
    in the clip if tracking is lost for a long stretch (e.g. a bystander
    standing between the bowler and camera for several seconds). Each
    seed only has to survive the gap to its nearest neighboring seed
    instead of one seed carrying the whole video — see
    main._walk_from_seed for how zones are split between seeds.

    camera_angle_override: 'side_on' | 'front_or_rear' | 'uncertain',
    when the coach has manually confirmed the filming angle up front.
    Verified directly on real footage that the geometry-based auto-detect
    (shoulder-width/height ratio) can be unstable for some bowlers' running
    styles — the same genuinely side-on setup produced ratios swinging
    from clearly-side-on to clearly-front-or-rear across different frames
    of the same clip, misclassifying it and disabling a real-fix release
    check that only applies to side-on footage. A human confirmation
    should win over that shaky auto-guess. None (default) uses auto-detect.

    precomputed: optional result dict already returned by
    extract_and_detect_events — reuses it instead of re-running extraction,
    for callers (the Streamlit UI) that already ran that stage to show the
    camera-angle confirmation before this function is called.
    """
    if precomputed is not None and precomputed.get("status") == "success":
        stage12 = precomputed
    else:
        stage12 = extract_and_detect_events(
            video_path, output_dir=output_dir,
            bowling_arm_override=bowling_arm_override,
            seed_point=seed_point, seed_frame_index=seed_frame_index,
            extra_seeds=extra_seeds, camera_angle_override=camera_angle_override,
        )
    if stage12["status"] != "success":
        return stage12

    df = stage12["df"]
    fps = stage12["fps"]
    bowling_arm = stage12["bowling_arm"]
    events = stage12["events"]

    # STAGE 3 — FRAME VALIDATION
    ffc_rows = df[df["frame"] == events["FFC"]]
    if ffc_rows.empty:
        return {
            "status": "failed",
            "stage": "frame_extraction",
            "message": (
                f"FFC frame {events['FFC']} not found in landmark data. "
                f"Video may be too short or landmarks dropped."
            )
        }
    ffc_row = ffc_rows.iloc[0]

    br_rows = df[df["frame"] == events["BR"]]
    if br_rows.empty:
        return {
            "status": "failed",
            "stage": "frame_extraction",
            "message": f"BR frame {events['BR']} not found in landmark data."
        }
    br_row = br_rows.iloc[0]
    lead_side = "left" if bowling_arm == "right" else "right"

    # STAGE 4 — METRIC CALCULATIONS
    knee_analysis = calculate_knee_bracing(ffc_row, lead_side=lead_side)
    knee_at_release = calculate_knee_bracing(br_row, lead_side=lead_side)
    lean_analysis = calculate_trunk_lean(br_row)
    head_stability = calculate_head_stability(df, events["BFC"], events["BR"])
    hip_separation = calculate_hip_shoulder_separation(df, events["FFC"])
    # Anchored on BR (the release frame — coach-confirmable, see the
    # Streamlit release-frame-confirmation step), not FFC. FFC's own
    # detected TIMING can be wrong for leaping bowlers (see
    # _find_grounded_reference_near's docstring) — but we don't actually
    # need FFC's timing to be right for this, we just need any nearby
    # frame where he's plausibly grounded, and searching near a verified
    # release frame is far more trustworthy than near an unverified one.
    height_reference_row = _find_grounded_reference_near(df, events["BR"], bowling_arm)
    if height_reference_row is None:
        # Genuinely nothing grounded nearby (e.g. he's airborne the whole
        # window) — fall back to the old FFC-based search rather than
        # silently having no reference at all.
        height_reference_row = _nearest_complete_row(
            df, events["FFC"], ["NOSE_y", "LEFT_ANKLE_y", "RIGHT_ANKLE_y"]
        )
    # Coach-confirmed wrist/ball position, when given, overrides the
    # tracked landmark entirely — see calculate_release_height_ratio_safe's
    # docstring for why (verified real, sustained MediaPipe mistracking
    # during a fast, blurred release swing that no automatic plausibility
    # filter could catch).
    wrist_override_x = events.get("wrist_override_x")
    wrist_override_y = events.get("wrist_override_y")
    wrist_override_norm = (
        (wrist_override_x, wrist_override_y)
        if wrist_override_x is not None and wrist_override_y is not None else None
    )
    release_height = calculate_release_height_ratio_safe(br_row, bowling_arm=bowling_arm,
                                                           reference_row=height_reference_row,
                                                           wrist_override_norm=wrist_override_norm)

    # FFC-to-Release knee angle delta ("yielding knee" check flagged in
    # external biomechanical audit): a static single-frame knee angle at
    # FFC can't show whether the knee then BENDS (yields) before release,
    # which is a real, separate coaching concern from the FFC angle alone.
    knee_delta = None
    knee_delta_status = "unavailable"
    if knee_analysis.get("status") == "success" and knee_at_release.get("status") == "success":
        knee_delta = round(knee_at_release["degrees"] - knee_analysis["degrees"], 1)
        knee_delta_status = "yielding" if knee_delta < -5.0 else ("braced" if knee_delta >= 0 else "minor_yield")

    # STAGE 5 — VIDEO GENERATION
    raw_output_video = os.path.join(output_dir, "annotated_raw.mp4")
    generate_fail_safe_video(video_path, raw_output_video, df, events, bowling_arm=bowling_arm,
                              camera_angle=stage12.get("camera_angle", "side_on"))
    web_safe_video_file = transcode_to_h264(raw_output_video)

    # STAGE 6 — SAFE KEY EXTRACTION
    # FIX: the old `.get(a) or .get(b) or .get(c)` pattern has a falsy-zero
    # bug — if a metric's real value is exactly 0.0 (e.g. a perfectly
    # upright 0.0-degree trunk lean, a genuinely ideal result), Python
    # treats 0.0 as falsy and the `or` chain incorrectly keeps searching,
    # silently turning a great result into None/N/A. Explicit None-checks
    # fix this — verified this is the actual key kinematics.py returns
    # ("degrees" for both trunk_lean and knee_bracing).
    def _first_non_none(d: dict, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
        return None

    trunk_lean_val = _first_non_none(lean_analysis, "trunk_lean_degrees", "degrees", "angle")
    knee_bracing_val = _first_non_none(knee_analysis, "front_knee_angle", "degrees", "angle")

    # ANATOMICAL PLAUSIBILITY GUARD (knee bracing only):
    # A human knee cannot physically be at ~0 degrees mid-delivery — that
    # would mean the joint folded completely in on itself. A near-zero
    # reading here almost always means the hip/knee/ankle landmarks
    # collapsed onto nearly the same point due to tracking failure (e.g.
    # on a very short/low-quality clip), and arccos(~1) returned ~0 as a
    # pure math artifact of that degeneracy, not a real measurement.
    # Confirmed against a real clip: this exact scenario produced "0.0°"
    # after the falsy-zero fix above started correctly passing through
    # real zero values — this guard distinguishes a genuine 0-degree
    # result (which never happens for THIS metric) from degenerate math.
    # NOTE: this threshold (5 degrees) is an engineering choice based on
    # basic human anatomy, not a cited biomechanics constant.
    # IMPORTANT: this guard is intentionally NOT applied to trunk_lean —
    # 0 degrees of trunk lean is a real, genuinely ideal result (a
    # perfectly upright bowler), so the same "near-zero is implausible"
    # logic would be wrong there.
    KNEE_ANGLE_IMPLAUSIBLE_THRESHOLD = 5.0  # degrees
    if knee_bracing_val is not None and knee_bracing_val < KNEE_ANGLE_IMPLAUSIBLE_THRESHOLD:
        knee_bracing_val = None

    # STAGE 7 — RETURN UNIFIED PAYLOAD
    return {
        "status": "success",
        "video_metadata": {
            "source_file": os.path.basename(video_path),
            "fps": fps,
            "total_frames": len(df)
        },
        "time_indices": {
            "back_foot_contact_frame": events["BFC"],
            "front_foot_contact_frame": events["FFC"],
            "ball_release_frame": events["BR"],
            "ball_release_confidence": events.get("BR_confidence", "high"),
            "ball_release_plausible_fraction": events.get("BR_plausible_fraction", 1.0),
            # Present only when the Streamlit UI overrode the auto-detected
            # BR with a coach-confirmed frame — real (auto, confirmed) label
            # pairs for Phase 2 training, not present for callers (CLI,
            # dual-camera) that don't run that confirmation step.
            "ball_release_frame_auto_detected": events.get("BR_auto_detected"),
            "ball_release_auto_confidence": events.get("BR_auto_confidence"),
            "front_foot_contact_frame_auto_detected": events.get("FFC_auto_detected"),
            "back_foot_contact_frame_auto_detected": events.get("BFC_auto_detected"),
        },
        "biomechanical_metrics": {
            "trunk_lean": {
                "degrees": trunk_lean_val,
                "tier": (lean_analysis.get("classification") or
                         lean_analysis.get("tier") or "Unknown"),
                "status": lean_analysis.get("status", "error"),
                "critique": lean_analysis.get("critique", "N/A")
            },
            "front_knee_bracing": {
                "degrees": knee_bracing_val,
                "tier": (knee_analysis.get("classification") or
                         knee_analysis.get("tier") or "Unknown"),
                "status": knee_analysis.get("status", "error"),
                "critique": knee_analysis.get("critique", "N/A"),
                "degrees_at_release": knee_at_release.get("degrees"),
                "yield_delta_degrees": knee_delta,
                "yield_status": knee_delta_status
            },
            "hip_shoulder_separation": hip_separation,
            "bowling_arm_detected": bowling_arm,
            "release_height": {
                "ratio": release_height.get("ratio"),
                "classification": (release_height.get("classification") or
                                    release_height.get("tier") or "Unknown"),
                "status": release_height.get("status", "error"),
                "debug_raw": release_height.get("debug_raw")
            },
            "head_stability": {
                "value": (head_stability.get("deviation_index") or
                          head_stability.get("value")),
                "classification": (head_stability.get("tier") or
                                    head_stability.get("classification") or
                                    "Unknown"),
                "status": head_stability.get("status", "error")
            }
        },
        "annotated_video_output": web_safe_video_file.replace("\\", "/")
    }