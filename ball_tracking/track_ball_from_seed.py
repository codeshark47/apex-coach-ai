"""
ball_tracking/track_ball_from_seed.py

Seeded ball tracker: starts from a human-confirmed (frame, x, y) — the
same trust model already used for bowler identity (main.py's
_walk_from_seed) and every other "coach click is ground truth" step in
this app — and follows the ball forward using a LOCAL search window
around each frame's predicted position, instead of scanning the whole
frame with a general detector.

WHY LOCAL SEARCH, NOT WHOLE-FRAME: confirmed directly on real footage
(2026-08-14) that whole-frame classical detection and unconstrained YOLO
inference both get dominated by the bowler's own moving body — an
elbow, a shoulder, a bright uniform edge all out-compete the much
smaller ball as "the most confident detection in the frame." A small
crop around the ball's own predicted next position structurally cannot
contain something as far away as an umpire's hat or the bowler's far
shoulder, which is what actually fixes the confusion — not a smarter
classifier, a smaller search space.

SIZE-TREND CONSISTENCY (2026-08-14, real coach observation): for a
camera positioned behind the bowling stumps, the ball is receding away
from camera in real 3D space for the whole flight — its on-screen size
shrinks frame to frame, smoothly, the same physical fact that makes its
POSITION move smoothly. A detection whose size jumps around instead of
following that trend is exactly the kind of thing that isn't really the
ball, even if a bare confidence score alone can't tell — position-only
scoring already proved too permissive on its own (see the 2026-08-14
validation: an accepted candidate jumped 50px in one frame against a
real ball moving ~2-6px between frames at this point in its flight).
Size gives a second, independent physical signal noise is unlikely to
satisfy by coincidence. Positive per-frame size delta = shrinking
(behind-the-bowler framing); negative = growing (behind-the-keeper,
ball approaching camera) — direction is inferred from the seed's own
first couple of confirmed observations, not assumed.

HONEST STATUS: validated against a short real human-labeled sequence
(2026-08-14) — that test found the underlying detector doesn't yet have
a reliable signal on the ball right at the release instant (smallest,
most motion-blurred moment), independent of this tracker's own logic.
Re-validate this same way every time the underlying model changes.
"""

import cv2
import numpy as np


def track_ball_from_seed(
    video_path: str,
    seed_frame: int,
    seed_xy: tuple,
    yolo_model,
    seed_size: float = None,
    max_frames_forward: int = 30,
    search_radius_start: float = 250.0,
    search_radius_growth: float = 60.0,
    search_radius_cap: float = 400.0,
    max_gap: int = 3,
    conf_threshold: float = 0.02,
    size_trend_tolerance: float = 0.6,
) -> dict:
    """
    Tracks forward from (seed_frame, seed_xy) — a coach's confirmed
    click — up to max_frames_forward frames or until the ball can't be
    found for max_gap consecutive frames (graceful stop, same pattern
    as main.py's identity walk — never fabricates a position it isn't
    reasonably confident of).

    seed_xy: (x, y) in ORIGINAL frame pixel coordinates.

    seed_size: optional starting ball diameter in px (e.g. from the
    coach's own radius calibration in label_tool.py, doubled). Only
    used once a real trend exists — a single starting size alone can't
    establish a direction (shrinking vs growing), so the first
    confirmed detection after the seed is still accepted on
    position+confidence alone; the size check only engages from the
    SECOND tracked point onward, once there are two real size samples
    to compare.

    search_radius grows the longer the trail goes unconfirmed (same
    reasoning as every other growing-tolerance walk in this app) but is
    capped — a real ball's frame-to-frame displacement is bounded by
    real physics at a given fps/distance, unlike, say, an identity walk
    that might need to tolerate longer gaps.

    DEFAULTS RAISED (2026-08-15, real validation against a long, dense
    human-labeled sequence — VID_20260411_092805.mp4, 24 real ground-
    truth points spanning a full delivery): the original defaults
    (search_radius_start=50) came from reasoning about typical frame-to-
    frame motion, not a measured one. Confirmed directly: the real ball
    moved ~173px in the first 2 frames after this clip's seed — a fast
    release, with no velocity estimate yet to compensate (velocity is
    only learned AFTER the first successful detection), so the search
    crop right after a seed needs real margin, not the smaller radius
    that's fine once a trend is established. At the old defaults, this
    real sequence failed after the seed alone; at these, it correctly
    picked up frames 23 and 25 within ~5-8px of the true (interpolated)
    position. Still an HONEST, PARTIAL result, not a fixed tracker: the
    same real test lost the trail again after frame 25 of a 74-frame
    flight — the search-radius bug was real and worth fixing, but it
    was never the ONLY limitation; the underlying detector's own recall
    across a full flight is the remaining, larger gap (see the project
    memory on ball-tracking strategy for the current, honest state).

    conf_threshold: LOWER than label_tool.py's AI_PREFILL_CONF_THRESHOLD
    (0.5) deliberately — this only searches a small crop near a
    physically-plausible position, which already does most of the work
    a high confidence bar exists for elsewhere (rejecting far-away,
    unrelated objects). The size-trend check below is what allows this
    to stay low without accepting noise, rather than raising the bar
    and losing real-but-uncertain detections instead. Lowered further
    (0.15->0.02) alongside the radius change above — the same real
    sequence's genuine detections at frames 23/25 scored only 0.05-0.23,
    below the old floor.

    size_trend_tolerance: fractional deviation from the expected
    (trend-extrapolated) size still accepted — e.g. 0.6 allows a
    candidate anywhere from 40% to 160% of the predicted size. Loose on
    purpose: real detections' box sizes are themselves noisy frame to
    frame (motion blur, partial occlusion), this only needs to catch a
    candidate that's obviously a different, larger/smaller object, not
    police normal measurement noise.

    Returns {"status": "success", "points": [(frame, x, y, conf, size), ...]}
    (points[0] is the seed itself, conf=1.0, size=seed_size or None)
    or {"status": "error", "message": ...}.
    """
    if not video_path or seed_xy is None:
        return {"status": "error", "message": "seed_xy and video_path are required."}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": f"Could not open video: {video_path}"}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    points = [(seed_frame, float(seed_xy[0]), float(seed_xy[1]), 1.0, seed_size)]
    last_xy = (float(seed_xy[0]), float(seed_xy[1]))
    velocity = (0.0, 0.0)  # (dx, dy) per frame, updated once 2+ points are confirmed
    last_size = seed_size
    size_velocity = None  # per-frame size delta once 2+ real size samples exist
    gap = 0
    radius = search_radius_start

    end_frame = min(total_frames - 1, seed_frame + max_frames_forward)
    for frame_idx in range(seed_frame + 1, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        pred_x = last_xy[0] + velocity[0]
        pred_y = last_xy[1] + velocity[1]
        # BUG FOUND (2026-08-14, real validation against ground truth): a
        # trend-extrapolated expected_size requires size_velocity, which
        # doesn't exist until the SECOND real detection — meaning the
        # very first tracked frame after the seed had no size check at
        # all, even when seed_size was given, and that's exactly the
        # frame a real test caught accepting an 87%-too-small candidate.
        # Falls back to last_size alone (a flat "still roughly this
        # size" check) until a real trend exists, instead of skipping
        # the check entirely for that one critical frame.
        if last_size is not None:
            expected_size = last_size + size_velocity if size_velocity is not None else last_size
        else:
            expected_size = None
        h, w = frame_bgr.shape[:2]
        x1 = int(max(0, pred_x - radius))
        y1 = int(max(0, pred_y - radius))
        x2 = int(min(w, pred_x + radius))
        y2 = int(min(h, pred_y + radius))

        found = None
        if x2 > x1 and y2 > y1:
            crop = frame_bgr[y1:y2, x1:x2]
            # YOLO expects BGR raw arrays the same way cv2 itself reads
            # them — real bug found in label_tool.py (2026-08-04) from
            # feeding it an RGB-converted frame by mistake, silently
            # swapping red/blue for every prediction. Reading directly
            # via cv2 above already keeps this in BGR, unchanged.
            results = yolo_model.predict(crop, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
            candidates = []
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes.xyxy[i].tolist()
                cand_size = ((bx2 - bx1) + (by2 - by1)) / 2
                cand_conf = boxes.conf[i].item()
                if expected_size is not None and expected_size > 0:
                    deviation = abs(cand_size - expected_size) / expected_size
                    if deviation > size_trend_tolerance:
                        continue  # fails the physical size-trend check — not the ball
                candidates.append((x1 + (bx1 + bx2) / 2, y1 + (by1 + by2) / 2, cand_conf, cand_size))
            if candidates:
                # Among candidates that passed the size-trend gate (or all
                # of them, if no trend exists yet to check against), the
                # most CONFIDENT one is still the best available signal.
                found = max(candidates, key=lambda c: c[2])

        if found is not None:
            fx, fy, fconf, fsize = found
            velocity = (fx - last_xy[0], fy - last_xy[1])
            if last_size is not None:
                size_velocity = fsize - last_size
            last_xy = (fx, fy)
            last_size = fsize
            points.append((frame_idx, fx, fy, fconf, fsize))
            gap = 0
            radius = search_radius_start
        else:
            gap += 1
            radius = min(search_radius_cap, radius + search_radius_growth)
            if gap > max_gap:
                break
            # Coast forward on the last known velocity while within the
            # gap tolerance, so a brief miss doesn't reset the predicted
            # position back to a stale point — same idea as main.py's
            # identity walk carrying its anchor through a short gap.
            last_xy = (last_xy[0] + velocity[0], last_xy[1] + velocity[1])
            if last_size is not None and size_velocity is not None:
                last_size = last_size + size_velocity

    cap.release()
    return {"status": "success", "points": points}
