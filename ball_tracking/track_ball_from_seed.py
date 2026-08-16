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

import math

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
    max_gap: int = 8,
    conf_threshold: float = 0.02,
    size_trend_tolerance: float = 0.6,
    recovery_gap_threshold: int = 2,
    recovery_radius: float = 500.0,
    recovery_max_speed_ratio: float = 3.0,
    min_baseline_speed: float = 5.0,
    stagnation_radius: float = 0.0,
    stagnation_window: int = 4,
    proximity_scale: float = 60.0,
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

    RECOVERY MODE (2026-08-15, real bounce found by tracing frames 43-50
    of the same benchmark sequence after the anchor/velocity fix above):
    the ball's trajectory reverses direction in image-Y right around
    frame 40-42 in this real clip — a bounce, physically impossible for
    any constant-velocity model to represent. Confirmed directly: once
    the linear extrapolation starts missing, the predicted search center
    drifts 50-190px from the true position within a handful of frames,
    since it keeps assuming the ball continues in its PRE-bounce
    direction. Once `gap >= recovery_gap_threshold` consecutive misses
    happen, this stops trusting that drifting prediction: the search
    crop re-centers on the last CONFIRMED real position (anchor_xy, not
    an extrapolated guess) with a wide fixed `recovery_radius`, and
    candidate selection drops the proximity-to-prediction weighting
    (meaningless once the "prediction" is just an anchor point, not a
    real guess) in favor of confidence + the size-trend check alone.
    `max_gap` raised 3->8 to give recovery mode enough attempts to
    actually re-acquire — the real bounce here took 4 consecutive misses
    (frames 43/45/46/47 all found nothing) before a real candidate
    reappeared.

    Still an HONEST, PARTIAL fix, not a solved tracker: this recovers
    from the DRIFT recovery mode is built for, but a real bounce also
    changes the ball's on-screen size/motion character in ways this
    hasn't been separately validated against — re-run the full-
    trajectory benchmark after any change here, same as always.

    stagnation_radius / stagnation_window (2026-08-16, real bug found
    from a coach's first live admin-panel test, on a clip outside the
    original benchmark): a NEW static-lock-on shape the speed-ratio
    check above doesn't cover. Traced the exact raw sequence — frames
    107-110 were genuine (smooth, 0.3-0.5 confidence), then frame 111
    jumped onto a background object and the tracker sat there for 29
    STRAIGHT frames (112-139), oscillating within 1-2px, confidence
    collapsed to 0.02-0.11. The speed-ratio filter didn't catch entry
    into it (the jump itself wasn't fast enough relative to the recent
    trend to look implausible — the problem wasn't speed, it was
    direction). But nothing catches what happens AFTER: a real ball in
    flight is never in the same few square pixels for several
    consecutive real frames — this checks exactly that, terminating the
    trail (not coasting through it) once it fires.

    FIRST VERSION USED A SINGLE-STEP THRESHOLD AND REGRESSED A REAL
    CLIP: comparing each new point only to the immediately-previous one
    looked right on the stuck clip, but broke a DIFFERENT, already-
    validated clip — a real ball naturally decelerating near an apex in
    that delivery had individual steps shrink to 3.5-6px, which a
    single-step 5px cutoff couldn't tell apart from genuine stagnation.
    Fixed by checking NET displacement across a short WINDOW of
    consecutive real detections instead of one step at a time: the truly
    stuck case still barely moves at all across 4 real frames (~1px net,
    confirmed on the same raw data), while the decelerating-but-real
    case covers real net distance across the same span (~13-31px,
    confirmed on the regression clip) even as its individual steps
    shrink. Net-over-a-window is what actually separates "slowing down"
    from "stopped," not any single frame-to-frame distance.

    Deliberately NOT paired with a hard directional/angle constraint
    (Gemini's other proposed check, "reject any candidate inconsistent
    with downward flight toward the pitch") — the PRIMARY benchmark's
    own verified ground truth includes the ball genuinely moving UP in
    image-space early in flight (already confirmed accurate against
    real labels), so a rigid down-and-forward rule would have rejected
    real, already-proven motion. Windowed stagnation alone explains and
    catches this specific failure without that risk.

    DISABLED BY DEFAULT (stagnation_radius=0.0) AS OF 2026-08-16, SAME
    DAY — proven wrong on real ground truth, not just a hunch. THIS
    SAME CLIP (VID_20260815_075824.mp4) turned out to already have 16
    real coach-labeled frames in the database — never checked before
    building or defending this filter. Converting them to this frame's
    pixel scale: frames 120-144 (24 real frames) sit in a tight
    (198-210, 288-308) cluster — the EXACT region this filter had been
    rejecting as "the static object" through two rounds of Gemini
    exchanges. Confirmed directly (whole-frame scan, sorted by distance
    to the real label): the closest candidate to the true position at
    every one of those frames is within 3.8-21.7px and matches the same
    size/confidence profile as what was being called a false positive —
    there is no separate, different real-ball candidate being missed.
    It's the same detection. The filter's core assumption — a real ball
    is never in the same few pixels for several consecutive frames — is
    simply FALSE for this camera framing: a ball traveling nearly along
    the camera's own optical axis (toward/away from it, not laterally
    across frame) has real, fast 3D motion but can legitimately show
    almost no lateral image-space displacement for an extended real
    stretch. Position alone cannot tell that apart from a truly static
    background object (confirmed separately, by direct visual pixel
    inspection, for the primary benchmark's frames 46/66 — that case
    really was static background, so this isn't "the whole idea was
    wrong," it's "position alone isn't sufficient evidence, camera-angle-
    dependent, and this project doesn't have a reliable second signal
    for it yet"). Tried size-trend (a real ball approaching camera
    should show real growth) as a second signal on this same real data —
    too noisy at this scale (5.6-7.4px, so +/-1px is a large relative
    swing) to safely threshold on. Left the mechanism in the code
    (calling it with a real stagnation_radius still works) rather than
    deleting it, since it may still be useful once a more reliable
    depth/size signal exists — just not trustworthy enough to ship
    enabled by default when the cost of being wrong is silently
    discarding real, correct tracking.

    recovery_max_speed_ratio (2026-08-15, real bug found by tracing
    frames 46/66 on the same benchmark, Gemini-prompted): recovery
    mode's wide radius fixed the bounce but opened a narrower new hole —
    a candidate can sit well within recovery_radius (336px, under the
    400px cap by then) while still being a persistent STATIC false
    positive (confirmed directly: frames 46 and 66 landed within 6px of
    each other, and of a third bad candidate rejected earlier at frame
    62 — the same fixed background feature, not the ball, getting
    detected repeatedly whenever the wide crop includes it). Distance
    alone can't tell that apart from a real, fast post-bounce ball. What
    can: reaching a static point implies a sudden, large SPEED relative
    to the ball's own last established rate — frame 46's implied speed
    was ~6x the real rate measured just 4 frames earlier, while the
    genuine bounce-recovery at frame 29 only implied ~0.7x (comfortably
    within normal variation). Rejecting recovery candidates whose
    implied speed exceeds `recovery_max_speed_ratio` times the last real
    velocity's magnitude catches the former without the latter — a
    physical continuity check, not a fixed pixel constant (same reason
    the distance-reject above uses current_radius instead of Gemini's
    original fixed 30px). Only applies once a real velocity exists
    (skipped for the first tracked point after the seed, same reasoning
    as expected_size above) — there's nothing yet to compare against.

    min_baseline_speed (2026-08-16, real bug found tracing clip3 — a
    clip outside the frame-46/66 benchmark above, where the ball travels
    almost straight along the camera's own optical axis): the ratio
    above divides by the ball's LAST established speed, which is fine
    when that speed is a normal in-flight rate but breaks down when it's
    near-zero. On this clip the ball legitimately had near-zero LATERAL
    velocity for a real physical reason (same effect documented for the
    disabled stagnation filter above — motion toward the camera doesn't
    show up as image-plane displacement), so the established rate right
    before the gap was 0.32px/frame. Every real subsequent detection —
    genuinely only ~4px/frame away — then measured as a 96x-124x "speed
    spike" against that near-zero baseline and was rejected, even though
    nothing about it was actually implausible. Flooring last_speed at
    min_baseline_speed (default 5px/frame, roughly one ball-width on
    this clip's scale) before computing the ratio fixes the near-zero
    blowup while leaving the original frame-46/66 rejections untouched —
    those had real established speeds well above this floor already, so
    max(last_speed, floor) there just returns last_speed unchanged.

    proximity_scale (2026-08-16, real bug found tracing frame 111 on a
    clip outside prior testing — see the stagnation notes above for the
    same clip): the OLD proximity weighting divided distance by
    current_radius (250-400px), a linear falloff so gentle that a real
    ball candidate 14.8px from the prediction (conf 0.025) still LOST to
    a static false positive 26.9px away (conf 0.041) — the confidence
    gap alone was enough to win even though the real ball was clearly
    the closer, more physically plausible candidate. Confirmed directly:
    the real ball WAS in the candidate list at frame 111, low confidence
    but present — this was a scoring-formula bug, not a detector recall
    failure the way it first looked. Fixed with a smaller, QUADRATIC
    falloff (proximity_scale=60px, squared instead of linear) so nearby
    candidates are preferred much more strongly at exactly the pixel
    distances that separate a real ball from a nearby false positive,
    without needing a hard cutoff — a dramatically more confident but
    farther candidate can still win, same as before, just not from a
    small, noisy confidence edge alone. Deliberately its own parameter,
    not reusing current_radius (250-500px) — that stays the outer hard-
    reject boundary (see the distance-reject note above), a completely
    different question ("could this possibly be real at all") from this
    one ("which of several plausible candidates is most likely real").

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
    # ANCHOR + PER-FRAME VELOCITY (2026-08-15, two real bugs found by
    # tracing an exact failure on the full-trajectory benchmark): anchor_xy/
    # anchor_frame are the position/frame of the LAST REAL detection (never
    # a coasted guess), and velocity is always a true per-frame rate.
    #
    # BUG 1 — velocity wasn't scaled by elapsed frames: the old code did
    # `velocity = (fx - last_xy[0], fy - last_xy[1])` where last_xy could
    # be several frames stale (after a multi-frame gap), so a real 3-frame
    # displacement got used AS a 1-frame velocity — a 3x overshoot fed
    # straight into the next frame's prediction. Traced directly on real
    # ground truth: after a gap from frame 20 to 23, this alone was enough
    # to send the next frame's search crop centered on the wrong spot.
    #
    # BUG 2 — coasting mutated the anchor itself: the old `last_xy = last_xy
    # + velocity` on every miss meant a run of misses compounded prediction
    # error into the position used to compute the NEXT real velocity too,
    # not just that frame's search center. Fixed by always predicting from
    # the fixed anchor + velocity * elapsed, never from a walked/mutated
    # running position — one clean extrapolation, not N accumulated ones.
    anchor_xy = (float(seed_xy[0]), float(seed_xy[1]))
    anchor_frame = seed_frame
    velocity = (0.0, 0.0)  # (dx, dy) per frame, updated once 2+ points are confirmed
    # SAME ANCHOR PRINCIPLE AS POSITION, APPLIED TO SIZE (2026-08-16, real
    # bug found tracing why clip3 stops at frame 111 even with stagnation
    # disabled and max_gap raised): expected_size used to compound
    # size_velocity once per MISSED frame (`last_size = last_size +
    # size_velocity` in the miss branch), the exact same per-miss-
    # mutation bug already fixed for position earlier tonight, just never
    # applied here. Confirmed directly: a real -0.38px/frame size trend,
    # compounded over the real 9-frame gap to frame 120, decayed the
    # expected size from 4.9px to 1.46px — so the real ball's actual
    # 7.19px size failed the size-trend tolerance check by 6x, rejected
    # before distance/speed checks even mattered. Fixed the same way as
    # position: anchor_size/anchor_size_frame only update on real
    # detections; expected size is always anchor_size + size_velocity *
    # elapsed, computed fresh each frame, never an accumulated value.
    anchor_size = seed_size
    anchor_size_frame = seed_frame
    size_velocity = None  # per-frame size delta once 2+ real size samples exist
    gap = 0
    radius = search_radius_start
    recent_real_positions = [(seed_frame, float(seed_xy[0]), float(seed_xy[1]))]  # rolling window, see stagnation docstring

    end_frame = min(total_frames - 1, seed_frame + max_frames_forward)
    for frame_idx in range(seed_frame + 1, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        in_recovery = gap >= recovery_gap_threshold
        if in_recovery:
            # See this function's RECOVERY MODE docstring section — the
            # linear extrapolation has already missed enough times that
            # trusting it further just searches deeper into the wrong
            # area. Re-anchor on the last CONFIRMED real position instead
            # of a compounding guess about where the ball's gone.
            #
            # RADIUS ALSO GROWS PROGRESSIVELY IN RECOVERY (2026-08-15, real
            # regression found immediately after adding the outlier-reject
            # filter above): jumping straight to the full recovery_radius
            # the moment recovery mode starts let a confident false
            # positive 393px from the last real position win on just the
            # THIRD frame since that anchor — physically implausible for
            # this clip's real per-frame speed even accounting for a
            # bounce (a bounce changes direction, it doesn't teleport the
            # ball). `radius` already grows on every miss regardless of
            # mode; reusing it here (capped at recovery_radius) means early
            # recovery attempts stay appropriately tight and only widen
            # toward the full recovery radius the longer the ball stays
            # lost — same principle as the outlier-reject filter, just
            # applied to recovery mode's own search too.
            pred_x, pred_y = anchor_xy
            current_radius = min(recovery_radius, radius)
        else:
            elapsed = frame_idx - anchor_frame
            pred_x = anchor_xy[0] + velocity[0] * elapsed
            pred_y = anchor_xy[1] + velocity[1] * elapsed
            current_radius = radius
        # BUG FOUND (2026-08-14, real validation against ground truth): a
        # trend-extrapolated expected_size requires size_velocity, which
        # doesn't exist until the SECOND real detection — meaning the
        # very first tracked frame after the seed had no size check at
        # all, even when seed_size was given, and that's exactly the
        # frame a real test caught accepting an 87%-too-small candidate.
        # Falls back to anchor_size alone (a flat "still roughly this
        # size" check) until a real trend exists, instead of skipping
        # the check entirely for that one critical frame.
        # RECOVERY MODE STOPS TREND-EXTRAPOLATING SIZE TOO (2026-08-16, real
        # bug found tracing clip3 past frame 111 even AFTER the anchor_size
        # fix above eliminated per-miss compounding): a size trend is
        # estimated from a SINGLE interval between two real detections — one
        # noisy sample, not an averaged rate. Extrapolating that one sample
        # forward is fine for a frame or two, but over a real 9-frame gap it
        # drove expected_size from 4.9px to 1.46px on this clip while the
        # ball's actual real size stayed ~7px, rejecting the true candidate
        # at frame 120 by a ~4x deviation — same shape of bug as the
        # position predictor already fixed for recovery mode above ("the
        # linear extrapolation has already missed enough times that trusting
        # it further just searches deeper into the wrong area"), just never
        # applied to size. Once in recovery, fall back to anchor_size flat
        # (a "still roughly this size" check, same reasoning as the
        # no-second-sample-yet fallback below) instead of riding a
        # one-sample trend further and further from anything real.
        if anchor_size is not None:
            if in_recovery:
                expected_size = anchor_size
            else:
                size_elapsed = frame_idx - anchor_size_frame
                expected_size = anchor_size + size_velocity * size_elapsed if size_velocity is not None else anchor_size
        else:
            expected_size = None
        h, w = frame_bgr.shape[:2]
        x1 = int(max(0, pred_x - current_radius))
        y1 = int(max(0, pred_y - current_radius))
        x2 = int(min(w, pred_x + current_radius))
        y2 = int(min(h, pred_y + current_radius))

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
                cand_x = x1 + (bx1 + bx2) / 2
                cand_y = y1 + (by1 + by2) / 2
                # HARD DISTANCE OUTLIER REJECT (2026-08-15, real gap found
                # tracing frame 28 on the benchmark, Gemini-prompted): the
                # distance-weighted scoring below only helps when there's
                # an ALTERNATIVE candidate to prefer over a far-off one —
                # frame 28 had exactly ONE candidate in the whole crop
                # (356px from the prediction, near a corner of the search
                # square), so max() over a single-element list picked it
                # regardless of how implausible the distance was. A candidate
                # outside the search radius's own INSCRIBED CIRCLE (square
                # crop corners are up to radius*sqrt(2) away, but the crop
                # was only ever meant to mean "within radius px") is
                # rejected outright here, not just down-weighted — never
                # accepted as a real detection even with nothing else to
                # compare it to. NOT a fixed pixel constant (Gemini's first
                # draft proposed 30px) — that would have broken the
                # already-validated 173px real jump documented above right
                # after the seed; scaling with current_radius (which
                # already reflects how much uncertainty this frame has) is
                # what makes this work at every stage of the walk, in
                # recovery mode too now that its own radius also grows
                # progressively instead of jumping straight to the max
                # (see the recovery-mode radius comment above — without
                # this, a confident false positive 393px from the last
                # real position won on the third frame of a real recovery
                # attempt, physically implausible for this clip's actual
                # per-frame speed).
                dist_from_pred = math.hypot(cand_x - pred_x, cand_y - pred_y)
                if dist_from_pred > current_radius:
                    continue
                # SPEED-RATIO REJECT (2026-08-15, real bug found tracing
                # frames 46/66, then WIDENED after tracing frame 59 —
                # see this function's docstring). Originally recovery-mode
                # only, but frame 59 proved a normal-mode candidate can
                # ALSO be an implausible-velocity false positive (a small
                # dark blob on the background gate, confirmed by eye) that
                # then corrupts the velocity baseline the NEXT recovery
                # check compares against — the static tree/net object at
                # frame 62/66 slipped through specifically because its
                # ratio was computed against that already-inflated
                # baseline. Applying this check universally (not just in
                # recovery) closes that hole at the source instead of
                # letting a bad normal-mode hit poison every check after
                # it. Only fires once a real velocity exists (nothing to
                # compare against on the very first tracked point).
                if velocity[0] != 0.0 or velocity[1] != 0.0:
                    elapsed_here = frame_idx - anchor_frame
                    implied_speed = math.hypot(cand_x - anchor_xy[0], cand_y - anchor_xy[1]) / elapsed_here
                    last_speed = math.hypot(velocity[0], velocity[1])
                    # MIN_BASELINE_SPEED FLOOR (2026-08-16, real bug found
                    # tracing clip3 past frame 111 — see docstring): this
                    # ratio divides by last_speed, so when the ball's last
                    # established rate is itself near-zero the check
                    # collapses — ANY subsequent real motion reads as a huge
                    # multiple of ~nothing. That's exactly what happened
                    # here: a ball travelling nearly along the camera's own
                    # optical axis has near-zero LATERAL velocity for real,
                    # physical reasons (same effect already documented for
                    # the disabled stagnation filter above), so the "last
                    # real velocity" anchor at frame 111 was 0.32px/frame —
                    # and the real ball's next genuine detection, ~4px/frame
                    # away, was rejected as a 96x-124x "speed spike" that
                    # was never actually implausible. Flooring last_speed at
                    # min_baseline_speed before dividing keeps this check
                    # doing its real job (catching a jump to a static
                    # object, still ~30-40x over any reasonable per-frame
                    # rate) without letting a legitimately-near-zero
                    # baseline make every subsequent frame look impossible.
                    effective_last_speed = max(last_speed, min_baseline_speed)
                    if implied_speed > effective_last_speed * recovery_max_speed_ratio:
                        continue
                candidates.append((cand_x, cand_y, cand_conf, cand_size))
            if candidates:
                # DISTANCE-WEIGHTED SELECTION (2026-08-15, real bug found by
                # tracing frame 28 of the benchmark): the physics prediction
                # was actually close to the true ball there, but the OLD
                # code picked whichever candidate had the highest raw
                # confidence, full stop — so a higher-confidence false
                # positive 195px away won over the real ball sitting near
                # the prediction. A real ball's position is constrained by
                # physical continuity; a false positive's isn't, so
                # proximity to the prediction is itself real evidence, not
                # just a tiebreaker. combined_score multiplies confidence by
                # a proximity factor (1.0 at the predicted point, falling to
                # a 0.05 floor at the edge of the search radius and beyond)
                # so spatial continuity can outweigh a raw confidence edge,
                # without letting one lucky-but-wrong high-confidence pixel
                # blob win just because nothing else fired as strongly.
                # In recovery mode, pred_x/pred_y is just the last known
                # real anchor, not a real guess about the current frame —
                # weighting by distance to it would penalize exactly the
                # post-bounce candidates recovery mode exists to find.
                # Confidence + the size-trend check (already applied
                # above) are the only trustworthy signals here.
                def _combined_score(c):
                    cx, cy, cconf, _ = c
                    if in_recovery:
                        return cconf
                    dist = math.hypot(cx - pred_x, cy - pred_y)
                    # QUADRATIC, TIGHTER-SCALE proximity (2026-08-16) —
                    # see proximity_scale in this function's docstring.
                    # current_radius (250-500px) is still the hard-reject
                    # boundary elsewhere; this is a separate, much
                    # smaller scale specifically for RANKING candidates
                    # that already passed that boundary.
                    proximity = max(0.05, 1.0 - dist / proximity_scale) ** 2
                    return cconf * proximity

                found = max(candidates, key=_combined_score)

        if found is not None:
            fx, fy, fconf, fsize = found
            # STAGNATION REJECT (2026-08-16) — see this function's
            # docstring. NET displacement across a short WINDOW of
            # consecutive real detections, not a single-step distance —
            # a single-step threshold regressed a real clip where the
            # ball genuinely decelerates near an apex (individual steps
            # shrink to a few px even though real net motion continues).
            # A truly stuck static object barely moves at all across the
            # whole window; real deceleration still covers real distance
            # over the same span even as it slows down.
            recent_real_positions.append((frame_idx, fx, fy))
            if len(recent_real_positions) > stagnation_window:
                recent_real_positions.pop(0)
            if len(recent_real_positions) == stagnation_window:
                _, ox, oy = recent_real_positions[0]
                if math.hypot(fx - ox, fy - oy) < stagnation_radius:
                    break
            elapsed_at_hit = frame_idx - anchor_frame
            velocity = ((fx - anchor_xy[0]) / elapsed_at_hit, (fy - anchor_xy[1]) / elapsed_at_hit)
            if anchor_size is not None:
                size_elapsed_at_hit = frame_idx - anchor_size_frame
                size_velocity = (fsize - anchor_size) / size_elapsed_at_hit
            anchor_xy = (fx, fy)
            anchor_frame = frame_idx
            anchor_size = fsize
            anchor_size_frame = frame_idx
            points.append((frame_idx, fx, fy, fconf, fsize))
            gap = 0
            radius = search_radius_start
        else:
            gap += 1
            radius = min(search_radius_cap, radius + search_radius_growth)
            if gap > max_gap:
                break
            # No coasting mutation needed for position OR size — next
            # iteration's pred_x/pred_y and expected_size both extrapolate
            # fresh from their fixed anchors over the growing elapsed
            # count, see above (same principle applied to both now).

    cap.release()
    return {"status": "success", "points": points}
