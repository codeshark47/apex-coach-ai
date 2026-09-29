"""
ball_tracking/pitch_calibration.py

Converts pixel positions in the fixed behind-the-stumps camera framing
(see detect_ball_classical.py / track_ball_candidates.py) into real-world
distances, using the stumps themselves as the only reliable known-size
reference object in frame — no separate calibration step or checkerboard
needed, since a fixed cricket setup already has two of them (near and far
stumps) at a known real distance apart (20.12m / 22 yards, the standard
popping-crease-to-popping-crease pitch length) and a known real width
(22.86cm / 9 inches, the standard stump line width).

HONEST LIMITATION — read before trusting any output from this file: this
builds a GROUND-PLANE homography (cv2.getPerspectiveTransform on 4 known
ground-level points). It has no way to know how far ABOVE the ground a
given pixel actually is — a ball at head height and a ball resting on the
pitch at the same image position would project to the same "ground"
point. A ball in real flight is elevated for almost its entire visible
trajectory, so any distance/speed computed this way is a real, honest
approximation with a real, uncorrected bias, not a validated
measurement — same standard already held for every other estimate in
this app, see the project memory on accuracy claims. Fixing this
properly needs either a second camera (stereo) or a physics model of the
ball's height at each point (which needs its OWN calibration this file
doesn't attempt). Do not present a number from this file as an accurate
speed to a coach without that caveat attached.

Stump pixel coordinates must be re-measured for each new camera position
(see the grid-overlay method used to measure them for the reference clip
this was built and validated against — not a universal constant).
"""

import numpy as np
import cv2
from scipy.optimize import minimize_scalar, least_squares

PITCH_LENGTH_M = 20.12  # popping crease to popping crease, standard
STUMP_LINE_WIDTH_M = 0.2286  # 9 inches, outer edge to outer edge
STUMP_HEIGHT_M = 0.7112  # 28 inches, standard stump height above ground
GRAVITY_M_S2 = 9.81


def build_ground_homography(near_left_px, near_right_px, far_left_px, far_right_px):
    """
    near_left_px / near_right_px / far_left_px / far_right_px: (x, y) pixel
    coordinates of the outer edges of the near and far stump lines, at
    GROUND level (where the stumps meet the pitch, not their tops).

    Returns a 3x3 homography mapping image pixel (x, y, 1) to real-world
    ground-plane (X, Y, 1) in meters, where Y=0 is the near stumps line,
    Y=PITCH_LENGTH_M is the far stumps line, and X=0 is the pitch
    centerline (positive X = toward the near-right/far-right stump).
    """
    half_w = STUMP_LINE_WIDTH_M / 2
    src = np.array([near_left_px, near_right_px, far_left_px, far_right_px], dtype=np.float32)
    dst = np.array([
        [-half_w, 0.0],
        [half_w, 0.0],
        [-half_w, PITCH_LENGTH_M],
        [half_w, PITCH_LENGTH_M],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def pixel_to_ground(homography, x_px, y_px):
    """
    Projects one image pixel onto the ground-plane homography. Returns
    (X_m, Y_m) — see build_ground_homography's docstring for axes.
    GROUND-PLANE ASSUMPTION APPLIES — see this module's docstring.
    """
    pt = np.array([[[float(x_px), float(y_px)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, homography)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def estimate_speed_kmh(homography, points, fps):
    """
    points: [(frame_index, x_px, y_px), ...] from a single trusted track
    (e.g. a BallTrack's .candidates), in increasing frame order.
    fps: the source video's real frame rate.

    Returns {"speed_kmh": ..., "distance_m": ..., "duration_s": ...,
    "ground_points": [(X_m, Y_m), ...]} using straight-line ground-plane
    distance between the first and last point, over the real elapsed
    time — NOT a frame-by-frame speed curve, since the ground-plane bias
    (see module docstring) would make instantaneous speeds between
    individual frames noisier and less trustworthy than one estimate
    over the whole trusted segment.
    """
    if len(points) < 2:
        return {"status": "error", "message": "Need at least 2 points to estimate speed."}

    ground_points = [pixel_to_ground(homography, x, y) for _, x, y in points]
    (x0, y0), (x1, y1) = ground_points[0], ground_points[-1]
    distance_m = float(np.hypot(x1 - x0, y1 - y0))

    frame0, frame_last = points[0][0], points[-1][0]
    duration_s = (frame_last - frame0) / fps
    if duration_s <= 0:
        return {"status": "error", "message": "Non-positive duration between first and last point."}

    speed_ms = distance_m / duration_s
    return {
        "status": "success",
        "speed_kmh": speed_ms * 3.6,
        "distance_m": distance_m,
        "duration_s": duration_s,
        "ground_points": ground_points,
    }


# ---------------------------------------------------------------------------
# FULL 3D CAMERA POSE (2026-09-27, real coach request: a working speed/
# pitch-map feature that can survive a ball actually being airborne).
#
# Everything above this line builds a flat GROUND-PLANE homography — it
# has no way to represent height at all, so a ball at head height and a
# ball resting on the pitch at the same image position map to the exact
# same real-world point. Traced directly on a real clip (IMG_4060.MOV,
# a genuine bouncer): the resulting "ground" positions jumped by tens of
# meters between adjacent tracked frames, and the one whole-segment
# speed number that came out (118.7 km/h) only looked plausible by
# coincidence, not because the math was sound — see project memory for
# the full writeup.
#
# Real fix: the calibration UI now also captures ONE stump-TOP point (a
# known height, STUMP_HEIGHT_M, above its own base) in addition to the 4
# stump-BASE points. That's a 5th real-world point that is NOT on the
# ground plane — enough to break the height ambiguity and solve the
# camera's actual 3D position and angle (a real "pose", not just a flat
# map), via cv2.solvePnP.
# ---------------------------------------------------------------------------

def _stump_object_points(half_w: float) -> np.ndarray:
    """The 6 known real-world points (meters), same axes as
    build_ground_homography (X sideways from centerline, Y down-pitch
    from the near stumps, Z height above ground) — 4 at ground level
    plus BOTH near stumps' TOPs (2026-09-27: originally just the
    near-left top; upgraded to both after a real coach mistake on
    IMG_3796.MOV showed why one alone isn't enough — see
    _check_top_point_assignment's docstring)."""
    return np.array([
        [-half_w, 0.0, 0.0],
        [half_w, 0.0, 0.0],
        [-half_w, PITCH_LENGTH_M, 0.0],
        [half_w, PITCH_LENGTH_M, 0.0],
        [-half_w, 0.0, STUMP_HEIGHT_M],
        [half_w, 0.0, STUMP_HEIGHT_M],
    ], dtype=np.float64)


def _check_top_point_assignment(near_left_px, near_right_px, near_left_top_px, near_right_top_px):
    """
    Real bug this catches (2026-09-27, IMG_3796.MOV): the coach's near-
    stump BASE clicks were mislabeled left/right (an easy, understandable
    mistake — camera-left vs. screen-left is genuinely ambiguous). This
    was NOT caught by the reprojection-error gate alone: swapping just
    the two base pixel VALUES while leaving the single top point
    attached to whichever argument slot it was in silently created a
    NEW, physically inconsistent correspondence (a "top" click paired
    with a base click from a DIFFERENT physical stump) that still
    numerically reprojected well enough to pass — then produced a
    physically absurd trajectory fit downstream (a delivery moving
    almost straight up/down, ~16 km/h) once applied to real tracked ball
    points far from the calibration region. A single top point has no
    way to catch this; it can only ever be paired one way.

    Real fix: require BOTH near stumps' tops (this function), and check
    the one thing that's true regardless of any left/right labeling
    mistake — each top point must be closer to its OWN base than to the
    other stump's base (a stump doesn't lean sideways by more than its
    own base-to-base separation). Returns an error message if not, or
    None if consistent. This makes the whole calibration self-checking
    instead of trusting the coach's semantic "which one is left"
    judgment at all.
    """
    def _dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    left_top_to_left = _dist(near_left_top_px, near_left_px)
    left_top_to_right = _dist(near_left_top_px, near_right_px)
    right_top_to_right = _dist(near_right_top_px, near_right_px)
    right_top_to_left = _dist(near_right_top_px, near_left_px)

    problems = []
    if left_top_to_right < left_top_to_left:
        problems.append("the near-LEFT stump-top click is closer to the near-RIGHT base than its own base")
    if right_top_to_left < right_top_to_right:
        problems.append("the near-RIGHT stump-top click is closer to the near-LEFT base than its own base")
    if problems:
        return ("Likely left/right mix-up on the near stumps: " + "; ".join(problems) +
                ". Redo the near stump clicks (base AND top), making sure each top is "
                "directly above its own base.")
    return None


def solve_camera_pose(near_left_px, near_right_px, far_left_px, far_right_px,
                       near_left_top_px, near_right_top_px,
                       image_width: int, image_height: int) -> dict:
    """
    Solves the camera's real 3D position and angle from the same 4
    stump-base points build_ground_homography uses, PLUS the tops of
    BOTH near stumps — the height information a flat ground-plane
    homography structurally cannot have (see this module's own
    "HONEST LIMITATION" docstring at the top of the file). With real
    points at a second, known height, the pose is no longer ambiguous,
    and a real ball position ABOVE the ground can finally be told apart
    from one resting on it.

    UPGRADED FROM ONE TOP POINT TO TWO (2026-09-27, real coach mistake,
    IMG_3796.MOV): the coach's near-stump BASE clicks were mislabeled
    left/right — an easy, understandable mistake (camera-left vs.
    screen-left is genuinely ambiguous), not carelessness. With only
    ONE top point, this was invisible to the reprojection-error check:
    swapping which base pixel occupied the "near_left"/"near_right"
    argument slots, while the single top point stayed fixed in its
    slot, silently created a NEW correspondence pairing that top with a
    DIFFERENT physical stump than it was actually clicked on — still
    numerically reprojected well enough to look "fixed" (17.5px -> 8px),
    then produced a physically absurd trajectory fit downstream (a
    delivery reading ~16 km/h, moving almost straight up/down) once
    applied to real tracked ball points far from the calibration
    region. (An earlier version of this function also chased a lens-
    distortion explanation for the original error pattern — direct
    testing showed error barely changing across the whole plausible
    k1 range, meaning distortion was NOT the real cause; the left/right
    mix-up was. That distortion term is kept — see below — because
    it's real and free, not because it explains this specific incident.)

    Requiring BOTH tops removes the ambiguity structurally: each top
    point must be closer to ITS OWN base than to the other stump's base
    (checked directly, see _check_top_point_assignment) — a hard,
    explainable, unmissable check that doesn't depend on trusting
    anyone's left/right semantic judgment at all.

    Assumes square pixels and the principal point at the image center —
    a reasonable approximation for an uncalibrated phone camera (a full
    checkerboard calibration would be more precise but needs physical
    equipment this app's filming setup doesn't use). Focal length and
    ONE radial lens-distortion term (k1) are NOT assumed or hand-
    measured: both are solved via a nested search for whichever (focal
    length, k1) pair makes cv2.solvePnP's recovered rotation/translation
    reproject all 6 real points back onto where they were actually
    clicked with the least error — the same self-consistent-solve
    principle already validated by hand in the project's 2026-08-15
    calibration-precision investigation (see project memory).

    Returns {"status": "success", "rvec", "tvec", "focal_length_px",
    "dist_coeffs" (shape (5,), only k1 nonzero), "principal_point_px",
    "reprojection_error_px" (mean), "max_reprojection_error_px"}, or
    {"status": "error", "message": ...} if the top-point consistency
    check fails, or any single one of the 6 points still can't be
    reprojected within 15px even with distortion allowed for. MAX, not
    mean, gates the reprojection check — confirmed directly that a
    single badly mis-clicked point barely moves the mean across 6
    points (the 4 coplanar ground points alone can still fit almost any
    focal length/distortion), but always dominates the max.
    """
    consistency_problem = _check_top_point_assignment(
        near_left_px, near_right_px, near_left_top_px, near_right_top_px)
    if consistency_problem:
        return {"status": "error", "message": consistency_problem}

    half_w = STUMP_LINE_WIDTH_M / 2
    object_points = _stump_object_points(half_w)
    image_points = np.array([
        near_left_px, near_right_px, far_left_px, far_right_px,
        near_left_top_px, near_right_top_px,
    ], dtype=np.float64)
    cx, cy = image_width / 2.0, image_height / 2.0

    def _solve_for_f(f, k1):
        if f <= 0:
            return None
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # EPNP, not the (default) ITERATIVE flag: this project's 6-point
        # set mixes 4 coplanar (ground) points with 2 non-coplanar
        # (stump-top) points. EPNP handles any n>=4 point configuration,
        # planar or not, without the ITERATIVE flag's own DLT initial-
        # guess restrictions on mixed coplanar/non-coplanar sets.
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist,
                                       flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
        return rvec, tvec, error

    def _best_error_for_k1(k1):
        def _f_error(f):
            result = _solve_for_f(float(f), k1)
            return result[2] if result is not None else 1e9
        # A broad, physically plausible search range rather than
        # assuming a focal length — real phone lenses at typical
        # resolutions land well inside [0.3x, 4x] image width for any
        # normal (non-fisheye, non-telephoto) lens.
        search = minimize_scalar(_f_error, bounds=(0.3 * image_width, 4.0 * image_width),
                                  method="bounded", options={"xatol": 1.0})
        return search.fun

    # NESTED search: for every candidate k1 tried by the outer search,
    # the inner search (_best_error_for_k1 -> _f_error) already finds
    # the genuinely best focal length for THAT k1 — so the outer search
    # is optimizing a well-defined 1D function of k1 alone, a valid way
    # to jointly solve two variables via two nested 1D searches instead
    # of one 2D one. |k1| > 1 would be an extreme, physically implausible
    # lens distortion for a normal (non-fisheye) phone camera -- bounding
    # it keeps the search from chasing an overfit on just 6 points.
    k1_search = minimize_scalar(_best_error_for_k1, bounds=(-1.0, 1.0),
                                 method="bounded", options={"xatol": 0.002})
    best_k1 = float(k1_search.x)

    # DISTORTION RELIABILITY CHECK (2026-09-29, real second-clip test,
    # IMG_3797.MOV): k1 landed at 0.999 -- pinned right at the bound,
    # not a converged interior value. Traced directly by extending the
    # search well past the bound: the error kept falling all the way to
    # k1~5-8 before turning back up -- a "best fit" more than 5x any
    # physically real phone lens's distortion, the signature of the
    # extra parameter absorbing measurement noise across just 6 points,
    # not capturing genuine lens characteristics (the earlier synthetic
    # test recovers a real k1=-0.12 to a sharp, INTERIOR minimum, unlike
    # this). A parameter sitting exactly on its search bound is the same
    # untrustworthy signal already used for a fitted trajectory
    # parameter (see fit_gravity_trajectory's z0-at-bound finding on
    # IMG_4060.MOV) -- refuse it here the same way: fall back to k1=0
    # (pure pinhole) rather than trust a boundary-pinned value.
    if abs(best_k1) > 0.95:
        best_k1 = 0.0

    def _f_error_at_best_k1(f):
        result = _solve_for_f(float(f), best_k1)
        return result[2] if result is not None else 1e9
    f_search = minimize_scalar(_f_error_at_best_k1, bounds=(0.3 * image_width, 4.0 * image_width),
                                method="bounded", options={"xatol": 1.0})
    best_f = float(f_search.x)

    dist = np.array([best_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    K = np.array([[best_f, 0, cx], [0, best_f, cy], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist,
                                   flags=cv2.SOLVEPNP_EPNP)
    # LM refinement: EPNP's closed-form solve is a fast approximation;
    # this squeezes out the remaining reprojection error against the
    # now-fixed best focal length + k1, same as the searches above
    # already did per-candidate.
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, K, dist, rvec, tvec)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    per_point_error = np.sqrt(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))
    max_error = float(per_point_error.max())
    # MAX, not RMS, gates acceptance: a single genuinely mis-clicked
    # point (confirmed directly — a 400px-off stump-top click) barely
    # moves the RMS across all 6 points (the 4 coplanar ground points
    # can still fit almost any focal length, diluting one bad point's
    # contribution to the average), but it always dominates the MAX.
    if max_error > 15.0:
        return {"status": "error",
                "message": f"Could not find a consistent camera pose (worst single-point "
                           f"reprojection error {max_error:.1f}px, even allowing for lens "
                           f"distortion) — check the 6 clicked points."}

    return {
        "status": "success",
        "rvec": rvec, "tvec": tvec, "focal_length_px": best_f, "dist_coeffs": dist,
        "principal_point_px": (cx, cy), "reprojection_error_px": float(per_point_error.mean()),
        "max_reprojection_error_px": max_error,
        # False when the joint search wanted to pin k1 at its bound and
        # was overridden back to 0 -- visible to a caller/log rather
        # than silently swapped, so this specific clip's calibration
        # can be flagged for a careful redo rather than trusted blind.
        "distortion_reliable": bool(best_k1 != 0.0 or abs(k1_search.x) <= 0.95),
    }


def project_3d_to_pixel(pose: dict, x_m: float, y_m: float, z_m: float):
    """Projects one real-world 3D point (same X/Y axes as
    build_ground_homography; Z = height above ground in meters) to a
    pixel position, using a solved pose from solve_camera_pose. Used to
    check a candidate trajectory (e.g. a physics/gravity fit) against
    real tracked pixels. Applies the pose's own solved lens-distortion
    term (dist_coeffs) if present -- `.get(...)` so an older/hand-built
    pose dict without one (e.g. in tests) still works, as pure pinhole."""
    f = pose["focal_length_px"]
    cx, cy = pose["principal_point_px"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = pose.get("dist_coeffs")
    point = np.array([[[x_m, y_m, z_m]]], dtype=np.float64)
    projected, _ = cv2.projectPoints(point, pose["rvec"], pose["tvec"], K, dist)
    return float(projected[0, 0, 0]), float(projected[0, 0, 1])


def pixel_to_ground_at_height(pose: dict, x_px: float, y_px: float, z_m: float):
    """
    The height-aware inverse of project_3d_to_pixel: given a pixel AND
    an ASSUMED real height (z_m, meters above ground), casts the real
    camera ray through that pixel and intersects it with the horizontal
    plane at that height, returning (X_m, Y_m).

    This is the actual fix for the bug traced on IMG_4060.MOV: instead
    of always assuming z_m=0 (ground) like pixel_to_ground/
    build_ground_homography do, a caller that knows (or is fitting) the
    ball's real height at each frame — e.g. from a gravity-parabola
    trajectory fit — can invert correctly at every point along the
    flight, not just at the ends. At z_m=0 this must agree with
    pixel_to_ground's own flat-homography answer (checked directly in
    tests/test_pitch_calibration.py) — it's a strict generalization,
    not a different method.

    Applies the pose's own solved lens-distortion term first, if
    present, via cv2.undistortPoints (2026-09-27 — a distorted pixel's
    ray direction isn't simply K^-1 @ [u,v,1] once distortion is real;
    undistortPoints removes it and, with no P/R given, returns exactly
    the normalized ideal-pinhole camera-ray direction this function
    already needs).
    """
    f = pose["focal_length_px"]
    cx, cy = pose["principal_point_px"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    R, _ = cv2.Rodrigues(pose["rvec"])
    tvec = pose["tvec"].reshape(3)
    camera_pos = -R.T @ tvec  # world-space camera center

    dist = pose.get("dist_coeffs")
    if dist is not None and np.any(dist):
        pixel = np.array([[[x_px, y_px]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pixel, K, dist)
        dir_cam = np.array([undistorted[0, 0, 0], undistorted[0, 0, 1], 1.0])
    else:
        pixel_h = np.array([x_px, y_px, 1.0], dtype=np.float64)
        dir_cam = np.linalg.inv(K) @ pixel_h
    dir_world = R.T @ dir_cam  # world-space ray direction (unnormalized)

    if abs(dir_world[2]) < 1e-9:
        raise ValueError("Camera ray is parallel to the ground plane — cannot intersect.")
    t = (z_m - camera_pos[2]) / dir_world[2]
    x_m = camera_pos[0] + t * dir_world[0]
    y_m = camera_pos[1] + t * dir_world[1]
    return float(x_m), float(y_m)


# ---------------------------------------------------------------------------
# GRAVITY-CONSTRAINED TRAJECTORY FIT (2026-09-27, Stage 3 of the coach's
# agreed speed/pitch-map roadmap — see project memory). Once a real 3D
# camera pose exists (solve_camera_pose above), the right way to get a
# trustworthy speed is fitting ONE physics-consistent flight path through
# EVERY real tracked point in a single phase, not a straight line between
# two endpoints — the whole-segment endpoint method (estimate_speed_kmh
# above) was traced directly on a real clip (IMG_4060.MOV) to blend
# pre- and post-bounce motion into one meaningless average, on top of
# its own flat-ground bias. This averages out real per-frame pixel noise
# across many points instead, the same principle already reasoned
# through with the coach on 2026-09-24 before this was built.
#
# MODEL: constant horizontal velocity (X, Y) + constant downward
# acceleration (gravity) in height (Z) -- i.e. standard projectile
# motion, no air resistance or Magnus/swing curvature. This is a real,
# disclosed simplification (same standard as every other estimate in
# this app): it fits the dominant straight-line-at-pace component of a
# delivery, which is what "how fast was that ball" actually asks, the
# same way a radar gun reports pace independent of how much a ball
# swung. It does NOT model sideways swing/seam curvature -- vx is a
# single constant, not a curve.
#
# PHASES: this model only holds within ONE continuous flight arc -- a
# bounce changes vertical (and often horizontal) velocity
# discontinuously, so a fit spanning across one would be just as wrong
# as the old endpoint method, for a different reason. detect_bounce_frame
# below finds the split directly from the real tracked pixels (no
# physics needed for that part), and estimate_release_speed_kmh fits
# only the pre-bounce phase -- the number a coach actually means by
# "bowling speed."
# ---------------------------------------------------------------------------

def fit_gravity_trajectory(pose: dict, points: list, fps: float) -> dict:
    """
    points: [(frame_index, x_px, y_px), ...], increasing frame order,
    ALL from a single continuous flight phase (no bounce in between --
    see detect_bounce_frame/estimate_release_speed_kmh for splitting a
    real tracked sequence first).

    Fits (X0, Y0, Z0, vx, vy, vz) -- one 3D release point and one 3D
    release velocity for this phase -- by minimizing reprojection error
    against EVERY real point at once (scipy.optimize.least_squares),
    not just the first and last. Returns the fitted state, the initial
    speed magnitude (the actual answer to "how fast"), and per-point
    reprojection residuals so a bad fit is visible, not hidden.

    Needs real redundancy, not just the bare mathematical minimum (6
    unknowns): requires at least 8 points, so the fit is genuinely
    overdetermined and residuals mean something.
    """
    if pose.get("status") != "success":
        # Real gap found on the first genuine real-clip test (2026-09-27,
        # IMG_3796.MOV): a failed solve_camera_pose call was being passed
        # straight through with no check, crashing on pose["focal_length_px"]
        # (a KeyError, not a helpful message) the first time a real
        # coach's calibration didn't converge cleanly.
        return {"status": "error",
                "message": "Cannot fit a trajectory without a valid camera pose "
                           "(solve_camera_pose did not succeed)."}
    if len(points) < 8:
        return {"status": "error",
                "message": f"Need at least 8 real tracked points for a trustworthy fit "
                           f"(got {len(points)})."}

    frame0 = points[0][0]
    t = np.array([(p[0] - frame0) / fps for p in points], dtype=np.float64)
    observed = np.array([[p[1], p[2]] for p in points], dtype=np.float64)

    def _residuals(params):
        x0, y0, z0, vx, vy, vz = params
        x_m = x0 + vx * t
        y_m = y0 + vy * t
        z_m = z0 + vz * t - 0.5 * GRAVITY_M_S2 * t ** 2
        predicted = np.array([project_3d_to_pixel(pose, x_m[i], y_m[i], z_m[i])
                               for i in range(len(t))])
        return (predicted - observed).ravel()

    # BOUNDED trust-region ('trf'), not the unbounded default ('lm') —
    # confirmed directly that unbounded LM can run away to a physically
    # nonsensical solution (millions of km/h) on a short flight-phase
    # arc, where a near-linear trajectory leaves z0/vz weakly
    # constrained (classic collinearity over a brief window). Bounds
    # are generous physical limits, not a tuned fit to any one clip: no
    # recorded delivery approaches 200 km/h, and the position bounds
    # cover any plausible release point/pitch width. vy > 0 only
    # (matches this module's own axis convention: Y increases from the
    # near stumps toward the far stumps, the direction a delivery
    # actually travels).
    bounds = (
        [-5.0, -5.0, 0.0, -25.0, 0.0, -25.0],
        [5.0, PITCH_LENGTH_M + 5.0, 3.5, 25.0, 60.0, 25.0],
    )

    # MULTI-START initial guess, not one single seed: confirmed directly
    # that ray-intersecting a real tracked pixel at an ARBITRARY assumed
    # height can land the intersection BEHIND the camera for pixels
    # close to it or high in frame (the ray only reaches a given height
    # at a positive, in-front-of-camera distance for SOME heights, not
    # any height) — this silently produces a wrong-signed, sometimes
    # wildly wrong velocity guess with no error raised, which a local
    # optimizer (even bounded) can fail to recover from. Instead: seed
    # position (x0, y0) from the single most reliable anchor (the FIRST
    # point, intersected at ground level z=0 -- always a valid forward
    # intersection for a camera looking down/forward at the pitch, per
    # this whole setup's own purpose), then try a small, cheap grid of
    # physically-plausible velocity/height candidates and keep whichever
    # one the optimizer converges to with the LOWEST final reprojection
    # error -- sidesteps needing the guess itself to already be close.
    x0_guess, y0_guess = pixel_to_ground_at_height(pose, points[0][1], points[0][2], 0.0)
    best_result, best_cost = None, np.inf
    for z0_guess in (0.5, 1.0, 1.8, 2.5):
        for vy_guess in (15.0, 30.0, 45.0):
            for vz_guess in (-5.0, 0.0, 5.0):
                initial = np.clip(
                    [x0_guess, y0_guess, z0_guess, 0.0, vy_guess, vz_guess],
                    bounds[0], bounds[1],
                )
                candidate = least_squares(_residuals, initial, method="trf",
                                           bounds=bounds, max_nfev=2000)
                if candidate.cost < best_cost:
                    best_cost, best_result = candidate.cost, candidate

    result = best_result
    residual_norms = np.sqrt(np.sum(result.fun.reshape(-1, 2) ** 2, axis=1))
    max_error = float(residual_norms.max())
    x0, y0, z0, vx, vy, vz = result.x

    # SAME GATE AS solve_camera_pose, needed for the identical reason
    # (2026-09-27, first real-clip test, IMG_3796.MOV): fed the WHOLE
    # 44-point labeled sequence (including what looks like pre-release
    # arm-swing motion, not real ball flight) through this fit and it
    # returned a confident 123 km/h with a mean reprojection error of
    # 278px, max 396px -- a physically absurd fit result, presented with
    # the exact same "status": "success" as a genuinely clean one until
    # this check existed. A trajectory that doesn't actually follow
    # gravity-only motion (arm swing, or points spanning across an
    # undetected phase change) shows up as a huge reprojection error,
    # the same signal solve_camera_pose already uses to catch a bad
    # calibration click -- this closes the same gap one level up.
    if max_error > 20.0:
        return {"status": "error",
                "message": f"These points don't fit a single, real flight phase well "
                           f"(worst point off by {max_error:.0f}px) -- likely includes "
                           f"pre-release motion, a missed bounce, or a tracking error, "
                           f"not a real gravity-only arc."}

    return {
        "status": "success",
        "release_point_m": (float(x0), float(y0), float(z0)),
        "velocity_ms": (float(vx), float(vy), float(vz)),
        "speed_kmh": float(np.hypot(np.hypot(vx, vy), vz) * 3.6),
        "mean_reprojection_error_px": float(residual_norms.mean()),
        "max_reprojection_error_px": max_error,
        "num_points": len(points),
    }


def detect_bounce_frame(points: list):
    """
    Finds where a real tracked/labeled sequence's own pixel trajectory
    shows a bounce -- the image-row (y_px) descends (ball falling) to a
    peak, then rises again (ball climbing after impact). Pure pixel-
    space signal, no camera pose or physics needed for this part, same
    idea already confirmed by eye on real data (IMG_4060.MOV's own
    labels show exactly this shape, peaking around frame 104-108).

    Returns the frame_index of the peak, or None if the sequence never
    turns around (still monotonically descending/ascending throughout
    -- e.g. release-to-bounce with no post-bounce points included yet).

    Deliberately simple and requires the peak to be a REAL interior
    point with genuine descent before AND ascent after (not just noise
    at one end) -- a single-frame wobble at the very start or end must
    not be mistaken for a bounce.
    """
    ys = [p[2] for p in points]
    peak_idx = int(np.argmax(ys))
    if peak_idx == 0 or peak_idx == len(ys) - 1:
        return None
    if not (ys[peak_idx] > ys[0] and ys[peak_idx] > ys[-1]):
        return None
    return points[peak_idx][0]


def estimate_release_speed_kmh(pose: dict, points: list, fps: float) -> dict:
    """
    The actual answer to "how fast was that ball": detects a bounce (if
    the given points include one) and fits fit_gravity_trajectory to
    ONLY the pre-bounce points -- release speed, the number a coach
    means by "bowling speed", not an average blended across the bounce.

    points: [(frame_index, x_px, y_px), ...] from a single trusted
    track spanning release through (optionally) the bounce and beyond.
    If no bounce is found in the given points, fits the whole sequence
    (assumes it's all pre-bounce).
    """
    bounce_frame = detect_bounce_frame(points)
    if bounce_frame is not None:
        # STRICTLY before, not <= : bounce_frame is the pixel-space peak
        # itself -- the moment of impact, the single least certain point
        # in the whole sequence (the ball is at/against the pitch,
        # likely partly obscured or deformed in a real frame). Confirmed
        # directly with synthetic data that including it as "pre-bounce"
        # can inject a real outlier into the fit -- excluding it is
        # strictly safer, and never costs more than one frame.
        phase_points = [p for p in points if p[0] < bounce_frame]
    else:
        phase_points = points

    result = fit_gravity_trajectory(pose, phase_points, fps)
    if result["status"] == "success":
        result["bounce_frame"] = bounce_frame
        result["phase"] = "pre-bounce (release speed)"
    return result
