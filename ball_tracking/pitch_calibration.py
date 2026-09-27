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
from scipy.optimize import minimize_scalar

PITCH_LENGTH_M = 20.12  # popping crease to popping crease, standard
STUMP_LINE_WIDTH_M = 0.2286  # 9 inches, outer edge to outer edge
STUMP_HEIGHT_M = 0.7112  # 28 inches, standard stump height above ground


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
    """The 5 known real-world points (meters), same axes as
    build_ground_homography (X sideways from centerline, Y down-pitch
    from the near stumps, Z height above ground) — 4 at ground level
    plus the near-left stump's TOP, the one point that isn't."""
    return np.array([
        [-half_w, 0.0, 0.0],
        [half_w, 0.0, 0.0],
        [-half_w, PITCH_LENGTH_M, 0.0],
        [half_w, PITCH_LENGTH_M, 0.0],
        [-half_w, 0.0, STUMP_HEIGHT_M],
    ], dtype=np.float64)


def solve_camera_pose(near_left_px, near_right_px, far_left_px, far_right_px,
                       near_left_top_px, image_width: int, image_height: int) -> dict:
    """
    Solves the camera's real 3D position and angle from the same 4
    stump-base points build_ground_homography uses, PLUS the top of the
    near-left stump — the one piece of information a flat ground-plane
    homography structurally cannot have (see this module's own
    "HONEST LIMITATION" docstring at the top of the file). With a real
    point at a second, known height, the pose is no longer ambiguous,
    and a real ball position ABOVE the ground can finally be told apart
    from one resting on it.

    Assumes a standard pinhole camera with square pixels and the
    principal point at the image center — a reasonable approximation
    for an uncalibrated phone camera (a full checkerboard calibration
    would be more precise but needs physical equipment this app's
    filming setup doesn't use). Focal length is NOT assumed or
    hand-measured: it's solved directly, via a 1D search for whichever
    focal length makes cv2.solvePnP's recovered rotation/translation
    reproject all 5 real points back onto where they were actually
    clicked with the least error. This is the same self-consistent-
    solve principle already validated by hand in the project's
    2026-08-15 calibration-precision investigation (see project
    memory) — automated here, and extended from one depth number to a
    full camera pose.

    Returns {"status": "success", "rvec", "tvec", "focal_length_px",
    "principal_point_px", "reprojection_error_px" (mean), and
    "max_reprojection_error_px"}, or {"status": "error", "message":
    ...} if any single one of the 5 points can't be reprojected within
    15px. MAX, not mean, gates acceptance — confirmed directly that a
    single badly mis-clicked point (e.g. the stump-top, off by 400px)
    barely moves the mean across all 5 points, since the 4 coplanar
    ground points alone can still fit almost any focal length; only
    the max reliably catches it.
    """
    half_w = STUMP_LINE_WIDTH_M / 2
    object_points = _stump_object_points(half_w)
    image_points = np.array([
        near_left_px, near_right_px, far_left_px, far_right_px, near_left_top_px,
    ], dtype=np.float64)
    cx, cy = image_width / 2.0, image_height / 2.0

    def _reprojection_error(f):
        f = float(f)
        if f <= 0:
            return 1e9
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        # EPNP, not the (default) ITERATIVE flag: this project's 5-point
        # set mixes 4 coplanar (ground) points with 1 non-coplanar
        # (stump-top) point, and OpenCV's ITERATIVE solver's own DLT
        # initial-guess step refuses anything under 6 points for that
        # mixed case (confirmed directly — it raises, not a tuning
        # choice). EPNP handles any n>=4 point configuration, planar or
        # not, which is exactly this case.
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None,
                                       flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return 1e9
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
        return float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))

    # A broad, physically plausible search range rather than assuming a
    # focal length — real phone lenses at typical resolutions land well
    # inside [0.3x, 4x] image width for any normal (non-fisheye,
    # non-telephoto) lens.
    search = minimize_scalar(
        _reprojection_error, bounds=(0.3 * image_width, 4.0 * image_width),
        method="bounded", options={"xatol": 1.0},
    )
    best_f, best_rms_error = float(search.x), float(search.fun)

    K = np.array([[best_f, 0, cx], [0, best_f, cy], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None,
                                   flags=cv2.SOLVEPNP_EPNP)
    # LM refinement: EPNP's closed-form solve is a fast approximation;
    # this squeezes out the remaining reprojection error against the
    # now-fixed best focal length, same as the 1D search above already
    # did per-candidate.
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, K, None, rvec, tvec)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
    per_point_error = np.sqrt(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))
    max_error = float(per_point_error.max())
    # MAX, not RMS, gates acceptance: a single genuinely mis-clicked
    # point (confirmed directly — a 400px-off stump-top click) barely
    # moves the RMS across all 5 points (the 4 coplanar ground points
    # can still fit almost any focal length, diluting one bad point's
    # contribution to the average), but it always dominates the MAX.
    if max_error > 15.0:
        return {"status": "error",
                "message": f"Could not find a consistent camera pose (worst single-point "
                           f"reprojection error {max_error:.1f}px) — check the 5 clicked "
                           f"points, especially the stump-top point."}

    return {
        "status": "success",
        "rvec": rvec, "tvec": tvec, "focal_length_px": best_f,
        "principal_point_px": (cx, cy), "reprojection_error_px": float(per_point_error.mean()),
        "max_reprojection_error_px": max_error,
    }


def project_3d_to_pixel(pose: dict, x_m: float, y_m: float, z_m: float):
    """Projects one real-world 3D point (same X/Y axes as
    build_ground_homography; Z = height above ground in meters) to a
    pixel position, using a solved pose from solve_camera_pose. Used to
    check a candidate trajectory (e.g. a physics/gravity fit) against
    real tracked pixels."""
    f = pose["focal_length_px"]
    cx, cy = pose["principal_point_px"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    point = np.array([[[x_m, y_m, z_m]]], dtype=np.float64)
    projected, _ = cv2.projectPoints(point, pose["rvec"], pose["tvec"], K, None)
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
    """
    f = pose["focal_length_px"]
    cx, cy = pose["principal_point_px"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    R, _ = cv2.Rodrigues(pose["rvec"])
    tvec = pose["tvec"].reshape(3)
    camera_pos = -R.T @ tvec  # world-space camera center

    pixel_h = np.array([x_px, y_px, 1.0], dtype=np.float64)
    dir_cam = np.linalg.inv(K) @ pixel_h
    dir_world = R.T @ dir_cam  # world-space ray direction (unnormalized)

    if abs(dir_world[2]) < 1e-9:
        raise ValueError("Camera ray is parallel to the ground plane — cannot intersect.")
    t = (z_m - camera_pos[2]) / dir_world[2]
    x_m = camera_pos[0] + t * dir_world[0]
    y_m = camera_pos[1] + t * dir_world[1]
    return float(x_m), float(y_m)
