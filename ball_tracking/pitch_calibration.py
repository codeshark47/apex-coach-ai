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

PITCH_LENGTH_M = 20.12  # popping crease to popping crease, standard
STUMP_LINE_WIDTH_M = 0.2286  # 9 inches, outer edge to outer edge


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
