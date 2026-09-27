"""
Tests for ball_tracking/pitch_calibration.py's Stage 3: the gravity-
constrained trajectory fit (fit_gravity_trajectory, detect_bounce_frame,
estimate_release_speed_kmh) — the piece that turns a correct 3D camera
pose (Stage 2, see test_pitch_calibration.py) into an actual trustworthy
speed number, by fitting one physics-consistent flight path through
every real tracked point in a phase instead of a straight line between
two endpoints.

Uses synthetic projectile-motion data with a KNOWN speed and a KNOWN
camera pose — the only way to verify the fit recovers the right answer,
since there's no independent ground truth for a real delivery's exact
speed at the pixel level.
"""
import numpy as np
import cv2
import pytest

from ball_tracking.pitch_calibration import (
    STUMP_LINE_WIDTH_M, STUMP_HEIGHT_M, PITCH_LENGTH_M, GRAVITY_M_S2,
    solve_camera_pose, project_3d_to_pixel,
    fit_gravity_trajectory, detect_bounce_frame, estimate_release_speed_kmh,
)

IMAGE_W, IMAGE_H = 1920, 1080
FPS = 60.0


def _synthetic_pose():
    """Same real, known camera pose as test_pitch_calibration.py
    (~4m behind stumps, ~1.5m high, tilted down — matches FullTrack
    AI's own documented setup, see project memory). Duplicated locally
    rather than imported, matching this project's established pattern
    of duplicating small stable test fixtures over adding cross-file
    coupling between test modules."""
    true_f = 1600.0
    cx, cy = IMAGE_W / 2.0, IMAGE_H / 2.0
    K = np.array([[true_f, 0, cx], [0, true_f, cy], [0, 0, 1]], dtype=np.float64)
    tilt_rad = np.radians(15)
    forward = np.array([0, np.cos(tilt_rad), -np.sin(tilt_rad)])
    right = np.array([1, 0, 0])
    down = np.cross(forward, right)
    R = np.array([right, down, forward])
    camera_pos = np.array([0.0, -4.0, 1.5])
    tvec = (-R @ camera_pos).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    true_pose = {"rvec": rvec, "tvec": tvec, "focal_length_px": true_f,
                 "principal_point_px": (cx, cy)}

    half_w = STUMP_LINE_WIDTH_M / 2

    def proj(x, y, z):
        pt = np.array([[[x, y, z]]], dtype=np.float64)
        p, _ = cv2.projectPoints(pt, rvec, tvec, K, None)
        return float(p[0, 0, 0]), float(p[0, 0, 1])

    solved = solve_camera_pose(
        proj(-half_w, 0, 0), proj(half_w, 0, 0),
        proj(-half_w, PITCH_LENGTH_M, 0), proj(half_w, PITCH_LENGTH_M, 0),
        proj(-half_w, 0, STUMP_HEIGHT_M), IMAGE_W, IMAGE_H,
    )
    assert solved["status"] == "success"
    return solved


def _projectile_points(pose, x0, y0, z0, vx, vy, vz, frame0, fps, n_frames, frame_step=2):
    """Generates real projectile-motion pixel points (no bounce) for a
    known (X0,Y0,Z0,vx,vy,vz) release state, projected through the
    given pose — i.e. what a perfect tracker would have reported."""
    points = []
    for i in range(n_frames):
        frame = frame0 + i * frame_step
        t = (frame - frame0) / fps
        x = x0 + vx * t
        y = y0 + vy * t
        z = z0 + vz * t - 0.5 * GRAVITY_M_S2 * t ** 2
        px, py = project_3d_to_pixel(pose, x, y, z)
        points.append((frame, px, py))
    return points


def test_fit_gravity_trajectory_recovers_known_speed():
    pose = _synthetic_pose()
    true_vx, true_vy, true_vz = 0.5, 35.0, 2.0  # ~35 m/s down-pitch dominant -> ~126 km/h
    true_speed_kmh = np.hypot(np.hypot(true_vx, true_vy), true_vz) * 3.6

    points = _projectile_points(pose, x0=0.0, y0=0.5, z0=2.0,
                                 vx=true_vx, vy=true_vy, vz=true_vz,
                                 frame0=100, fps=FPS, n_frames=12)
    result = fit_gravity_trajectory(pose, points, FPS)

    assert result["status"] == "success"
    assert result["speed_kmh"] == pytest.approx(true_speed_kmh, rel=0.02)
    assert result["max_reprojection_error_px"] < 1.0


def test_fit_gravity_trajectory_is_robust_to_realistic_click_noise():
    """The actual point of fitting through every point instead of just
    two endpoints: real coach clicks aren't pixel-perfect. Adds ~2px
    Gaussian noise (a realistic click-precision estimate, not a
    worst-case) to every point and confirms the fit still recovers the
    true speed within a reasonable tolerance -- the noise should
    average out across many points, not multiply through like the old
    two-point endpoint method."""
    pose = _synthetic_pose()
    true_vx, true_vy, true_vz = 0.5, 35.0, 2.0
    true_speed_kmh = np.hypot(np.hypot(true_vx, true_vy), true_vz) * 3.6

    points = _projectile_points(pose, x0=0.0, y0=0.5, z0=2.0,
                                 vx=true_vx, vy=true_vy, vz=true_vz,
                                 frame0=100, fps=FPS, n_frames=14)
    rng = np.random.default_rng(42)
    noisy_points = [(f, x + rng.normal(0, 2.0), y + rng.normal(0, 2.0)) for f, x, y in points]

    result = fit_gravity_trajectory(pose, noisy_points, FPS)
    assert result["status"] == "success"
    assert result["speed_kmh"] == pytest.approx(true_speed_kmh, rel=0.08)


def test_fit_gravity_trajectory_requires_minimum_points():
    pose = _synthetic_pose()
    points = _projectile_points(pose, 0.0, 0.5, 2.0, 0.5, 35.0, 2.0, 100, FPS, n_frames=5)
    result = fit_gravity_trajectory(pose, points, FPS)
    assert result["status"] == "error"


def test_detect_bounce_frame_finds_a_real_peak():
    # A real image-space signature: y_px descends to a peak then rises
    # (matches IMG_4060.MOV's own real labeled data, see project memory).
    points = [(100, 500.0, 900.0), (102, 520.0, 1100.0), (104, 540.0, 1400.0),
              (106, 545.0, 1700.0), (108, 550.0, 1650.0), (110, 560.0, 1400.0),
              (112, 570.0, 1200.0)]
    assert detect_bounce_frame(points) == 106


def test_detect_bounce_frame_returns_none_when_monotonic():
    # Pure release-to-bounce arc with no reversal yet (real pre-bounce data).
    points = [(100, 500.0, 900.0), (102, 520.0, 1000.0), (104, 540.0, 1100.0),
              (106, 560.0, 1200.0)]
    assert detect_bounce_frame(points) is None


def test_estimate_release_speed_kmh_uses_only_the_prebounce_phase():
    """The actual point of this feature: a fit spanning across a real
    bounce must not blend two different speeds into one wrong average
    — it must isolate the release-phase points and report THEIR speed."""
    pose = _synthetic_pose()
    true_vx, true_vy, true_vz = 0.2, 38.0, 1.5  # release-phase speed
    true_release_speed_kmh = np.hypot(np.hypot(true_vx, true_vy), true_vz) * 3.6

    z0 = 2.0
    # Real time-to-impact for this release state (solve 0 = z0 + vz*t - 0.5*g*t^2).
    t_bounce = (true_vz + np.hypot(true_vz, 0) + np.sqrt(true_vz ** 2 + 2 * GRAVITY_M_S2 * z0)) / GRAVITY_M_S2
    t_bounce = (true_vz + np.sqrt(true_vz ** 2 + 2 * GRAVITY_M_S2 * z0)) / GRAVITY_M_S2

    prebounce = _projectile_points(pose, 0.0, 0.5, z0, true_vx, true_vy, true_vz,
                                    frame0=100, fps=FPS, n_frames=10, frame_step=2)
    # Keep only real pre-bounce points (t <= t_bounce).
    prebounce = [p for p in prebounce if (p[0] - 100) / FPS <= t_bounce]

    # A deliberately DIFFERENT, much slower post-bounce phase, starting
    # from the bounce point with an upward vz (a real bounce reversal).
    bounce_frame = prebounce[-1][0] + 2
    x_b = 0.0 + true_vx * t_bounce
    y_b = 0.5 + true_vy * t_bounce
    postbounce = _projectile_points(pose, x_b, y_b, 0.05, true_vx * 0.8, true_vy * 0.6, 8.0,
                                     frame0=bounce_frame, fps=FPS, n_frames=8, frame_step=2)

    all_points = prebounce + postbounce
    result = estimate_release_speed_kmh(pose, all_points, FPS)

    assert result["status"] == "success"
    assert result["bounce_frame"] is not None
    # Must recover the PRE-bounce speed, not something blended with the
    # much slower/different post-bounce phase.
    assert result["speed_kmh"] == pytest.approx(true_release_speed_kmh, rel=0.05)
