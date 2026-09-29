"""
Tests for ball_tracking/pitch_calibration.py's 3D camera pose solver
(2026-09-27) — the fix for a real, traced bug: the older flat
ground-plane homography has no way to represent height, so a ball at
head height and one resting on the pitch project to the same point.
On a real clip (IMG_4060.MOV, a genuine bouncer) this produced chaotic,
tens-of-meters jumps between adjacent tracked frames.

Uses synthetic data with a KNOWN camera pose — the only way to verify
correctness directly, since there's no independent ground truth for a
real camera's exact position/angle. Real-clip validation (does this
produce a plausible speed on real footage) is a separate, later step
that needs a genuinely tracked trajectory to fit against.
"""
import numpy as np
import cv2
import pytest

from ball_tracking.pitch_calibration import (
    STUMP_LINE_WIDTH_M, STUMP_HEIGHT_M, PITCH_LENGTH_M,
    solve_camera_pose, project_3d_to_pixel, pixel_to_ground_at_height,
    build_ground_homography, pixel_to_ground,
)

IMAGE_W, IMAGE_H = 1920, 1080


def _synthetic_pose(true_f=1600.0, camera_height=1.5, camera_back=4.0):
    """A real, known camera pose matching FullTrack AI's own documented
    setup (see project memory): ~4m behind the near stumps, ~1.5m high,
    tilted down toward the pitch."""
    cx, cy = IMAGE_W / 2.0, IMAGE_H / 2.0
    K = np.array([[true_f, 0, cx], [0, true_f, cy], [0, 0, 1]], dtype=np.float64)
    # Camera sits behind Y=0 (negative Y) and above the ground, tilted
    # down. Build world->camera rotation directly: camera looks mostly
    # along +Y (down the pitch) and slightly down.
    tilt_rad = np.radians(15)
    # Camera-space axes expressed in world space: forward, right, down.
    forward = np.array([0, np.cos(tilt_rad), -np.sin(tilt_rad)])
    right = np.array([1, 0, 0])
    down = np.cross(forward, right)
    R_world_to_cam = np.array([right, down, forward])  # rows = camera axes in world coords
    camera_pos_world = np.array([0.0, -camera_back, camera_height])
    tvec = -R_world_to_cam @ camera_pos_world
    rvec, _ = cv2.Rodrigues(R_world_to_cam)
    return {"rvec": rvec, "tvec": tvec.reshape(3, 1),
            "focal_length_px": true_f, "principal_point_px": (cx, cy)}, K


def _project(true_pose, K, x_m, y_m, z_m, dist=None):
    pt = np.array([[[x_m, y_m, z_m]]], dtype=np.float64)
    proj, _ = cv2.projectPoints(pt, true_pose["rvec"], true_pose["tvec"], K, dist)
    return float(proj[0, 0, 0]), float(proj[0, 0, 1])


def _stump_pixels(true_pose, K, dist=None):
    half_w = STUMP_LINE_WIDTH_M / 2
    return {
        "near_left_px": _project(true_pose, K, -half_w, 0.0, 0.0, dist),
        "near_right_px": _project(true_pose, K, half_w, 0.0, 0.0, dist),
        "far_left_px": _project(true_pose, K, -half_w, PITCH_LENGTH_M, 0.0, dist),
        "far_right_px": _project(true_pose, K, half_w, PITCH_LENGTH_M, 0.0, dist),
        "near_left_top_px": _project(true_pose, K, -half_w, 0.0, STUMP_HEIGHT_M, dist),
        "near_right_top_px": _project(true_pose, K, half_w, 0.0, STUMP_HEIGHT_M, dist),
    }


def _solve(px, image_w=IMAGE_W, image_h=IMAGE_H):
    return solve_camera_pose(
        px["near_left_px"], px["near_right_px"], px["far_left_px"], px["far_right_px"],
        px["near_left_top_px"], px["near_right_top_px"], image_w, image_h,
    )


def test_solve_camera_pose_recovers_a_consistent_pose_from_synthetic_stumps():
    true_pose, K = _synthetic_pose()
    px = _stump_pixels(true_pose, K)
    result = _solve(px)
    assert result["status"] == "success"
    assert result["reprojection_error_px"] < 1.0
    # Focal length should be recovered close to the true synthetic value.
    assert abs(result["focal_length_px"] - true_pose["focal_length_px"]) < 5.0


def test_solved_pose_correctly_distinguishes_ground_from_elevated_points():
    """The actual bug being fixed: a flat ground-plane homography maps
    an elevated point to the SAME place as a ground point at the same
    pixel — a solved 3D pose must not do that."""
    true_pose, K = _synthetic_pose()
    px = _stump_pixels(true_pose, K)
    pose = _solve(px)
    assert pose["status"] == "success"

    # A ball at real-world (0m sideways, 10m down the pitch), at two
    # different heights, must project to two DIFFERENT pixel positions.
    ground_px = project_3d_to_pixel(pose, 0.0, 10.0, 0.0)
    elevated_px = project_3d_to_pixel(pose, 0.0, 10.0, 1.5)
    assert abs(ground_px[1] - elevated_px[1]) > 20.0  # clearly different rows


def test_pixel_to_ground_at_height_recovers_the_true_3d_point():
    true_pose, K = _synthetic_pose()
    px = _stump_pixels(true_pose, K)
    pose = _solve(px)
    assert pose["status"] == "success"

    true_x, true_y, true_z = 0.05, 12.3, 0.8
    ball_px = project_3d_to_pixel(pose, true_x, true_y, true_z)
    recovered_x, recovered_y = pixel_to_ground_at_height(pose, ball_px[0], ball_px[1], true_z)
    assert recovered_x == pytest.approx(true_x, abs=0.05)
    assert recovered_y == pytest.approx(true_y, abs=0.05)


def test_pixel_to_ground_at_height_agrees_with_flat_homography_at_ground_level():
    """At z_m=0, the new height-aware inversion must agree with the
    existing flat build_ground_homography/pixel_to_ground — it's a
    strict generalization, not a different method."""
    true_pose, K = _synthetic_pose()
    px = _stump_pixels(true_pose, K)
    pose = _solve(px)
    assert pose["status"] == "success"

    H = build_ground_homography(px["near_left_px"], px["near_right_px"],
                                 px["far_left_px"], px["far_right_px"])

    true_x, true_y = -0.05, 8.0
    ground_px = project_3d_to_pixel(pose, true_x, true_y, 0.0)

    x_flat, y_flat = pixel_to_ground(H, *ground_px)
    x_3d, y_3d = pixel_to_ground_at_height(pose, *ground_px, z_m=0.0)
    assert x_3d == pytest.approx(x_flat, abs=0.05)
    assert y_3d == pytest.approx(y_flat, abs=0.05)


def test_solve_camera_pose_recovers_pose_despite_real_lens_distortion():
    """A real, moderate lens-distortion magnitude, baked into the
    synthetic points the same way a real lens would, must still be
    recovered accurately by the joint (focal length, k1) search."""
    true_pose, K = _synthetic_pose()
    true_k1 = -0.12  # a real, moderate barrel-distortion magnitude
    true_dist = np.array([true_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    px = _stump_pixels(true_pose, K, dist=true_dist)

    result = _solve(px)
    assert result["status"] == "success"
    assert result["max_reprojection_error_px"] < 1.0
    assert result["dist_coeffs"][0] == pytest.approx(true_k1, abs=0.02)


def test_solve_camera_pose_reports_error_on_inconsistent_points():
    true_pose, K = _synthetic_pose()
    px = _stump_pixels(true_pose, K)
    # Corrupt the stump-top click badly (as if mis-clicked far from the
    # real stump top) -- no focal length should reproject this well.
    px["near_left_top_px"] = (px["near_left_top_px"][0] + 400, px["near_left_top_px"][1] + 400)
    result = _solve(px)
    assert result["status"] == "error"


def test_solve_camera_pose_catches_a_real_left_right_mixup():
    """Real regression test (2026-09-27, IMG_3796.MOV): the coach
    mislabeled which near stump was "left" vs "right" -- an easy,
    understandable mistake, not carelessness. With only one top point
    (the original design), this was invisible to the reprojection
    check: it silently paired the surviving top click with the WRONG
    base, still reprojected well enough to look fixed, then produced a
    physically absurd downstream speed (~16 km/h, moving almost
    straight up/down). Requiring both stumps' tops catches this
    directly and explainably, rather than a mysterious high error."""
    true_pose, K = _synthetic_pose()
    px = _stump_pixels(true_pose, K)
    # Swap only the BASE points -- exactly the coach's real mistake --
    # while both top points stay correctly paired with their real base.
    swapped = dict(px)
    swapped["near_left_px"], swapped["near_right_px"] = px["near_right_px"], px["near_left_px"]

    result = _solve(swapped)
    assert result["status"] == "error"
    assert "left/right" in result["message"].lower()
