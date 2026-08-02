"""
tests/test_orchestrator_metrics.py

Regression tests for orchestrator.py metric functions that had real,
confirmed bugs found on real footage this session:

  - calculate_hip_shoulder_separation: a NaN landmark used to compare
    False against every tier threshold and silently fall through to a
    confident (and false) "Blocked rotation" tier, instead of being
    flagged as a tracking failure.
  - calculate_release_height_ratio_safe: an implausibility ceiling meant
    to catch a MISTRACKED wrist was also rejecting a coach's directly
    confirmed release point — discarding a human's ground-truth
    observation on the theory that it must be a tracking glitch.
  - _find_grounded_reference_near: "grounded" only checked ankle-vs-knee/
    hip ORDERING, not whether the resulting nose-to-ankle span was even
    physically plausible — found on a real rear-view clip where a frame
    passed that ordering check with a compressed, implausible span
    (inaccurate ankle landmark, not a real crouch), got returned
    immediately, and then failed calculate_release_height_ratio_safe's
    own too-small-to-divide-by floor one step later — "N/A" instead of
    searching further out for a frame that was actually usable.
"""

import os
import subprocess

import numpy as np
import pandas as pd
import pytest

import orchestrator as o


def _hip_shoulder_row(**overrides):
    """A row where hips and shoulders are both clearly separated by a
    real rotation (~30 degrees) — a plausible mid-delivery pose."""
    base = {
        "frame": 10,
        "LEFT_SHOULDER_x": 0.40, "LEFT_SHOULDER_y": 0.30,
        "RIGHT_SHOULDER_x": 0.60, "RIGHT_SHOULDER_y": 0.35,
        "LEFT_HIP_x": 0.45, "LEFT_HIP_y": 0.50,
        "RIGHT_HIP_x": 0.55, "RIGHT_HIP_y": 0.50,
    }
    base.update(overrides)
    return base


class TestHipShoulderSeparation:
    def test_nan_landmark_returns_none_not_blocked_rotation(self):
        """The exact bug found on real footage: a NaN shoulder landmark
        used to fall through to tier='Blocked rotation', status='success'
        — a confident, false coaching claim generated from a pure
        tracking failure."""
        df = pd.DataFrame([_hip_shoulder_row(LEFT_SHOULDER_x=np.nan)])
        result = o.calculate_hip_shoulder_separation(df, ffc_frame=10)
        assert result["degrees"] is None
        assert result["status"] == "error"
        assert result["tier"] != "Blocked rotation"

    def test_missing_frame_returns_error(self):
        df = pd.DataFrame([_hip_shoulder_row(frame=1)])
        result = o.calculate_hip_shoulder_separation(df, ffc_frame=999)
        assert result["status"] == "error"

    def test_wraparound_case_produces_small_real_separation(self):
        """Regression test for the documented wraparound bug: when
        shoulder/hip angles straddle the +/-180 boundary, the OLD fold
        produced a negative nonsense value. This case (shoulder ~178,
        hip ~-178.7) should now report a small, physically real
        separation instead."""
        # Construct shoulder/hip vectors whose arctan2 angles straddle
        # the +/-180 boundary.
        row = _hip_shoulder_row(
            LEFT_SHOULDER_x=0.30, LEFT_SHOULDER_y=0.500,
            RIGHT_SHOULDER_x=0.70, RIGHT_SHOULDER_y=0.503,
            LEFT_HIP_x=0.70, LEFT_HIP_y=0.500,
            RIGHT_HIP_x=0.30, RIGHT_HIP_y=0.497,
        )
        df = pd.DataFrame([row])
        result = o.calculate_hip_shoulder_separation(df, ffc_frame=10)
        assert result["status"] == "success"
        # Must be a small real separation, not a negative/nonsense value
        # or a value anywhere near the raw (unwrapped) ~356 degree diff.
        assert 0 <= result["degrees"] <= 90

    def test_optimal_stretch_tier_boundary(self):
        assert o.calculate_hip_shoulder_separation(
            pd.DataFrame([_hip_shoulder_row()]), ffc_frame=10
        )["status"] == "success"


def _release_row(wrist_y=0.40, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20):
    return pd.Series({
        "RIGHT_WRIST_y": wrist_y,
        "LEFT_ANKLE_y": ankle_y, "RIGHT_ANKLE_y": ankle_y + 0.02,
        "LEFT_KNEE_y": knee_y, "LEFT_HIP_y": hip_y,
        "NOSE_y": nose_y,
    })


class TestReleaseHeightRatio:
    def test_missing_landmark_returns_error(self):
        row = pd.Series({"RIGHT_WRIST_y": None})
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right")
        assert result["status"] == "error"
        assert result["ratio"] is None

    def test_ankle_above_knee_is_flagged_implausible(self):
        """At release the plant foot is grounded — an ankle ABOVE the
        knee/hip in the frame indicates a mistracked landmark, not a
        real reading."""
        row = _release_row(ankle_y=0.10)  # ankle above knee/hip - implausible
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["status"] == "error"
        assert "implausible" in result["classification"].lower()

    def test_automatic_implausible_ratio_rejected_without_override(self):
        """An automatic (untouched) reading with a ratio outside
        physical bounds must be rejected — this is the case where 'this
        is probably a tracking glitch' is a reasonable inference. Uses a
        normal, plausible body height (nose/ankle span 0.65) so this
        specifically exercises the ratio-implausibility ceiling, not the
        separate 'body height too small' floor."""
        row = _release_row(wrist_y=-0.05, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["status"] == "error"
        assert result["classification"] == "Measurement error — verify camera angle"

    def test_same_implausible_ratio_accepted_with_coach_override(self):
        """THE regression-critical case found on real footage: a
        coach-confirmed wrist_override_norm must bypass the
        implausibility ceiling entirely — a human's direct observation
        is ground truth, not a tracking glitch to second-guess.
        Uses a normal, plausible body height (nose/ankle span 0.65) —
        only the OVERRIDDEN wrist position is extreme (ratio ~1.4,
        above the 1.30 ceiling that would reject an automatic reading)."""
        row = _release_row(ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(
            row, bowling_arm="right", reference_row=row,
            wrist_override_norm=(0.5, -0.05),
        )
        assert result["status"] == "success"
        assert result["ratio"] > 1.30

    def test_standard_mid_arm_release_classification(self):
        row = _release_row(wrist_y=0.40, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["status"] == "success"
        assert result["classification"] in (
            "Standard Mid-Arm Release", "High-Release Leverage", "Low-Sling Action",
        )


def _tracking_row(frame, nose_y, ankle_y, knee_y=0.70, hip_y=0.55):
    """LEFT-side landmarks only — lead side for a right-arm bowler."""
    return {
        "frame": frame, "NOSE_y": nose_y,
        "LEFT_ANKLE_y": ankle_y, "LEFT_KNEE_y": knee_y, "LEFT_HIP_y": hip_y,
    }


class TestFindGroundedReferenceNear:
    def test_skips_ordinally_grounded_frame_with_implausible_span(self):
        """The exact bug found on real rear-view footage: frame 10 has the
        ankle correctly below the knee/hip (passes the old check) but only
        0.03 away from the nose — a physically compressed span from a bad
        ankle reading, not a real crouch. Frame 12 is a genuinely normal,
        fully plausible standing reference a few frames away and must be
        preferred over the nearer-but-implausible frame 10."""
        df = pd.DataFrame([
            _tracking_row(frame=10, nose_y=0.53, ankle_y=0.56, knee_y=0.55, hip_y=0.54),
            _tracking_row(frame=11, nose_y=0.20, ankle_y=0.10, knee_y=0.25, hip_y=0.22),
            _tracking_row(frame=12, nose_y=0.20, ankle_y=0.85, knee_y=0.70, hip_y=0.55),
        ])
        ref = o._find_grounded_reference_near(df, frame_idx=10, bowling_arm="right", max_search=5)
        assert ref is not None
        assert ref["frame"] == 12

    def test_accepts_frame_at_idx_when_span_is_plausible(self):
        row = _tracking_row(frame=50, nose_y=0.20, ankle_y=0.85, knee_y=0.70, hip_y=0.55)
        df = pd.DataFrame([row])
        ref = o._find_grounded_reference_near(df, frame_idx=50, bowling_arm="right")
        assert ref is not None
        assert ref["frame"] == 50

    def test_returns_none_when_nothing_in_range_is_plausible(self):
        df = pd.DataFrame([
            _tracking_row(frame=10, nose_y=0.53, ankle_y=0.56, knee_y=0.55, hip_y=0.54),
        ])
        ref = o._find_grounded_reference_near(df, frame_idx=10, bowling_arm="right", max_search=3)
        assert ref is None


class TestLandmarksCsvPath:
    """Regression test for a real bug found during a broader audit:
    streamlit_app.py hardcoded "output/landmarks.csv" for Speed
    Estimation / Run-Up Analysis regardless of camera mode, but Dual
    Camera's pipeline never writes that file — only
    "landmarks_side.csv"/"landmarks_rear.csv". This either silently hid
    Speed/Run-Up entirely in Dual Camera mode, or worse, silently read a
    STALE landmarks.csv left over from an earlier Single Camera run in
    the same session — computing Dual Camera's numbers from a
    completely different, unrelated delivery."""

    def test_dual_camera_uses_the_side_stream_csv(self):
        assert o.landmarks_csv_path("Dual Camera") == os.path.join("output", "landmarks_side.csv")

    def test_single_camera_uses_the_single_stream_csv(self):
        assert o.landmarks_csv_path("Single Camera") == os.path.join("output", "landmarks.csv")

    def test_respects_a_custom_output_dir(self):
        assert o.landmarks_csv_path("Dual Camera", output_dir="tmp_out") == os.path.join("tmp_out", "landmarks_side.csv")


class _FakeUploadedFile:
    """Minimal stand-in for Streamlit's UploadedFile — just enough for
    save_uploaded_video_capped, which only touches .name and .getbuffer()."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content


class TestSaveUploadedVideoCapped:
    """Regression test for a real production crash found from a coach's
    device test: a native 4K (2160x3840) HEVC recording straight off a
    phone camera crashed the app during Execute Analysis — every clip
    tested before that had gone through WhatsApp first, which
    re-compresses to well under 1080p, so this pipeline had never
    actually been asked to decode/process a full-resolution native
    recording. MediaPipe/OpenCV/ffmpeg all pay a per-frame cost
    proportional to pixel count, and nothing capped that anywhere.

    ffmpeg itself isn't available in this test environment, so these
    cover the one behavior that's safe and meaningful to verify without
    it: never fail outright, never silently drop the coach's video, even
    if the downscale step itself can't run."""

    def test_falls_back_to_original_bytes_when_ffmpeg_is_missing(self, tmp_path, monkeypatch):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda name: None)

        content = b"not a real video, just bytes to prove passthrough"
        fake_file = _FakeUploadedFile("clip.mp4", content)
        dest = str(tmp_path / "saved.mp4")

        o.save_uploaded_video_capped(fake_file, dest)

        assert os.path.exists(dest)
        with open(dest, "rb") as f:
            assert f.read() == content

    def test_creates_the_destination_directory_if_missing(self, tmp_path, monkeypatch):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda name: None)

        content = b"video bytes"
        fake_file = _FakeUploadedFile("clip.mov", content)
        dest = str(tmp_path / "nested" / "dir" / "saved.mp4")

        o.save_uploaded_video_capped(fake_file, dest)

        assert os.path.exists(dest)

    def test_raises_instead_of_falling_back_when_ffmpeg_fails_on_this_file(self, tmp_path, monkeypatch):
        """BUG FOUND (2026-08-02): an iPhone 17 Pro native recording crashed
        the whole shared Streamlit Cloud process outright. The old fallback
        treated a failed downscale attempt the same as "ffmpeg not
        installed" and silently handed the untouched (often even more
        demanding) original to the rest of the pipeline — which just moves
        the same crash a few steps later instead of preventing it. A real
        per-file failure must now surface as a catchable error, not a
        silent, dangerous fallback."""
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/ffmpeg")

        class _FailedResult:
            returncode = 1
            stderr = b"some ffmpeg failure output"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FailedResult())

        fake_file = _FakeUploadedFile("clip.mp4", b"pretend this is a huge native 4K HDR file")
        dest = str(tmp_path / "saved.mp4")

        with pytest.raises(RuntimeError):
            o.save_uploaded_video_capped(fake_file, dest)

        assert not os.path.exists(dest)

    def test_raises_instead_of_falling_back_when_ffmpeg_times_out(self, tmp_path, monkeypatch):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/ffmpeg")

        def _raise_timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=180)

        monkeypatch.setattr(subprocess, "run", _raise_timeout)

        fake_file = _FakeUploadedFile("clip.mp4", b"pretend this is a huge native 4K HDR file")
        dest = str(tmp_path / "saved.mp4")

        with pytest.raises(RuntimeError):
            o.save_uploaded_video_capped(fake_file, dest)

        assert not os.path.exists(dest)
