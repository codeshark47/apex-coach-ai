"""
tests/test_run_up_analysis.py

Regression test for a real, confirmed bug: the peak-detection window in
_detect_contacts_for_foot was `max(2, int(fps * 0.05))`, which floors to
2 at every frame rate this app has ever actually seen on real footage
(25fps, 27fps, 30fps — even "slow-mo" clips end up here once WhatsApp or
similar normalizes the frame rate, see the ball-tracking work this same
session). Verified directly: a window of 2 finds ZERO peaks even on a
perfectly clean, noise-free synthetic stride signal with 6 unambiguous
real peaks — near a true peak the curve is locally flat, so a point only
2 frames away is barely lower than the peak and never clears the
prominence margin. This is very likely why run-up analysis looked
completely broken on real coaching footage — not occasionally, but on
nearly every clip, since real footage is almost always in the exact fps
range this was silently guaranteed to fail on.
"""

import numpy as np
import pandas as pd

import run_up_analysis as rua


def _synthetic_stride_df(fps=30, n_frames=90, cadence_hz=2.0, amplitude=0.05):
    """A clean, idealized stride signal — foot vertical position
    oscillating sinusoidally, no tracking gaps, no noise. Not realistic
    running dynamics, but a controlled way to test the peak-detection
    algorithm in isolation from real-footage data-quality problems."""
    t = np.arange(n_frames) / fps
    heel_y = 0.5 + amplitude * np.sin(2 * np.pi * cadence_hz * t)
    return pd.DataFrame({
        "frame": np.arange(n_frames),
        "LEFT_HEEL_y": heel_y,
        "RIGHT_HEEL_y": heel_y,
        "LEFT_ANKLE_y": heel_y,
        "RIGHT_ANKLE_y": heel_y,
    })


class TestGetSeriesDoesNotFabricateEdgeGaps:
    """Regression test for a real bug found during a broader audit:
    _get_series's interpolate() call had no limit_area="inside", so a
    leading or trailing run of missing frames (e.g. the bowler not yet
    confidently detected at the very start of the run-up, a scenario
    this module's own docstring describes as common) got filled with a
    frozen, fabricated value copied from the nearest real detection —
    the same "flat line erases a genuine footfall peak" failure this
    module was rewritten to fix for interior gaps, just left open at the
    clip's edges."""

    def test_leading_gap_stays_nan_not_a_frozen_value(self):
        fps = 30
        vals = [np.nan] * 10 + list(np.arange(10, 30, dtype=float))
        df = pd.DataFrame({"LEFT_HEEL_y": vals})

        series = rua._get_series(df, "LEFT_HEEL_y", frame_width=640, frame_height=480,
                                  is_x=False, fps=fps)

        assert np.isnan(series[:10]).all()

    def test_short_interior_gap_is_still_bridged_normally(self):
        """The fix must not disable interior gap-filling — only stop
        fabricating data at the clip's edges."""
        vals = [10.0, 11.0, 12.0, np.nan, np.nan, 15.0, 16.0]
        df = pd.DataFrame({"LEFT_HEEL_y": vals})

        series = rua._get_series(df, "LEFT_HEEL_y", frame_width=640, frame_height=480,
                                  is_x=False, fps=30)

        assert not np.isnan(series[3:5]).any()


class TestDetectRunUpStrides:
    def test_clean_synthetic_stride_signal_at_30fps_is_detected(self):
        """The exact regression case: at 30fps (the most common real
        frame rate in this app), a clean, unambiguous stride pattern
        must actually be detected, not silently return zero contacts."""
        df = _synthetic_stride_df(fps=30, n_frames=90, cadence_hz=2.0)
        result = rua.detect_run_up_strides(df, bfc_frame_idx=90, fps=30,
                                            frame_width=478, frame_height=850)
        assert result["status"] == "success"
        assert result["stride_count"] > 0

    def test_clean_synthetic_stride_signal_at_25fps_is_detected(self):
        """Same regression, at the other common real-world frame rate
        (WhatsApp-compressed clips have consistently landed on 25 or 30
        fps this session, never anything higher)."""
        df = _synthetic_stride_df(fps=25, n_frames=75, cadence_hz=2.0)
        result = rua.detect_run_up_strides(df, bfc_frame_idx=75, fps=25,
                                            frame_width=478, frame_height=850)
        assert result["status"] == "success"
        assert result["stride_count"] > 0

    def test_flat_signal_correctly_reports_no_contacts(self):
        """Not everything should detect a stride — a genuinely flat
        signal (no real foot motion) must still correctly report no
        contacts, not be broken by the window-size fix in the other
        direction (false positives on noise)."""
        df = pd.DataFrame({
            "frame": np.arange(60),
            "LEFT_HEEL_y": [0.5] * 60,
            "RIGHT_HEEL_y": [0.5] * 60,
            "LEFT_ANKLE_y": [0.5] * 60,
            "RIGHT_ANKLE_y": [0.5] * 60,
        })
        result = rua.detect_run_up_strides(df, bfc_frame_idx=60, fps=30,
                                            frame_width=478, frame_height=850)
        assert result["status"] == "error"

    def test_too_short_clip_reports_insufficient_runup(self):
        df = _synthetic_stride_df(fps=30, n_frames=20)
        result = rua.detect_run_up_strides(df, bfc_frame_idx=5, fps=30,
                                            frame_width=478, frame_height=850)
        assert result["status"] == "insufficient_runup"
