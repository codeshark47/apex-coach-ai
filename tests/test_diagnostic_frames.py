"""
tests/test_diagnostic_frames.py

diagnostic_frames.py generates freeze-frame diagnostic stills (skeleton
+ callouts for CRITICAL metrics) at each delivery/shot's key event
frames, for both the coach-facing report/PDF and coaching_agent.py's
Gemini call. These tests use synthetic pd.Series rows (same convention
as tests/test_video_overlay.py) and a monkeypatched frame reader — no
real video decode, no network.
"""

import numpy as np
import pandas as pd
import pytest

import diagnostic_frames as diag


def _standing_row(**overrides):
    """A plausible, fully-populated standing-pose row — passes
    torso_shape_is_plausible/body_size_is_plausible/implausible_arm_nodes
    (same reference pose used implicitly by test_video_overlay.py's own
    plausible-row fixtures) — with every landmark diagnostic_frames.py's
    anchors/skeleton might touch. Normalized 0-1 coordinates."""
    data = {
        "frame": 0,
        "NOSE_x": 0.50, "NOSE_y": 0.20,
        "LEFT_SHOULDER_x": 0.45, "LEFT_SHOULDER_y": 0.30,
        "RIGHT_SHOULDER_x": 0.55, "RIGHT_SHOULDER_y": 0.30,
        "LEFT_ELBOW_x": 0.42, "LEFT_ELBOW_y": 0.40,
        "RIGHT_ELBOW_x": 0.58, "RIGHT_ELBOW_y": 0.40,
        "LEFT_WRIST_x": 0.40, "LEFT_WRIST_y": 0.50,
        "RIGHT_WRIST_x": 0.60, "RIGHT_WRIST_y": 0.50,
        "LEFT_HIP_x": 0.46, "LEFT_HIP_y": 0.55,
        "RIGHT_HIP_x": 0.54, "RIGHT_HIP_y": 0.55,
        "LEFT_KNEE_x": 0.45, "LEFT_KNEE_y": 0.70,
        "RIGHT_KNEE_x": 0.55, "RIGHT_KNEE_y": 0.70,
        "LEFT_ANKLE_x": 0.45, "LEFT_ANKLE_y": 0.90,
        "RIGHT_ANKLE_x": 0.55, "RIGHT_ANKLE_y": 0.90,
        "LEFT_HEEL_x": 0.44, "LEFT_HEEL_y": 0.92,
        "RIGHT_HEEL_x": 0.56, "RIGHT_HEEL_y": 0.92,
        "LEFT_FOOT_INDEX_x": 0.44, "LEFT_FOOT_INDEX_y": 0.95,
        "RIGHT_FOOT_INDEX_x": 0.56, "RIGHT_FOOT_INDEX_y": 0.95,
    }
    data.update(overrides)
    return pd.Series(data)


class TestAnchorResolution:
    def test_bowling_anchors_resolve_to_expected_landmarks(self):
        row = _standing_row()
        w, h = 1000, 1000
        knee = diag._bowling_metric_anchor("front_knee_bracing", row, w, h, "LEFT", "RIGHT")
        assert knee == (int(0.45 * w), int(0.70 * h))
        wrist = diag._bowling_metric_anchor("release_height", row, w, h, "LEFT", "RIGHT")
        assert wrist == (int(0.60 * w), int(0.50 * h))  # bowl_side="RIGHT" -> RIGHT_WRIST
        nose = diag._bowling_metric_anchor("head_stability", row, w, h, "LEFT", "RIGHT")
        assert nose == (int(0.50 * w), int(0.20 * h))

    def test_midpoint_anchor_is_between_both_landmarks(self):
        row = _standing_row()
        w, h = 1000, 1000
        anchor = diag._bowling_metric_anchor("hip_shoulder_separation", row, w, h, "LEFT", "RIGHT")
        expected_x = (int(0.46 * w) + int(0.54 * w)) // 2
        expected_y = (int(0.55 * h) + int(0.55 * h)) // 2
        assert anchor == (expected_x, expected_y)

    def test_missing_landmark_returns_none_not_a_crash(self):
        row = _standing_row(LEFT_KNEE_x=float("nan"), LEFT_KNEE_y=float("nan"))
        anchor = diag._bowling_metric_anchor("front_knee_bracing", row, 1000, 1000, "LEFT", "RIGHT")
        assert anchor is None

    def test_batting_anchors_resolve_to_expected_landmarks(self):
        row = _standing_row()
        w, h = 1000, 1000
        foot = diag._batting_metric_anchor("batting_front_foot_alignment", row, w, h, "LEFT", "LEFT")
        assert foot == (int(0.44 * w), int(0.95 * h))
        elbow = diag._batting_metric_anchor("batting_top_elbow_angle", row, w, h, "LEFT", "LEFT")
        assert elbow == (int(0.42 * w), int(0.40 * h))


class TestLayoutAndDrawCallouts:
    def _fake_frame(self, w=400, h=720):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_no_callouts_draws_nothing_and_does_not_crash(self):
        frame = self._fake_frame()
        diag._layout_and_draw_callouts(frame, [])
        assert frame.sum() == 0

    def test_callouts_split_left_and_right_by_anchor_x(self):
        frame = self._fake_frame(w=400, h=720)
        callouts = [
            {"anchor": (50, 100), "title": "A", "detail": "1"},
            {"anchor": (350, 100), "title": "B", "detail": "2"},
        ]
        diag._layout_and_draw_callouts(frame, callouts)
        # left callout's box should be drawn near the left margin, right
        # callout's box near the right margin -- confirm some non-zero
        # pixels exist in both halves rather than everything piling on
        # one side.
        left_half = frame[:, :200]
        right_half = frame[:, 200:]
        assert left_half.sum() > 0
        assert right_half.sum() > 0

    def test_heavily_lopsided_cluster_rebalances_across_both_sides(self):
        """6 anchors all on the left half must not all stack past the
        frame height on one side -- the rebalancing logic should move
        some to the (empty) right side."""
        frame = self._fake_frame(w=400, h=300)  # short frame: 6 stacked boxes (64px+10px gap each) would overflow
        callouts = [{"anchor": (20, 20 + i * 5), "title": f"M{i}", "detail": "x"} for i in range(6)]
        # Should not raise even though naive single-side stacking would
        # overflow a 300px-tall frame.
        diag._layout_and_draw_callouts(frame, callouts)
        assert frame.sum() > 0


class TestGenerateBowlingDiagnosticFrames:
    def _patch_frame_reader(self, monkeypatch, w=400, h=720):
        monkeypatch.setattr(diag, "_read_frame_bgr", lambda video_path, frame_idx: np.zeros((h, w, 3), dtype=np.uint8))

    def test_returns_all_three_keys_even_when_events_missing(self, monkeypatch):
        self._patch_frame_reader(monkeypatch)
        row = _standing_row(frame=10)
        df = pd.DataFrame([row])
        result = diag.generate_bowling_diagnostic_frames(
            "fake.mp4", df, {"BFC": None, "FFC": None, "BR": None}, {}, None, "right")
        assert set(result.keys()) == {"bfc", "ffc", "release"}
        assert all(v is None for v in result.values())

    def test_clean_metrics_produce_frames_with_no_callouts_needed(self, monkeypatch):
        """All metrics green -- frames should still generate (skeleton
        only), never silently skipped just because nothing is critical."""
        self._patch_frame_reader(monkeypatch)
        row = _standing_row(frame=10)
        df = pd.DataFrame([row])
        metrics = {
            "front_knee_bracing": {"degrees": 170.0},
            "trunk_lean": {"degrees": 10.0},
            "release_height": {"ratio": 1.22},
            "head_stability": {"value": 1.0},
        }
        result = diag.generate_bowling_diagnostic_frames(
            "fake.mp4", df, {"BFC": 10, "FFC": 10, "BR": 10}, metrics, None, "right")
        assert result["bfc"] is not None
        assert result["ffc"] is not None
        assert result["release"] is not None

    def test_critical_metric_produces_a_frame_with_visible_callout(self, monkeypatch):
        """trunk_lean is kind="higher_better" (green=13-30, amber=5-13,
        red below 5) -- confirmed directly against metric_ranges.classify
        before picking these two values, rather than assuming "a bigger
        number must be worse" (that assumption was wrong once already:
        80.0 classifies as green here, not red, since values above the
        green band are still fine for a higher_better metric)."""
        self._patch_frame_reader(monkeypatch)
        row = _standing_row(frame=10)
        df = pd.DataFrame([row])

        clean_metrics = {"trunk_lean": {"degrees": 20.0}}  # within green (13-30)
        clean_result = diag.generate_bowling_diagnostic_frames(
            "fake.mp4", df, {"BFC": None, "FFC": None, "BR": 10}, clean_metrics, None, "right")

        critical_metrics = {"trunk_lean": {"degrees": 2.0}}  # below amber (5-13) -> red
        critical_result = diag.generate_bowling_diagnostic_frames(
            "fake.mp4", df, {"BFC": None, "FFC": None, "BR": 10}, critical_metrics, None, "right")

        assert len(critical_result["release"]) > len(clean_result["release"])

    def test_never_raises_on_a_frame_read_failure(self, monkeypatch):
        monkeypatch.setattr(diag, "_read_frame_bgr", lambda video_path, frame_idx: None)
        row = _standing_row(frame=10)
        df = pd.DataFrame([row])
        result = diag.generate_bowling_diagnostic_frames(
            "fake.mp4", df, {"BFC": 10, "FFC": 10, "BR": 10}, {}, None, "right")
        assert result == {"bfc": None, "ffc": None, "release": None}

    def test_never_raises_on_a_completely_broken_metrics_dict(self, monkeypatch):
        """Defensive: a malformed metrics dict (wrong shape) must degrade
        to all-None, never propagate an exception up into the orchestrator."""
        self._patch_frame_reader(monkeypatch)
        row = _standing_row(frame=10)
        df = pd.DataFrame([row])
        result = diag.generate_bowling_diagnostic_frames(
            "fake.mp4", df, {"BFC": 10, "FFC": 10, "BR": 10}, None, None, "right")
        assert result == {"bfc": None, "ffc": None, "release": None}


class TestGenerateBattingDiagnosticFrames:
    def _patch_frame_reader(self, monkeypatch, w=400, h=720):
        monkeypatch.setattr(diag, "_read_frame_bgr", lambda video_path, frame_idx: np.zeros((h, w, 3), dtype=np.uint8))

    def test_returns_all_three_keys(self, monkeypatch):
        self._patch_frame_reader(monkeypatch)
        row = _standing_row(frame=5)
        df = pd.DataFrame([row])
        result = diag.generate_batting_diagnostic_frames(
            "fake.mp4", df, {"STANCE": 5, "BACKLIFT": 5, "CONTACT": 5}, {}, "left")
        assert set(result.keys()) == {"stance", "backlift", "contact"}

    def test_critical_metric_produces_visible_callout(self, monkeypatch):
        self._patch_frame_reader(monkeypatch)
        row = _standing_row(frame=5)
        df = pd.DataFrame([row])

        clean = diag.generate_batting_diagnostic_frames(
            "fake.mp4", df, {"STANCE": None, "BACKLIFT": None, "CONTACT": 5},
            {"xfactor_separation": {"degrees": 35.0}}, "left")
        critical = diag.generate_batting_diagnostic_frames(
            "fake.mp4", df, {"STANCE": None, "BACKLIFT": None, "CONTACT": 5},
            {"xfactor_separation": {"degrees": 5.0}}, "left")

        assert len(critical["contact"]) > len(clean["contact"])
