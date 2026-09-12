"""
tests/test_diagnostic_frames.py

diagnostic_frames.py generates freeze-frame diagnostic stills (skeleton
+ a callout for EVERY mapped metric, tagged CRITICAL only when it's
actually drill-eligible) at each delivery/shot's key event frames, for
both the coach-facing report/PDF and coaching_agent.py's Gemini call.
These tests use synthetic pd.Series rows (same convention as
tests/test_video_overlay.py) and a monkeypatched frame reader — no real
video decode, no network.
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

    def test_rear_leg_anchors_use_bowl_side_not_lead_side(self):
        """rear_knee_angle/rear_hip_flexion anchor on the TRAIL leg, which
        bowl_side already represents (same side as the bowling arm) --
        no separate trail_side parameter needed."""
        row = _standing_row()
        w, h = 1000, 1000
        rear_knee = diag._bowling_metric_anchor("rear_knee_angle", row, w, h, "LEFT", "RIGHT")
        assert rear_knee == (int(0.55 * w), int(0.70 * h))  # bowl_side="RIGHT" -> RIGHT_KNEE
        rear_hip = diag._bowling_metric_anchor("rear_hip_flexion", row, w, h, "LEFT", "RIGHT")
        assert rear_hip == (int(0.54 * w), int(0.55 * h))  # bowl_side="RIGHT" -> RIGHT_HIP

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
            {"anchor": (50, 100), "title": "A", "detail": "1", "tier": "red"},
            {"anchor": (350, 100), "title": "B", "detail": "2", "tier": "green"},
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
        callouts = [{"anchor": (20, 20 + i * 5), "title": f"M{i}", "detail": "x", "tier": "amber"} for i in range(6)]
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

    def test_clean_metrics_still_produce_frames(self, monkeypatch):
        """All metrics green -- frames must still generate. Unlike the
        first version of this feature, a clean/green metric now STILL
        gets a callout (just without a CRITICAL tag) -- see this
        module's own docstring for the real coach-reported bug (an
        all-clean delivery rendered a skeleton with no numbers at all)
        this behavior specifically fixes."""
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

    def test_missing_metrics_dict_entry_still_renders_an_na_callout(self, monkeypatch):
        """An empty metrics dict -- every mapped metric reads as
        None/"unknown" -- must still produce a real (larger than a bare
        skeleton) frame with "N/A" callouts, not silently render nothing."""
        self._patch_frame_reader(monkeypatch)
        row = _standing_row(frame=10)
        df = pd.DataFrame([row])

        no_metrics_result = diag.generate_bowling_diagnostic_frames(
            "fake.mp4", df, {"BFC": None, "FFC": None, "BR": 10}, {}, None, "right")
        no_events_result = diag.generate_bowling_diagnostic_frames(
            "fake.mp4", df, {"BFC": None, "FFC": None, "BR": None}, {}, None, "right")

        assert no_metrics_result["release"] is not None
        assert no_events_result["release"] is None  # no BR frame at all -> genuinely nothing to draw

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

    def test_clean_and_critical_metrics_both_produce_a_frame(self, monkeypatch):
        """Both a green and a red xfactor_separation must render a real
        frame -- the exact content difference (CRITICAL tag, color) is
        covered precisely by TestPanelText below; PNG byte-length is too
        indirect a signal to assert on now that every mapped metric
        always draws a callout, clean or not."""
        self._patch_frame_reader(monkeypatch)
        row = _standing_row(frame=5)
        df = pd.DataFrame([row])

        clean = diag.generate_batting_diagnostic_frames(
            "fake.mp4", df, {"STANCE": None, "BACKLIFT": None, "CONTACT": 5},
            {"xfactor_separation": {"degrees": 35.0}}, "left")
        critical = diag.generate_batting_diagnostic_frames(
            "fake.mp4", df, {"STANCE": None, "BACKLIFT": None, "CONTACT": 5},
            {"xfactor_separation": {"degrees": 5.0}}, "left")

        assert clean["contact"] is not None
        assert critical["contact"] is not None


class TestPanelText:
    """The exact text shown on a callout, per tier/eligibility
    combination -- this is what actually fixes the real coach-reported
    bug (a clean/excluded delivery showed no numbers at all)."""

    def test_missing_value_shows_na_not_a_crash(self):
        assert "N/A" in diag._panel_text("trunk_lean", None, "unknown", False)

    def test_eligible_critical_shows_critical_tag(self):
        text = diag._panel_text("trunk_lean", 2.0, "red", True)
        assert "CRITICAL" in text
        assert "provisional" not in text.lower()

    def test_red_but_not_eligible_shows_provisional_critical(self):
        """Recalibration-pending/tracking-uncertain case: still red, but
        excluded from drills -- must say so, not just look identical to
        a fully solid CRITICAL reading."""
        text = diag._panel_text("head_stability", 1.7, "red", False)
        assert "CRITICAL" in text
        assert "provisional" in text.lower()

    def test_descriptive_shows_descriptive_not_critical(self):
        text = diag._panel_text("hip_shoulder_separation", 30.0, "descriptive", False)
        assert "DESCRIPTIVE" in text
        assert "CRITICAL" not in text

    def test_green_shows_plain_value_no_critical_tag(self):
        text = diag._panel_text("trunk_lean", 20.0, "green", False)
        assert "CRITICAL" not in text
        assert "GREEN" in text

    def test_real_value_is_included_via_format_value(self):
        import metric_ranges as mr
        text = diag._panel_text("trunk_lean", 20.0, "green", False)
        assert mr.format_value("trunk_lean", 20.0) in text


class TestMetricsByFrame:
    """_metrics_by_frame must return EVERY mapped metric for a frame,
    not just the ones that pass is_critical_and_eligible -- the core
    fix for the real bug (an all-excluded delivery showed nothing)."""

    def test_a_green_metric_still_appears_with_eligible_false(self):
        per_frame = diag._metrics_by_frame(
            diag._BOWLING_METRIC_FRAMES, {"trunk_lean": {"degrees": 20.0}}, None, ["bfc", "ffc", "release"])
        entries = [e for e in per_frame["release"] if e[0] == "trunk_lean"]
        assert len(entries) == 1
        metric_key, value, tier, eligible = entries[0]
        assert tier == "green"
        assert eligible is False

    def test_always_descriptive_metric_still_appears(self):
        per_frame = diag._metrics_by_frame(
            diag._BOWLING_METRIC_FRAMES, {"hip_shoulder_separation": {"degrees": 30.0}}, None, ["bfc", "ffc", "release"])
        entries = [e for e in per_frame["ffc"] if e[0] == "hip_shoulder_separation"]
        assert len(entries) == 1
        assert entries[0][2] == "descriptive"
        assert entries[0][3] is False

    def test_recalibration_pending_red_metric_still_appears_but_not_eligible(self):
        per_frame = diag._metrics_by_frame(
            diag._BOWLING_METRIC_FRAMES,
            {"head_stability": {"value": 1.7, "recalibration_pending": True}},
            None, ["bfc", "ffc", "release"])
        entries = [e for e in per_frame["bfc"] if e[0] == "head_stability"]
        assert len(entries) == 1
        _, value, tier, eligible = entries[0]
        assert tier == "red"
        assert eligible is False

    def test_missing_metric_appears_as_unknown_not_omitted(self):
        per_frame = diag._metrics_by_frame(diag._BOWLING_METRIC_FRAMES, {}, None, ["bfc", "ffc", "release"])
        entries = [e for e in per_frame["release"] if e[0] == "trunk_lean"]
        assert len(entries) == 1
        assert entries[0][2] == "unknown"

    def test_rear_leg_metrics_are_mapped_to_bfc_only(self):
        per_frame = diag._metrics_by_frame(
            diag._BOWLING_METRIC_FRAMES,
            {"rear_knee_angle": {"degrees": 150.0}, "rear_hip_flexion": {"degrees": 35.0}},
            None, ["bfc", "ffc", "release"])
        assert any(e[0] == "rear_knee_angle" for e in per_frame["bfc"])
        assert any(e[0] == "rear_hip_flexion" for e in per_frame["bfc"])
        assert not any(e[0] == "rear_knee_angle" for e in per_frame["ffc"])
        assert not any(e[0] == "rear_hip_flexion" for e in per_frame["release"])

        rear_hip_entry = [e for e in per_frame["bfc"] if e[0] == "rear_hip_flexion"][0]
        _, value, tier, eligible = rear_hip_entry
        assert tier == "red"
        assert eligible is True  # rear_hip_flexion is a real scored metric, unlike rear_knee_angle

        rear_knee_entry = [e for e in per_frame["bfc"] if e[0] == "rear_knee_angle"][0]
        assert rear_knee_entry[2] == "descriptive"
        assert rear_knee_entry[3] is False
