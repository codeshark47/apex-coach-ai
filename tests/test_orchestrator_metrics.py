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
        used to fall through to tier='Blocked rotation' (renamed 'Low
        Separation' 2026-08-07, see TestHipShoulderTierText below),
        status='success' — a confident, false coaching claim generated
        from a pure tracking failure."""
        df = pd.DataFrame([_hip_shoulder_row(LEFT_SHOULDER_x=np.nan)])
        result = o.calculate_hip_shoulder_separation(df, ffc_frame=10)
        assert result["degrees"] is None
        assert result["status"] == "error"
        assert result["tier"] not in ("Blocked rotation", "Low Separation", "Moderate Separation", "High Separation")

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

    def test_computes_successfully_for_a_plausible_rotated_pose(self):
        assert o.calculate_hip_shoulder_separation(
            pd.DataFrame([_hip_shoulder_row()]), ffc_frame=10
        )["status"] == "success"


class TestHipShoulderTierText:
    """FIX (2026-08-07, real literature audit + a real coach test that
    surfaced it): "Optimal stretch"/"Moderate separation"/"Blocked
    rotation" were value judgments using unsourced 25/15-degree cutoffs.
    This metric is now always-descriptive (real research — Senington, Lee
    & Williams, 2018 — shows separation varies by bowling action type, not
    skill), so the raw tier text is now purely descriptive of magnitude,
    with no "good/bad" framing. Confirmed live: Gemini's coaching
    narrative had repeated "described as blocked rotation" straight from
    this field even though the ZONE correctly said DESCRIPTIVE."""

    def test_high_separation(self):
        row = _hip_shoulder_row(RIGHT_SHOULDER_y=0.42)  # verified 30.96 degrees
        result = o.calculate_hip_shoulder_separation(pd.DataFrame([row]), ffc_frame=10)
        assert result["status"] == "success"
        assert result["degrees"] >= 25.0
        assert result["tier"] == "High Separation"

    def test_moderate_separation(self):
        row = _hip_shoulder_row(RIGHT_SHOULDER_y=0.38)  # verified 21.8 degrees
        result = o.calculate_hip_shoulder_separation(pd.DataFrame([row]), ffc_frame=10)
        assert result["status"] == "success"
        assert 15.0 <= result["degrees"] < 25.0
        assert result["tier"] == "Moderate Separation"

    def test_low_separation(self):
        row = _hip_shoulder_row(RIGHT_SHOULDER_y=0.35)  # verified 14.04 degrees, default row
        result = o.calculate_hip_shoulder_separation(pd.DataFrame([row]), ffc_frame=10)
        assert result["status"] == "success"
        assert result["degrees"] < 15.0
        assert result["tier"] == "Low Separation"
        assert result["tier"] not in ("Blocked rotation", "Optimal stretch", "Moderate separation")


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


class TestReleaseHeightClassificationThresholds:
    """FIX (2026-08-07, real bug found on a live clip): this raw
    classification string is computed independently of metric_ranges.
    RANGES["release_height"] and drifted out of sync with it when the
    real literature audit re-sourced that range to 1.18/1.08 (Felton et
    al. 2018) — confirmed live, a 60.9% ratio (deep red by the real
    bounds) showed the passable-sounding "Low-Sling Action" from a stale
    0.75 threshold that no longer matched anything in metric_ranges.py.
    Pins the real, current thresholds down directly so they can't drift
    again without a test noticing."""

    def test_high_release_leverage_at_the_real_green_floor(self):
        row = _release_row(wrist_y=0.07, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["status"] == "success"
        assert result["ratio"] >= 1.18
        assert result["classification"] == "High-Release Leverage"

    def test_standard_mid_arm_at_the_real_amber_floor(self):
        row = _release_row(wrist_y=0.148, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["status"] == "success"
        assert 1.08 <= result["ratio"] < 1.18
        assert result["classification"] == "Standard Mid-Arm Release"

    def test_low_sling_below_the_real_amber_floor(self):
        row = _release_row(wrist_y=0.55, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["status"] == "success"
        assert result["ratio"] < 1.08
        assert result["classification"] == "Low-Sling Action"

    def test_default_call_reports_head_ankle_span_source_not_recalibrating(self):
        """Every pre-existing caller (not passing segment_sum_body_height)
        must see body_height_source="head_ankle_span" and
        recalibration_pending=False/absent — the 2026-08-05 fix must be
        fully opt-in, never silently change behavior for a caller that
        hasn't started passing the new baseline yet."""
        row = _release_row(wrist_y=0.40, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["debug_raw"]["body_height_source"] == "head_ankle_span"
        assert result.get("recalibration_pending", False) is False

    def test_segment_sum_baseline_used_when_provided_and_flagged_pending(self):
        """When a real segment_sum_body_height is supplied, it becomes the
        denominator (not the reference frame's raw head-ankle span), and
        the result is explicitly flagged recalibration_pending=True — see
        calculate_release_height_ratio_safe's own comment for why: the
        0.85/0.75 tier cutoffs were tuned against the OLD, more lenient
        measurement and haven't been re-validated against this stricter
        one yet."""
        row = _release_row(wrist_y=0.40, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(
            row, bowling_arm="right", reference_row=row, segment_sum_body_height=0.60,
        )
        assert result["status"] == "success"
        assert result["debug_raw"]["body_height_source"] == "segment_sum"
        assert result["debug_raw"]["body_height"] == 0.60
        assert result["recalibration_pending"] is True
        # ratio = abs(0.85 - 0.40) / 0.60
        assert result["ratio"] == round(0.45 / 0.60, 4)

    def test_real_bug_scenario_bent_reference_frame_no_longer_gives_a_nonsense_ratio(self):
        """Regression test for the real, reported bug: a reference frame
        that visibly showed the bowler bent forward near the ground
        produced raw_head_ankle_span=0.0531 and a 240% ratio reported as
        "OPTIMAL / High-Release Leverage" with no warning. A segment-sum
        baseline representing his real standing height (unaffected by
        that frame's bend) must produce a sane ratio instead."""
        bent_reference_row = _release_row(
            wrist_y=0.40, ankle_y=0.90, hip_y=0.87, knee_y=0.88, nose_y=0.86,
        )  # nose/hip/knee/ankle all compressed near the ground - a real bent posture
        br_row = pd.Series({"RIGHT_WRIST_y": 0.40, "LEFT_ANKLE_y": 0.90, "RIGHT_ANKLE_y": 0.90})

        old_result = o.calculate_release_height_ratio_safe(
            br_row, bowling_arm="right", reference_row=bent_reference_row,
        )
        # Old method: either a nonsense >>1.30 ratio or an outright rejection
        # ("Body height too small") - both are the real, reported failure.
        assert old_result["status"] == "error"

        new_result = o.calculate_release_height_ratio_safe(
            br_row, bowling_arm="right", reference_row=bent_reference_row,
            segment_sum_body_height=0.65,  # a plausible real standing-height baseline
        )
        assert new_result["status"] == "success"
        assert new_result["ratio"] < 1.30
        assert new_result["recalibration_pending"] is True

    def test_real_bug_scenario_upright_reference_frame_ignores_a_bad_segment_sum(self):
        """Regression test for a real bug found on the coach's own live-
        demo session (2026-08-13): release_height came back 24%, later
        traced to segment_sum_body_height being computed from early run-up
        frames and rescaled to match the release frame's own camera
        distance via shoulder width — which is an unreliable proxy
        specifically AT release, because the bowling action rotates the
        torso (that's what hip-shoulder separation measures), shrinking
        apparent shoulder width from rotation, not distance. Confirmed on
        the real clip: the release-adjacent reference frame was visibly,
        verifiably upright with a raw span implying a plausible ~130%
        (matching the video), while the rescaled segment_sum came out
        SMALLER than that raw span — backwards for a measure that's
        supposed to represent fully-extended standing height.

        Fix: when the reference frame is independently verified upright
        (the ONLY case segment_sum_body_height's own compression problem
        doesn't apply — see the bent-frame test above), trust its raw
        span directly instead of a segment-sum baseline that needs a
        fragile rescale to be comparable. This is the exact inverse of
        the bent-frame test above: there, a bad raw span (compressed by a
        real bend) is corrected by a good segment_sum; here, a bad/
        mismatched segment_sum must not override a good raw span."""
        upright_reference_row = pd.Series({
            "NOSE_y": 0.20, "LEFT_SHOULDER_y": 0.25, "RIGHT_SHOULDER_y": 0.25,
            "LEFT_HIP_y": 0.55, "LEFT_ANKLE_y": 0.85, "RIGHT_ANKLE_y": 0.87,
            "LEFT_KNEE_y": 0.70,
        })
        br_row = pd.Series({"RIGHT_WRIST_y": 0.05, "LEFT_ANKLE_y": 0.85, "RIGHT_ANKLE_y": 0.87})

        result = o.calculate_release_height_ratio_safe(
            br_row, bowling_arm="right", reference_row=upright_reference_row,
            # Deliberately a bad/mismatched baseline (much smaller than the
            # verified-upright raw span of 0.65) — simulates the real
            # rescale-mismatch bug. Must be ignored, not used.
            segment_sum_body_height=0.20,
        )
        assert result["status"] == "success"
        assert result["debug_raw"]["body_height_source"] == "head_ankle_span"
        assert result["debug_raw"]["reference_frame_verified_upright"] is True
        assert result["debug_raw"]["body_height"] == pytest.approx(0.65, abs=0.01)
        assert result["recalibration_pending"] is False
        # ratio = abs(0.85 - 0.05) / 0.65 ~= 1.23, NOT abs(0.85-0.05)/0.20 = 4.0
        assert result["ratio"] == pytest.approx(1.2308, abs=0.01)


class TestReleaseFrameTrackingUncertain:
    """Regression coverage for a real bug found testing a live clip
    (2026-08-07): release_height came back 35.1% ("Low-Sling Action", a
    confident verdict) on a delivery whose OWN speed-estimate section
    already said tracking around release was too unstable for a reliable
    reading — the same wrist landmark this ratio's numerator depends on.
    br_tracking_confidence threads that existing signal through instead of
    computing anything new."""

    def test_low_confidence_flags_the_result_without_changing_the_ratio(self):
        row = _release_row(wrist_y=0.40, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        high = o.calculate_release_height_ratio_safe(
            row, bowling_arm="right", reference_row=row, br_tracking_confidence="high",
        )
        low = o.calculate_release_height_ratio_safe(
            row, bowling_arm="right", reference_row=row, br_tracking_confidence="low",
        )
        assert high["release_frame_tracking_uncertain"] is False
        assert low["release_frame_tracking_uncertain"] is True
        # The flag discloses, it never alters the actual measurement.
        assert high["ratio"] == low["ratio"]

    def test_default_call_with_no_confidence_arg_does_not_flag(self):
        """Every pre-existing caller (not passing br_tracking_confidence)
        must see release_frame_tracking_uncertain=False — this must be
        fully opt-in, same discipline as segment_sum_body_height's own
        backward-compatibility test above."""
        row = _release_row(wrist_y=0.40, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["release_frame_tracking_uncertain"] is False

    def test_coach_confirmed_wrist_point_bypasses_the_flag(self):
        """A human directly clicking the real wrist/ball position isn't
        subject to the tracker's own confidence — same reasoning as the
        1.30 implausibility ceiling already skipping wrist_override_norm."""
        row = _release_row(wrist_y=0.40, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(
            row, bowling_arm="right", reference_row=row,
            wrist_override_norm=(0.5, 0.40), br_tracking_confidence="low",
        )
        assert result["release_frame_tracking_uncertain"] is False


class TestSegmentSumBodyHeight:
    """
    _compute_segment_sum_body_height: the real body-height reference
    fixing the 240% false-reading bug — see its own docstring and
    calculate_release_height_ratio_safe's for the full story. Sums
    nose-to-shoulder + shoulder-to-hip + hip-to-knee + knee-to-ankle
    (Euclidean, not a raw vertical projection) across several plausible,
    early (pre-BFC) frames, and takes the median.
    """

    @staticmethod
    def _upright_row(frame, nose_y=0.30, sh_y=0.35, hip_y=0.55, knee_y=0.70, ankle_y=0.90):
        """A genuinely upright, stacked pose - the "reliably running,
        not bent" early-run-up frame this function is designed to find.
        Lead side for a right-arm bowler is LEFT (matches bowling_arm="right")."""
        return {
            "frame": frame,
            "NOSE_x": 0.50, "NOSE_y": nose_y,
            "LEFT_SHOULDER_x": 0.45, "LEFT_SHOULDER_y": sh_y,
            "RIGHT_SHOULDER_x": 0.55, "RIGHT_SHOULDER_y": sh_y,
            "LEFT_HIP_x": 0.48, "LEFT_HIP_y": hip_y,
            "LEFT_KNEE_x": 0.48, "LEFT_KNEE_y": knee_y,
            "LEFT_ANKLE_x": 0.48, "LEFT_ANKLE_y": ankle_y,
        }

    def test_computes_a_real_segment_sum_including_the_head(self):
        """Deliberately includes the head (nose-to-shoulder) segment, not
        just torso+thigh+shin - the OLD method measured ankle-to-NOSE (a
        near-full-body span), so omitting the head here would make this
        baseline systematically smaller by a head+neck length even on a
        perfectly good clip, silently recalibrating every threshold."""
        rows = [self._upright_row(f) for f in range(12)]
        df = pd.DataFrame(rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=15)
        # head 0.05 + torso 0.20 + thigh 0.15 + shin 0.20 = 0.60 (all straight
        # vertical segments in this synthetic row, so Euclidean == the
        # simple y-difference)
        assert result == pytest.approx(0.60, abs=0.01)

    def test_running_gait_with_a_lifted_lead_leg_does_not_get_excluded(self):
        """REAL BUG FOUND (2026-08-05): the original version required the
        WHOLE leg chain (ankle-knee-hip-shoulder-nose) to be strictly
        stacked top to bottom, same as _find_grounded_reference_near's
        "grounded" check. That's wrong here — the bowler is RUNNING during
        early run-up, so the lead leg is very often mid-swing (knee
        lifted, ankle temporarily ABOVE the knee), which is completely
        normal gait, not a tracking problem. On one real clip that
        rejected all but 3 of 169 otherwise-good early frames, right on
        the old MIN_PLAUSIBLE_SAMPLES=3 floor. Only the TORSO needs to be
        upright (no genuine trunk bend) — a lifted lead leg mid-stride
        must still count."""
        rows = [self._upright_row(f) for f in range(15)]
        # Every other frame has the lead (LEFT) ankle lifted ABOVE the
        # knee - a normal running swing phase, not a bent-over torso.
        for i in range(0, 15, 2):
            rows[i] = self._upright_row(i, knee_y=0.75, ankle_y=0.60)
        df = pd.DataFrame(rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=20)
        assert result is not None

    def test_high_percentile_captures_true_extension_not_bent_leg_underestimates(self):
        """A bent knee mid-stride can only make the 2D hip-knee-ankle
        segment look SHORTER than the bowler's true, fully-extended leg
        length (foreshortening), never longer. Most running-gait frames
        will therefore UNDERESTIMATE the true body height — using a high
        percentile (not the median, and not the max, which a single
        landmark glitch could inflate) recovers something close to the
        real, fully-extended value instead of being dragged down by the
        many naturally-bent frames."""
        # 9 frames with a visibly bent lead leg (shorter thigh+shin ->
        # smaller segment sum, simulating ordinary mid-stride bending),
        # 3 frames genuinely fully extended (the true ~0.60 value).
        bent_rows = [self._upright_row(f, knee_y=0.62, ankle_y=0.72) for f in range(9)]
        extended_rows = [self._upright_row(f + 9) for f in range(3)]
        df = pd.DataFrame(bent_rows + extended_rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=15)
        # Must land near the true, fully-extended ~0.60 - a median across
        # this same data would land much closer to the bent-leg value.
        assert result == pytest.approx(0.60, abs=0.02)

    def test_rejects_a_genuinely_bent_over_torso(self):
        """The real failure mode this function guards against: a genuine
        trunk bend (nose not above shoulder/hip) — unlike a lifted leg,
        this must still be excluded.

        VALUE UPDATED (2026-08-07, independent-per-segment redesign): a
        minority of compressed head/torso rows now gets washed out by
        THAT segment's own 90th percentile specifically, while thigh/shin
        — unaffected by the bad rows (same knee/hip/ankle values as the
        good rows) — independently contribute their own full, undiminished
        90th percentile. Summing four independently-maximized segments can
        legitimately land slightly ABOVE what one combined-per-frame
        percentile gave (0.60), since each segment's best evidence no
        longer has to occur in the same single frame — a real, intended
        consequence of no longer being bottlenecked by whichever body part
        tracks worst in any one frame (see the real 2026-08-07 clip this
        whole redesign was found on: thigh/shin data was 10x scarcer than
        head/torso data early in a real run-up)."""
        good_rows = [self._upright_row(f) for f in range(12)]
        bad_rows = [self._upright_row(f, nose_y=0.86, sh_y=0.87, hip_y=0.88) for f in range(12, 15)]
        df = pd.DataFrame(good_rows + bad_rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=20)
        assert result == pytest.approx(0.631, abs=0.01)

    @staticmethod
    def _distant_row(frame, scale=0.2, center=0.5):
        """A genuinely SMALLER, more distant subject — every landmark
        scaled toward the frame center, simulating a real person standing
        much farther from the camera (same relative pose, smaller on
        screen). Used to simulate a batsman/bystander visible before the
        bowler enters frame."""
        base = TestSegmentSumBodyHeight._upright_row(frame)
        return {
            k: (frame if k == "frame" else center + (v - center) * scale)
            for k, v in base.items()
        }

    def test_real_bug_scenario_distant_bystander_before_bowler_no_longer_contaminates(self):
        """Regression test for a real bug the coach caught (2026-08-07):
        on his actual clip, the batsman was visible (small, distant) for
        the first ~75 frames before the bowler entered frame — confirmed
        directly on that clip that shoulder width jumped ~3.7x right at
        the entry point. Even correct seeding on the bowler didn't fully
        prevent the identity walk from bridging back through the gap and
        locking onto the distant bystander for a stretch of frames (a
        real, documented limitation of that walk, not fixed here). This
        function's own scale-consistency guard must now exclude those
        distant frames using the genuinely-bowler frames near the window
        end as the reference scale, instead of quietly blending a much
        smaller, unrelated person into the body-height baseline."""
        bystander_rows = [self._distant_row(f) for f in range(20)]
        bowler_rows = [self._upright_row(f) for f in range(20, 35)]
        df = pd.DataFrame(bystander_rows + bowler_rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=35)
        assert result is not None
        # Must reflect the real bowler's ~0.60 body height, not a value
        # dragged down by 20 much-smaller bystander frames.
        assert result == pytest.approx(0.60, abs=0.03)

    def test_bystander_frame_that_coincidentally_matches_bowler_scale_still_excluded(self):
        """FIX (2026-08-08, after independently evaluating a Gemini
        suggestion — see [[feedback_evaluate_external_ai_advice]]): the
        first version of the scale guard checked each frame's scale in
        isolation and leaked 8 of 35 real contaminated frames through on
        the coach's actual clip, because a bystander's own pose variation
        let a few individual frames' values coincidentally fall inside
        the bowler's normal range. This is that exact scenario: an
        otherwise-small, DISTANT bystander with one frame that happens to
        match the bowler's scale (e.g. leaning toward camera), sitting on
        the wrong side of a real gap. A per-frame-only check would wrongly
        include it; the backward walk must not, since it isn't reachable
        by an unbroken run back from the trusted anchor near the window's
        end. Made numerically detectable: the leaking frame is scaled
        LARGER than the true bowler (not just "big enough to pass"), so if
        it wrongly slipped into a 90th-percentile aggregate the result
        would measurably shift upward — this isn't just a plausible-
        looking number, it's evidence the frame was truly excluded."""
        bystander_rows = [self._distant_row(f) for f in range(20)]
        bystander_rows[10] = self._distant_row(10, scale=1.4)  # briefly matches/exceeds bowler scale
        bowler_rows = [self._upright_row(f) for f in range(20, 35)]
        df = pd.DataFrame(bystander_rows + bowler_rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=35)
        clean_result = o._compute_segment_sum_body_height(
            pd.DataFrame([self._distant_row(f) for f in range(20)][:10]
                         + [self._distant_row(f) for f in range(20)][11:] + bowler_rows),
            "right", search_end_frame=35,
        )
        assert result is not None
        # If the inflated frame 10 had leaked in, a 90th-percentile
        # aggregate over the bowler-era samples would be measurably
        # pulled toward its larger value - confirm it wasn't.
        assert result == pytest.approx(clean_result, abs=0.001)

    def test_real_bug_scenario_close_camera_setup_frames_no_longer_inflate_the_baseline(self):
        """Regression test for a real bug found from the coach's actual
        saved session data (2026-08-10): release_height came back 24%
        ("Low-Sling Action") on a well-tracked clip (M.Rauf.mp4), and the
        debug_raw showed segment_sum_body_height at 1.012 — over 6x the
        SAME reference frame's raw head-ankle span (0.164), physically
        impossible for one real person.

        Traced directly: the scale-consistency guard (added 2026-08-07/08
        to reject a distant bystander) only ever had a FLOOR
        (SCALE_MIN_RATIO) — nothing rejected a frame LARGER than the
        reference. Confirmed on the real clip: frames 25-34, early in the
        search window, had shoulder width up to 0.60 against a near-BFC
        reference of 0.115 — a 5.2x jump, almost certainly the camera
        being held close to someone during setup before the run-up
        starts, not the bowler at any real filming distance. Those
        oversized frames' segment lengths dominated 3 of the 4
        independently-maximized 90th-percentile sums.

        This is that exact scenario, scaled down: a block of oversized
        early frames (scale=5x the real bowler) before the genuine
        bowler-era frames near the window end. Without the ceiling, the
        old floor-only check would have let them straight through
        (5x is well above the 0.5x floor) and inflated the result well
        past the true ~0.60 baseline."""
        oversized_rows = [self._distant_row(f, scale=5.0) for f in range(10)]
        bowler_rows = [self._upright_row(f) for f in range(20, 35)]
        df = pd.DataFrame(oversized_rows + bowler_rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=35)
        assert result is not None
        # Must reflect the real bowler's ~0.60 body height, not a value
        # inflated by the 10 oversized camera-setup frames.
        assert result == pytest.approx(0.60, abs=0.03)

    def test_real_bug_scenario_scarce_leg_landmarks_no_longer_forces_a_fallback(self):
        """Regression test for the real bug found on a live clip
        (2026-08-07): 67 of 93 early frames had a real nose+shoulders+hip
        detection, but only 6 also had the knee+ankle (legs frequently out
        of frame/undetected that early, a real framing limitation, not a
        tracking glitch) — requiring the full chain threw away the 61
        extra good head/torso frames and left too few samples to trust,
        forcing a silent fallback to the OLD single-frame method this
        function exists to replace. This is the same shape of scenario,
        scaled down: plenty of head/torso frames, only just enough
        (MIN_LEG_SEGMENT_SAMPLES) leg frames — must now succeed instead of
        returning None."""
        rows = []
        for f in range(30):
            row = self._upright_row(f)
            if f >= 6:  # only the first 6 frames keep real leg landmarks
                row["LEFT_KNEE_x"] = np.nan
                row["LEFT_KNEE_y"] = np.nan
                row["LEFT_ANKLE_x"] = np.nan
                row["LEFT_ANKLE_y"] = np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=35)
        assert result is not None
        assert result == pytest.approx(0.60, abs=0.02)

    def test_returns_none_with_too_few_plausible_frames(self):
        """Too little evidence to trust a baseline from - caller must fall
        back to the old single-frame method rather than trust this."""
        rows = [self._upright_row(f) for f in range(5)]  # below MIN_PLAUSIBLE_SAMPLES
        df = pd.DataFrame(rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=10)
        assert result is None

    def test_returns_none_when_no_frames_are_plausible(self):
        rows = [self._upright_row(f, hip_y=0.20, sh_y=0.35) for f in range(12)]  # hip above shoulder - genuinely inverted
        df = pd.DataFrame(rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=15)
        assert result is None

    def test_narrow_window_excludes_frames_outside_the_range(self):
        """search_start_frame (roadmap item #1, 2026-08-06): bounds the
        scan on BOTH sides, unlike the wide 0..end run-up scan the ratio
        baseline uses. This is what lets the absolute-cm height estimate
        sample frames near BFC (close to the stump-calibration plane's
        depth) instead of far-away early run-up frames - see
        _compute_segment_sum_body_height's docstring for why a wide-window
        baseline produced a real 444cm implausible reading when reused for
        that feature."""
        # Frames 0-11: plausible, but a deliberately different (smaller)
        # segment-sum height - must be excluded, they're before the
        # narrow window and would silently pollute the result otherwise.
        outside_rows = [
            self._upright_row(f, nose_y=0.40, sh_y=0.45, hip_y=0.55, knee_y=0.62, ankle_y=0.70)
            for f in range(12)
        ]
        # Frames 90-101, inside [85, 115): the true ~0.60 segment sum.
        inside_rows = [self._upright_row(f) for f in range(90, 102)]
        df = pd.DataFrame(outside_rows + inside_rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=115, search_start_frame=85)
        assert result == pytest.approx(0.60, abs=0.01)

    def test_narrow_window_returns_none_when_too_few_frames_inside_it(self):
        """Plenty of plausible frames exist overall, but only a handful
        fall inside the narrow BFC-centered window - must still fall back
        to None (caller treats this as "can't estimate"), not silently
        borrow frames from outside the requested range to hit the
        minimum."""
        outside_rows = [self._upright_row(f) for f in range(60)]  # all before the window
        inside_rows = [self._upright_row(f) for f in range(90, 95)]  # only 5, below MIN_PLAUSIBLE_SAMPLES
        df = pd.DataFrame(outside_rows + inside_rows)
        result = o._compute_segment_sum_body_height(df, "right", search_end_frame=115, search_start_frame=85)
        assert result is None


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
    save_uploaded_video_capped, which only touches .name, .size, and
    .getbuffer(). size defaults to len(content) (real Streamlit behavior)
    but can be overridden to simulate a huge upload without actually
    allocating huge fake bytes."""

    def __init__(self, name: str, content: bytes, size: int = None):
        self.name = name
        self._content = content
        self.size = len(content) if size is None else size

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

    def test_rejects_grossly_oversized_upload_before_reading_it_into_memory(self, tmp_path, monkeypatch):
        """BUG FOUND (2026-08-02, same iPhone 17 Pro Max incident): the
        server log showed the app going silent with NO warning printed —
        but every ffmpeg failure path prints one before giving up. That
        means the crash likely happened even earlier: reading the raw
        upload into memory, which runs before ffmpeg is ever invoked.
        A grossly oversized upload must be rejected on its reported .size
        alone, without ever calling .getbuffer()."""
        import shutil as _shutil
        # If getbuffer() were called, this would prove the size check was
        # skipped -- fail loudly rather than actually allocating 400MB.
        def _explode():
            raise AssertionError("getbuffer() should never be called for an oversized upload")
        monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/ffmpeg")

        fake_file = _FakeUploadedFile("clip.mp4", b"tiny placeholder", size=400 * 1024 * 1024)
        fake_file.getbuffer = _explode
        dest = str(tmp_path / "saved.mp4")

        with pytest.raises(RuntimeError, match="too large"):
            o.save_uploaded_video_capped(fake_file, dest)

        assert not os.path.exists(dest)

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

    @staticmethod
    def _mock_source_fps(monkeypatch, fps):
        """cv2.VideoCapture is used here just to read the raw upload's own
        fps before deciding whether to cap it — stub it so the test
        controls that reading without needing a real video file."""
        import cv2 as _cv2

        class _FakeCapture:
            def get(self, prop):
                return fps
            def release(self):
                pass

        monkeypatch.setattr(_cv2, "VideoCapture", lambda path: _FakeCapture())

    def test_caps_frame_rate_for_a_genuine_slow_mo_recording(self, tmp_path, monkeypatch):
        """Regression test for a real bug the coach reported: a native
        slow-mo recording (100MB, comfortably under the size cap) crashed
        the app, while the SAME clip re-compressed through WhatsApp first
        worked fine. Traced directly: this function only ever capped
        RESOLUTION — a slow-mo capture at 120-240fps+ still handed the
        SAME frame count to MediaPipe afterward (just resized), several
        times the frames of a normal video of the same real-world
        duration. WhatsApp's own re-encode happens to flatten frame rate
        too, not just resolution, which is almost certainly the real
        reason that "fixed" it. The ffmpeg command must now include an
        explicit -r cap when the source exceeds it."""
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/ffmpeg")
        self._mock_source_fps(monkeypatch, 240.0)

        captured_cmd = []

        def _fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            with open(cmd[-1], "wb") as f:
                f.write(b"fake encoded output")
            class _Result:
                returncode = 0
                stderr = b""
            return _Result()

        monkeypatch.setattr(subprocess, "run", _fake_run)

        fake_file = _FakeUploadedFile("slowmo.mp4", b"pretend this is a 240fps slow-mo recording")
        dest = str(tmp_path / "saved.mp4")
        o.save_uploaded_video_capped(fake_file, dest)

        assert "-r" in captured_cmd
        assert captured_cmd[captured_cmd.index("-r") + 1] == "60"

    def test_does_not_cap_frame_rate_for_a_normal_recording(self, tmp_path, monkeypatch):
        """A normal 30fps recording must pass through with no -r flag at
        all — this cap exists for genuine slow-mo, not the common case,
        and an unnecessary -r flag risks its own re-encode artifacts."""
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/ffmpeg")
        self._mock_source_fps(monkeypatch, 30.0)

        captured_cmd = []

        def _fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            with open(cmd[-1], "wb") as f:
                f.write(b"fake encoded output")
            class _Result:
                returncode = 0
                stderr = b""
            return _Result()

        monkeypatch.setattr(subprocess, "run", _fake_run)

        fake_file = _FakeUploadedFile("normal.mp4", b"pretend this is a normal 30fps recording")
        dest = str(tmp_path / "saved.mp4")
        o.save_uploaded_video_capped(fake_file, dest)

        assert "-r" not in captured_cmd
