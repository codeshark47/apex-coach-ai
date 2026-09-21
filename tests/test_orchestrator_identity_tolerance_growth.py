"""
tests/test_orchestrator_identity_tolerance_growth.py

Regression tests for _select_identity_consistent_candidate's distance
tolerance growing with how many frames away the reference is
(2026-09-19).

REAL MEASURED BUG, not a guess: a fixed IDENTITY_CONSISTENCY_MAX_DIST
doesn't distinguish "a different person nearby right now" from "the SAME
person, legitimately further from their last confirmed position because
more time has passed." Confirmed directly on the coach's real clip: a
genuine, continuously-moving detection recovered by the ROI-crop
fallback at the coach's own confirmed FFC/BR frames (21-24 frames from
the nearest reference) measured 0.169-0.181 -- just over the fixed 0.15
base and incorrectly rejected -- while a genuine bystander in this same
investigation has always measured around 0.45, a completely different
order of magnitude. Mirrors main._walk_from_seed's own existing
position-matching radius, which already grows with frames since the
last confirmed match.
"""

import pandas as pd

import orchestrator as o


def _ref_df(ref_frame, nose_xy):
    return pd.DataFrame([{
        "frame": ref_frame, "NOSE_x": nose_xy[0], "NOSE_y": nose_xy[1],
        "LEFT_HIP_x": nose_xy[0] - 0.01, "RIGHT_HIP_x": nose_xy[0] + 0.01,
        "LEFT_HIP_y": nose_xy[1] + 0.06, "RIGHT_HIP_y": nose_xy[1] + 0.06,
    }])


class TestIdentityToleranceGrowsWithGapLength:
    def test_a_short_gap_uses_close_to_the_base_tolerance(self):
        """Close to the reference frame, the tolerance should still be
        close to the base (0.15) -- must not become permissive
        immediately, or this reopens the original bystander-lock bug.
        At 1 frame away and 30fps, effective tolerance is 0.15 + 0.02 =
        0.17 -- a distance clearly beyond that must still be rejected."""
        df = _ref_df(50, (0.30, 0.52))
        candidates = [{"NOSE": (0.55, 0.52, 0.99)}]  # distance 0.25, well beyond 0.17
        selected = o._select_identity_consistent_candidate(df, 51, candidates, fps=30.0)
        assert selected is None

    def test_a_real_measured_case_a_long_gap_recovers_a_genuinely_moved_person(self):
        """The exact real scenario this fix targets: 21 frames away (at
        ~30fps, ~0.7s), a distance of 0.169 -- just over the fixed 0.15
        base -- must now be accepted, matching the real measurement from
        the coach's own clip."""
        df = _ref_df(53, (0.299, 0.523))
        candidates = [{"NOSE": (0.299 + 0.169, 0.523, 0.99)}]  # dist = 0.169
        selected = o._select_identity_consistent_candidate(df, 74, candidates, fps=29.97)
        assert selected is not None

    def test_a_bystander_at_the_measured_real_distance_is_still_rejected_even_over_a_long_gap(self):
        """The growth must stay well under the real measured bystander
        distance (~0.45) even for a long gap -- this fix must not
        reopen the original bug it's built on top of."""
        df = _ref_df(53, (0.299, 0.523))
        candidates = [{"NOSE": (0.299 + 0.45, 0.523, 0.99)}]  # real bystander-scale distance
        selected = o._select_identity_consistent_candidate(df, 90, candidates, fps=29.97)  # 37 frames away
        assert selected is None

    def test_growth_is_capped_not_unbounded(self):
        """An extremely long gap must not make the tolerance effectively
        infinite -- IDENTITY_CONSISTENCY_MAX_DIST_CAP still applies."""
        df = _ref_df(0, (0.30, 0.52))
        candidates = [{"NOSE": (0.30 + o.IDENTITY_CONSISTENCY_MAX_DIST_CAP + 0.05, 0.52, 0.99)}]
        selected = o._select_identity_consistent_candidate(df, 500, candidates, fps=29.97)
        assert selected is None

    def test_default_fps_is_used_when_caller_does_not_pass_one(self):
        """Backward-compat: existing/direct callers that don't pass fps
        must not break (defaults to a reasonable assumption)."""
        df = _ref_df(53, (0.299, 0.523))
        candidates = [{"NOSE": (0.299 + 0.169, 0.523, 0.99)}]
        selected = o._select_identity_consistent_candidate(df, 74, candidates)  # no fps kwarg
        assert selected is not None
