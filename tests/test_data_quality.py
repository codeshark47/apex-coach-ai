"""
tests/test_data_quality.py

assess_quality() is the guard that decides whether a whole session's
numbers can be trusted, based purely on how many of the 5 metrics came
back None — never on a "does this look plausible" guess (see the
module's own docstring for why). These tests exist to make sure that
count is always accurate, including for metrics whose real, legitimate
value can look exactly like a placeholder (0.0 degrees of trunk lean is
a genuinely perfect score, not a sign of failure).
"""

import metric_ranges as mr
import data_quality as dq


def _all_metrics_present():
    """A metrics dict where all 5 metrics genuinely computed — every
    field is a real, present value in its own optimal/green band."""
    return {
        "front_knee_bracing": {"degrees": 170.0, "status": "success"},
        "hip_shoulder_separation": {"degrees": 30.0, "status": "success"},
        "trunk_lean": {"degrees": 5.0, "status": "success"},
        "release_height": {"ratio": 0.90, "status": "success"},
        "head_stability": {"value": "0.01", "status": "success"},
    }


class TestAssessQuality:
    def test_all_five_present_is_high_confidence_zero_missing(self):
        result = dq.assess_quality(_all_metrics_present())
        assert result["confidence"] == "high"
        assert result["missing_count"] == 0
        assert result["missing_metrics"] == []

    def test_genuine_zero_values_are_not_counted_as_missing(self):
        """The exact end-to-end guard for this session's core bug: a
        REAL 0.0-degree trunk lean and a REAL near-zero head-stability
        reading must both be correctly recognized as present, valid
        data — not confused with the None a genuine failure returns.
        This only holds because kinematics.py returns None (not 0.0) on
        failure — this test guards the CONTRACT between the two
        modules, not just data_quality.py in isolation."""
        metrics = _all_metrics_present()
        metrics["trunk_lean"]["degrees"] = 0.0
        metrics["head_stability"]["value"] = "0.00"
        result = dq.assess_quality(metrics)
        assert "trunk_lean" not in result["missing_metrics"]
        assert "head_stability" not in result["missing_metrics"]
        assert result["confidence"] == "high"

    def test_three_missing_metrics_is_low_confidence(self):
        metrics = _all_metrics_present()
        metrics["trunk_lean"]["degrees"] = None
        metrics["hip_shoulder_separation"]["degrees"] = None
        metrics["release_height"]["ratio"] = None
        result = dq.assess_quality(metrics)
        assert result["missing_count"] == 3
        assert result["confidence"] == "low"
        assert set(result["missing_metrics"]) == {
            "trunk_lean", "hip_shoulder_separation", "release_height",
        }

    def test_two_missing_metrics_stays_high_confidence(self):
        """LOW_CONFIDENCE_THRESHOLD is 3 — exactly 2 missing must NOT
        trip low confidence (boundary test, one below the threshold)."""
        metrics = _all_metrics_present()
        metrics["trunk_lean"]["degrees"] = None
        metrics["hip_shoulder_separation"]["degrees"] = None
        result = dq.assess_quality(metrics)
        assert result["missing_count"] == 2
        assert result["confidence"] == "high"

    def test_all_five_missing_is_low_confidence(self):
        metrics = {
            "front_knee_bracing": {"degrees": None},
            "hip_shoulder_separation": {"degrees": None},
            "trunk_lean": {"degrees": None},
            "release_height": {"ratio": None},
            "head_stability": {"value": None},
        }
        result = dq.assess_quality(metrics)
        assert result["missing_count"] == 5
        assert result["confidence"] == "low"

    def test_completely_empty_metrics_dict_counts_all_as_missing(self):
        """No keys at all (e.g. a totally failed extraction) — every
        metric_ranges.extract_metric_value lookup returns None via
        .get() chains, none of this should raise."""
        result = dq.assess_quality({})
        assert result["missing_count"] == len(mr.all_metric_keys())
        assert result["confidence"] == "low"
