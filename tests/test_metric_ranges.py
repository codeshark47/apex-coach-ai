"""
tests/test_metric_ranges.py

metric_ranges.py is the single source of truth for green/amber/red
classification, feeding the PDF, dashboard, and Gemini narrative alike —
per the module's own docstring, nothing should duplicate these bounds
elsewhere. These tests pin down the exact boundary behavior for each
metric "kind" (higher_better / lower_better / band) so a future edit
can't silently shift a classification tier without a test noticing.
"""

import pytest

import metric_ranges as mr


class TestClassifyHigherBetter:
    """front_knee_bracing: green=(160,180), amber=(145,160)."""

    def test_green_at_lower_boundary(self):
        assert mr.classify("front_knee_bracing", 160.0) == "green"

    def test_green_above_upper_boundary_still_green(self):
        """A higher_better metric has no upper red ceiling — values
        above the green band's top are still fine, per the documented
        fix for release_height (a genuinely high release/leaping action
        isn't a fault)."""
        assert mr.classify("front_knee_bracing", 179.9) == "green"
        assert mr.classify("front_knee_bracing", 200.0) == "green"

    def test_amber_band(self):
        assert mr.classify("front_knee_bracing", 150.0) == "amber"

    def test_amber_lower_boundary_is_amber_not_red(self):
        assert mr.classify("front_knee_bracing", 145.0) == "amber"

    def test_just_below_amber_is_red(self):
        assert mr.classify("front_knee_bracing", 144.9) == "red"


class TestClassifyLowerBetter:
    """trunk_lean: green=(0,20), amber=(20,35)."""

    def test_green_at_zero(self):
        assert mr.classify("trunk_lean", 0.0) == "green"

    def test_green_upper_boundary(self):
        assert mr.classify("trunk_lean", 20.0) == "green"

    def test_amber_band(self):
        assert mr.classify("trunk_lean", 30.0) == "amber"

    def test_amber_upper_boundary_is_amber_not_red(self):
        assert mr.classify("trunk_lean", 35.0) == "amber"

    def test_just_above_amber_is_red(self):
        assert mr.classify("trunk_lean", 35.1) == "red"

    def test_below_green_floor_still_green(self):
        """lower_better has no floor — values below the green band's
        bottom are still fine (can't have "too little" trunk lean)."""
        assert mr.classify("trunk_lean", -5.0) == "green"


class TestClassifyBand:
    """hip_shoulder_separation: green=(25,50), amber=(15,25), amber_high=(50,65)."""

    def test_green_middle(self):
        assert mr.classify("hip_shoulder_separation", 35.0) == "green"

    def test_low_amber(self):
        assert mr.classify("hip_shoulder_separation", 20.0) == "amber"

    def test_low_red(self):
        assert mr.classify("hip_shoulder_separation", 5.0) == "red"

    def test_high_amber(self):
        """Regression test for a real bug this session: the high-side
        amber/red split was inverted (amber band computed as zero-width,
        everything above green immediately red) until fixed."""
        assert mr.classify("hip_shoulder_separation", 60.0) == "amber"

    def test_high_red(self):
        assert mr.classify("hip_shoulder_separation", 84.0) == "red"

    def test_band_kind_raises_without_amber_high_configured(self):
        """A 'band' kind metric MUST define amber_high — this should
        fail loudly, not silently misclassify, if that's ever missing."""
        from dataclasses import replace
        broken = replace(mr.RANGES["hip_shoulder_separation"], amber_high=None)
        original = mr.RANGES["hip_shoulder_separation"]
        mr.RANGES["hip_shoulder_separation"] = broken
        try:
            with pytest.raises(ValueError):
                mr.classify("hip_shoulder_separation", 60.0)
        finally:
            mr.RANGES["hip_shoulder_separation"] = original


class TestClassifyUnknownAndInvalid:
    def test_none_value_is_unknown(self):
        assert mr.classify("trunk_lean", None) == "unknown"

    def test_nan_value_is_unknown(self):
        assert mr.classify("trunk_lean", float("nan")) == "unknown"

    def test_non_numeric_value_is_unknown(self):
        assert mr.classify("trunk_lean", "not a number") == "unknown"

    def test_unknown_metric_key_raises(self):
        """No metric is ever scored without an explicit boundary — an
        unrecognized key must raise, never silently guess."""
        with pytest.raises(KeyError):
            mr.classify("made_up_metric", 10.0)


class TestExtractMetricValue:
    def test_extracts_each_metric_from_its_own_subfield(self):
        metrics = {
            "front_knee_bracing": {"degrees": 165.0},
            "hip_shoulder_separation": {"degrees": 30.0},
            "trunk_lean": {"degrees": 10.0},
            "release_height": {"ratio": 0.9},
            "head_stability": {"value": "0.01"},
        }
        assert mr.extract_metric_value(metrics, "front_knee_bracing") == 165.0
        assert mr.extract_metric_value(metrics, "hip_shoulder_separation") == 30.0
        assert mr.extract_metric_value(metrics, "trunk_lean") == 10.0
        assert mr.extract_metric_value(metrics, "release_height") == 0.9
        assert mr.extract_metric_value(metrics, "head_stability") == "0.01"

    def test_head_stability_falls_back_to_deviation_index(self):
        """head_stability has been stored under two different key names
        across this codebase's history (value vs deviation_index) — the
        extractor must check both."""
        metrics = {"head_stability": {"deviation_index": "0.02"}}
        assert mr.extract_metric_value(metrics, "head_stability") == "0.02"

    def test_missing_metric_dict_returns_none(self):
        assert mr.extract_metric_value({}, "trunk_lean") is None


class TestDescribeRangeMatchesClassify:
    """Regression test for a real bug found in an actual PDF report: a
    145% release-height reading showed Zone=OPTIMAL (correct — classify()
    has no ceiling for higher_better) directly next to a range string
    reading "Optimal 85%-130%" (145% visibly outside it) AND a red
    "measurement error likely" warning below — three contradictory
    signals for a value the code was specifically changed to protect
    from a false flag. describe_range() and measurement_warning() had
    drifted from classify() when release_height's kind was fixed."""

    def test_higher_better_describes_an_open_ended_optimal_band(self):
        """Must not show a closed "X%-Y%" range that implies a ceiling
        classify() doesn't actually enforce."""
        desc = mr.describe_range("release_height")
        assert "130%" not in desc
        assert "85%+" in desc

    def test_a_high_release_height_is_not_contradicted_by_the_range_text(self):
        value = 1.45
        assert mr.classify("release_height", value) == "green"
        desc = mr.describe_range("release_height")
        # The description must not state an upper bound this green value
        # falls outside of.
        assert "130%" not in desc

    def test_high_release_height_no_longer_triggers_a_stale_warning(self):
        """The real implausibility ceiling for this metric lives upstream
        (calculate_release_height_ratio_safe, rejects > 1.30 before a
        value ever reaches here) — this function must not re-impose a
        second, lower, contradictory one."""
        assert mr.measurement_warning("release_height", 1.45) is None
        assert mr.measurement_warning("release_height", 1.18) is None
