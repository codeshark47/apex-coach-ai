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
    """release_height: green=(1.18,1.28), amber=(1.08,1.18) — real
    thresholds re-sourced 2026-08-06 (was front_knee_bracing before that
    metric became always-descriptive; see TestAlwaysDescriptiveMetrics)."""

    def test_green_at_lower_boundary(self):
        assert mr.classify("release_height", 1.18) == "green"

    def test_green_above_upper_boundary_still_green(self):
        """A higher_better metric has no upper red ceiling — values
        above the green band's top are still fine, per the documented
        fix for release_height (a genuinely high release/leaping action
        isn't a fault)."""
        assert mr.classify("release_height", 1.279) == "green"
        assert mr.classify("release_height", 2.0) == "green"

    def test_amber_band(self):
        assert mr.classify("release_height", 1.12) == "amber"

    def test_amber_lower_boundary_is_amber_not_red(self):
        assert mr.classify("release_height", 1.08) == "amber"

    def test_just_below_amber_is_red(self):
        assert mr.classify("release_height", 1.079) == "red"


class TestClassifyLowerBetter:
    """head_stability: green=(0.0,0.08), amber=(0.08,0.15) — was trunk_lean
    before that metric's direction was reversed to higher_better (real
    research shows more forward lean at release correlates with faster
    ball speed; see TestTrunkLeanReversedDirection)."""

    def test_green_at_zero(self):
        assert mr.classify("head_stability", 0.0) == "green"

    def test_green_upper_boundary(self):
        assert mr.classify("head_stability", 0.08) == "green"

    def test_amber_band(self):
        assert mr.classify("head_stability", 0.1) == "amber"

    def test_amber_upper_boundary_is_amber_not_red(self):
        assert mr.classify("head_stability", 0.15) == "amber"

    def test_just_above_amber_is_red(self):
        assert mr.classify("head_stability", 0.151) == "red"

    def test_below_green_floor_still_green(self):
        """lower_better has no floor — values below the green band's
        bottom are still fine (can't have "too little" head stability)."""
        assert mr.classify("head_stability", -0.01) == "green"


class TestClassifyBand:
    """batting_xfactor_separation: green=(25,50), amber=(15,25),
    amber_high=(50,65) — was hip_shoulder_separation before that metric
    became always-descriptive (see TestAlwaysDescriptiveMetrics); same
    numbers, since batting_xfactor_separation reuses the identical
    computation and bounds."""

    def test_green_middle(self):
        assert mr.classify("batting_xfactor_separation", 35.0) == "green"

    def test_low_amber(self):
        assert mr.classify("batting_xfactor_separation", 20.0) == "amber"

    def test_low_red(self):
        assert mr.classify("batting_xfactor_separation", 5.0) == "red"

    def test_high_amber(self):
        """Regression test for a real bug this session: the high-side
        amber/red split was inverted (amber band computed as zero-width,
        everything above green immediately red) until fixed."""
        assert mr.classify("batting_xfactor_separation", 60.0) == "amber"

    def test_high_red(self):
        assert mr.classify("batting_xfactor_separation", 84.0) == "red"

    def test_band_kind_raises_without_amber_high_configured(self):
        """A 'band' kind metric MUST define amber_high — this should
        fail loudly, not silently misclassify, if that's ever missing."""
        from dataclasses import replace
        broken = replace(mr.RANGES["batting_xfactor_separation"], amber_high=None)
        original = mr.RANGES["batting_xfactor_separation"]
        mr.RANGES["batting_xfactor_separation"] = broken
        try:
            with pytest.raises(ValueError):
                mr.classify("batting_xfactor_separation", 60.0)
        finally:
            mr.RANGES["batting_xfactor_separation"] = original


class TestAlwaysDescriptiveMetrics:
    """REAL LITERATURE AUDIT (2026-08-06): front_knee_bracing and
    hip_shoulder_separation have no universal validated pass/fail band for
    ANY bowler_type, including pace — unlike trunk_lean/release_height/
    head_stability, which do have a real pace band. front_knee_bracing is
    a real technique CLASSIFICATION (Extended-Knee >=170deg vs Flexed-Knee
    <170deg — Portus, Mason, Elliott, Pfitzner & Done, 2004), not a
    higher-is-better scale; hip_shoulder_separation varies by bowling
    action type (front-on/side-on/mixed), not skill (Senington, Lee &
    Williams)."""

    def test_front_knee_bracing_is_descriptive_for_pace_default(self):
        assert mr.classify("front_knee_bracing", 170.0) == "descriptive"
        assert mr.classify("front_knee_bracing", 150.0) == "descriptive"

    def test_front_knee_bracing_is_descriptive_for_pace_explicit(self):
        assert mr.classify("front_knee_bracing", 170.0, "pace") == "descriptive"

    def test_front_knee_bracing_wrist_spin_override_still_works(self):
        """The one real, validated override (Goswami et al. 2016) must
        still fire — _ALWAYS_DESCRIPTIVE_METRICS only applies where no
        override exists for that specific bowler_type."""
        assert mr.classify("front_knee_bracing", 160.0, "wrist_spin") == "green"

    def test_hip_shoulder_separation_is_descriptive_for_every_bowler_type(self):
        assert mr.classify("hip_shoulder_separation", 35.0) == "descriptive"
        assert mr.classify("hip_shoulder_separation", 35.0, "pace") == "descriptive"
        assert mr.classify("hip_shoulder_separation", 35.0, "wrist_spin") == "descriptive"
        assert mr.classify("hip_shoulder_separation", 35.0, "finger_spin") == "descriptive"

    def test_has_validated_range_matches_classify(self):
        assert mr.has_validated_range("front_knee_bracing", None) is False
        assert mr.has_validated_range("front_knee_bracing", "wrist_spin") is True
        assert mr.has_validated_range("hip_shoulder_separation", None) is False
        assert mr.has_validated_range("release_height", None) is True

    def test_descriptive_note_classifies_extended_vs_flexed_knee(self):
        extended = mr.descriptive_note("front_knee_bracing", 175.0)
        flexed = mr.descriptive_note("front_knee_bracing", 150.0)
        assert "Extended-Knee" in extended
        assert "Flexed-Knee" in flexed
        assert "Portus" in extended and "Portus" in flexed

    def test_descriptive_note_hip_shoulder_separation_explains_action_type(self):
        note = mr.descriptive_note("hip_shoulder_separation")
        assert "action type" in note.lower()
        assert "Senington" in note


class TestTrunkLeanReversedDirection:
    """DIRECTION FIX (2026-08-06): trunk_lean used to score less forward
    lean as better (lower_better, green 0-20deg). Real research (Elliott
    1986; Portus et al. 2004; Worthington et al. 2013a) shows MORE forward
    lean at release correlates with FASTER ball speed — now higher_better,
    green=(13,30), amber=(5,13), sourced from Felton, Lister, Worthington
    & King (2018/19): elite male mean ~20.5 degrees forward flexion at
    release (159.5 +/- 7.8 on an anatomical-180-neutral convention)."""

    def test_green_at_real_elite_mean(self):
        assert mr.classify("trunk_lean", 20.5) == "green"

    def test_green_at_lower_boundary(self):
        assert mr.classify("trunk_lean", 13.0) == "green"

    def test_very_upright_no_longer_reads_as_optimal(self):
        """The OLD range called 0 degrees "green" (Optimal Upright
        Posture) — real research says minimal forward lean is now the
        concerning end, not the ideal one."""
        assert mr.classify("trunk_lean", 0.0) == "red"

    def test_amber_transition_band(self):
        assert mr.classify("trunk_lean", 8.0) == "amber"

    def test_high_lean_still_green_no_established_ceiling(self):
        """No real upper ceiling exists in the literature for "too much"
        forward lean at release — same reasoning as release_height's own
        no-ceiling fix."""
        assert mr.classify("trunk_lean", 45.0) == "green"


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
        assert "118%+" in desc

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


class TestFormatValuePercentUnits:
    """BUG FOUND (2026-08-03, while restyling the batting UI): format_value
    and describe_range's fv() helper used to multiply EVERY unit="%" metric
    by 100 unconditionally, which is only correct for a metric stored as a
    0-1 fraction (release_height, e.g. 0.85 -> "85%"). batting_weight_transfer's
    calculate_weight_transfer already returns a 0-100+ number (e.g. 52.0
    meaning 52%) -- the old code turned that into "5200%". Fixed with an
    explicit already_percent flag on MetricRange rather than guessing from
    the unit string alone."""

    def test_fraction_based_metric_still_multiplies_by_100(self):
        assert mr.format_value("release_height", 0.85) == "85%"

    def test_already_percent_metric_is_not_multiplied_again(self):
        assert mr.format_value("batting_weight_transfer", 52.0) == "52%"
        assert mr.format_value("batting_weight_transfer", 100.0) == "100%"

    def test_describe_range_does_not_inflate_an_already_percent_metric(self):
        desc = mr.describe_range("batting_weight_transfer")
        assert "40%+" in desc
        assert "4000%" not in desc


class TestBowlerTypeClassification:
    """
    Spin-bowling support (2026-08-04): bowler_type defaults to None, which
    must behave IDENTICALLY to every pre-existing call site that doesn't
    pass it — pace bowling's classification must never change. A spin
    bowler_type with no entry in SPIN_RANGE_OVERRIDES for a given metric
    must classify as "descriptive" (a real, measured value with no
    invented pass/fail verdict) rather than silently falling back to
    pace's band — a real spinner's normal technique could otherwise get
    falsely flagged against fast-bowling-calibrated numbers.

    front_knee_bracing/wrist_spin is sourced from Goswami, Srivastava &
    Rajpoot (2016) — 5 real leg-spin bowlers, front/lead knee angle at
    release: mean 162.4 +/- 13.3 deg, observed range 134-187 (their
    "Knee Joint LEFT" column; all 5 bowlers were right-handed, so the
    front/lead leg is the left one, same convention kinematics.py uses).
    """

    def test_no_bowler_type_matches_default_two_arg_call(self):
        """Every existing call site in the codebase calls classify(key,
        value) with no third argument — bowler_type must default to
        exactly that same pace behavior, not require every caller to be
        updated."""
        assert mr.classify("front_knee_bracing", 160.0) == mr.classify("front_knee_bracing", 160.0, None)
        assert mr.classify("front_knee_bracing", 150.0) == mr.classify("front_knee_bracing", 150.0, None)

    def test_pace_bowler_type_is_identical_to_default(self):
        assert mr.classify("front_knee_bracing", 150.0, "pace") == mr.classify("front_knee_bracing", 150.0)

    def test_wrist_spin_front_knee_uses_the_real_override(self):
        # Pace would call all three of these "amber" or "red" (below its
        # 160/145 thresholds) — a real wrist-spinner's naturally lower,
        # more-flexed knee at release is normal technique, not a flaw.
        assert mr.classify("front_knee_bracing", 160.0, "wrist_spin") == "green"
        assert mr.classify("front_knee_bracing", 150.0, "wrist_spin") == "green"
        assert mr.classify("front_knee_bracing", 134.0, "wrist_spin") == "green"
        assert mr.classify("front_knee_bracing", 125.0, "wrist_spin") == "amber"
        assert mr.classify("front_knee_bracing", 110.0, "wrist_spin") == "red"

    def test_finger_spin_front_knee_has_no_override_so_is_descriptive(self):
        """No real, correctly-mapped source was found for finger-spin
        front-knee angle — must show as descriptive, not inherit pace's
        band or wrist-spin's override."""
        assert mr.classify("front_knee_bracing", 160.0, "finger_spin") == "descriptive"
        assert mr.classify("front_knee_bracing", 100.0, "finger_spin") == "descriptive"

    @pytest.mark.parametrize("bowler_type", ["finger_spin", "wrist_spin"])
    @pytest.mark.parametrize("metric_key", [
        "hip_shoulder_separation", "trunk_lean", "release_height", "head_stability",
    ])
    def test_metrics_with_no_spin_override_are_descriptive_for_both_spin_types(self, metric_key, bowler_type):
        """Only front_knee_bracing/wrist_spin has a real override — every
        other metric, for either spin type, must never silently borrow
        pace's band."""
        assert mr.classify(metric_key, 30.0, bowler_type) == "descriptive"

    def test_descriptive_never_fires_for_a_missing_value(self):
        """A None/NaN value is "unknown" regardless of bowler_type — never
        conflate "we have no benchmark" with "we have no measurement"."""
        assert mr.classify("hip_shoulder_separation", None, "wrist_spin") == "unknown"
        assert mr.classify("hip_shoulder_separation", float("nan"), "finger_spin") == "unknown"

    def test_describe_range_states_no_benchmark_for_a_descriptive_metric(self):
        """hip_shoulder_separation is always-descriptive now (real audit,
        2026-08-06 — see TestAlwaysDescriptiveMetrics), so its real text
        is the same technique-dependence note regardless of bowler_type,
        not a generic "no {style} benchmark" message."""
        desc = mr.describe_range("hip_shoulder_separation", "wrist_spin")
        assert "action type" in desc.lower()
        assert "senington" in desc.lower()

    def test_describe_range_reflects_the_real_wrist_spin_knee_override(self):
        desc = mr.describe_range("front_knee_bracing", "wrist_spin")
        assert "134" in desc
        # Must NOT describe pace's 160-180 band for a wrist-spin bowler.
        assert "160" not in desc

    def test_describe_range_default_is_unaffected(self):
        """Same regression this whole class exists to prevent, applied to
        describe_range: the pre-existing 2-arg call must be untouched.
        front_knee_bracing is always-descriptive for pace now (real audit,
        2026-08-06 — see TestAlwaysDescriptiveMetrics), so the real text
        is the Extended/Flexed-Knee classification note, not the old
        160-180 band description."""
        assert mr.describe_range("front_knee_bracing") == mr.describe_range("front_knee_bracing", None)
        assert "Portus" in mr.describe_range("front_knee_bracing")
