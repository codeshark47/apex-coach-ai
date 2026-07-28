"""
tests/test_calibration.py

Regression test for a real production incident: the live app hit
Streamlit Cloud's memory limit and crashed ("This app has gone over its
resource limits") during ordinary coach testing. Root cause:
_extract_reference_frame_cached had no max_entries/ttl, so every distinct
frame any coach ever scrubbed to (via the frame-jump box or slider),
on any video, since the app's last reboot stayed cached forever as a
multi-MB decoded image array — and Streamlit Cloud runs one shared
process for every visitor, so this never resets between coaches or
sessions. This test guards against the bound silently being removed
again (e.g. by someone "cleaning up" the decorator).
"""

import calibration as cal


class TestImplausibilityWarning:
    def test_real_reported_case_is_flagged(self):
        """The exact case reported directly: 'popping crease to popping
        crease' (20.12m) calibrated to 0.457155 m/px, which implies the
        two clicked points were only ~44px apart — nowhere near 'full
        pitch in frame'. Must be flagged, not silently accepted."""
        calibration = cal.compute_scale(
            point_a_px=(100, 200), point_b_px=(144, 200),
            real_world_distance_m=20.12, reference_label="popping crease to popping crease",
        )
        warning = cal.implausibility_warning(calibration, frame_width_px=848)
        assert warning is not None

    def test_realistic_full_pitch_calibration_is_not_flagged(self):
        """A genuinely correct full-pitch calibration — points spanning
        most of a real frame width — must NOT be flagged."""
        calibration = cal.compute_scale(
            point_a_px=(50, 300), point_b_px=(798, 300),
            real_world_distance_m=20.12, reference_label="popping crease to popping crease",
        )
        warning = cal.implausibility_warning(calibration, frame_width_px=848)
        assert warning is None

    def test_close_up_stump_width_calibration_is_not_flagged(self):
        """A short real-world reference distance (stump width) can
        legitimately have a small pixel span on a close-up shot — must
        not be flagged just because the span is small."""
        calibration = cal.compute_scale(
            point_a_px=(400, 300), point_b_px=(420, 300),
            real_world_distance_m=0.2286, reference_label="stump width",
        )
        warning = cal.implausibility_warning(calibration, frame_width_px=848)
        assert warning is None


class TestReferenceFrameCacheIsBounded:
    def test_cache_has_a_bounded_entry_count(self):
        info = cal._extract_reference_frame_cached._info
        assert info.max_entries is not None
        assert info.max_entries > 0

    def test_cache_entries_expire(self):
        info = cal._extract_reference_frame_cached._info
        assert info.ttl is not None
        assert info.ttl > 0
