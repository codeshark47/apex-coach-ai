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


class TestReferenceFrameCacheIsBounded:
    def test_cache_has_a_bounded_entry_count(self):
        info = cal._extract_reference_frame_cached._info
        assert info.max_entries is not None
        assert info.max_entries > 0

    def test_cache_entries_expire(self):
        info = cal._extract_reference_frame_cached._info
        assert info.ttl is not None
        assert info.ttl > 0
