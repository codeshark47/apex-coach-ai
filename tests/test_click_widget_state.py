"""
tests/test_click_widget_state.py

Two unrelated bugs, both fixed in click_widget_state.py because
streamlit_app.py runs UI code at import time and can't be imported
directly in a test:

1. TestNextClickGeneration: clicking "Reset" on a calibration/wrist-
   correction/seed-point marker didn't reset anything, and a misclicked
   point stayed forever. Root cause: the click widget's key never
   changed at the default (1x, no zoom) zoom level regardless of the
   caller's tracked point state, so Streamlit replayed the component's
   last-known click value on every rerun a reset button caused —
   immediately re-appending the point that was just cleared.

2. TestSeedConfirmationStatus: a real production crash. Repeated
   MediaPipe PoseLandmarker creation was measured to permanently leak
   memory even after proper .close() calls, and the single-seed case
   (no extra confirmations) had no confirmation gate at all — every
   click adjusting the primary seed point re-triggered a full, leaking
   extraction pass.
"""

import click_widget_state as cws


class TestNextClickGeneration:
    def test_same_state_does_not_bump_generation(self):
        state = {}
        gen1 = cws.next_click_generation(state, "calib", None, [])
        gen2 = cws.next_click_generation(state, "calib", None, [])
        assert gen1 == gen2

    def test_adding_a_point_bumps_generation(self):
        state = {}
        gen_empty = cws.next_click_generation(state, "calib", None, [])
        gen_one_point = cws.next_click_generation(
            state, "calib", None, [{"point": (10, 20)}]
        )
        assert gen_one_point != gen_empty

    def test_reset_bumps_generation_even_though_state_matches_the_original_empty_state(self):
        """The exact regression case: empty -> point1 -> point2 -> RESET
        back to empty. The generation after reset must differ from every
        generation seen so far, including the very first "empty" one —
        otherwise the widget key repeats a string Streamlit already has
        a stale click cached against, and the bug resurfaces."""
        state = {}
        seen_generations = []

        seen_generations.append(cws.next_click_generation(state, "calib", None, []))
        seen_generations.append(cws.next_click_generation(
            state, "calib", None, [{"point": (10, 20)}]
        ))
        seen_generations.append(cws.next_click_generation(
            state, "calib", None, [{"point": (10, 20)}, {"point": (30, 40)}]
        ))
        # RESET — back to the exact same shape as the very first call
        gen_after_reset = cws.next_click_generation(state, "calib", None, [])

        assert gen_after_reset not in seen_generations

    def test_moving_a_single_marker_point_bumps_generation(self):
        """Covers the wrist/release-point and seed-point flows, which
        track one marker_point rather than a list of extra_markers."""
        state = {}
        gen1 = cws.next_click_generation(state, "single_wrist", (100, 200), [])
        gen2 = cws.next_click_generation(state, "single_wrist", (150, 250), [])
        assert gen1 != gen2

    def test_resetting_single_marker_point_to_none_bumps_generation(self):
        state = {}
        gen_with_point = cws.next_click_generation(state, "single_wrist", (100, 200), [])
        gen_after_reset = cws.next_click_generation(state, "single_wrist", None, [])
        assert gen_after_reset != gen_with_point

    def test_different_key_prefixes_track_independent_generations(self):
        """Calibration, wrist-correction, and seed-point flows all share
        this same function — their counters must not interfere with
        each other in the same session_state dict."""
        state = {}
        cws.next_click_generation(state, "calib", None, [{"point": (1, 1)}])
        wrist_gen = cws.next_click_generation(state, "single_wrist", None, [])
        assert wrist_gen == 1  # unaffected by calibration's own bumps


class TestSeedConfirmationStatus:
    """Regression tests for a real production crash: repeated MediaPipe
    PoseLandmarker creation was measured to permanently leak ~30-40MB
    per extraction call, even after .close() and gc.collect(). The
    single-seed case (no extra confirmations) previously had NO
    confirmation gate at all, so every click adjusting the primary seed
    point's placement re-triggered a full, leaking extraction — a live
    session's log showed 11 extraction passes in 90 seconds of ordinary
    seed placement, and the app was OOM-killed shortly after."""

    def test_single_seed_with_no_extras_is_not_ready_by_default(self):
        """THE exact regression: this case used to return ready=True
        unconditionally. It must now require explicit confirmation like
        any other seed configuration."""
        state = {}
        is_ready, _ = cws.seed_confirmation_status(state, "single", (100, 200), 0, [])
        assert is_ready is False

    def test_single_seed_becomes_ready_after_explicit_confirmation(self):
        state = {}
        _, pending_identity = cws.seed_confirmation_status(state, "single", (100, 200), 0, [])
        cws.lock_seed_confirmation(state, "single", pending_identity)
        is_ready, _ = cws.seed_confirmation_status(state, "single", (100, 200), 0, [])
        assert is_ready is True

    def test_adjusting_the_seed_point_after_confirming_requires_reconfirmation(self):
        """A coach who re-clicks to nudge the seed point after already
        confirming must not silently reuse the OLD confirmation for a
        DIFFERENT point."""
        state = {}
        _, pending_identity = cws.seed_confirmation_status(state, "single", (100, 200), 0, [])
        cws.lock_seed_confirmation(state, "single", pending_identity)

        is_ready, _ = cws.seed_confirmation_status(state, "single", (105, 205), 0, [])
        assert is_ready is False

    def test_extra_seeds_case_still_requires_confirmation(self):
        """The original (already-fixed) case must still work: adding
        extra confirmations also requires an explicit continue."""
        state = {}
        is_ready, _ = cws.seed_confirmation_status(
            state, "single", (100, 200), 0, [{"point": (300, 400), "frame": 10}]
        )
        assert is_ready is False

    def test_confirmed_extra_seeds_configuration_stays_ready(self):
        state = {}
        extra_seeds = [{"point": (300, 400), "frame": 10}]
        _, pending_identity = cws.seed_confirmation_status(state, "single", (100, 200), 0, extra_seeds)
        cws.lock_seed_confirmation(state, "single", pending_identity)
        is_ready, _ = cws.seed_confirmation_status(state, "single", (100, 200), 0, extra_seeds)
        assert is_ready is True

    def test_different_key_prefixes_track_independent_confirmations(self):
        """Dual Camera's side/rear streams share this same function —
        confirming one must not silently confirm the other."""
        state = {}
        _, side_identity = cws.seed_confirmation_status(state, "side", (1, 1), 0, [])
        cws.lock_seed_confirmation(state, "side", side_identity)

        rear_ready, _ = cws.seed_confirmation_status(state, "rear", (1, 1), 0, [])
        assert rear_ready is False
