"""
tests/test_click_widget_state.py

Regression tests for the real bug reported directly: clicking "Reset"
on a calibration/wrist-correction/seed-point marker didn't reset
anything, and a misclicked point stayed forever. Root cause: the click
widget's key never changed at the default (1x, no zoom) zoom level
regardless of the caller's tracked point state, so Streamlit replayed
the component's last-known click value on every rerun a reset button
caused — immediately re-appending the point that was just cleared.
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
