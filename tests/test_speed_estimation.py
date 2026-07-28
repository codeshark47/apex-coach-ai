"""
tests/test_speed_estimation.py

Regression tests for a real reported production bug: a coach reported a
properly-calibrated setup (clicking both popping-crease ends correctly)
still returning ~600 km/h release-arm speed — physically impossible
(fastest recorded deliveries are ~161 km/h).

Root cause: _corroborated_peak_speed_px_s exists specifically to reject
isolated single-frame tracking glitches (verified on real footage: raw
wrist tracking during a fast, blurred swing shows both genuine sustained
bursts AND one-frame jumps that snap straight back, a classic glitch
signature) — but when corroboration failed for EVERY frame in the
window (the window is pure noise, exactly the case the check exists to
catch), the old code fell back to `max(speeds.values())`, the single
WORST, most glitch-prone reading, with zero filtering. That inverted
fallback is almost certainly what produced the 600 km/h reading. Fixed
to return None (a clear "unreliable" signal) instead, and the smoothed-
dataframe fallback path (which had the exact same unprotected-max bug)
now goes through the same corroboration check.
"""

import numpy as np
import pandas as pd

import speed_estimation as se


class TestCorroboratedPeakSpeed:
    def test_isolated_single_frame_glitch_is_rejected(self):
        """The case this function was originally built for: one frame
        teleports far away and snaps straight back — no neighbor
        corroborates it — must be excluded from the result."""
        positions = {
            0: (100.0, 100.0, 1.0),
            1: (105.0, 100.0, 1.0),
            2: (900.0, 900.0, 1.0),  # isolated glitch: huge jump...
            3: (108.0, 100.0, 1.0),  # ...and snaps straight back
            4: (112.0, 100.0, 1.0),
        }
        peak = se._corroborated_peak_speed_px_s(positions, fps=30)
        # The real, corroborated motion is ~3-4px/frame -> ~90-120px/s.
        # The glitch (~800px jump) would be ~24000px/s if it leaked through.
        assert peak is not None
        assert peak < 500

    def test_real_reported_bug_pure_noise_window_returns_none_not_worst_value(self):
        """THE exact regression: every frame-pair speed is an isolated
        spike with no corroborating neighbor (pure tracking noise, e.g.
        heavy motion blur right at release). Must return None — NOT the
        single worst (highest, most glitch-prone) reading in the window,
        which is what produced the reported ~600km/h."""
        positions = {
            0: (100.0, 100.0, 1.0),
            1: (900.0, 100.0, 1.0),   # huge jump
            2: (150.0, 800.0, 1.0),   # huge jump, different direction
            3: (700.0, 50.0, 1.0),    # huge jump, different direction again
            4: (120.0, 400.0, 1.0),   # huge jump, different direction again
        }
        peak = se._corroborated_peak_speed_px_s(positions, fps=30)
        assert peak is None

    def test_genuine_sustained_burst_is_kept(self):
        """A real delivery: several consecutive frames all show similar,
        elevated speed. Must NOT be thrown away just because it's fast —
        only isolated, uncorroborated spikes should be rejected."""
        positions = {
            0: (100.0, 100.0, 1.0),
            1: (150.0, 100.0, 1.0),
            2: (200.0, 100.0, 1.0),
            3: (250.0, 100.0, 1.0),
            4: (300.0, 100.0, 1.0),
        }
        peak = se._corroborated_peak_speed_px_s(positions, fps=30)
        assert peak is not None
        assert peak == 50.0 * 30  # 50px/frame * 30fps, every step agrees


class TestComputeReleaseArmSpeedNeverFabricatesImpossibleNumbers:
    def _df_with_glitch(self, fps=30, n_frames=40, br_idx=20):
        """A wrist landmark series that's calm except for pure noise
        right around release — the real-world failure mode reported
        (heavy motion blur exactly at the moment of fastest arm motion)."""
        frame_width, frame_height = 848, 478
        rng = np.random.default_rng(42)
        x = np.full(n_frames, 400.0)
        y = np.full(n_frames, 200.0)
        window = 3
        for i in range(max(0, br_idx - window), min(n_frames, br_idx + window + 1)):
            x[i] = 400.0 + rng.uniform(-350, 350)
            y[i] = 200.0 + rng.uniform(-150, 150)
        df = pd.DataFrame({
            "frame": np.arange(n_frames),
            "RIGHT_WRIST_x": x / frame_width,
            "RIGHT_WRIST_y": y / frame_height,
            "LEFT_WRIST_x": np.full(n_frames, 0.1),
            "LEFT_WRIST_y": np.full(n_frames, 0.1),
        })
        return df, frame_width, frame_height

    def test_noisy_release_window_never_reports_physically_impossible_speed(self):
        df, fw, fh = self._df_with_glitch()
        events = {"BFC": 0, "FFC": 10, "BR": 20}
        result = se.compute_release_arm_speed(
            df, events, fps=30, frame_width=fw, frame_height=fh,
            meters_per_pixel=0.025,  # a plausible, correctly-calibrated scale
            video_path=None, bowling_arm_override="right",
        )
        # Must never silently report something like 600km/h as "success" —
        # either a sane number, or an honest error, never both a "success"
        # status AND a physically-impossible number.
        if result["status"] == "success":
            assert 0 < result["kmh"] <= 200
        else:
            assert result["status"] == "error"
