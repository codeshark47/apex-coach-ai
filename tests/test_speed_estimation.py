"""
tests/test_speed_estimation.py

Regression tests for TWO real reported production bugs, both traced to
the same underlying flaw in how peak release-arm speed was estimated:
a properly-calibrated setup reported BOTH physically impossible speeds
(~600 km/h; fastest recorded deliveries are ~161 km/h) on some clips AND
suspiciously slow ones (~5-20 km/h, implausible for genuine bowling
effort) on others — not two separate bugs, one shared cause.

ROOT CAUSE (found on a full re-investigation, not assumed from the first
fix): the previous "corroborated hop" approach required a frame-to-frame
hop's speed to be matched by a comparably-fast NEIGHBORING hop before
trusting it — built to catch a jump-then-snap-back tracking glitch,
which does produce two adjacent, similarly-large hops. But the genuine
peak wrist speed AT ball release is, by definition, a brief, sharp
spike: release is the one instant the arm moves fastest, meaningfully
faster than the frames just before/after it — that's what makes it the
release point. That exact signature (one frame far exceeding its
immediate neighbors) is what the old check was built to reject as
noise. So on a real, fast, clean swing, the TRUE peak often failed its
own "comparable neighbor" test and got discarded — while an unrelated,
genuinely noisy corner of the window that happened to have two
similarly-sized adjacent hops could still pass. This explains BOTH
symptoms from one mechanism.

FIX: _corroborated_peak_speed_px_s now fits a straight line (least
squares) to x(t)/y(t) over a small window centered on each candidate
frame (a simplified Savitzky-Golay derivative) and uses the FITTED
slope as that frame's velocity, gated by the fit's R^2 (fit quality).
Real coherent motion — even a brief, sharp peak — fits a short window
well (high R^2); a tracking glitch or pure noise does not. fit_radius=2,
min_r_squared=0.9 were tuned empirically (see the session that added
this) against synthetic glitch/noise/sustained-motion scenarios,
including a 1000-trial sweep confirming pure random noise passes the
R^2 gate only ~0.2% of the time.

Test window sizes below (~15-21 frames) match how this is actually
called in production — the raw re-extraction window around release is
comfortably wider than one fit window, so a single bad frame doesn't
have to poison the only candidate; a real, clean window elsewhere in
the same clip can still be found and used.
"""

import numpy as np
import pandas as pd
import pytest

import speed_estimation as se


def _positions_from_xy(xs, ys, start_frame=0):
    return {start_frame + i: (x, y, 1.0) for i, (x, y) in enumerate(zip(xs, ys))}


class TestCorroboratedPeakSpeed:
    def test_isolated_single_frame_glitch_surrounded_by_good_data_is_rejected(self):
        """The case this function exists to guard against: one frame
        teleports far away (heavy motion blur misdetection), with plenty
        of genuinely tracked, stationary frames on both sides. The glitch
        must not leak through as a fabricated high speed."""
        n = 21
        x = np.full(n, 100.0)
        y = np.full(n, 100.0)
        x[10] = 900.0  # single-frame glitch
        y[10] = 900.0
        peak = se._corroborated_peak_speed_px_s(_positions_from_xy(x, y), fps=30)
        assert peak is not None
        assert peak < 100  # genuinely near-stationary, not the glitch's ~35000px/s

    def test_real_reported_bug_pure_noise_window_returns_none_not_worst_value(self):
        """THE original regression: a window that's pure tracking noise
        end to end (heavy motion blur right at release, no genuine
        coherent motion anywhere) must return None — NOT the single
        worst (highest, most glitch-prone) reading, which is what
        produced the originally reported ~600km/h."""
        n = 21
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 800, n)
        y = rng.uniform(0, 800, n)
        peak = se._corroborated_peak_speed_px_s(_positions_from_xy(x, y), fps=30)
        assert peak is None

    def test_genuine_sustained_burst_is_kept(self):
        """A real delivery: many consecutive frames all show the same
        steady, elevated speed. Must not be thrown away just because
        it's fast."""
        n = 21
        x = np.arange(n) * 50.0
        y = np.full(n, 100.0)
        peak = se._corroborated_peak_speed_px_s(_positions_from_xy(x, y), fps=30)
        assert peak is not None
        assert abs(peak - 50.0 * 30) < 0.01  # 50px/frame * 30fps, every step agrees

    def test_genuine_sharp_brief_peak_is_kept_not_rejected_for_lacking_a_comparable_neighbor(self):
        """THE specific blind spot found in the OLD approach: a real
        release swing is slow on approach, has a brief (few-frame) sharp
        speed burst right at release, then slows again — the burst is
        NOT comparable in magnitude to its immediate neighbors (that's
        what makes it the peak), which is exactly what made the old
        "needs a comparable neighbor" check discard genuine peaks. Must
        be detected, close to its true magnitude, not diluted to near
        the slow surrounding speed."""
        n = 21
        x = np.zeros(n)
        for i in range(n):
            if i < 9:
                x[i] = i * 2.0  # slow approach, ~60px/s
            elif i < 13:
                x[i] = x[8] + (i - 8) * 40.0  # sharp burst, ~1200px/s
            else:
                x[i] = x[12] + (i - 12) * 3.0  # slow follow-through
        y = np.full(n, 100.0)
        peak = se._corroborated_peak_speed_px_s(_positions_from_xy(x, y), fps=30)
        assert peak is not None
        assert peak > 1000  # close to the true ~1200px/s burst, not ~60-90px/s

    def test_pure_noise_false_positive_rate_is_low(self):
        """Statistical sanity check on the R^2 gate itself: across many
        independent pure-noise windows, only a small fraction should
        randomly happen to look enough like a line to pass — verified
        empirically at ~0.2% over 1000 trials when this was tuned."""
        n = 21
        false_positives = 0
        trials = 200
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            x = rng.uniform(0, 800, n)
            y = rng.uniform(0, 800, n)
            peak = se._corroborated_peak_speed_px_s(_positions_from_xy(x, y), fps=30)
            if peak is not None:
                false_positives += 1
        assert false_positives / trials < 0.05


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

    def test_tracking_unstable_failure_carries_a_machine_readable_reason(self):
        """FIX (2026-08-07, real bug found on a live clip): this failure
        used to be identifiable only by string-matching the human message
        — orchestrator.py's release_height/head_stability tracking-
        uncertain flag relies on a DIFFERENT signal (detect_delivery_
        events' br_confidence, a coarser whole-window aggregate) that can
        stay "high" even when this stricter frame-level check fails,
        confirmed live: the UI showed this exact instability message for
        the speed estimate while release_height's own warning never
        appeared. streamlit_app.py now cross-checks this "reason" field
        directly instead of relying on br_confidence alone — this test
        pins down that the field actually exists and is set correctly
        whenever this specific failure fires."""
        df, fw, fh = self._df_with_glitch()  # fixed seed=42 - deterministic
        events = {"BFC": 0, "FFC": 10, "BR": 20}
        result = se.compute_release_arm_speed(
            df, events, fps=30, frame_width=fw, frame_height=fh,
            meters_per_pixel=0.025, video_path=None, bowling_arm_override="right",
        )
        assert result["status"] == "error"
        assert result["reason"] == "tracking_unstable"


class TestComputeEstimatedStandingHeight:
    """
    compute_estimated_standing_height (2026-08-05): automatic bowler-height
    estimate from the segment-sum body-height baseline
    (orchestrator._compute_segment_sum_body_height) converted through
    stump calibration — a coach explicitly said no coach will realistically
    supply a bowler's real height manually, so this has to be fully
    automatic or not exist at all. Same never-invent-a-scale and
    never-return-an-implausible-value discipline as
    compute_release_height_absolute right above it.
    """

    def test_not_calibrated_when_no_scale_available(self):
        result = se.compute_estimated_standing_height(0.60, frame_height=1080, meters_per_pixel=None)
        assert result["status"] == "not_calibrated"

    def test_no_baseline_when_segment_sum_is_none(self):
        """A clip too short/noisy to find enough plausible early run-up
        frames must say so, not silently fall back to guessing."""
        result = se.compute_estimated_standing_height(None, frame_height=1080, meters_per_pixel=0.0006)
        assert result["status"] == "no_baseline"

    def test_plausible_real_height_succeeds(self):
        # segment_sum=0.60 (normalized), frame_height=1080px,
        # meters_per_pixel=0.0006 (a plausible stump-height-calibrated
        # scale for a bowler filling most of the frame) ->
        # 0.60 * 1080 * 0.0006 = 0.3888m... use a scale that lands in a
        # real human range instead, e.g. meters_per_pixel=0.0027:
        # 0.60 * 1080 * 0.0027 = 1.7496m = 174.96cm
        result = se.compute_estimated_standing_height(0.60, frame_height=1080, meters_per_pixel=0.0027)
        assert result["status"] == "success"
        assert 100 < result["cm"] < 230
        assert result["cm"] == pytest.approx(175.0, abs=0.1)

    def test_implausible_height_is_flagged_not_trusted(self):
        """A wildly-off calibration or tracking failure would otherwise
        produce a nonsense height (e.g. 50cm or 400cm) — must be flagged
        as an error, never presented as a real reading."""
        # Same segment_sum as above, but a scale an order of magnitude off.
        result = se.compute_estimated_standing_height(0.60, frame_height=1080, meters_per_pixel=0.05)
        assert result["status"] == "error"
