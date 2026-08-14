"""
Unit tests for main._walk_from_seed's appearance-gating logic.

Uses synthetic landmarks/histograms (no video, no MediaPipe) so these run
fast and test the walk/gating LOGIC directly, independent of whether real
footage happens to exercise a position/appearance conflict. The real-footage
appearance-signal validation (does an HSV histogram actually discriminate
two different real people) was done separately against the real 173-frame
clip and is not repeated here.
"""
import numpy as np
import pytest

import main


class _FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _torso_landmarks(cx, cy, spread=0.02):
    """33-slot landmark list with just the torso indices (0, 11, 12, 23, 24)
    populated meaningfully — the only ones _centroid_xy/_bbox_from_landmarks
    read."""
    landmarks = [_FakeLandmark(0.0, 0.0) for _ in range(33)]
    landmarks[0] = _FakeLandmark(cx, cy - spread)        # nose
    landmarks[11] = _FakeLandmark(cx - spread, cy)         # left shoulder
    landmarks[12] = _FakeLandmark(cx + spread, cy)         # right shoulder
    landmarks[23] = _FakeLandmark(cx - spread, cy + spread)  # left hip
    landmarks[24] = _FakeLandmark(cx + spread, cy + spread)  # right hip
    return landmarks


def _hist(peak_bin):
    h = np.zeros((16, 16), dtype=np.float32)
    h[peak_bin] = 1.0
    return h


_HIST_A = _hist((2, 2))     # "the real bowler"'s appearance
_HIST_B = _hist((14, 14))   # a completely different appearance


def test_appearance_gate_rejects_a_positionally_plausible_but_wrong_looking_candidate():
    """
    Direct regression test for the real bug found validating against real
    footage (2026-08-10): the appearance check used to be skipped whenever
    the gap since the last successful match was small — but a match resets
    that gap to 1 regardless of whether the match was RIGHT, so a candidate
    that's wrong but happens to sit within the (very tight, gap=1) position
    tolerance was never appearance-checked at all. This builds exactly that
    scenario: a wrong-looking candidate sitting well within position
    tolerance of the anchor, immediately after a fresh confirmation.
    """
    total = 6
    frame_candidates = [[] for _ in range(total)]
    frame_hists = [[] for _ in range(total)]

    # Seed + two confirmations build a profile of consistent appearance A.
    frame_candidates[0] = [_torso_landmarks(0.50, 0.50)]
    frame_hists[0] = [_HIST_A]
    frame_candidates[1] = [_torso_landmarks(0.50, 0.50)]
    frame_hists[1] = [_HIST_A]
    frame_candidates[2] = [_torso_landmarks(0.50, 0.50)]
    frame_hists[2] = [_HIST_A]

    # Frame 3: ONLY a wrong-looking candidate, positioned just 0.01 away
    # from the anchor — well within the gap=1 position tolerance (~0.02),
    # but visually nothing like the profile.
    frame_candidates[3] = [_torso_landmarks(0.51, 0.50)]
    frame_hists[3] = [_HIST_B]

    result = main._walk_from_seed(0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=3)

    assert result[3] is None, "a visually-mismatched candidate must not be accepted just because it's positionally close"


def test_appearance_gate_accepts_a_genuinely_continuing_candidate():
    """Same setup, but the frame-3 candidate actually matches the profile's
    appearance — must still be accepted, proving the gate isn't just
    rejecting everything."""
    total = 6
    frame_candidates = [[] for _ in range(total)]
    frame_hists = [[] for _ in range(total)]

    frame_candidates[0] = [_torso_landmarks(0.50, 0.50)]
    frame_hists[0] = [_HIST_A]
    frame_candidates[1] = [_torso_landmarks(0.50, 0.50)]
    frame_hists[1] = [_HIST_A]
    frame_candidates[2] = [_torso_landmarks(0.50, 0.50)]
    frame_hists[2] = [_HIST_A]
    frame_candidates[3] = [_torso_landmarks(0.51, 0.50)]
    frame_hists[3] = [_HIST_A]

    result = main._walk_from_seed(0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=3)

    assert result[3] is not None, "a visually-consistent, positionally-plausible candidate should still be accepted"


def test_appearance_gate_does_not_reject_before_enough_profile_history_exists():
    """Right after the seed, before APPEARANCE_MIN_PROFILE frames have been
    confirmed, there isn't enough data to judge appearance yet — must fail
    open to the existing position-only behavior rather than reject
    everything blind."""
    total = 3
    frame_candidates = [[] for _ in range(total)]
    frame_hists = [[] for _ in range(total)]

    frame_candidates[0] = [_torso_landmarks(0.50, 0.50)]
    frame_hists[0] = [_HIST_A]
    # Only ONE prior confirmation before this — not enough profile history
    # (APPEARANCE_MIN_PROFILE == 3) for the appearance gate to engage yet.
    frame_candidates[1] = [_torso_landmarks(0.50, 0.50)]
    frame_hists[1] = [_HIST_B]  # even a mismatched appearance should pass here

    result = main._walk_from_seed(0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=1)

    assert result[1] is not None, "with insufficient profile history, the walk must fail open on position alone"


def test_walk_still_works_with_no_seed_appearance_data():
    """If the seed frame's histogram is None (e.g. a degenerate crop), the
    walk must not crash and should still track on position alone."""
    total = 2
    frame_candidates = [[_torso_landmarks(0.50, 0.50)], [_torso_landmarks(0.50, 0.50)]]
    frame_hists = [[None], [None]]

    result = main._walk_from_seed(0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=1)

    assert result[0] is not None
    assert result[1] is not None
