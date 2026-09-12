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

    result, _ = main._walk_from_seed(0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=3)

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

    result, _ = main._walk_from_seed(0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=3)

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

    result, _ = main._walk_from_seed(0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=1)

    assert result[1] is not None, "with insufficient profile history, the walk must fail open on position alone"


def test_walk_still_works_with_no_seed_appearance_data():
    """If the seed frame's histogram is None (e.g. a degenerate crop), the
    walk must not crash and should still track on position alone."""
    total = 2
    frame_candidates = [[_torso_landmarks(0.50, 0.50)], [_torso_landmarks(0.50, 0.50)]]
    frame_hists = [[None], [None]]

    result, _ = main._walk_from_seed(0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=1)

    assert result[0] is not None
    assert result[1] is not None


class TestSeedFrameMatchTolerance:
    """
    REAL BUG (2026-09-12, coach-reported): the seed-frame match itself has
    NO appearance check (empty profile at that exact point) — it's pure
    nearest-candidate-within-SEED_MATCH_TOLERANCE. Confirmed on real
    footage: a click aimed at the bowler still matched the batter instead.
    SEED_MATCH_TOLERANCE was tightened 0.2 -> 0.08 as the first line of
    defense — these tests pin the new, tighter boundary down directly.
    """

    def test_a_candidate_just_outside_the_tightened_radius_is_not_matched(self):
        """0.09 away from the click — inside the OLD 0.2 tolerance (would
        have matched before this fix) but outside the new 0.08 one."""
        frame_candidates = [[_torso_landmarks(0.59, 0.50)]]
        frame_hists = [[_HIST_A]]
        result, chosen_hist = main._walk_from_seed(
            0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=0)
        assert result[0] is None
        assert chosen_hist is None

    def test_a_candidate_just_inside_the_tightened_radius_still_matches(self):
        frame_candidates = [[_torso_landmarks(0.55, 0.50)]]
        frame_hists = [[_HIST_A]]
        result, chosen_hist = main._walk_from_seed(
            0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=0)
        assert result[0] is not None
        assert chosen_hist is _HIST_A

    def test_trust_seed_match_false_discards_a_real_match(self):
        """The cross-seed appearance override: even a real, in-range
        candidate must be ignored when trust_seed_match=False, falling
        back to position-only tracking from the raw click instead."""
        frame_candidates = [[_torso_landmarks(0.50, 0.50)]]
        frame_hists = [[_HIST_A]]
        result, chosen_hist = main._walk_from_seed(
            0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=0,
            trust_seed_match=False)
        assert result[0] is None
        assert chosen_hist is None


class TestPriorProfileWarmStart:
    """
    REAL BUG (2026-09-12, traced on an actual coach-reported failure):
    when a seed's own exact-frame match finds nobody (the target was
    briefly too small/distant to detect — legitimate, not a bug by
    itself), the walk used to start with a completely empty appearance
    profile, so it fails open on position alone for the first
    APPEARANCE_MIN_PROFILE (3) matches — long enough for a wrong-but-
    nearby person to become "confirmed" before the real target is ever
    seen in that zone. prior_profile lets the caller warm-start the
    profile from the coach's OTHER confirmed seeds elsewhere in the clip.
    """

    def test_without_prior_profile_a_wrong_but_close_candidate_is_accepted_first(self):
        """Baseline (no fix engaged): confirms the vulnerability is real
        — with an empty profile, the first plausible-position candidate
        wins even though it looks nothing like the real target."""
        frame_candidates = [[], [_torso_landmarks(0.50, 0.50)]]
        frame_hists = [[], [_HIST_B]]  # a wrong-looking candidate, close to the click
        result, chosen_hist = main._walk_from_seed(
            0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=1)
        assert chosen_hist is None  # nothing at the seed frame itself
        assert result[1] is not None  # accepted anyway, on position alone

    def test_with_prior_profile_the_same_wrong_candidate_is_rejected(self):
        """Same exact scenario, but the caller supplies a real appearance
        reference from other seeds (all looking like _HIST_A) — the
        wrong-looking candidate must now be rejected from frame 1."""
        frame_candidates = [[], [_torso_landmarks(0.50, 0.50)]]
        frame_hists = [[], [_HIST_B]]
        result, _ = main._walk_from_seed(
            0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=1,
            prior_profile=[_HIST_A, _HIST_A, _HIST_A])
        assert result[1] is None

    def test_with_prior_profile_a_genuinely_matching_candidate_is_still_accepted(self):
        """The warm start must not become a blanket rejection — a
        candidate that DOES look like the reference profile should still
        be picked up."""
        frame_candidates = [[], [_torso_landmarks(0.50, 0.50)]]
        frame_hists = [[], [_HIST_A]]
        result, _ = main._walk_from_seed(
            0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=1,
            prior_profile=[_HIST_A, _HIST_A, _HIST_A])
        assert result[1] is not None

    def test_a_real_seed_match_takes_priority_over_prior_profile(self):
        """If this seed's OWN exact-frame match succeeded, that real
        histogram seeds the profile — prior_profile must never override
        or get mixed in ahead of real, direct evidence for this seed."""
        frame_candidates = [[_torso_landmarks(0.50, 0.50)]]
        frame_hists = [[_HIST_B]]
        result, chosen_hist = main._walk_from_seed(
            0, (0.50, 0.50), frame_candidates, frame_hists, fps=30, lo_bound=0, hi_bound=0,
            prior_profile=[_HIST_A, _HIST_A, _HIST_A])
        assert chosen_hist is _HIST_B


class TestSeedAppearanceMajorityCheck:
    """
    main._seed_appearance_majority_ok — cross-checks each seed's matched
    appearance against the OTHERS, since the coach confirms the SAME
    bowler at every seed. The credible second layer of defense for the
    same real bug as TestSeedFrameMatchTolerance above: even a seed whose
    click WAS close enough to match a candidate can still have matched
    the wrong person if that person happened to be the closer one.
    """

    def test_fewer_than_three_real_histograms_returns_all_true(self):
        """Not enough data to vote with — must not guess from 1-2 points."""
        assert main._seed_appearance_majority_ok([_HIST_A, _HIST_B]) == [True, True]
        assert main._seed_appearance_majority_ok([_HIST_A, None]) == [True, True]
        assert main._seed_appearance_majority_ok([None, None, None]) == [True, True, True]

    def test_three_agreeing_seeds_are_all_ok(self):
        assert main._seed_appearance_majority_ok([_HIST_A, _HIST_A, _HIST_A]) == [True, True, True]

    def test_one_outlier_among_three_agreeing_seeds_is_flagged(self):
        """4 real seed clicks, 3 of which matched the same-looking bowler
        and 1 which matched something else entirely — the exact real
        scenario this whole fix exists for."""
        result = main._seed_appearance_majority_ok([_HIST_A, _HIST_A, _HIST_A, _HIST_B])
        assert result == [True, True, True, False]

    def test_no_clear_majority_flags_nothing(self):
        """Three seeds that all look equally DIFFERENT from each other —
        no reliable majority to vote with, so nothing gets overridden
        rather than guessing which one (if any) is "correct"."""
        hist_c = _hist((8, 8))
        result = main._seed_appearance_majority_ok([_HIST_A, _HIST_B, hist_c])
        assert result == [True, True, True]
