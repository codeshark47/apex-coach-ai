"""
track_ball_candidates.py

Phase 2, Step 2: link per-frame ball CANDIDATES (from
detect_ball_classical.py) into candidate TRACKS across consecutive
frames, and score those tracks by physical plausibility — a real ball in
flight moves a meaningful, fairly consistent distance frame to frame in
a fairly consistent direction; a tree leaf, a shadow edge, or a player's
hand flickering in and out of the foreground mask does not link into a
long, smooth chain the same way.

WHY THIS EXISTS: verified directly on real footage (see
detect_ball_classical.py's test results) that raw candidates alone are
far too noisy to use — tens of false positives per frame. This doesn't
fix that by tuning the detector further; it exploits a DIFFERENT signal
(motion over time, not appearance in one frame) that noise generally
lacks and a real ball has by definition.

HONEST STATUS: this is a simple greedy nearest-neighbor linker, not a
proper multi-hypothesis tracker (no Kalman filter, no occlusion
handling yet) — deliberately, so its actual behavior on real data can be
inspected before investing in something more complex. It surfaces
candidate TRACKS ranked by plausibility, not a confirmed answer — the
top-ranked track still needs a human to confirm it's actually the ball
before it's trustworthy for anything.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ball_tracking.detect_ball_classical import BallCandidate


@dataclass
class BallTrack:
    candidates: List[BallCandidate]  # one per frame, in increasing frame order

    @property
    def start_frame(self) -> int:
        return self.candidates[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.candidates[-1].frame_index

    @property
    def length(self) -> int:
        return len(self.candidates)

    @property
    def total_displacement_px(self) -> float:
        if len(self.candidates) < 2:
            return 0.0
        dx = self.candidates[-1].x_px - self.candidates[0].x_px
        dy = self.candidates[-1].y_px - self.candidates[0].y_px
        return float(np.hypot(dx, dy))

    @property
    def velocity_consistency(self) -> float:
        """
        1.0 = perfectly steady frame-to-frame speed, 0.0 = wildly erratic.
        A real ball's speed changes smoothly (gravity, drag) over a few
        frames; noise blobs that happen to link together usually don't.

        BUG FOUND AND FIXED (2026-08-15, real validation against a real,
        human-labeled ball trajectory in real footage): the correct chain
        for a genuine delivery was found by the linker (verified: 6/6 of
        its points matching real ball-position labels within 15px), but
        scored velocity_consistency=0.0 — worse than pure noise. Traced
        directly to the source video, not the algorithm: consecutive
        frames 489/490 are near pixel-identical (mean abs diff 0.13,
        vs. 0.7-3.9 for genuinely different neighboring frames) — a
        duplicate/stuttered frame, a real WhatsApp-compression artifact,
        not the ball stopping. The raw step sequence for this real track
        was 12.9, 0.0, 0.0, 12.8, 0.0, 12.4, ... — a perfectly consistent
        ~12px real step every OTHER frame, interleaved with fake 0px
        steps from duplicate frames, which is about as erratic as a
        coefficient-of-variation calculation can see. A genuinely
        stationary ball is physically impossible mid-flight, so a ~0px
        step is itself evidence of a duplicate frame, not real ball
        motion — excluding those steps (instead of averaging them in as
        if they were real data) measures what this is actually meant to
        measure. Re-verified after this fix: the same real track's score
        went from dead last to a real, competitive rank (see the project
        memory on ball-tracking strategy for the exact before/after).
        """
        if len(self.candidates) < 3:
            return 0.0
        speeds = []
        for a, b in zip(self.candidates, self.candidates[1:]):
            speeds.append(float(np.hypot(b.x_px - a.x_px, b.y_px - a.y_px)))
        speeds = np.array(speeds)
        # A near-zero step between consecutive real video frames is not a
        # real "the ball briefly stopped" reading — it is what a
        # duplicate/stuttered source frame looks like to this detector.
        # 1.0px is well below any real ball's per-frame displacement at
        # this fps/distance, so this only strips duplicate-frame
        # artifacts, not genuine slow motion.
        real_speeds = speeds[speeds > 1.0]
        if len(real_speeds) < 2:
            return 0.0
        if real_speeds.mean() < 1e-6:
            return 0.0
        cv = real_speeds.std() / real_speeds.mean()  # coefficient of variation
        return float(max(0.0, 1.0 - min(cv, 1.0)))

    def plausibility_score(self) -> float:
        """
        BUG FOUND AND FIXED (2026-08-15, real validation against real
        bowler+batter footage from a fixed tripod behind the stumps): the
        original formula (length * (0.5 + 0.5*velcons)) gave every track
        a 0.5-per-frame floor regardless of how erratic it was, so a
        long (40-52 frame), completely incoherent chain (velcons ~0,
        confirmed by direct pixel inspection to be wind/net/building
        texture flicker, not the ball) still outscored the short, smooth
        chains a real ball actually produces in this framing. Squaring
        velocity_consistency (instead of a linear 0.5-floor blend) makes
        genuine erraticism collapse the score toward zero instead of
        merely halving it, so length can no longer compensate for
        incoherence the way it did before.
        """
        return self.length * (self.velocity_consistency ** 2)


def link_candidates_into_tracks(
    candidates_by_frame: dict,
    max_link_distance_px: float = 40.0,
    min_track_length: int = 6,
    max_track_length: int = 25,
    min_total_displacement_px: float = 15.0,
) -> List[BallTrack]:
    """
    Greedy nearest-neighbor linker: for each candidate in frame N, link
    to the closest unclaimed candidate in frame N+1 within
    max_link_distance_px. Chains that survive the length window and
    min_total_displacement_px filters are returned, sorted by
    plausibility (best first).

    max_link_distance_px: how far the ball can plausibly move between
    consecutive frames — depends entirely on fps and camera distance,
    same caveat as detect_ball_classical's radius bounds. Not a
    validated constant; expect to tune per real footage.

    TRIED AND REJECTED (2026-08-15, real evidence, not a design guess):
    velocity-predicted linking — extending each chain toward last + (last
    - second_last) instead of toward the raw last point, the same
    predict-then-search idea track_ball_from_seed.py uses successfully
    for the seeded tracker. Tested directly against the same real,
    human-labeled ball trajectory this module is validated against: it
    made the result WORSE, not better — the one clean, correctly-ranked
    real-ball chain this file DOES find (frames 486-502, 6/6 real
    positions matched, rank #1 of 363) dropped to a worse, lower-ranked,
    partially-wrong chain (5/7 match, rank ~154 of 450) once prediction
    was in play. Root cause: extrapolating velocity from only 2 raw,
    already-slightly-noisy detections has no trusted anchor and no
    second corroborating signal (track_ball_from_seed.py has both — a
    human-confirmed seed point, and a ball-size trend check) to keep a
    bad early estimate from compounding. Reverted rather than keep a
    regression that merely sounded like an improvement. Left here so
    this isn't tried again without remembering why it failed.

    min_track_length / max_track_length (2026-08-15, real validation): a
    real ball's flight from release to the batsman, at 30fps and this
    camera's distance, physically fits in roughly this many frames — a
    chain of only 2-5 candidates is too short to trust as "smooth" (looks
    consistent by pure chance far more easily than a longer one can), and
    a chain past ~25 frames is almost certainly background clutter that
    happened to link (a real ball doesn't stay in frame that long at this
    framing). Rejecting both ends outright, rather than just letting
    plausibility_score() rank them lower, is what stops a long noisy
    chain from ever competing with a real short one in the first place —
    same reasoning as the recurring net/pole/banner false-positive
    lesson elsewhere in this project: bound the SHAPE of what's
    acceptable, don't just hope the score sorts it out. Tied to 30fps and
    this specific camera distance, same as every other bound in this
    module — not a universal constant, revisit per camera setup.

    min_total_displacement_px: rejects tracks that stay roughly in one
    place for their whole length — plausible for a genuinely stationary
    false positive (a bright patch, a stump reflection), implausible for
    a ball actually in flight, which is what this pipeline cares about
    finding.
    """
    frame_indices = sorted(candidates_by_frame.keys())
    if not frame_indices:
        return []

    # active_chains: list of (last_candidate, [candidates in chain])
    active_chains: List[List[BallCandidate]] = []
    completed_chains: List[List[BallCandidate]] = []

    for i, frame_idx in enumerate(frame_indices):
        this_frame_candidates = list(candidates_by_frame[frame_idx])
        claimed = set()

        still_active = []
        for chain in active_chains:
            last = chain[-1]
            # only try to extend if this frame is the immediate next one
            # a chain touched — otherwise the trail has gone cold.
            if frame_idx - last.frame_index > 1:
                completed_chains.append(chain)
                continue
            best_idx, best_dist = None, None
            for j, cand in enumerate(this_frame_candidates):
                if j in claimed:
                    continue
                d = float(np.hypot(cand.x_px - last.x_px, cand.y_px - last.y_px))
                if d <= max_link_distance_px and (best_dist is None or d < best_dist):
                    best_idx, best_dist = j, d
            if best_idx is not None:
                claimed.add(best_idx)
                chain.append(this_frame_candidates[best_idx])
                still_active.append(chain)
            else:
                completed_chains.append(chain)
        active_chains = still_active

        # any candidate not claimed by an existing chain starts a new one
        for j, cand in enumerate(this_frame_candidates):
            if j not in claimed:
                active_chains.append([cand])

    completed_chains.extend(active_chains)

    tracks = [BallTrack(candidates=c) for c in completed_chains
              if min_track_length <= len(c) <= max_track_length]
    tracks = [t for t in tracks if t.total_displacement_px >= min_total_displacement_px]
    tracks.sort(key=lambda t: t.plausibility_score(), reverse=True)
    return tracks
