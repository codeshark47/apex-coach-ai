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
        """
        if len(self.candidates) < 3:
            return 0.0
        speeds = []
        for a, b in zip(self.candidates, self.candidates[1:]):
            speeds.append(float(np.hypot(b.x_px - a.x_px, b.y_px - a.y_px)))
        speeds = np.array(speeds)
        if speeds.mean() < 1e-6:
            return 0.0
        cv = speeds.std() / speeds.mean()  # coefficient of variation
        return float(max(0.0, 1.0 - min(cv, 1.0)))

    def plausibility_score(self) -> float:
        """
        Combines track length and velocity consistency into one ranking
        score. Weighting (length dominates) is an engineering choice —
        a longer track is stronger evidence than a short-but-smooth one,
        since a 3-frame chain can look "smooth" by pure chance far more
        easily than a 15-frame one can.
        """
        return self.length * (0.5 + 0.5 * self.velocity_consistency)


def link_candidates_into_tracks(
    candidates_by_frame: dict,
    max_link_distance_px: float = 40.0,
    min_track_length: int = 5,
    min_total_displacement_px: float = 15.0,
) -> List[BallTrack]:
    """
    Greedy nearest-neighbor linker: for each candidate in frame N, link
    to the closest unclaimed candidate in frame N+1 within
    max_link_distance_px. Chains that survive min_track_length and
    min_total_displacement_px filters are returned, sorted by
    plausibility (best first).

    max_link_distance_px: how far the ball can plausibly move between
    consecutive frames — depends entirely on fps and camera distance,
    same caveat as detect_ball_classical's radius bounds. Not a
    validated constant; expect to tune per real footage.

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
              if len(c) >= min_track_length]
    tracks = [t for t in tracks if t.total_displacement_px >= min_total_displacement_px]
    tracks.sort(key=lambda t: t.plausibility_score(), reverse=True)
    return tracks
