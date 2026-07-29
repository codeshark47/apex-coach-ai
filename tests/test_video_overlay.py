"""
tests/test_video_overlay.py

Regression test for a real bug found from a direct visual comparison:
the coach flagged skeleton joint dots as looking oversized and
"covering the body" specifically when the bowler is far from camera
(smaller on screen) — despite two earlier rounds of shrinking the base
dot size, the problem persisted.

Root cause: outline and core radii were each computed independently via
the same fixed floor (4px). Verified directly on a real rear-view clip:
body height (the per-frame scale reference) ranged from ~120px
(follow-through, most distant frame) to ~539px across one delivery. At
120px, BOTH the outline (base 6px) and core (base 4px) radii rounded
below the 4px floor and got clamped to the identical value — collapsing
the two-tone ring into one flat blob, and one that's proportionally
LARGER relative to the now-smaller body than at the 400px reference
scale this was tuned against (4px is 1.5% of 400px, but >3% of 120px).

Fixed in joint_marker_radii: the outline is now always core + a small
gap that itself scales down (with its own lower floor), so the ring
can never collapse into the core, and the whole marker keeps shrinking
with distance instead of hitting the old floor's wall.
"""

import pandas as pd

import video_overlay as vo


class TestJointMarkerRadii:
    def test_matches_old_fixed_values_at_reference_scale(self):
        """render_scale=1.0 is the reference size this was always tuned
        against — must render identically to the old hardcoded (4, 6),
        i.e. zero behavior change for a normally-sized subject."""
        core_r, outline_r = vo.joint_marker_radii(4, render_scale=1.0)
        assert core_r == 4
        assert outline_r == 6

    def test_real_reported_case_outline_and_core_no_longer_collide(self):
        """THE exact regression: render_scale corresponding to a ~120px
        body height against the 400px reference (0.3) — the old code
        collapsed both radii to the same floored value (4, 4). The ring
        must survive: outline strictly greater than core."""
        core_r, outline_r = vo.joint_marker_radii(4, render_scale=0.3)
        assert outline_r > core_r

    def test_distant_subject_dot_is_smaller_than_reference_dot(self):
        """The actual complaint: dots must get SMALLER, not stay pinned
        to the same floored size, as the subject gets more distant."""
        core_ref, outline_ref = vo.joint_marker_radii(4, render_scale=1.0)
        core_far, outline_far = vo.joint_marker_radii(4, render_scale=0.3)
        assert core_far < core_ref
        assert outline_far < outline_ref

    def test_never_shrinks_below_the_legibility_floor(self):
        """Even at an extreme scale, the core dot must stay a visible
        point, not vanish to sub-pixel/zero."""
        core_r, outline_r = vo.joint_marker_radii(4, render_scale=0.01)
        assert core_r >= 2
        assert outline_r > core_r

    def test_hero_node_size_bonus_keeps_the_ring_visible(self):
        """Callers add a node_extra bonus to BOTH radii for emphasized
        joints (e.g. the lead knee) — the ring gap must survive that
        bonus being added identically to both radii."""
        core_r, outline_r = vo.joint_marker_radii(4, render_scale=0.65)
        node_extra = 3
        assert (outline_r + node_extra) > (core_r + node_extra)


def _torso_row(shoulder_w, hip_w, torso_h, shoulder_y=0.3, hip_y=None):
    """A minimal row with just the 8 landmark columns
    torso_shape_is_plausible needs, laid out so shoulder/hip width and
    torso height come out to the requested normalized (0-1) values."""
    if hip_y is None:
        hip_y = shoulder_y + torso_h
    return pd.Series({
        "LEFT_SHOULDER_x": 0.5 - shoulder_w / 2, "LEFT_SHOULDER_y": shoulder_y,
        "RIGHT_SHOULDER_x": 0.5 + shoulder_w / 2, "RIGHT_SHOULDER_y": shoulder_y,
        "LEFT_HIP_x": 0.5 - hip_w / 2, "LEFT_HIP_y": hip_y,
        "RIGHT_HIP_x": 0.5 + hip_w / 2, "RIGHT_HIP_y": hip_y,
    })


class TestTorsoShapeIsPlausible:
    """Regression test for a real bug found from a coach-downloaded
    rear-view clip: a bowler bending over to pick up the ball (small,
    distant, a hard case for pose estimation) produced a MediaPipe
    reading with shoulders/hips splayed nearly 3x wider than the torso
    is tall — drawn as-is, with no plausibility check at all, as a
    bizarre "tent/spider" shape instead of a body."""

    def test_normal_standing_pose_is_plausible(self):
        row = _torso_row(shoulder_w=0.15, hip_w=0.12, torso_h=0.2)
        assert vo.torso_shape_is_plausible(row) is True

    def test_real_reported_case_is_rejected(self):
        """The exact regression: shoulders/hips ~3x the torso height —
        physically impossible for any human pose."""
        row = _torso_row(shoulder_w=0.45, hip_w=0.45, torso_h=0.15)
        assert vo.torso_shape_is_plausible(row) is False

    def test_bent_over_but_still_human_pose_is_plausible(self):
        """A real bent-over pose compresses torso height without
        blowing shoulder/hip width out sideways — must not be rejected
        just for being bent over."""
        row = _torso_row(shoulder_w=0.14, hip_w=0.11, torso_h=0.09)
        assert vo.torso_shape_is_plausible(row) is True

    def test_missing_landmarks_default_to_plausible(self):
        """Not enough data to judge either way — must not become a new
        reason frames go missing beyond the specific failure this
        guards against."""
        row = pd.Series({"LEFT_SHOULDER_x": 0.4, "LEFT_SHOULDER_y": 0.3})
        assert vo.torso_shape_is_plausible(row) is True


class TestLimbSegmentIsPlausible:
    """Regression test for a real bug found on a coach-downloaded
    rear/front-view clip: during a fast, motion-blurred release swing,
    the bowling arm tracked correctly (raised high, holding the ball)
    while the OTHER arm's wrist floated to a position with nothing
    visible there in the real frame — connected by a bone segment far
    longer than any real arm segment, reading as a phantom second limb.
    Front/rear view can't just hide the "other" arm like side-on does
    (both are genuinely visible from that angle), so this checks each
    arm segment's own length instead."""

    def test_normal_forearm_length_is_plausible(self):
        # torso_h=0.2 (typical); a forearm ~0.5x torso length is normal.
        assert vo.limb_segment_is_plausible((0.5, 0.4), (0.55, 0.5), torso_h=0.2) is True

    def test_real_reported_case_floating_wrist_is_rejected(self):
        """A wrist floating far from the elbow/shoulder — segment length
        several times the torso height — must be rejected."""
        shoulder = (0.5, 0.3)
        floating_wrist = (0.5, 0.9)  # 0.6 away vertically alone
        assert vo.limb_segment_is_plausible(shoulder, floating_wrist, torso_h=0.2) is False

    def test_fully_extended_arm_is_still_plausible(self):
        """A real, fully extended arm (e.g. reaching overhead at
        release) must not be rejected just for being long."""
        assert vo.limb_segment_is_plausible((0.5, 0.3), (0.5, 0.55), torso_h=0.2) is True

    def test_zero_torso_height_defaults_to_plausible(self):
        """Can't judge without a usable body-size reference — must not
        become a new reason frames go missing."""
        assert vo.limb_segment_is_plausible((0.1, 0.1), (0.9, 0.9), torso_h=0.0) is True
