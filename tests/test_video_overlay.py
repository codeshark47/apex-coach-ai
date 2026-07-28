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
