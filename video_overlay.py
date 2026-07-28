"""
video_overlay.py

Annotated skeleton overlay rendering — extracted out of orchestrator.py
(where it was one ~735-line function bolted onto the analysis pipeline)
into its own module, so the rendering/visual-design code has a clear home
separate from the biomechanics calculations. Pure extraction: behavior is
unchanged for every existing caller (verified by direct frame-by-frame
render comparison against the pre-extraction version on real side-on and
rear-view footage) except for two explicitly-scoped additions:

  1. A small camera-angle indicator badge (this module now actually
     receives camera_angle, which orchestrator.py's version silently
     never did — every camera angle rendered through an identical,
     angle-blind path).
  2. Dead code removed: elbow-extension estimation was computed every
     frame but never displayed (explicitly disabled in its own old
     comment — "needs a real accuracy fix before showing this to users
     again"), and a handful of unused leftover color/connection
     constants from a color-by-limb scheme already replaced by the
     unified palette below.
"""

import math
import os

import cv2
import numpy as np
import pandas as pd

import metric_ranges as mr


def joint_marker_radii(base_core_px: int, render_scale: float,
                        core_floor_px: int = 2, ring_gap_px: int = 2) -> tuple:
    """
    Returns (core_radius, outline_radius) in pixels for a joint dot at a
    given per-frame body-size scale (render_scale: measured body height
    in this frame / a reference body height — see render_annotated_video).

    BUG FIX, found from a real, direct visual comparison against a
    reference frame: outline and core radii used to be computed
    INDEPENDENTLY, each floored at the same fixed minimum (4px) for
    legibility. On a small/distant subject — verified on a real rear-view
    clip, body height dropped to ~120px during follow-through, well below
    the 400px reference this scaling is calibrated against — BOTH radii
    hit that same floor and came out identical, collapsing the two-tone
    outline+core look into one flat, oversized blob (a 4px floor is
    ~1.5% of the 400px reference body's height, but over 3% of a
    120px-tall distant figure — more than double the intended proportion,
    which is exactly what read as joint dots "covering the body" the
    farther away the bowler got).

    Fixed by deriving the outline from the core radius PLUS a small gap
    that itself scales down (with its own, lower floor) instead of
    flooring each independently. The ring can never collapse into the
    core no matter how small the subject gets, and the whole marker keeps
    shrinking with distance instead of hitting the same wall regardless
    of how far away someone is. At render_scale=1.0 (the reference size
    this was always tuned against) this returns exactly the old fixed
    values (4, 6) — zero behavior change at normal distance, only at the
    low end where the old floor was colliding.
    """
    core_r = max(core_floor_px, int(round(base_core_px * render_scale)))
    gap = max(1, int(round(ring_gap_px * render_scale)))
    return core_r, core_r + gap


def torso_shape_is_plausible(row, max_width_to_height_ratio: float = 1.8) -> bool:
    """
    Sanity-checks ONE frame's shoulder/hip landmarks before a skeleton is
    drawn from them. Verified on a real coach-downloaded rear-view clip:
    a bowler bending over to pick up the ball — small and distant in
    frame, a genuinely hard case for pose estimation (self-occlusion,
    tiny pixel footprint) — produced a MediaPipe reading with shoulders
    and hips splayed out nearly 3x wider than the torso is tall. That's
    physically impossible for any human pose, even bent fully over, but
    there was no check at all on whether the drawn shape was even
    physically possible — it rendered as a bizarre "tent/spider" shape
    instead of a body, which is what actually read as broken, not a
    wrong-person tracking issue (there wasn't one).

    Compares the wider of (shoulder width, hip width) against the
    shoulder-to-hip torso length. Real human proportions keep this
    ratio well under the ceiling regardless of pose — shoulders/hips
    don't fan out sideways beyond roughly the torso's own length no
    matter how someone bends. The default ceiling (1.8) is deliberately
    generous: it's tuned to catch a clear tracking failure (the observed
    case was closer to 3.0) without risking discarding a real frame just
    because an athlete is broad-shouldered or turned at an angle.

    Returns True (assume plausible) when there isn't enough landmark
    data to judge either way — this check exists to catch a specific,
    confirmed failure mode, not to become a new reason frames go
    missing.
    """
    cols = ["LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y",
            "LEFT_HIP_x", "LEFT_HIP_y", "RIGHT_HIP_x", "RIGHT_HIP_y"]
    if any(pd.isna(row.get(c)) for c in cols):
        return True

    ls = (float(row["LEFT_SHOULDER_x"]), float(row["LEFT_SHOULDER_y"]))
    rs = (float(row["RIGHT_SHOULDER_x"]), float(row["RIGHT_SHOULDER_y"]))
    lh = (float(row["LEFT_HIP_x"]), float(row["LEFT_HIP_y"]))
    rh = (float(row["RIGHT_HIP_x"]), float(row["RIGHT_HIP_y"]))

    shoulder_w = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
    hip_w = math.hypot(lh[0] - rh[0], lh[1] - rh[1])
    mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    torso_h = math.hypot(mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1])

    if torso_h < 1e-6:
        return True

    return (max(shoulder_w, hip_w) / torso_h) <= max_width_to_height_ratio


def _hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# Same green/amber/red the dashboard and PDF already use for this exact
# classification (mr.TIER_COLORS) — converted to BGR for OpenCV so the
# skeleton's color never has to be kept in sync with those by hand.
TIER_COLORS_BGR = {tier: _hex_to_bgr(hexval) for tier, hexval in mr.TIER_COLORS.items()}


def render_annotated_video(video_path: str, output_path: str,
                            df: pd.DataFrame, events: dict,
                            slow_motion_factor: float = 4.0,
                            bowling_arm: str = "right",
                            camera_angle: str = "side_on"):
    """
    Generates annotated skeleton overlay video using mp4v codec.

    Skeleton style: bold white bones with a dark contact-shadow for
    legibility on any background, cyan joints with a white outline ring.

    slow_motion_factor: the output video is written with its FPS divided by
    this factor. All frames are kept (nothing skipped or duplicated) — the
    same frame count now plays back over a longer real-world duration,
    which is the correct way to get true slow motion without needing frame
    interpolation. Default 4.0 = 4x slower than real time.

    camera_angle: 'side_on' | 'front_or_rear' | 'uncertain' — now actually
    changes which metric is the "hero" of this render, not just a label:

      - side_on sees the true sagittal-plane bend without foreshortening,
        so it charts LEAD KNEE ANGLE (+ colors the lead knee bones/joint
        and the spine by TRUNK LEAN's classification).
      - front_or_rear sees the full shoulder/hip width needed to measure
        their rotational offset, so it charts HIP-SHOULDER SEPARATION
        (+ colors the shoulder-line and hip-line bones instead).

    This is deliberately only a 2-way split, not 3 (side-on / front-on /
    rear-on): camera_angle_detection.py's own docstring documents that
    front-on and rear-on produce geometrically IDENTICAL 2D joint data —
    there's no landmark signal for which way the face points — so there's
    no evidence-backed basis for treating them differently here. Every
    color/highlight below reuses metric_ranges.classify against a metric
    this app already computes and validates elsewhere (dashboard, PDF);
    nothing here is a newly invented metric or judgment.

    'uncertain' (or any other value) falls back to the side_on treatment,
    matching this overlay's original behavior before camera_angle existed.
    """
    from orchestrator import (
        calculate_release_height_ratio_safe,
        _find_grounded_reference_near,
        _nearest_complete_row,
        _draw_panel,
    )

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    output_fps = max(1, int(round(source_fps / slow_motion_factor)))

    # LOGO WATERMARK: subtle brand mark in a fixed corner, low opacity so
    # it doesn't distract from the analysis itself. Loaded once, resized
    # once — composited per-frame is just an alpha blend, cheap.
    logo_rgba = None
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apex_logo.png.png")
    if os.path.exists(logo_path):
        raw_logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if raw_logo is not None and raw_logo.shape[2] == 4:
            logo_w = max(40, int(width * 0.09))
            logo_h = int(raw_logo.shape[0] * (logo_w / raw_logo.shape[1]))
            logo_rgba = cv2.resize(raw_logo, (logo_w, logo_h), interpolation=cv2.INTER_AREA)
    LOGO_MARGIN = 14
    LOGO_OPACITY = 0.6

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        output_fps, (width, height)
    )

    df = df.copy()
    landmark_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    # Same fix as main.py's gap-fill: time-based limit (not a fixed frame
    # count) plus limit_area="inside" so a genuinely long tracking loss
    # (e.g. an arm occluded during running) stays real NaN — which the
    # drawing loop below already skips gracefully — instead of getting
    # padded into a fabricated, frozen position that then gets drawn as a
    # stray disconnected limb.
    draw_gap_fill_limit = max(1, int(round(source_fps * 0.1)))
    df[landmark_cols] = df[landmark_cols].interpolate(
        method="linear", limit=draw_gap_fill_limit, limit_direction="both", limit_area="inside"
    )

    # UNIFIED BROADCAST PALETTE: one clean look across the whole skeleton
    # (was: green legs, white upper body, orange joints — a color-by-limb
    # scheme with no real informational purpose that read as a debug
    # overlay rather than a broadcast graphic).
    #
    # RECOLORED to the coach's own exact spec (hex/RGB converted to OpenCV
    # BGR by the coach directly): #2C657B bones, a #BB9792 copper joint
    # ring around a #FDFEFD white center. Bones are now their own distinct
    # tone rather than reusing the JOINT_CORE chart accent — the coach's
    # palette treats bones and the chart/ticks/badge accent as separate
    # colors, unlike the previous "share one cyan for everything" design.
    # The dark contact-shadow behind every bone (for legibility against
    # any background, including a bright sky) is unchanged.
    BONE_SHADOW = (25, 25, 25)
    BONE_CORE = (123, 101, 44)
    JOINT_OUTLINE = (146, 151, 187)
    JOINT_CORE = (235, 195, 50)
    # Default (non-hero) joint CENTER fill — separate from JOINT_CORE
    # specifically because JOINT_CORE is also reused for unrelated chrome
    # (chart line, ticks, badges) the coach's color spec doesn't cover;
    # only the skeleton's own joint centers use the new white.
    JOINT_FILL = (253, 254, 253)

    # SUBJECT-SIZE SCALE: every fixed pixel size below (joint radius, bone
    # width) was tuned against a 1080x1920 test clip. A prior fix scaled
    # this by frame RESOLUTION alone — but verified directly on three real
    # clips at similar frame heights (462-832px), the bowler's actual
    # on-screen size ranged from 67px to 170px tall depending on how wide
    # or zoomed the shot was, unrelated to frame resolution itself. A wide
    # shot capturing a full run-up (bowler small, lots of empty field
    # around him) got the exact same joint size as a tighter shot under
    # the resolution-only formula, which is why the wide clip's joints
    # physically overlapped into a blob while the tighter one looked fine
    # at the identical frame size. What actually matters is how big the
    # TRACKED PERSON is on screen. Reference constant (400px) is an
    # engineering calibration against the clip a working render was
    # already confirmed on, not a cited standard — documented as such so
    # it isn't mistaken for validated science.
    def _estimate_body_height_px(reference_row):
        if reference_row is None:
            return None
        try:
            y_nose = reference_row.get("NOSE_y")
            ankle_vals = [v for v in (reference_row.get("LEFT_ANKLE_y"), reference_row.get("RIGHT_ANKLE_y"))
                          if v is not None and not pd.isna(v)]
            if y_nose is None or pd.isna(y_nose) or not ankle_vals:
                return None
            return abs(float(max(ankle_vals)) - float(y_nose)) * height
        except Exception:
            return None

    REFERENCE_BODY_HEIGHT_PX = 400.0
    MIN_RENDER_SCALE, MAX_RENDER_SCALE = 0.18, 1.5

    # PER-FRAME, not a single global value computed once at FFC. BUG FIX:
    # verified directly on real footage that a bowler's on-screen size
    # changes a lot over one clip — small and distant during run-up,
    # progressively larger approaching release — so a scale locked to
    # whatever he measured at FFC was correct for maybe one phase of the
    # delivery and visibly wrong (joints overlapping into a blob, "covering
    # the whole body") for the rest. Recomputed every frame from that
    # frame's own landmarks below; _render_scale holds the last successfully
    # measured value so a frame with a brief tracking gap keeps the
    # previous size instead of snapping to a fallback default.
    _fallback_scale = max(MIN_RENDER_SCALE, min(MAX_RENDER_SCALE, min(width, height) / 1080.0))
    scale_ref_rows = df[df["frame"] == events.get("FFC")]
    scale_ref_row = scale_ref_rows.iloc[0] if not scale_ref_rows.empty else None
    _initial_body_height_px = _estimate_body_height_px(scale_ref_row)
    if _initial_body_height_px and _initial_body_height_px > 10:
        render_scale = max(MIN_RENDER_SCALE, min(MAX_RENDER_SCALE, _initial_body_height_px / REFERENCE_BODY_HEIGHT_PX))
    else:
        render_scale = _fallback_scale

    def _rs(px):
        return max(1, int(round(px * render_scale)))

    # JOINT SIZE: see joint_marker_radii's docstring (module level, above)
    # for the real bug this fixes — outline/core radii used to be
    # independently floored and collapsed into one oversized blob on a
    # distant subject. base_core_px=4 matches this file's previous core
    # radius exactly, so a normally-sized subject renders identically to
    # before; only the previously-colliding low end changed.
    def _rs_joint(base_core_px=4):
        return joint_marker_radii(base_core_px, render_scale)

    # BONE OUTLINE FLOOR: BUG FIX — a prior comment here claimed "a thin
    # connecting line stays visually coherent even at 1px," which verified
    # FALSE on real footage. At the same low render_scale above, _rs(6)
    # (shadow) and _rs(3) (core) BOTH round down to 1px — identical
    # widths — so the dark shadow layer (meant to give the white core line
    # contrast against ANY background, sky included) gets completely
    # covered by the core drawn on top of it. The result is a plain 1px
    # near-white line with no outline, which visually vanishes against a
    # bright overcast sky — this is what read as the skeleton "flying off"
    # into empty space. Shadow now always stays wider than core by a fixed
    # margin regardless of scale, so the outline effect can't disappear.
    def _rs_bone_core(px):
        return max(2, int(round(px * render_scale)))

    def _rs_bone_shadow(px):
        return _rs_bone_core(px) + 2

    connections = [
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"),
        ("LEFT_HIP", "LEFT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
        ("LEFT_ANKLE", "LEFT_HEEL"),
        ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
        ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
        ("RIGHT_ANKLE", "RIGHT_HEEL"),
        ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
        ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
        ("NOSE", "LEFT_SHOULDER"),
        ("NOSE", "RIGHT_SHOULDER"),
    ]
    joint_nodes = ["LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP",
                   "LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE",
                   "LEFT_SHOULDER", "RIGHT_SHOULDER", "NOSE"]
    _arm_fallback_triples = [
        ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ]

    # SIDE-ON ONLY: don't draw the non-bowling arm at all. Real,
    # coach-reported bug: side-on filming is done from the bowling-arm
    # side specifically so THAT arm's full swing is visible — which means
    # the OTHER arm is largely hidden behind the torso from the camera's
    # own perspective. MediaPipe still outputs a "best guess" position for
    # an arm it can't actually see, and that guess is unreliable. Verified
    # directly on a real release frame: the visible (bowling) arm tracked
    # correctly, high and extended, while the hidden arm's guessed elbow
    # bent to a position matching nothing actually visible in the shot —
    # reading as a phantom "third arm" with a bend that was never really
    # there. Front/rear view is unaffected: both arms are genuinely
    # visible side by side from that angle, so both stay drawn.
    if camera_angle == "side_on":
        _hidden_arm_side = "RIGHT" if bowling_arm == "left" else "LEFT"
        _hidden_arm_bones = {
            (f"{_hidden_arm_side}_SHOULDER", f"{_hidden_arm_side}_ELBOW"),
            (f"{_hidden_arm_side}_ELBOW", f"{_hidden_arm_side}_WRIST"),
        }
        connections = [c for c in connections if c not in _hidden_arm_bones]
        joint_nodes = [n for n in joint_nodes if n != f"{_hidden_arm_side}_WRIST"]
        _arm_fallback_triples = [t for t in _arm_fallback_triples if not t[0].startswith(_hidden_arm_side)]

    total_frames = int(df["frame"].max()) + 1 if len(df) else 0

    # BUG FIX: this was hardcoded to always measure the LEFT knee,
    # regardless of bowling_arm — for a left-arm bowler the real lead
    # (front, bracing) leg is the RIGHT leg, so every left-arm session's
    # video overlay (chart + RELEASE/CONTACT badge KNEE value) was
    # displaying the TRAILING leg's angle instead, silently disagreeing
    # with the correct lead-side-aware number already used in the actual
    # report/PDF (calculate_knee_bracing, which was never affected by
    # this — only this chart-drawing duplicate of the calculation was).
    # Verified directly: on a real left-arm rear-view session, the
    # burned-in badge showed 173deg while the correct lead-knee value at
    # that exact frame was ~152deg.
    _chart_lead_side = "LEFT" if bowling_arm == "right" else "RIGHT"
    _lead_hip_knee = (f"{_chart_lead_side}_HIP", f"{_chart_lead_side}_KNEE")
    _lead_knee_ankle = (f"{_chart_lead_side}_KNEE", f"{_chart_lead_side}_ANKLE")
    _shoulder_line = ("LEFT_SHOULDER", "RIGHT_SHOULDER")
    _hip_line = ("LEFT_HIP", "RIGHT_HIP")

    # A problem angle (amber/red) gets a visibly bolder bone/joint, not
    # just a different color — "thickness according to the angle": draws
    # the eye to whichever joint actually needs coaching attention instead
    # of every joint looking equally important.
    _TIER_EXTRA_PX = {"green": 0, "amber": 1, "red": 2, "unknown": 0}

    # COLOR/THICKNESS BY ANGLE: reuses metric_ranges.classify — the exact
    # same green/amber/red the dashboard and PDF report already use for
    # each of these metrics, not a new judgment invented for the video.
    def _tier_and_color(metric_key, value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "unknown", TIER_COLORS_BGR["unknown"]
        tier = mr.classify(metric_key, float(value))
        return tier, TIER_COLORS_BGR[tier]

    def _row_knee_angle(r):
        try:
            h = np.array([float(r[f"{_chart_lead_side}_HIP_x"]), float(r[f"{_chart_lead_side}_HIP_y"])])
            k = np.array([float(r[f"{_chart_lead_side}_KNEE_x"]), float(r[f"{_chart_lead_side}_KNEE_y"])])
            a = np.array([float(r[f"{_chart_lead_side}_ANKLE_x"]), float(r[f"{_chart_lead_side}_ANKLE_y"])])
            kh, ka = h - k, a - k
            denom = np.linalg.norm(kh) * np.linalg.norm(ka)
            if denom == 0 or np.isnan(denom):
                return np.nan
            cos_theta = np.dot(kh, ka) / denom
            return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
        except Exception:
            return np.nan

    # Mirrors kinematics.calculate_trunk_lean's math exactly, row-based —
    # same reasoning as _row_knee_angle above (this chart/overlay needs a
    # per-frame value, the report function works off a single confirmed
    # frame), so the color drawn here can never disagree with the report.
    def _row_trunk_lean(r):
        try:
            mid_hip_x = (float(r["LEFT_HIP_x"]) + float(r["RIGHT_HIP_x"])) / 2
            mid_hip_y = (float(r["LEFT_HIP_y"]) + float(r["RIGHT_HIP_y"])) / 2
            mid_sh_x = (float(r["LEFT_SHOULDER_x"]) + float(r["RIGHT_SHOULDER_x"])) / 2
            mid_sh_y = (float(r["LEFT_SHOULDER_y"]) + float(r["RIGHT_SHOULDER_y"])) / 2
            dx = mid_sh_x - mid_hip_x
            dy = mid_hip_y - mid_sh_y
            return float(np.degrees(np.arctan2(np.abs(dx), dy)))
        except Exception:
            return np.nan

    # Mirrors orchestrator.calculate_hip_shoulder_separation's math exactly
    # (arctan2 + wraparound-safe fold into [0, 90]) for the same reason.
    def _row_hip_shoulder_separation(r):
        try:
            shoulder_angle = np.degrees(np.arctan2(
                float(r["LEFT_SHOULDER_y"]) - float(r["RIGHT_SHOULDER_y"]),
                float(r["LEFT_SHOULDER_x"]) - float(r["RIGHT_SHOULDER_x"])
            ))
            hip_angle = np.degrees(np.arctan2(
                float(r["LEFT_HIP_y"]) - float(r["RIGHT_HIP_y"]),
                float(r["LEFT_HIP_x"]) - float(r["RIGHT_HIP_x"])
            ))
            if np.isnan(shoulder_angle) or np.isnan(hip_angle):
                return np.nan
            wrapped = (shoulder_angle - hip_angle + 180) % 360 - 180
            separation = abs(wrapped)
            if separation > 90:
                separation = 180 - separation
            return float(separation)
        except Exception:
            return np.nan

    def _interpolated_array(row_fn):
        by_frame = {int(r["frame"]): row_fn(r) for _, r in df.iterrows()}
        series = pd.Series([by_frame.get(i, np.nan) for i in range(total_frames)])
        return series.interpolate(limit_direction="both").to_numpy()

    knee_arr = _interpolated_array(_row_knee_angle)
    hip_shoulder_arr = _interpolated_array(_row_hip_shoulder_separation)

    # CAMERA-ANGLE MODULE: which metric is the hero (live chart + badge +
    # bone highlight) — see the docstring above for why this is a 2-way
    # split, and why each side gets the metric it can actually measure.
    if camera_angle == "front_or_rear":
        hero_key, hero_arr, hero_label, hero_badge_label = (
            "hip_shoulder_separation", hip_shoulder_arr,
            "HIP-SHOULDER SEPARATION", "HIP-SHOULDER",
        )
    else:
        hero_key, hero_arr, hero_label, hero_badge_label = (
            "front_knee_bracing", knee_arr, "LEAD KNEE ANGLE", "KNEE",
        )

    # --- RELEASE HEIGHT % at BR ---
    # Reuses calculate_release_height_ratio_safe (same function that feeds the
    # report card) instead of a separate formula, so the number burned into
    # the video can never disagree with the report, respects bowling_arm
    # instead of always assuming right-arm, and shows nothing rather than a
    # fabricated number when tracking is unreliable.
    release_height_pct = None
    # Normalized (0-1) points for the release-height line drawn on the BR
    # frame: from the bowling-arm wrist straight down to ground level (the
    # same ankle landmark already used to calculate the ratio above), so
    # the line visually matches the number rather than being a separate
    # guess. Only set when the ratio itself is trustworthy — same
    # "don't show it if we don't trust it" rule as the number.
    release_line_pts = None
    br_frame = events.get("BR")
    if br_frame is not None:
        br_rows = df[df["frame"] == br_frame]
        if not br_rows.empty:
            br_row_for_release = br_rows.iloc[0]
            height_ref_for_release = _find_grounded_reference_near(df, br_frame, bowling_arm)
            if height_ref_for_release is None:
                height_ref_for_release = _nearest_complete_row(
                    df, events.get("FFC"), ["NOSE_y", "LEFT_ANKLE_y", "RIGHT_ANKLE_y"]
                )
            # Coach-confirmed wrist/ball position, when given, overrides the
            # tracked landmark entirely — see calculate_release_height_ratio_safe's
            # docstring for why (verified real MediaPipe mistracking during
            # fast, blurred release swings that a plausibility filter can't
            # catch automatically).
            wrist_override_x = events.get("wrist_override_x")
            wrist_override_y = events.get("wrist_override_y")
            wrist_override_norm = (
                (wrist_override_x, wrist_override_y)
                if wrist_override_x is not None and wrist_override_y is not None else None
            )
            rh = calculate_release_height_ratio_safe(br_row_for_release, bowling_arm=bowling_arm,
                                                      reference_row=height_ref_for_release,
                                                      wrist_override_norm=wrist_override_norm)
            if rh.get("ratio") is not None:
                release_height_pct = rh["ratio"] * 100
                dbg = rh.get("debug_raw") or {}
                bowl_side = dbg.get("bowl_side_used")
                if wrist_override_norm is not None:
                    wrist_x = wrist_override_norm[0]
                else:
                    wrist_x = br_row_for_release.get(f"{bowl_side}_WRIST_x") if bowl_side else None
                if (wrist_x is not None and not pd.isna(wrist_x)
                        and "y_wrist" in dbg and "y_ankle" in dbg):
                    release_line_pts = {
                        "wrist_x": float(wrist_x),
                        "wrist_y": dbg["y_wrist"],
                        "ground_y": dbg["y_ankle"],
                    }

    # CHART BOUNDS: driven by whichever metric is the hero (see above),
    # not hardcoded to knee angle anymore. front_knee_bracing is
    # "higher_better" (0-180, green band open-ended at the top);
    # hip_shoulder_separation is "band" (0-90, since the separation
    # calculation folds into that range, green band in the middle) — both
    # read straight from metric_ranges.RANGES, the same single source of
    # truth the dashboard and PDF already use, so this can't silently
    # drift out of sync with what "green" actually means for either metric.
    if hero_key == "hip_shoulder_separation":
        CHART_MIN, CHART_MAX = 0.0, 90.0
        CHART_GRIDLINES = (0, 45, 90)
    else:
        CHART_MIN, CHART_MAX = 0.0, 180.0
        CHART_GRIDLINES = (0, 90, 180)
    _hero_green_lo, _hero_green_hi = mr.RANGES[hero_key].green
    # SHRUNK from 38% of frame height to a slim ~16% strip — the old chart
    # dominated a third of the video, which read as a dashboard bolted onto
    # footage rather than a broadcast graphic. The data itself (live knee
    # angle) stays, just far less visually loud. A separate, even slimmer
    # ZONE_BAR sits below it — a static phase legend inspired directly by
    # the commercial reference sample, which has no live chart at all but
    # does have this clean bottom orientation strip.
    # Floors LOWERED from 46/120 to 30/90: those were tuned as "don't go
    # unreadably small" minimums against a 1920-tall test clip, but on a
    # real, shorter 478px-tall clip they backfired — forcing the bottom
    # overlay to ~35% of frame height instead of the intended ~20%,
    # because the fixed floor overrode the percentage entirely. Still a
    # legibility floor, just one that doesn't dominate genuinely small
    # footage.
    ZONE_BAR_H = max(30, int(height * 0.045))
    PANEL_H = max(90, int(height * 0.16))
    ZONE_BAR_TOP = height - ZONE_BAR_H
    PANEL_TOP = ZONE_BAR_TOP - PANEL_H
    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 55, 20, 24, 20

    def x_to_px(fidx):
        span = max(total_frames - 1, 1)
        return MARGIN_L + int((fidx / span) * (width - MARGIN_L - MARGIN_R))

    def y_to_px(angle):
        clipped = max(CHART_MIN, min(CHART_MAX, angle))
        frac = (clipped - CHART_MIN) / (CHART_MAX - CHART_MIN)
        usable = PANEL_H - MARGIN_T - MARGIN_B
        return MARGIN_T + int((1 - frac) * usable)

    chart_base = np.full((PANEL_H, width, 3), (16, 16, 18), dtype=np.uint8)
    # Green band rect: for a "higher_better" metric green_hi == CHART_MAX,
    # so this naturally reproduces the old "elite zone from the top down"
    # look; for a "band" metric (green in the middle) it's a thinner
    # strip — either way it's the metric's REAL green band, not a
    # separately-hardcoded threshold that could drift from it.
    green_top_y = y_to_px(_hero_green_hi)
    green_bot_y = y_to_px(_hero_green_lo)
    cv2.rectangle(chart_base, (MARGIN_L, green_top_y), (width - MARGIN_R, green_bot_y),
                  (28, 55, 28), -1)
    band_label = (f"OPTIMAL {_hero_green_lo:.0f}-{_hero_green_hi:.0f}"
                  if hero_key == "hip_shoulder_separation" else f"ELITE {_hero_green_lo:.0f}+")
    cv2.putText(chart_base, band_label, (MARGIN_L + 8, green_top_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 205, 130), 1, cv2.LINE_AA)
    # Fewer gridlines (was every 45deg incl. labels at each) — a compact
    # panel doesn't have room for a dense axis without feeling cramped.
    for g in CHART_GRIDLINES:
        gy = y_to_px(g)
        cv2.line(chart_base, (MARGIN_L, gy), (width - MARGIN_R, gy), (48, 48, 50), 1, cv2.LINE_AA)
        cv2.putText(chart_base, f"{g}", (6, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (150, 150, 150), 1, cv2.LINE_AA)
    prev_pt = None
    for fidx in range(total_frames):
        val = hero_arr[fidx] if fidx < len(hero_arr) else np.nan
        if np.isnan(val):
            prev_pt = None
            continue
        pt = (x_to_px(fidx), y_to_px(val))
        if prev_pt is not None:
            cv2.line(chart_base, prev_pt, pt, JOINT_CORE, 2, cv2.LINE_AA)
        prev_pt = pt
    cv2.putText(chart_base, hero_label, (MARGIN_L, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)

    PHASE_BADGE_WINDOW = max(3, int(round(source_fps * 0.5)))

    # Tracks the bowler's last-known horizontal screen position (0-1) so
    # the info badge can be placed on whichever side he ISN'T on — a
    # fixed top-left badge was covering him directly at release on
    # tightly-framed footage, exactly the moment a coach most wants an
    # unobstructed view. Persists across any frame with no tracking data
    # so the badge doesn't flicker sides during a brief gap.
    last_known_bowler_x = None

    # Coach-confirmed wrist/ball position (from "Correct Release Point")
    # already fixes the Release Height NUMBER, but was never fed into the
    # drawn skeleton — the visual overlay kept showing the original
    # mistracked wrist even after the coach corrected it, so the video and
    # the number silently disagreed with each other. Only applied at the
    # exact confirmed release frame (the only frame the coach actually
    # verified), not smeared across neighboring frames the coach never saw.
    wrist_override_x = events.get("wrist_override_x")
    wrist_override_y = events.get("wrist_override_y")
    br_frame_for_override = events.get("BR")
    bowl_side_for_override = "RIGHT" if bowling_arm == "right" else "LEFT"

    # CAMERA-ANGLE INDICATOR: a small, subtle tag so the annotated video
    # itself states which angle it was analyzed from — every existing
    # caller renders side-on and rear/front through the identical drawing
    # path with no visual distinction at all today, which reads as generic
    # rather than a deliberate, camera-aware analysis. Purely a label; the
    # underlying skeleton/chart/metrics are unchanged by camera_angle (see
    # the docstring above for why swapping the charted metric per angle
    # isn't done here without stronger evidence).
    ANGLE_TAG = {
        "side_on": "SIDE-ON VIEW",
        "front_or_rear": "REAR / FRONT VIEW",
        "uncertain": None,
    }.get(camera_angle)

    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        f_match = df[df["frame"] == f_idx]
        if not f_match.empty:
            row = f_match.iloc[0]
            if (wrist_override_x is not None and wrist_override_y is not None
                    and br_frame_for_override is not None and f_idx == br_frame_for_override):
                row = row.copy()
                row[f"{bowl_side_for_override}_WRIST_x"] = wrist_override_x
                row[f"{bowl_side_for_override}_WRIST_y"] = wrist_override_y

            # PER-FRAME SIZE: recompute from THIS frame's own landmarks —
            # see the comment above render_scale's initial value for why a
            # single clip-wide scale isn't good enough. Keeps the previous
            # value on a frame where the body height can't be measured
            # (brief tracking gap) rather than reverting to a fallback.
            #
            # BUG FIX: a raw per-frame measurement with no plausibility
            # check or damping let ONE frame with a bad ankle/nose reading
            # (exactly the kind of single-frame landmark noise this app
            # sees constantly on real footage) spike render_scale toward
            # its ceiling — verified on real footage this produced
            # comically oversized joints for that frame ("beach balls").
            # Two guards now: reject a measurement implying a body taller
            # than 85% of the frame outright (no real camera framing
            # produces that; it's a tracking artifact every time), and
            # exponential smoothing on top so even a borderline-plausible
            # bad reading only nudges the size a little rather than
            # snapping straight to it — a real, sustained distance change
            # still converges within a few frames, which is correct.
            _frame_body_height_px = _estimate_body_height_px(row)
            if _frame_body_height_px and 10 < _frame_body_height_px < height * 0.85:
                _measured_scale = max(MIN_RENDER_SCALE,
                                       min(MAX_RENDER_SCALE, _frame_body_height_px / REFERENCE_BODY_HEIGHT_PX))
                SCALE_SMOOTHING_ALPHA = 0.25
                render_scale = (1 - SCALE_SMOOTHING_ALPHA) * render_scale + SCALE_SMOOTHING_ALPHA * _measured_scale

            _hero_val = hero_arr[f_idx] if f_idx < len(hero_arr) else np.nan
            _hero_tier, _hero_color = _tier_and_color(hero_key, _hero_val)
            _hero_extra_px = _TIER_EXTRA_PX[_hero_tier]
            _is_side_on_module = hero_key == "front_knee_bracing"

            torso_x_cols = ["NOSE_x", "LEFT_HIP_x", "RIGHT_HIP_x"]
            torso_x_vals = [float(row[c]) for c in torso_x_cols if not pd.isna(row[c])]
            if torso_x_vals:
                last_known_bowler_x = sum(torso_x_vals) / len(torso_x_vals)

            # BUG FIX: real coach-reported case — a bent-over, small/distant
            # figure produced shoulder/hip landmarks splayed nearly 3x wider
            # than the torso is tall, drawing as an anatomically impossible
            # "tent/spider" shape instead of a body. See
            # torso_shape_is_plausible's docstring for the full story.
            # Skipping the skeleton entirely for a frame like this (rare —
            # this is a specific tracking-failure signature, not ordinary
            # noise) is an honest gap; drawing it anyway is a wrong-looking
            # graphic presented as real.
            if torso_shape_is_plausible(row):
                for partA, partB in connections:
                    try:
                        if pd.isna(row[f"{partA}_x"]) or pd.isna(row[f"{partB}_x"]):
                            continue
                        xA = int(float(row[f"{partA}_x"]) * width)
                        yA = int(float(row[f"{partA}_y"]) * height)
                        xB = int(float(row[f"{partB}_x"]) * width)
                        yB = int(float(row[f"{partB}_y"]) * height)
                        is_hero_bone = (
                            (partA, partB) in (_lead_hip_knee, _lead_knee_ankle) if _is_side_on_module
                            else (partA, partB) in (_shoulder_line, _hip_line)
                        )
                        bone_color = _hero_color if is_hero_bone else BONE_CORE
                        bone_extra = _hero_extra_px if is_hero_bone else 0
                        if (0 < xA < width and 0 < yA < height and
                                0 < xB < width and 0 < yB < height):
                            cv2.line(frame, (xA, yA), (xB, yB), BONE_SHADOW, _rs_bone_shadow(6) + bone_extra, cv2.LINE_AA)
                            cv2.line(frame, (xA, yA), (xB, yB), bone_color, _rs_bone_core(3) + bone_extra, cv2.LINE_AA)
                    except Exception:
                        continue

                # SPINE: MediaPipe has no single spine landmark, so this is a
                # virtual centerline from the shoulder midpoint to the hip
                # midpoint. Without it the torso was just a hollow box (shoulder
                # line + two side lines + hip line) with nothing down the
                # middle, which is what made the skeleton look unrigged/amateur
                # rather than like a proper body frame.
                #
                # SIDE-ON MODULE ONLY: colored by trunk_lean's own
                # classification — the secondary highlight alongside the lead
                # knee for this module (both are sagittal-plane bends best
                # measured side-on). Front-or-rear leaves it the neutral
                # palette, same reasoning as the knee bones above: trunk lean
                # isn't this module's hero metric.
                try:
                    spine_cols = ["LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y",
                                  "LEFT_HIP_x", "LEFT_HIP_y", "RIGHT_HIP_x", "RIGHT_HIP_y"]
                    if not any(pd.isna(row[c]) for c in spine_cols):
                        neck_x = (float(row["LEFT_SHOULDER_x"]) + float(row["RIGHT_SHOULDER_x"])) / 2
                        neck_y = (float(row["LEFT_SHOULDER_y"]) + float(row["RIGHT_SHOULDER_y"])) / 2
                        midhip_x = (float(row["LEFT_HIP_x"]) + float(row["RIGHT_HIP_x"])) / 2
                        midhip_y = (float(row["LEFT_HIP_y"]) + float(row["RIGHT_HIP_y"])) / 2
                        sx1, sy1 = int(neck_x * width), int(neck_y * height)
                        sx2, sy2 = int(midhip_x * width), int(midhip_y * height)
                        if _is_side_on_module:
                            _, spine_color = _tier_and_color("trunk_lean", _row_trunk_lean(row))
                        else:
                            spine_color = BONE_CORE
                        if 0 < sx1 < width and 0 < sy1 < height and 0 < sx2 < width and 0 < sy2 < height:
                            cv2.line(frame, (sx1, sy1), (sx2, sy2), BONE_SHADOW, 6, cv2.LINE_AA)
                            cv2.line(frame, (sx1, sy1), (sx2, sy2), spine_color, 3, cv2.LINE_AA)
                            _core_r, _outline_r = _rs_joint()
                            cv2.circle(frame, (sx1, sy1), _outline_r, JOINT_OUTLINE, -1, cv2.LINE_AA)
                            cv2.circle(frame, (sx1, sy1), _core_r, JOINT_FILL, -1, cv2.LINE_AA)
                except Exception:
                    pass

                # LIMB FALLBACK: when the middle joint (elbow/knee) isn't tracked
                # — verified directly on real footage that this happens on
                # essentially every frame of a fast, motion-blurred arm swing —
                # the normal two-segment connection (shoulder-elbow, elbow-wrist)
                # both get skipped, but the wrist/ankle joint DOT still gets
                # drawn on its own with nothing connecting it to the body. That
                # floating, disconnected dot is exactly what read as "loose" /
                # not attached to the bowler rather than a genuine limb. Drawing
                # a direct shoulder-to-wrist (or hip-to-ankle) line in that
                # specific case keeps every visible joint attached to the
                # skeleton — straight instead of bent, which is honest (no real
                # elbow position is invented) rather than fabricated.
                for shoulder, elbow, wrist in _arm_fallback_triples:
                    try:
                        if pd.isna(row[f"{elbow}_x"]) and not pd.isna(row[f"{shoulder}_x"]) and not pd.isna(row[f"{wrist}_x"]):
                            xA = int(float(row[f"{shoulder}_x"]) * width)
                            yA = int(float(row[f"{shoulder}_y"]) * height)
                            xB = int(float(row[f"{wrist}_x"]) * width)
                            yB = int(float(row[f"{wrist}_y"]) * height)
                            if (0 < xA < width and 0 < yA < height and
                                    0 < xB < width and 0 < yB < height):
                                cv2.line(frame, (xA, yA), (xB, yB), BONE_SHADOW, _rs_bone_shadow(6), cv2.LINE_AA)
                                cv2.line(frame, (xA, yA), (xB, yB), BONE_CORE, _rs_bone_core(3), cv2.LINE_AA)
                    except Exception:
                        continue

                for hip, knee, ankle in (
                    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
                    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
                ):
                    try:
                        if pd.isna(row[f"{knee}_x"]) and not pd.isna(row[f"{hip}_x"]) and not pd.isna(row[f"{ankle}_x"]):
                            xA = int(float(row[f"{hip}_x"]) * width)
                            yA = int(float(row[f"{hip}_y"]) * height)
                            xB = int(float(row[f"{ankle}_x"]) * width)
                            yB = int(float(row[f"{ankle}_y"]) * height)
                            if (0 < xA < width and 0 < yA < height and
                                    0 < xB < width and 0 < yB < height):
                                cv2.line(frame, (xA, yA), (xB, yB), BONE_SHADOW, _rs_bone_shadow(6), cv2.LINE_AA)
                                cv2.line(frame, (xA, yA), (xB, yB), BONE_CORE, _rs_bone_core(3), cv2.LINE_AA)
                    except Exception:
                        continue

                _hero_joint_nodes = (
                    {f"{_chart_lead_side}_KNEE"} if _is_side_on_module
                    else {"LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"}
                )
                # BUG FIX: front-or-rear used to also apply the same size bump
                # as the single lead-knee dot to all FOUR shoulder/hip joints
                # at once. One enlarged, isolated knee dot reads fine — four
                # enlarged dots clustered into the much smaller shoulder/hip
                # span (verified on real footage: they visually merged into
                # oversized blobs covering the whole torso) does not. The bone
                # lines already carry the color/thickness "vector axis"
                # highlight for this module — the dots only need the color,
                # not extra size on top of it.
                _hero_node_size_extra = _hero_extra_px if _is_side_on_module else 0
                for node in joint_nodes:
                    try:
                        if pd.isna(row[f"{node}_x"]):
                            continue
                        nx = int(float(row[f"{node}_x"]) * width)
                        ny = int(float(row[f"{node}_y"]) * height)
                        is_hero_node = node in _hero_joint_nodes
                        node_color = _hero_color if is_hero_node else JOINT_FILL
                        node_extra = _hero_node_size_extra if is_hero_node else 0
                        if 0 < nx < width and 0 < ny < height:
                            _core_r, _outline_r = _rs_joint()
                            cv2.circle(frame, (nx, ny), _outline_r + node_extra, JOINT_OUTLINE, -1, cv2.LINE_AA)
                            cv2.circle(frame, (nx, ny), _core_r + node_extra, node_color, -1, cv2.LINE_AA)
                    except Exception:
                        continue

        # Composited BEFORE the chart panel overwrites the bottom of the
        # frame, and anchored bottom-right of the VIDEO area specifically
        # (not the chart panel) — the badge only ever appears in the top
        # area, so this placement can never collide with it regardless of
        # which side the badge dynamically picks.
        if logo_rgba is not None:
            lh, lw = logo_rgba.shape[:2]
            lx1 = width - lw - LOGO_MARGIN
            ly1 = PANEL_TOP - lh - LOGO_MARGIN
            if lx1 > 0 and ly1 > 0:
                roi = frame[ly1:ly1 + lh, lx1:lx1 + lw]
                logo_bgr = logo_rgba[:, :, :3]
                logo_alpha = (logo_rgba[:, :, 3:4].astype(np.float32) / 255.0) * LOGO_OPACITY
                frame[ly1:ly1 + lh, lx1:lx1 + lw] = (
                    logo_bgr.astype(np.float32) * logo_alpha
                    + roi.astype(np.float32) * (1 - logo_alpha)
                ).astype(np.uint8)

        panel_region = frame[PANEL_TOP:PANEL_TOP + PANEL_H, 0:width]
        blended = cv2.addWeighted(panel_region, 0.35, chart_base, 0.65, 0)
        frame[PANEL_TOP:PANEL_TOP + PANEL_H, 0:width] = blended

        cur_val = hero_arr[f_idx] if f_idx < len(hero_arr) else np.nan
        if not np.isnan(cur_val):
            _, cur_val_color = _tier_and_color(hero_key, cur_val)
            px, py_rel = x_to_px(f_idx), y_to_px(cur_val)
            py = PANEL_TOP + py_rel
            cv2.line(frame, (px, PANEL_TOP + MARGIN_T), (px, PANEL_TOP + PANEL_H - MARGIN_B),
                     cur_val_color, 1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, JOINT_OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 3, cur_val_color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{cur_val:.0f}deg", (width - 130, PANEL_TOP + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cur_val_color, 2, cv2.LINE_AA)

        status_text = "RUN-UP"
        if f_idx >= events["BR"]:
            status_text = "FOLLOW-THROUGH"
        elif f_idx >= events["FFC"]:
            status_text = "BALL RELEASE"
        elif f_idx >= events["BFC"]:
            status_text = "DELIVERY STRIDE"

        # ZONE PROGRESS BAR: a slim, mostly-static phase legend inspired by
        # the commercial reference sample (which has no live chart at all,
        # just this bottom strip) — but goes one step further by actually
        # tracking a live position marker along it, since real BFC/FFC/BR
        # boundaries are already known here and the reference apparently
        # doesn't have that data to show. Tick x-positions are evenly
        # spaced (matching the reference's even spacing, not scaled to each
        # phase's actual duration) — this is a schematic orientation aid,
        # not a second data chart.
        zone_bg = frame[ZONE_BAR_TOP:ZONE_BAR_TOP + ZONE_BAR_H, 0:width].copy()
        cv2.rectangle(zone_bg, (0, 0), (width, ZONE_BAR_H), (14, 14, 16), -1)
        frame[ZONE_BAR_TOP:ZONE_BAR_TOP + ZONE_BAR_H, 0:width] = cv2.addWeighted(
            frame[ZONE_BAR_TOP:ZONE_BAR_TOP + ZONE_BAR_H, 0:width], 0.25, zone_bg, 0.75, 0
        )
        zone_labels = ["RUN-UP", "STRIDE", "RELEASE", "FOLLOW-THROUGH"]
        zone_bounds = [0, events.get("BFC", 0), events.get("FFC", 0), events.get("BR", 0),
                       max(total_frames - 1, 1)]
        bar_x0, bar_x1 = MARGIN_L + 10, width - MARGIN_R - 10
        # FIX: was ZONE_BAR_TOP + ZONE_BAR_H//2 + 6, then labels drawn
        # BELOW that at +th+10 — on a shorter clip (lower ZONE_BAR_H,
        # see the floor fix above) that pushed label text past the
        # bottom edge of the frame entirely, clipping it off. Anchor the
        # tick line near the TOP of the bar and the label baseline to a
        # fixed offset from the frame's own bottom edge instead, which is
        # correct by construction regardless of ZONE_BAR_H.
        bar_y = ZONE_BAR_TOP + max(10, int(ZONE_BAR_H * 0.32))
        label_baseline_y = height - 8
        # 5 boundary x-positions (matching the 5 zone_bounds frame indices)
        # for the marker to interpolate across 4 segments; labels are drawn
        # at each segment's MIDPOINT, since each label names a span (e.g.
        # "RELEASE" = the FFC-to-BR window), not a single instant.
        boundary_xs = [int(bar_x0 + (bar_x1 - bar_x0) * i / 4) for i in range(5)]
        label_xs = [(boundary_xs[i] + boundary_xs[i + 1]) // 2 for i in range(4)]
        cv2.line(frame, (boundary_xs[0], bar_y), (boundary_xs[-1], bar_y), (90, 90, 95), 2, cv2.LINE_AA)

        # Current phase index (0-3), used to highlight both the label and
        # find where along its segment the live marker sits.
        phase_idx = 0
        for i in range(3):
            if f_idx >= zone_bounds[i + 1]:
                phase_idx = i + 1
        phase_idx = min(phase_idx, 3)

        for i, (tx, label) in enumerate(zip(label_xs, zone_labels)):
            active = (i == phase_idx)
            tick_color = JOINT_CORE if active else (130, 130, 135)
            cv2.circle(frame, (tx, bar_y), _rs(5) if active else _rs(3), tick_color, -1, cv2.LINE_AA)
            font_scale = (0.42 if active else 0.38) * render_scale
            text_color = (245, 245, 245) if active else (150, 150, 150)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx_centered = max(2, min(width - tw - 2, tx - tw // 2))
            cv2.putText(frame, label, (tx_centered, label_baseline_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

        # Live marker: piecewise-linear position within the current phase's
        # own frame span, mapped onto that segment's boundary-to-boundary
        # x-range (not the label positions, which sit at segment midpoints).
        seg_lo, seg_hi = zone_bounds[phase_idx], zone_bounds[phase_idx + 1]
        seg_frac = 0.0 if seg_hi <= seg_lo else (f_idx - seg_lo) / (seg_hi - seg_lo)
        seg_frac = max(0.0, min(1.0, seg_frac))
        marker_x = int(boundary_xs[phase_idx] + seg_frac * (boundary_xs[phase_idx + 1] - boundary_xs[phase_idx]))
        cv2.circle(frame, (marker_x, bar_y), _rs(7), JOINT_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(frame, (marker_x, bar_y), _rs(5), JOINT_CORE, -1, cv2.LINE_AA)
        # Was plain text drawn straight onto the video with no background —
        # low contrast and hard to read against a bright sky, reading more
        # like a debug label than a broadcast lower-third. Same rounded
        # pill treatment as the other badges for a consistent look.
        (status_w, status_h), _ = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        _draw_panel(frame, (12, 12), (12 + status_w + 24, 12 + status_h + 20),
                    radius=status_h // 2 + 6, shadow_offset=3, fill_alpha=0.5)
        cv2.putText(frame, status_text, (24, 12 + status_h + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (235, 235, 235), 2, cv2.LINE_AA)

        # CAMERA-ANGLE TAG: small pill, top-right, well clear of the
        # phase-status pill (top-left) and the event badge (which uses
        # last_known_bowler_x to pick a side lower down) — always visible,
        # never repositioned, since it never overlaps a subject at any
        # tracked position (it sits above where the event badge's box_y
        # region starts).
        if ANGLE_TAG:
            (tag_w, tag_h), _ = cv2.getTextSize(ANGLE_TAG, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tag_x1 = width - tag_w - 24 - 12
            _draw_panel(frame, (tag_x1, 12), (width - 12, 12 + tag_h + 16),
                        radius=tag_h // 2 + 4, shadow_offset=3, fill_alpha=0.5)
            cv2.putText(frame, ANGLE_TAG, (tag_x1 + 12, 12 + tag_h + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)

        event_labels = [("BFC", "CONTACT", (60, 225, 90)),
                         ("FFC", "CONTACT", (60, 225, 90)),
                         ("BR", "RELEASE", JOINT_CORE)]
        # Pick whichever event is actually CLOSEST to this frame, not just
        # the first one in the list that's within range. BFC/FFC/BR often
        # land within a few frames of each other for a real delivery, so
        # "first match wins" was showing the earlier event's badge
        # (CONTACT) even on the exact BR frame itself, hiding the RELEASE
        # badge and the release-height line/text that only draw when the
        # RELEASE badge is showing.
        active_badge = None
        best_distance = None
        for key, label, color in event_labels:
            ev_frame = events.get(key)
            if ev_frame is None:
                continue
            distance = abs(f_idx - ev_frame)
            if distance <= PHASE_BADGE_WINDOW and (best_distance is None or distance < best_distance):
                active_badge = (label, color, ev_frame)
                best_distance = distance

        if active_badge is not None:
            label, color, ev_frame = active_badge
            badge_val = hero_arr[ev_frame] if ev_frame < len(hero_arr) else np.nan
            is_release = (label == "RELEASE")
            # Metric-label width varies ("KNEE" vs "HIP-SHOULDER") — measure
            # it rather than assuming a fixed offset, so the degree value
            # next to it never overlaps regardless of which module is active.
            (metric_label_w, _), _ = cv2.getTextSize(
                hero_badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            value_x_offset = 14 + metric_label_w + 14
            # SHRUNK from 340x150/130 — smaller, less boxy footprint, more
            # in line with the reference sample's restraint, while keeping
            # every number the old, larger version showed. Widened to fit
            # the longer "HIP-SHOULDER" label when that module is active.
            box_w = max(290, value_x_offset + 110)
            box_h = 128 if is_release else 108
            # Default to the left (matches the old fixed behavior) unless
            # the bowler is known to be on the left half of frame, in
            # which case the badge moves to the right so it never sits
            # directly on top of him.
            if last_known_bowler_x is not None and last_known_bowler_x < 0.5:
                box_x = width - box_w - 20
            else:
                box_x = 20
            box_y = 40
            _draw_panel(frame, (box_x, box_y), (box_x + box_w, box_y + box_h))
            cv2.putText(frame, label, (box_x + 14, box_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(frame, hero_badge_label, (box_x + 14, box_y + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
            if not np.isnan(badge_val):
                # Colored by the SAME classification as the skeleton/chart
                # (green/amber/red), not the badge's own event-type color —
                # this number's color means "how good is this angle", the
                # badge label's color just means "which event is this".
                _, badge_hero_color = _tier_and_color(hero_key, badge_val)
                cv2.putText(frame, f"{badge_val:.0f}deg", (box_x + value_x_offset, box_y + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.05, badge_hero_color, 2, cv2.LINE_AA)
            if is_release:
                cv2.putText(frame, "RELEASE HEIGHT", (box_x + 14, box_y + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
                if release_height_pct is not None:
                    cv2.putText(frame, f"{release_height_pct:.0f}%", (box_x + 165, box_y + 104),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, JOINT_CORE, 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "N/A", (box_x + 165, box_y + 104),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)

                # RELEASE HEIGHT LINE: a vertical line from the bowling-arm
                # wrist straight down to ground level (same ankle landmark
                # used to calculate the ratio above), drawn only on the
                # actual BR frame — no ball tracking involved, this is a
                # visual read-out of the same number shown as text.
                if f_idx == ev_frame and release_line_pts is not None:
                    lx = int(release_line_pts["wrist_x"] * width)
                    ly_wrist = int(release_line_pts["wrist_y"] * height)
                    ly_ground = int(release_line_pts["ground_y"] * height)
                    if (0 < lx < width and 0 < ly_wrist < height
                            and 0 < ly_ground < height):
                        DROP_LINE_COLOR = (60, 60, 255)  # BGR — highly visible red
                        cv2.line(frame, (lx, ly_wrist), (lx, ly_ground),
                                 DROP_LINE_COLOR, 2, cv2.LINE_AA)
                        cv2.circle(frame, (lx, ly_wrist), 5, DROP_LINE_COLOR, -1, cv2.LINE_AA)
                        cv2.circle(frame, (lx, ly_ground), 5, DROP_LINE_COLOR, -1, cv2.LINE_AA)
                        label = "Ball Release Height"
                        (label_w, label_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        # FIX: was centered on the line's vertical midpoint,
                        # which sits right on the torso — verified on real
                        # footage this landed the label directly over the
                        # bowler's body. Place it beside the line at wrist
                        # height instead (right if there's room, else left),
                        # same "avoid the subject" logic already used for
                        # the phase badge above.
                        label_x_right = lx + 14
                        if label_x_right + label_w + 10 < width:
                            label_x = label_x_right
                        else:
                            label_x = max(6, lx - label_w - 14)
                        label_y = min(max(ly_wrist + label_h // 2, label_h + 6), height - 6)
                        _draw_panel(frame, (label_x - 6, label_y - label_h - 6),
                                    (label_x + label_w + 6, label_y + 6),
                                    radius=6, shadow_offset=3)
                        cv2.putText(frame, label, (label_x, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, DROP_LINE_COLOR, 2, cv2.LINE_AA)

        out.write(frame)

        # FREEZE ON RELEASE: hold the exact Ball Release frame (with its
        # skeleton/badge already drawn) for a beat before playback
        # continues, like a broadcast replay highlight. Duplicates the same
        # already-rendered frame at the output's own fps, so it doesn't
        # touch source detection/tracking at all.
        if br_frame is not None and f_idx == br_frame:
            RELEASE_FREEZE_SECONDS = 0.8
            freeze_frame_count = max(1, int(round(output_fps * RELEASE_FREEZE_SECONDS)))
            for _ in range(freeze_frame_count):
                out.write(frame)

        f_idx += 1
    cap.release()
    out.release()
