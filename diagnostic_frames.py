"""
diagnostic_frames.py

Freeze-frame diagnostic stills at each delivery/shot's key event moments
(bowling: BFC/FFC/Release; batting: Stance/Backlift/Contact) — skeleton
overlay plus callout boxes with leader lines on whichever metrics are
CRITICAL at that moment. Two consumers: the coach-facing report/PDF
(a still image is easier to read than scrubbing a moving overlay for
5-7 simultaneous flags), and coaching_agent.py, which attaches these
same images to its existing Gemini call so the AI narrative can
reference what's visually shown, not just recite numbers.

Deliberately a SEPARATE file from video_overlay.py/batting_video_overlay.py
— this is a genuinely different rendering mode (one seeked still frame
composited with callouts, no cv2.VideoWriter loop) serving BOTH sports
with one shared layout algorithm, rather than bolting a new concern onto
either sport-specific video-encoding file.

Skeleton bone/joint lists and colors below are duplicated from
batting_video_overlay.py rather than imported — same reasoning that
file's own docstring already gives for duplicating out of video_overlay.py
a second time: this is ~15 lines of small, stable data, and the
established bias in this codebase is duplication over cross-file
coupling for something this size. The plausibility GATES themselves
(torso_shape_is_plausible etc.) are genuinely shared logic, not just
data, and ARE imported directly from video_overlay.py.

SHOW EVERY MAPPED METRIC, TAG ONLY THE DRILL-ELIGIBLE ONES (2026-09-XX,
real bug found from an actual coach test): the first version of this
file only ever drew a callout for a metric that passed
metric_ranges.is_critical_and_eligible() — meaning a delivery where
every metric was either DESCRIPTIVE, RECALIBRATION-PENDING, TRACKING-
UNCERTAIN, or simply not critical rendered a skeleton with NO numbers
on it at all, which is what the coach actually hit and flagged. Every
metric mapped to a given frame is now ALWAYS drawn with its real value
and its real tier color (green/amber/red/descriptive/unknown, straight
from metric_ranges.classify() — never invented here). is_critical_and_
eligible() still runs, but now only decides whether to ALSO print a
"CRITICAL" tag on that same panel — it no longer decides whether the
panel exists. This keeps the freeze-frame and the drill-prescription
logic in agreement about which metric actually warrants a drill,
without making non-qualifying metrics invisible.
"""

import cv2
import pandas as pd

import metric_ranges as mr
import monitoring
from video_overlay import (
    torso_shape_is_plausible,
    body_size_is_plausible,
    implausible_arm_nodes,
    torso_centroid,
    _torso_height,
    TIER_COLORS_BGR,
)

_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"), ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"), ("LEFT_HEEL", "LEFT_FOOT_INDEX"), ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"), ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"), ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
    ("NOSE", "LEFT_SHOULDER"), ("NOSE", "RIGHT_SHOULDER"),
]
_JOINT_NODES = ["LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP",
                "LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE",
                "LEFT_SHOULDER", "RIGHT_SHOULDER", "NOSE"]

BONE_SHADOW = (25, 25, 25)
BONE_CORE = (123, 101, 44)
JOINT_OUTLINE = (146, 151, 187)
JOINT_FILL = (253, 254, 253)

_CALLOUT_BOX_W = 260
_CALLOUT_BOX_H = 64
_CALLOUT_MARGIN_X = 20
_CALLOUT_GAP_Y = 10
_CALLOUT_TOP_Y = 30

# Only these two bowling metrics carry recalibration/tracking-uncertain
# flags at all (confirmed directly against orchestrator.py's own return
# payload and coaching_agent.py's prompt, which only ever brackets these
# two with "[RECALIBRATION PENDING]"/"[TRACKING UNCERTAIN]"). Batting has
# neither concept anywhere (confirmed: zero matches repo-wide) — every
# batting metric naturally reads as False/False via the same .get(..., "")
# fallback, no separate batting-specific branch needed.
_RECALIBRATION_FLAG_KEYS = {"release_height": "recalibration_pending", "head_stability": "recalibration_pending"}
_TRACKING_UNCERTAIN_FLAG_KEYS = {
    "release_height": "release_frame_tracking_uncertain",
    "head_stability": "release_window_tracking_uncertain",
}

_BOWLING_METRIC_FRAMES = {
    "front_knee_bracing": ["ffc"],
    "trunk_lean": ["release"],
    "hip_shoulder_separation": ["ffc"],  # always DESCRIPTIVE (see metric_ranges._ALWAYS_DESCRIPTIVE_METRICS) — never actually flags, kept for completeness
    "release_height": ["release"],
    "head_stability": ["bfc", "release"],  # a whole-window (BFC->BR) metric — shown on both bookends rather than forced onto one arbitrary frame
}

_BATTING_METRIC_FRAMES = {
    "batting_head_movement": ["stance", "contact"],  # also a whole-window metric
    "batting_front_foot_alignment": ["contact"],
    "batting_weight_transfer": ["contact"],
    "batting_downswing_plane": ["backlift", "contact"],
    "batting_top_elbow_angle": ["contact"],
    "batting_front_knee_flexion": ["contact"],
    "batting_xfactor_separation": ["contact"],
}


def _read_frame_bgr(video_path: str, frame_index: int):
    """Seeks directly to frame_index, returns the raw BGR frame or None.
    Plain cv2 (not calibration.py's @st.cache_data-wrapped
    extract_reference_frame) since this runs inside orchestrator.py /
    batting_orchestrator.py, not a Streamlit page script — and returns
    BGR directly, matching every drawing call below (extract_reference_
    frame returns RGB, which would need an extra color-space round trip
    for no benefit here)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _draw_skeleton(frame_bgr, row) -> bool:
    """Draws bones+joints in place, same plausibility gates as
    batting_video_overlay.py. Returns False (draws nothing) when the row
    fails plausibility — callers treat that as "don't fabricate a
    skeleton on bad data," skipping this diagnostic frame entirely
    rather than drawing a misleading one."""
    height, width = frame_bgr.shape[:2]
    torso_h = _torso_height(row)
    estimated_body_height_px = torso_h * 2.2 * height if torso_h > 0 else 0
    if not (torso_h > 0 and torso_shape_is_plausible(row) and body_size_is_plausible(estimated_body_height_px)):
        return False

    implausible_nodes = implausible_arm_nodes(row, torso_h)
    for a, b in _CONNECTIONS:
        xa, ya = row.get(f"{a}_x"), row.get(f"{a}_y")
        xb, yb = row.get(f"{b}_x"), row.get(f"{b}_y")
        if any(pd.isna(v) for v in (xa, ya, xb, yb)):
            continue
        if a in implausible_nodes or b in implausible_nodes:
            continue
        pa = (int(float(xa) * width), int(float(ya) * height))
        pb = (int(float(xb) * width), int(float(yb) * height))
        cv2.line(frame_bgr, pa, pb, BONE_SHADOW, 6, cv2.LINE_AA)
        cv2.line(frame_bgr, pa, pb, BONE_CORE, 3, cv2.LINE_AA)

    for node in _JOINT_NODES:
        if node in implausible_nodes:
            continue
        x, y = row.get(f"{node}_x"), row.get(f"{node}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        p = (int(float(x) * width), int(float(y) * height))
        cv2.circle(frame_bgr, p, 6, JOINT_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, p, 4, JOINT_FILL, -1, cv2.LINE_AA)
    return True


def _landmark_px(row, name, width, height):
    x, y = row.get(f"{name}_x"), row.get(f"{name}_y")
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None
    return (int(float(x) * width), int(float(y) * height))


def _midpoint_px(row, name_a, name_b, width, height):
    pa = _landmark_px(row, name_a, width, height)
    pb = _landmark_px(row, name_b, width, height)
    if pa is None or pb is None:
        return None
    return ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)


def _torso_centroid_px(row, width, height):
    c = torso_centroid(row)
    if c is None:
        return None
    return (int(c[0] * width), int(c[1] * height))


def _bowling_metric_anchor(metric_key, row, width, height, lead_side, bowl_side):
    if metric_key == "front_knee_bracing":
        return _landmark_px(row, f"{lead_side}_KNEE", width, height)
    if metric_key == "trunk_lean":
        return _torso_centroid_px(row, width, height)
    if metric_key == "hip_shoulder_separation":
        return _midpoint_px(row, "LEFT_HIP", "RIGHT_HIP", width, height)
    if metric_key == "release_height":
        return _landmark_px(row, f"{bowl_side}_WRIST", width, height)
    if metric_key == "head_stability":
        return _landmark_px(row, "NOSE", width, height)
    return None


def _batting_metric_anchor(metric_key, row, width, height, front_side, top_hand_side):
    if metric_key == "batting_head_movement":
        return _landmark_px(row, "NOSE", width, height)
    if metric_key == "batting_front_foot_alignment":
        return _landmark_px(row, f"{front_side}_FOOT_INDEX", width, height)
    if metric_key == "batting_weight_transfer":
        return _midpoint_px(row, "LEFT_HIP", "RIGHT_HIP", width, height)
    if metric_key == "batting_downswing_plane":
        return _midpoint_px(row, "LEFT_WRIST", "RIGHT_WRIST", width, height)
    if metric_key == "batting_top_elbow_angle":
        return _landmark_px(row, f"{top_hand_side}_ELBOW", width, height)
    if metric_key == "batting_front_knee_flexion":
        return _landmark_px(row, f"{front_side}_KNEE", width, height)
    if metric_key == "batting_xfactor_separation":
        return _midpoint_px(row, "LEFT_SHOULDER", "RIGHT_SHOULDER", width, height)
    return None


def _draw_panel(frame_bgr, pt1, pt2, **kwargs):
    """Local import — matches the existing precedent of video_overlay.
    render_annotated_video's own local `from orchestrator import ...,
    _draw_panel, ...` rather than importing orchestrator (a much heavier
    module) at this file's top level."""
    from orchestrator import _draw_panel as _op
    _op(frame_bgr, pt1, pt2, **kwargs)


def _fit_font_scale(text, max_width_px, start_scale, min_scale, font=cv2.FONT_HERSHEY_SIMPLEX,
                     thickness=1):
    """Shrinks font scale (down to min_scale) until text fits within
    max_width_px. Found via real testing (2026-09-XX): several metric
    labels reused verbatim from metric_ranges.RANGES (e.g. 'Front Foot
    Alignment (vs. Shot Target)') are long enough to overflow a
    fixed-width callout box and run past the frame edge at a fixed font
    size. Auto-shrinking keeps every label exactly as-reviewed in
    metric_ranges.py (no new, unreviewed abbreviations invented here)
    while guaranteeing it actually fits."""
    scale = start_scale
    while scale > min_scale:
        (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if w <= max_width_px:
            break
        scale -= 0.05
    return max(scale, min_scale)


def _panel_text(metric_key: str, value, tier: str, eligible: bool) -> str:
    """The second line of a callout: the real formatted value, plus a
    suffix reflecting exactly why it does/doesn't count as a drill
    target — never inventing a number metric_ranges didn't already
    compute. mr.format_value() assumes a real numeric value, so a
    missing/unknown reading is handled before ever reaching it (calling
    it on None would raise, e.g. float(None) for a "%"-unit metric)."""
    if value is None or tier == "unknown":
        return "N/A — no reading this delivery"
    formatted = mr.format_value(metric_key, value)
    if eligible:
        return f"{formatted} — CRITICAL"
    if tier == "descriptive":
        return f"{formatted} — DESCRIPTIVE"
    if tier == "red":
        # Genuinely red but excluded from drills (recalibration-pending
        # or tracking-uncertain this delivery) — still real information,
        # just not solid enough to act on yet. See is_critical_and_
        # eligible's own docstring for the two flags this covers.
        return f"{formatted} — CRITICAL (provisional)"
    return f"{formatted} — {tier.upper()}"


def _draw_callout(frame_bgr, anchor_px, title, detail, tier, box_x, box_y, box_w, side):
    """One callout: rounded panel + two lines of text + a leader line
    from the box's inner edge to the anchor point, plus a small circle
    marking the anchor itself — the same 'line + endpoint circle(s)'
    idiom video_overlay.py's existing release-height drop-line already
    uses, generalized from always-vertical-one-fixed-metric to a
    diagonal line to any joint. Color is always the real tier color
    (green/amber/red/descriptive/unknown) — a callout is drawn for
    every mapped metric now, not just critical ones, so the accent color
    is what actually communicates severity at a glance."""
    box_h = _CALLOUT_BOX_H
    _draw_panel(frame_bgr, (box_x, box_y), (box_x + box_w, box_y + box_h))
    accent = TIER_COLORS_BGR.get(tier, TIER_COLORS_BGR["unknown"])
    text_max_w = box_w - 24  # 12px padding each side

    title_scale = _fit_font_scale(title, text_max_w, start_scale=0.5, min_scale=0.32)
    cv2.putText(frame_bgr, title, (box_x + 12, box_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, accent, 1, cv2.LINE_AA)

    detail_scale = _fit_font_scale(detail, text_max_w, start_scale=0.55, min_scale=0.35)
    cv2.putText(frame_bgr, detail, (box_x + 12, box_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX, detail_scale, (255, 255, 255), 1, cv2.LINE_AA)

    inner_x = box_x + box_w if side == "left" else box_x
    inner_y = box_y + box_h // 2
    cv2.line(frame_bgr, (inner_x, inner_y), anchor_px, accent, 2, cv2.LINE_AA)
    cv2.circle(frame_bgr, anchor_px, 5, accent, -1, cv2.LINE_AA)
    cv2.circle(frame_bgr, anchor_px, 7, (255, 255, 255), 1, cv2.LINE_AA)


def _layout_and_draw_callouts(frame_bgr, callouts: list):
    """callouts: [{"anchor": (x,y), "title": str, "detail": str, "tier": str}, ...].
    Buckets left/right by anchor x vs. frame center, rebalances toward
    the lighter side if one has 2+ more than the other (so a heavy
    cluster on one side never overflows the frame height), stacks
    top-to-bottom by anchor y within each side.

    box_w scales with frame width (2026-09-XX, real bug found visually
    inspecting an actual rendered frame): this app's videos range from
    ~400px-wide downscaled portrait clips to full-resolution uploads —
    a fixed 260px box works fine on a wide frame but two side-by-side
    260px boxes physically cannot fit on a 400px-wide one at all
    (520px + margins > 400px). Scaling box_w down on narrow frames,
    with _fit_font_scale already handling the text-legibility side of
    the same problem, means this degrades gracefully instead of boxes
    overlapping or running off-frame."""
    if not callouts:
        return
    height, width = frame_bgr.shape[:2]
    box_w = max(140, min(_CALLOUT_BOX_W, int(width * 0.42)))
    center_x = width / 2
    left = [c for c in callouts if c["anchor"][0] < center_x]
    right = [c for c in callouts if c["anchor"][0] >= center_x]

    while len(left) - len(right) >= 2:
        left.sort(key=lambda c: -abs(c["anchor"][0] - center_x))
        right.append(left.pop(0))
    while len(right) - len(left) >= 2:
        right.sort(key=lambda c: -abs(c["anchor"][0] - center_x))
        left.append(right.pop(0))

    left.sort(key=lambda c: c["anchor"][1])
    right.sort(key=lambda c: c["anchor"][1])

    y = _CALLOUT_TOP_Y
    for c in left:
        _draw_callout(frame_bgr, c["anchor"], c["title"], c["detail"], c["tier"],
                      _CALLOUT_MARGIN_X, y, box_w, side="left")
        y += _CALLOUT_BOX_H + _CALLOUT_GAP_Y

    y = _CALLOUT_TOP_Y
    for c in right:
        box_x = width - _CALLOUT_MARGIN_X - box_w
        _draw_callout(frame_bgr, c["anchor"], c["title"], c["detail"], c["tier"],
                      box_x, y, box_w, side="right")
        y += _CALLOUT_BOX_H + _CALLOUT_GAP_Y


def _encode_png(frame_bgr):
    ok, buf = cv2.imencode(".png", frame_bgr)
    return buf.tobytes() if ok else None


def _metrics_by_frame(metric_frames: dict, metrics: dict, bowler_type,
                       frame_keys: list) -> dict:
    """Shared pass for both sports: for EVERY metric mapped in
    metric_frames (regardless of tier), computes its real value, real
    tier (green/amber/red/descriptive/unknown, from metric_ranges.
    classify()), and whether it's drill-eligible (is_critical_and_
    eligible()) — then files it under every frame_key it's assigned to.
    Every mapped metric always appears here; eligibility is carried
    along for the CALLER to decide whether to add a "CRITICAL" tag, not
    used here to hide anything (see this module's own docstring for the
    real bug this fixes). Returns
    {frame_key: [(metric_key, value, tier, eligible), ...]}."""
    per_frame = {fk: [] for fk in frame_keys}
    for metric_key, mapped_frame_keys in metric_frames.items():
        if metric_key.startswith("batting_"):
            value = mr.extract_batting_metric_value(metrics, metric_key)
        else:
            value = mr.extract_metric_value(metrics, metric_key)
        m = metrics.get(_metric_dict_key(metric_key), {})
        m = m if isinstance(m, dict) else {}
        recalibration_pending = bool(m.get(_RECALIBRATION_FLAG_KEYS.get(metric_key, ""), False))
        tracking_uncertain = bool(m.get(_TRACKING_UNCERTAIN_FLAG_KEYS.get(metric_key, ""), False))
        tier = mr.classify(metric_key, value, bowler_type)
        eligible = mr.is_critical_and_eligible(metric_key, value, bowler_type, recalibration_pending, tracking_uncertain)
        for fk in mapped_frame_keys:
            per_frame[fk].append((metric_key, value, tier, eligible))
    return per_frame


def _metric_dict_key(metric_key: str) -> str:
    """Maps a metric_ranges key to the key it's actually stored under in
    the metrics dict — identical for bowling, but batting_* keys strip
    the "batting_" prefix (e.g. "batting_head_movement" -> "head_movement",
    matching extract_batting_metric_value's own lookup table)."""
    if metric_key.startswith("batting_"):
        return {
            "batting_head_movement": "head_movement",
            "batting_front_foot_alignment": "front_foot_alignment",
            "batting_weight_transfer": "weight_transfer",
            "batting_downswing_plane": "downswing_plane",
            "batting_top_elbow_angle": "top_elbow_angle",
            "batting_front_knee_flexion": "front_knee_flexion",
            "batting_xfactor_separation": "xfactor_separation",
        }.get(metric_key, metric_key)
    return metric_key


def generate_bowling_diagnostic_frames(video_path: str, df: pd.DataFrame, events: dict,
                                        metrics: dict, bowler_type, bowling_arm: str) -> dict:
    """Returns {"bfc": bytes|None, "ffc": bytes|None, "release": bytes|None}.
    Never raises — any failure degrades to all-None so a drawing bug can
    never break the analysis this runs alongside."""
    result = {"bfc": None, "ffc": None, "release": None}
    try:
        lead_side = "LEFT" if bowling_arm == "right" else "RIGHT"
        bowl_side = "RIGHT" if bowling_arm == "right" else "LEFT"
        frame_map = {"bfc": events.get("BFC"), "ffc": events.get("FFC"), "release": events.get("BR")}
        per_frame_metrics = _metrics_by_frame(
            _BOWLING_METRIC_FRAMES, metrics, bowler_type, list(frame_map.keys()))

        for frame_key, frame_idx in frame_map.items():
            if frame_idx is None:
                continue
            frame_bgr = _read_frame_bgr(video_path, frame_idx)
            if frame_bgr is None:
                continue
            rows = df[df["frame"] == frame_idx]
            if rows.empty:
                continue
            row = rows.iloc[0]
            if not _draw_skeleton(frame_bgr, row):
                continue

            callouts = []
            for metric_key, value, tier, eligible in per_frame_metrics[frame_key]:
                anchor = _bowling_metric_anchor(
                    metric_key, row, frame_bgr.shape[1], frame_bgr.shape[0], lead_side, bowl_side)
                if anchor is None:
                    continue
                callouts.append({
                    "anchor": anchor,
                    "title": mr.RANGES[metric_key].label.upper(),
                    "detail": _panel_text(metric_key, value, tier, eligible),
                    "tier": tier,
                })
            _layout_and_draw_callouts(frame_bgr, callouts)
            result[frame_key] = _encode_png(frame_bgr)
    except Exception as e:
        monitoring.capture(e)
    return result


def generate_batting_diagnostic_frames(video_path: str, df: pd.DataFrame, events: dict,
                                        metrics: dict, batting_hand: str) -> dict:
    """Returns {"stance": bytes|None, "backlift": bytes|None, "contact": bytes|None}.
    Never raises. batting_hand doubles as both front_side and
    top_hand_side, matching batting_orchestrator.run_batting_analysis's
    own call convention (front_side=batting_hand, top_hand_side=batting_hand)."""
    result = {"stance": None, "backlift": None, "contact": None}
    try:
        front_side = "LEFT" if batting_hand == "left" else "RIGHT"
        top_hand_side = front_side
        frame_map = {
            "stance": events.get("STANCE"),
            "backlift": events.get("BACKLIFT"),
            "contact": events.get("CONTACT"),
        }
        per_frame_metrics = _metrics_by_frame(
            _BATTING_METRIC_FRAMES, metrics, None, list(frame_map.keys()))

        for frame_key, frame_idx in frame_map.items():
            if frame_idx is None:
                continue
            frame_bgr = _read_frame_bgr(video_path, frame_idx)
            if frame_bgr is None:
                continue
            rows = df[df["frame"] == frame_idx]
            if rows.empty:
                continue
            row = rows.iloc[0]
            if not _draw_skeleton(frame_bgr, row):
                continue

            callouts = []
            for metric_key, value, tier, eligible in per_frame_metrics[frame_key]:
                anchor = _batting_metric_anchor(
                    metric_key, row, frame_bgr.shape[1], frame_bgr.shape[0], front_side, top_hand_side)
                if anchor is None:
                    continue
                callouts.append({
                    "anchor": anchor,
                    "title": mr.RANGES[metric_key].label.upper(),
                    "detail": _panel_text(metric_key, value, tier, eligible),
                    "tier": tier,
                })
            _layout_and_draw_callouts(frame_bgr, callouts)
            result[frame_key] = _encode_png(frame_bgr)
    except Exception as e:
        monitoring.capture(e)
    return result
