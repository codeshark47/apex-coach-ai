"""
ball_tracking/label_tool.py

Direct click-to-label tool for building ball_tracking_labels ground
truth — replaces the CapCut-circle workflow entirely, for two reasons:

1. SPEED: CapCut requires finding the ball, dragging a circle to the
   right position and size, and exporting a whole new video file, one
   clip at a time. This tool shows a frame, the coach clicks the ball's
   center (or clicks "No ball visible" to skip), and it auto-advances —
   no dragging, no export step.

2. CORRECTNESS: found during a broader audit (2026-08-01/02) that the
   CapCut workflow draws the circle directly onto the video's own
   pixels, so every training image had a bright, solid-colored,
   consistently-shaped ring sitting on the ball — a far easier thing
   for a network to learn to detect than the ball itself. An inpainting
   fix removed the ring's color but left a locally-smoothed patch the
   model then learned to detect instead (confirmed directly: a trained
   checkpoint's predicted box sat exactly on the inpainting scar, not
   the ball). This tool never draws anything onto the video at all —
   the coach clicks on the untouched original frame, and that exact
   same untouched frame becomes the training image. There is no marker,
   drawn or removed, for a model to learn as a shortcut.

Deliberately ISOLATED from the production app, same as every other file
in this package (see ball_tracking/__init__.py) — not imported by, and
does not import from, streamlit_app.py or any file it depends on.

AI PRE-FILL (2026-08-02): once a trained checkpoint exists under
training/runs/*/weights/best.pt, each new frame is run through it before
the coach ever sees it. A confident detection pre-fills the marker
(shown in ORANGE, distinct from a coach's own RED click) so the coach
reviews/confirms instead of clicking from a blank frame every time —
this is the "use V1 to help label V2" bootstrap the project's own
ball-tracking strategy always planned for, not a shortcut around
verification: the coach still has to look at every frame and either
confirm or correct it, same as before. No model found yet (e.g. a fresh
checkout) just falls back to the original click-from-scratch flow.

Run locally (not deployed to Streamlit Cloud):
    streamlit run ball_tracking/label_tool.py
"""

import bisect
import datetime
import glob
import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


def _log(msg: str):
    """Prints to the terminal running `streamlit run` (not the browser) —
    diagnostic trail for the recurring 'switches back to a different video'
    report (2026-08-02), which leaves no trace in the browser itself since
    the coach sees no page reload when it happens. Timestamped so it can be
    correlated against Supabase's created_at on the labels that did save."""
    print(f"[label_tool {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import profile_store as store
from ball_tracking.track_ball_from_seed import track_ball_from_seed

st.set_page_config(page_title="Ball Labeling Tool", layout="wide")

SEARCH_DIRS = [
    r"C:\Users\Shoaib\Downloads",
    r"C:\Users\Shoaib\Downloads\for phase two",
]
# Excluded for the same reason noted in prepare_dataset.py/project memory:
# not the coach's own footage, and not even real bowling footage (a
# competitor's equipment setup video, downloaded via a video-ripping
# service as a product reference, not training material).
EXCLUDED_FILENAMES = {
    "vidssave.com Fulltrack AI - How to Set Up + Equipment Needed 720P.mp4",
    # Confirmed by direct visual inspection (2026-08-02) to be pre-processed
    # analysis output — skeleton overlay + "Analysis Phase: ..." text burned
    # in — despite having no filename pattern in common with the app's own
    # "Annotated_*" exports (shared through some route that stripped the
    # original name to a hash). A content-based auto-scan for this was
    # attempted and abandoned — too many false positives (e.g. ordinary
    # bright sky triggering a "white text banner" check) to trust; see the
    # in-app "does this look right?" confirmation step instead.
    "161faba7ab0673e9c72eb0f69588f54e.mp4",
    "59f53f357cbf020127bf08311934b554.mp4",
    "c261db899adb35ca76504fd7e0c582c2.mp4",
}
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v")

# BUG FOUND: the Downloads folder is shared with the main app's own
# generated output (skeleton-overlay analysis result videos), not just
# raw source footage — these have the skeleton, phase labels, and chart
# panel burned into every frame, exactly the "burned-in artifact" problem
# this whole tool exists to avoid. Filtered out by the same naming
# convention the main app already uses for every one of these exports.
EXCLUDED_PREFIXES = ("Annotated_",)

SAMPLE_EVERY_N_FRAMES = 3
MAX_DISPLAY_WIDTH = 960
DEFAULT_RADIUS_FRACTION = 0.02  # of frame width — first-frame starting guess only

# RAISED from 0.35 (2026-08-14, real coach report): that value was
# calibrated against generic background noise (a tree branch, a patch of
# grass) sitting at 0.03-0.19 confidence — comfortably below 0.35. But a
# front-on clip surfaced a DIFFERENT, harder class of false positive the
# original calibration never saw: round, ball-colored lookalikes (a
# crease marking, an umpire's hat, a batter's thumb) score meaningfully
# higher than generic noise precisely because they share real visual
# similarity with a ball, not because the model is confused at random.
# Raised to sit at the floor of this model's own documented genuine-
# detection band (0.5-0.8, same real evidence as before) instead of
# just above generic noise — trades fewer assisted frames for fewer
# confidently-wrong ones. Still a starting point, not a final constant.
AI_PREFILL_CONF_THRESHOLD = 0.5

# How many frames ahead of the last coach-confirmed position the tracked
# pre-fill (see below) will follow before giving up and falling back to
# the blind whole-frame scan — matches track_ball_from_seed's own
# default, generous enough to cover this tool's max frame-sampling
# interval (10) with real margin.
TRACK_MAX_GAP_FRAMES = 30

# HARD-NEGATIVE MINING (2026-08-04): real evaluation of ball_v1-6/v1-7
# found the model's false positives cluster on specific lookalike
# objects (a glove was the clearest repeat offender), not random noise.
# Skipping a frame with one of these on screen previously threw the
# information away — the coach recognized exactly what confused a
# detector but the tool never asked. These categories let that get
# captured as a confirmed non-ball training example instead (see
# _save_hard_negative below and prepare_dataset.py's background-image
# handling). "No — just skip" must stay first (index 0) — code below
# treats that as the disabled/default state.
#
# EXPANDED (2026-09-25, real coach request + real confirmed case): the
# IMG_3755.MOV investigation found a confusable-object class not in the
# original 4 (a non-striker's dark pad/ankle strap) — this list was
# only ever grown reactively, one confirmed real confusion at a time.
# Rather than wait for each new class to bite first, added every
# plausible near-pitch lookalike a real match/net session can show: the
# original 4 kept exactly as-is (never rename an existing category —
# prepare_dataset.py's notes text and any coach's prior labeling history
# reference these exact strings), plus shoes, nets/fencing, poles/posts,
# a bare head, and an elbow/knee (skin-toned round joints at small scale
# and low resolution are a real, plausible ball-lookalike, same
# reasoning as the original glove/pad findings).
HARD_NEGATIVE_CATEGORIES = [
    "No — just skip",
    "Glove", "Pad/guard", "Helmet", "Other shiny/round object",
    "Shoe", "Net/fence", "Pole/post", "Head", "Elbow/knee",
]


@st.cache_resource(show_spinner="Loading AI ball detector (one-time)...")
def _load_yolo_model():
    """
    Loads the most recently trained YOLO checkpoint to pre-fill the ball
    position guess on each frame — the coach reviews/confirms instead of
    clicking from a blank frame every time. Auto-picks the newest
    training/runs/*/weights/best.pt by modification time so this never
    needs updating by hand after a retrain (see training/train_yolo.py).

    Returns None if no trained checkpoint exists yet (e.g. a fresh
    checkout before the first training run) — the rest of this file
    treats that as "no pre-fill available," falling back to the
    original click-from-scratch flow with no error. A pre-fill is a
    nice-to-have speedup, never a requirement for the tool to work.
    """
    candidates = glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "runs", "*", "weights", "best.pt")
    )
    if not candidates:
        _log("No trained checkpoint found under training/runs/*/weights/best.pt — "
             "AI pre-fill disabled, falling back to click-from-scratch.")
        return None
    latest = max(candidates, key=os.path.getmtime)

    from ultralytics import YOLO
    model = YOLO(latest)
    # Ultralytics' first .predict() call pays a one-time warmup cost
    # (~2.5s measured directly) that has nothing to do with per-frame
    # inference speed (~40ms measured, after warmup) — pay that cost once
    # here at load time (cached for the whole session by st.cache_resource),
    # not on whatever frame the coach happens to land on first.
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), conf=0.99, verbose=False)
    _log(f"AI pre-fill model loaded: {latest}")
    return model


def _discover_videos() -> list:
    """One entry per distinct filename across both search dirs — same
    dedup-by-name convention prepare_dataset.py already relies on."""
    seen = {}
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name in EXCLUDED_FILENAMES:
                continue
            if name.startswith(EXCLUDED_PREFIXES):
                continue
            if not name.lower().endswith(VIDEO_EXTENSIONS):
                continue
            if name not in seen:
                seen[name] = os.path.join(d, name)
    return sorted(seen.items())


def _already_labeled_counts(client) -> dict:
    """How many frames are already stored per clip — lets the picker
    show progress instead of the coach having to remember what they've
    already done."""
    rows = []
    start, page_size = 0, 1000
    while True:
        result = (
            client.table("ball_tracking_labels")
            .select("source_video_filename")
            .range(start, start + page_size - 1)
            .execute()
        )
        page = result.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    counts = {}
    for r in rows:
        counts[r["source_video_filename"]] = counts.get(r["source_video_filename"], 0) + 1
    return counts


def _last_labeled_frame_index(client, video_name: str):
    """Highest frame_index already saved for this clip, or None. Used to
    resume past work instead of restarting at frame 0 — see the resume
    logic in main() for why this matters."""
    result = (
        client.table("ball_tracking_labels")
        .select("frame_index")
        .eq("source_video_filename", video_name)
        .order("frame_index", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0]["frame_index"] if result.data else None


def _upsert_run(client, video_name: str, fps: float, frame_w: int, frame_h: int,
                 total_frames: int, frame_idx: int, x: float, y: float, radius: float):
    """
    One ball_tracking_runs row per clip, merging each newly-labeled
    frame's position/radius into its raw_candidates dict.

    BUG FOUND: ball_tracking_runs has no unique constraint on
    source_video_filename alone (only ball_tracking_labels does, on
    (source_video_filename, frame_index)) — an upsert with
    on_conflict="source_video_filename" fails outright with a real
    Postgres error ("no unique or exclusion constraint matching the ON
    CONFLICT specification"), not something that can be swallowed or
    retried. Read-then-insert-or-update explicitly instead, rather than
    requiring a schema migration before labeling can even start — a
    single coach clicking through frames one at a time has no real
    concurrent-write risk here.
    """
    existing = (
        client.table("ball_tracking_runs")
        .select("id,raw_candidates,frames_with_candidates")
        .eq("source_video_filename", video_name)
        .execute()
    )
    candidate = {"x_px": x, "y_px": y, "radius_px": radius}

    if existing.data:
        row = existing.data[0]
        raw_candidates = row.get("raw_candidates") or {}
        raw_candidates[str(frame_idx)] = candidate
        client.table("ball_tracking_runs").update({
            "raw_candidates": raw_candidates,
            "frames_with_candidates": len(raw_candidates),
        }).eq("id", row["id"]).execute()
    else:
        client.table("ball_tracking_runs").insert({
            "source_video_filename": video_name,
            "camera_setup_label": "direct_click_v1",
            "detector_name": "human_click",
            "fps": fps,
            "frame_width": frame_w,
            "frame_height": frame_h,
            "total_frames": total_frames,
            "frames_with_candidates": 1,
            "raw_candidates": {str(frame_idx): candidate},
        }).execute()


def _save_hard_negative(client, video_name: str, frame_idx: int, category: str):
    """
    Confirmed hard-negative example: the coach has determined there is NO
    ball in this frame, but a specific lookalike (glove/pad/helmet/other)
    is visible — exactly the false-positive pattern found by direct
    inspection of ball_v1-6/v1-7's predictions (the model latching onto a
    glove). Stored as an ordinary ball_tracking_labels row with NULL
    coordinates — the schema already documents this exact case ("null if
    ball not visible/identifiable this frame"), so no migration is
    needed. prepare_dataset.py reads these and writes a plain background
    image (no matching .txt label) — valid YOLO for "zero objects here,"
    which teaches the model this object is a confirmed non-ball rather
    than simply unseen/unlabeled.
    """
    client.table("ball_tracking_labels").upsert({
        "source_video_filename": video_name,
        "frame_index": frame_idx,
        "ball_x_px": None,
        "ball_y_px": None,
        "labeled_by": "direct_click_v1",
        "notes": f"HARD NEGATIVE: {category} visible here, coach confirmed no ball in frame.",
    }, on_conflict="source_video_filename,frame_index").execute()


# PITCH CALIBRATION (2026-09-25, real coach request): captures the same
# 4 ground-level stump corners pitch_calibration.build_ground_homography()
# already knows how to turn into a real ground-plane mapping, right in
# this tool instead of a separate manual grid-overlay measurement —
# matches FullTrack AI's own documented calibration step ("line up both
# sets of stumps"), see project memory. Stored per-clip on
# ball_tracking_runs.pitch_calibration (nullable jsonb — see
# add_pitch_calibration.sql, must be run once against the live Supabase
# project before this reads/writes anything).
PITCH_CALIBRATION_POINTS = [
    ("near_left_px", "NEAR stumps — LEFT edge (base, where it meets the ground)"),
    ("near_right_px", "NEAR stumps — RIGHT edge (base, where it meets the ground)"),
    ("far_left_px", "FAR stumps — LEFT edge (base, where it meets the ground)"),
    ("far_right_px", "FAR stumps — RIGHT edge (base, where it meets the ground)"),
]


def _load_pitch_calibration(client, video_name: str):
    """Returns the saved {"near_left_px": [x,y], ...} dict for this clip,
    or None if it has never been calibrated (or add_pitch_calibration.sql
    hasn't been run yet against this Supabase project — a missing column
    raises a real Postgres error rather than silently returning nothing,
    so it doesn't look identical to a genuinely un-calibrated clip)."""
    result = (
        client.table("ball_tracking_runs")
        .select("pitch_calibration")
        .eq("source_video_filename", video_name)
        .execute()
    )
    if result.data and result.data[0].get("pitch_calibration"):
        return result.data[0]["pitch_calibration"]
    return None


def _save_pitch_calibration(client, video_name: str, fps: float, frame_w: int,
                             frame_h: int, total_frames: int, calibration: dict):
    """Same read-then-insert-or-update pattern as _upsert_run above (see
    its docstring: ball_tracking_runs has no unique constraint on
    source_video_filename alone, so a plain upsert on that column fails
    outright) — a clip may already have a runs row from ball labeling, a
    prior calibration, both, or neither."""
    existing = (
        client.table("ball_tracking_runs")
        .select("id")
        .eq("source_video_filename", video_name)
        .execute()
    )
    if existing.data:
        client.table("ball_tracking_runs").update({
            "pitch_calibration": calibration,
        }).eq("id", existing.data[0]["id"]).execute()
    else:
        client.table("ball_tracking_runs").insert({
            "source_video_filename": video_name,
            "camera_setup_label": "direct_click_v1",
            "detector_name": "human_click",
            "fps": fps,
            "frame_width": frame_w,
            "frame_height": frame_h,
            "total_frames": total_frames,
            "frames_with_candidates": 0,
            "pitch_calibration": calibration,
        }).execute()


def _render_pitch_calibration_ui(client, video_name: str, video_path: str,
                                  fps: float, total_frames: int):
    """
    Self-contained 4-click flow: near-left, near-right, far-left,
    far-right stump BASE (ground level, not the tips) — the exact same
    4 points pitch_calibration.build_ground_homography() expects, in the
    same order. Deliberately separate from the frame-by-frame ball-
    labeling loop below (own frame picker, own click state) since
    calibrating the camera position is a one-time-per-clip task, not
    something that advances frame by frame.

    Real-world reference distances are fixed constants (pitch_calibration.
    PITCH_LENGTH_M / STUMP_LINE_WIDTH_M — standard cricket dimensions),
    never asked from the coach here, so there's no way to enter a wrong
    number by hand.
    """
    st.subheader("📐 Pitch calibration (stumps)")
    st.caption(
        "Click the base of each stump line (where the stumps meet the ground, not "
        "their tips) — near set then far set, left edge then right edge on each. "
        "Same idea as FullTrack AI's own setup step: line up both sets of stumps."
    )

    existing = _load_pitch_calibration(client, video_name)
    if existing and not st.session_state.get("label_tool_calib_redo"):
        st.success("✅ This clip already has a saved calibration.")
        cols = st.columns(4)
        for col, (key, label) in zip(cols, PITCH_CALIBRATION_POINTS):
            pt = existing.get(key)
            col.metric(label.split(" — ")[1].split(" (")[0], f"{pt}" if pt else "—")
        if st.button("🔁 Redo this clip's calibration"):
            st.session_state.label_tool_calib_redo = True
            st.rerun()
        return

    calib_frame_key = f"label_tool_calib_frame_{video_name}"
    calib_points_key = f"label_tool_calib_points_{video_name}"
    if calib_points_key not in st.session_state:
        st.session_state[calib_points_key] = {}
    points = st.session_state[calib_points_key]

    ref_frame_idx = st.number_input(
        "Reference frame to calibrate from (pick one where both stump sets are "
        "clearly visible and nobody is standing on them)",
        min_value=0, max_value=max(0, total_frames - 1),
        value=st.session_state.get(calib_frame_key, 0), key=calib_frame_key,
    )
    frame_rgb = _load_frame(video_path, ref_frame_idx)
    if frame_rgb is None:
        st.warning("Could not read that frame.")
        return
    orig_w = frame_rgb.shape[1]

    next_idx = len(points)
    if next_idx < len(PITCH_CALIBRATION_POINTS):
        key, label = PITCH_CALIBRATION_POINTS[next_idx]
        st.info(f"Click: **{label}**")
    else:
        st.success("All 4 points placed.")

    from streamlit_image_coordinates import streamlit_image_coordinates

    img = Image.fromarray(frame_rgb)
    scale = min(1.0, MAX_DISPLAY_WIDTH / orig_w)
    disp_img = img.resize((int(orig_w * scale), int(frame_rgb.shape[0] * scale)))
    draw = ImageDraw.Draw(disp_img)
    point_colors = [(255, 60, 60), (60, 200, 60), (60, 140, 255), (255, 200, 0)]
    for i, (key, _label) in enumerate(PITCH_CALIBRATION_POINTS):
        if key in points:
            px, py = points[key]
            dx, dy = px * scale, py * scale
            draw.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], outline=point_colors[i], width=3)
            draw.text((dx + 8, dy - 8), str(i + 1), fill=point_colors[i])

    click = streamlit_image_coordinates(disp_img, key=f"label_tool_calib_click_{video_name}_{ref_frame_idx}_{next_idx}")
    if click is not None and next_idx < len(PITCH_CALIBRATION_POINTS):
        key, _label = PITCH_CALIBRATION_POINTS[next_idx]
        points[key] = (click["x"] / scale, click["y"] / scale)
        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("↩️ Undo last point", disabled=not points):
            last_key = list(points.keys())[-1]
            del points[last_key]
            st.rerun()
    with col2:
        if st.button("🔄 Start over"):
            st.session_state[calib_points_key] = {}
            st.rerun()
    with col3:
        ready = len(points) == len(PITCH_CALIBRATION_POINTS)
        if st.button("✅ Save calibration", disabled=not ready, use_container_width=True):
            calibration = {
                key: list(points[key]) for key, _label in PITCH_CALIBRATION_POINTS
            }
            calibration["reference_frame"] = int(ref_frame_idx)
            _save_pitch_calibration(client, video_name, fps, orig_w, frame_rgb.shape[0],
                                     total_frames, calibration)
            st.session_state[calib_points_key] = {}
            st.session_state.label_tool_calib_redo = False
            st.success("Saved.")
            st.rerun()


def _load_frame(video_path: str, frame_idx: int):
    # REVERTED (2026-08-02): briefly cached one VideoCapture handle per
    # video to cut per-frame latency (~275ms -> ~153ms). That crashed the
    # whole Streamlit process ("Assertion fctx->async_lock failed" from
    # libavcodec's frame-threaded decoder) after ~30 minutes of real use —
    # Streamlit can run two script executions for the same session back to
    # back closely enough that a shared, cached VideoCapture's blocking
    # native .read() call from one run can still be in flight when the next
    # run's .set()/.read() touches the same object, and OpenCV's decoder
    # isn't safe under that kind of concurrent access. A fresh capture per
    # call has no shared native state to race on, so it can't hit this —
    # worth the latency to not crash the coach's whole labeling session.
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _display_image_with_marker(frame_rgb: np.ndarray, point, radius: float,
                                marker_color=(255, 60, 60)):
    """Downscaled preview for display, with a marker circle drawn at
    `point` (original-image pixel coords) if one exists yet. Returns the
    PIL image to show and the scale factor to convert a click on the
    DISPLAYED image back to original pixel coordinates.

    marker_color distinguishes an unreviewed AI suggestion (orange, see
    main()) from a coach's own click (red, the original/default) — the
    coach needs to be able to tell at a glance whether a marker is
    something they confirmed or something still waiting on their
    judgment, not just "a marker is present."
    """
    img = Image.fromarray(frame_rgb)
    orig_w, orig_h = img.size
    scale = min(1.0, MAX_DISPLAY_WIDTH / orig_w)
    disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((disp_w, disp_h))

    if point is not None:
        draw = ImageDraw.Draw(img)
        dx, dy = point[0] * scale, point[1] * scale
        dr = max(3, radius * scale)
        draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], outline=marker_color, width=3)

    return img, scale


def main():
    from streamlit_image_coordinates import streamlit_image_coordinates

    st.title("🎯 Ball Labeling Tool")
    st.caption(
        "Click the ball's center each frame. No dragging, no CapCut, no export — "
        "just click and it moves on."
    )

    # BUG FOUND directly from a coach report (2026-08-02): re-scanning the
    # actual folders on every single rerun (every click) meant ANY transient
    # inconsistency in that scan — a file mid-download, a cloud-sync
    # hiccup, antivirus briefly touching a file — could make the CURRENTLY
    # selected video look "missing" for one render, which triggered the
    # safety fallback below to reset the selection outright. That reset
    # also wiped label_tool_video_confirmed, forcing the "does this look
    # like raw footage?" screen to reappear — reported as "keeps pushing me
    # back to frame one" and confirmed to happen without ever restarting
    # the app, ruling out a simpler explanation. Fix: scan the folders ONCE
    # per session (cached in session_state), not on every rerun — the list
    # only changes when the coach explicitly asks it to via the refresh
    # button, so a transient filesystem hiccup elsewhere can't cascade into
    # resetting an in-progress video's state anymore.
    if "label_tool_videos" not in st.session_state:
        _log("SESSION INIT — label_tool_videos not in session_state, scanning fresh "
             "(either this session's first run, or the session was reset).")
        st.session_state.label_tool_videos = _discover_videos()
    if st.sidebar.button("🔄 Refresh video list", help="Pick up new files added to the folders since this session started."):
        _log("Manual refresh button clicked — rescanning video folders.")
        st.session_state.label_tool_videos = _discover_videos()
        st.rerun()
    videos = st.session_state.label_tool_videos
    if not videos:
        st.error(f"No video files found in {SEARCH_DIRS}.")
        return

    client = store.get_client()
    if "label_tool_counts" not in st.session_state:
        st.session_state.label_tool_counts = _already_labeled_counts(client)
    counts = st.session_state.label_tool_counts

    # Keyed by filename, not list position, so the selection survives even
    # if the (now session-cached, only-refreshed-on-request) list ever
    # does change — see BUG note above for why position-based indexing
    # was fragile.
    video_names = [name for name, _ in videos]
    name_to_path = dict(videos)
    # If the previously-selected file genuinely disappeared (moved,
    # renamed, deleted, or a stale selection from before a manual refresh),
    # Streamlit's selectbox raises outright rather than falling back —
    # clear the stale value first so it just defaults to the first option.
    _stale_choice = st.session_state.get("label_tool_video_choice")
    if _stale_choice not in video_names:
        if _stale_choice is not None:  # None here is just normal first-ever load, not an anomaly
            _log(f"STALE SELECTION — '{_stale_choice}' not found in the current "
                 f"{len(video_names)}-video list, clearing it so the picker falls back "
                 f"to its default option.")
        st.session_state.pop("label_tool_video_choice", None)
    # ROOT CAUSE FOUND (2026-08-02, via the coach's terminal log): format_func
    # used to bake the live "N frames already labeled" count into each
    # option's displayed text. Confirming a frame updates that count for the
    # CURRENTLY selected video in the very same rerun that also (re)submits
    # the selectbox's own value — the log showed session_state literally
    # holding the formatted display string ("IMG_3082.MOV  (3 frames already
    # labeled)") instead of the filename right after a count changed, which
    # then failed the video_names membership check above and fell back to
    # the first video alphabetically (Abu Bakar.MOV) — matching every
    # reported incident exactly. Fix: the option label must never change
    # while it's selected. Show the raw filename only in the dropdown, and
    # put the live count in a separate caption underneath instead.
    video_name = st.sidebar.selectbox("Pick a video", video_names, key="label_tool_video_choice")
    video_path = name_to_path[video_name]
    st.sidebar.caption(f"{counts.get(video_name, 0)} frames already labeled for this clip.")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Reset per-clip session state whenever the chosen video changes.
    #
    # BUG FOUND (2026-08-02, reported again after the earlier _discover_videos
    # caching fix): coach got sent back to the "does this look like raw
    # footage?" screen and lost their frame position mid-labeling, with no
    # visible page reload — ruled out via a direct Supabase query that no
    # labeled data was actually lost (a clip being worked on that session
    # still had exactly the frames the coach had confirmed, nothing more,
    # nothing less). That means the in-memory session state was silently
    # reset by something below the app's control (e.g. a dropped/reconnected
    # Streamlit connection) rather than by a code bug in the save path.
    # Instead of chasing that trigger further, resume state is now derived
    # from Supabase — the durable source of truth — rather than trusted
    # session_state: a clip with any saved labels is treated as already
    # confirmed as raw footage (the only way those rows could exist), and
    # labeling resumes right after the last saved frame instead of frame 0.
    # Worst case after any future reset is one redundant rerender, not lost
    # progress or re-skipping past an entire run-up again.
    previous_video = st.session_state.get("label_tool_current_video")
    if previous_video != video_name:
        if previous_video is None:
            _log(f"FIRST VIDEO THIS SESSION — '{video_name}' selected.")
        else:
            _log(f"VIDEO CHANGED — '{previous_video}' -> '{video_name}' "
                 f"(picker key is currently '{st.session_state.get('label_tool_video_choice')}').")
        st.session_state.label_tool_current_video = video_name
        st.session_state.label_tool_radius = None
        st.session_state.label_tool_history = []  # for undo: list of (frame_idx, had_ball)
        already_labeled = counts.get(video_name, 0) > 0
        st.session_state.label_tool_video_confirmed = already_labeled
        st.session_state.label_tool_frame_ptr = 0
        if already_labeled:
            last_idx = _last_labeled_frame_index(client, video_name)
            if last_idx is not None:
                default_sampled = list(range(0, total_frames, SAMPLE_EVERY_N_FRAMES))
                st.session_state.label_tool_frame_ptr = bisect.bisect_right(default_sampled, last_idx)
        _log(f"  -> confirmed={st.session_state.label_tool_video_confirmed} "
             f"frame_ptr={st.session_state.label_tool_frame_ptr}")

    # ONE-TIME SAFETY CHECK per video, before any labeling starts: found
    # directly (2026-08-02) that some files in this folder are the main
    # app's own pre-processed analysis output (skeleton + phase-label text
    # burned in) despite having no filename in common with the app's own
    # "Annotated_*" export convention — shared through some route that
    # stripped the original name. A pixel-color auto-scan for this was
    # tried and abandoned (too many false positives, e.g. bright sky
    # mistaken for a text banner) — a human's one-second glance at the
    # first frame is far more reliable than that heuristic turned out to be.
    if not st.session_state.get("label_tool_video_confirmed"):
        first_frame = _load_frame(video_path, 0)
        if first_frame is not None:
            st.image(first_frame, caption="First frame of this clip", width=500)
        st.warning(
            "⚠️ Quick check before labeling: does this look like RAW footage "
            "(no skeleton, no dots, no text overlay)? If it already has "
            "analysis graphics drawn on it, don't label it — pick a different video."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Yes, this is raw footage — start labeling", use_container_width=True):
                st.session_state.label_tool_video_confirmed = True
                st.rerun()
        with col_b:
            if st.button("🚫 No, this is a processed video — skip it", use_container_width=True):
                st.error(f"Noted — pick a different video from the sidebar. ({video_name} skipped)")
        return

    mode = st.sidebar.radio(
        "Mode", ["Label ball positions", "📐 Calibrate pitch (stumps)"],
        help="Pitch calibration is a one-time-per-clip step (both stump sets' ground "
             "positions) — doesn't need to be redone per frame, unlike ball labeling.",
    )
    if mode == "📐 Calibrate pitch (stumps)":
        _render_pitch_calibration_ui(client, video_name, video_path, fps, total_frames)
        return

    sample_n = st.sidebar.number_input(
        "Label every Nth frame", min_value=1, max_value=10, value=SAMPLE_EVERY_N_FRAMES,
        help="Consecutive frames are highly redundant — every 2nd-3rd frame is usually enough.",
    )
    sampled_indices = list(range(0, total_frames, sample_n))

    ptr = st.session_state.label_tool_frame_ptr
    if ptr >= len(sampled_indices):
        st.success(f"✅ All sampled frames done for {video_name}. Pick another video from the sidebar.")
        return

    frame_idx = sampled_indices[ptr]
    st.progress(ptr / max(1, len(sampled_indices) - 1))
    st.caption(f"**{video_name}** — frame {ptr + 1} of {len(sampled_indices)} (video frame #{frame_idx})")

    # BUG REPORTED (2026-08-02, real coach test, "fairly often"): the frame
    # image sometimes doesn't visually load at all, but the confirm/skip
    # buttons are still there and clickable — the coach described clicking
    # confirm on an AI suggestion without being able to see whether it was
    # right (this time it happened to be correct, verified directly, but
    # that was luck, not something to rely on). Wrapping the load+inference
    # step in a spinner at least makes "still working" visually distinct
    # from "silently stuck" — a manual reload button (below) is the actual
    # escape hatch if the image still doesn't show up after that.
    with st.spinner("Loading frame..."):
        frame_rgb = _load_frame(video_path, frame_idx)
        if frame_rgb is None:
            st.warning("Could not read this frame — skipping.")
            st.session_state.label_tool_frame_ptr += 1
            st.rerun()
            return

        orig_w = frame_rgb.shape[1]

        pending_point_key = f"label_tool_pending_{video_name}_{frame_idx}"
        pending_source_key = f"label_tool_pending_source_{video_name}_{frame_idx}"
        ai_radius_key = f"label_tool_ai_radius_{video_name}_{frame_idx}"

        # PRE-FILL: only runs the FIRST time this frame is shown this
        # session (pending_point_key not set yet) — never overwrites a
        # coach's own click or a prior visit to this same frame (e.g. after
        # "Undo last" stepping back to it). The coach still reviews and
        # confirms every single frame either way — this only removes the
        # "click from nothing" step when a guess is already available.
        #
        # TRACKED PRE-FILL, tried FIRST (2026-08-14, real coach ask: "the
        # AI should learn from human clicks"): the whole-frame AI scan
        # below treats every frame as independent — no memory of the
        # frame before it, which is exactly why it kept latching onto
        # unrelated things (a crease marking, an umpire's hat) once the
        # confidence threshold was low enough to fire at all. If the
        # coach already confirmed a REAL position earlier in this same
        # clip, that's a far stronger starting point: track_ball_from_
        # seed follows forward from that confirmed (frame, x, y, size)
        # using a small search crop plus the size-trend check validated
        # against real ground truth earlier today, instead of scanning
        # the whole frame blind. Shown in a THIRD color (teal) distinct
        # from the AI orange and the coach's own red, so it's clear at a
        # glance this suggestion came from following the ball, not a
        # fresh per-frame guess.
        last_confirmed_key = f"label_tool_last_confirmed_{video_name}"
        if pending_point_key not in st.session_state:
            last_confirmed = st.session_state.get(last_confirmed_key)
            tracked_this_frame = False
            if last_confirmed is not None:
                lc_frame, lc_x, lc_y, lc_size = last_confirmed
                gap = frame_idx - lc_frame
                if 0 < gap <= TRACK_MAX_GAP_FRAMES:
                    yolo_model = _load_yolo_model()
                    if yolo_model is not None:
                        track_result = track_ball_from_seed(
                            video_path, lc_frame, (lc_x, lc_y), yolo_model,
                            seed_size=lc_size, max_frames_forward=gap,
                        )
                        if track_result["status"] == "success":
                            _tp = track_result["points"]
                            if _tp and _tp[-1][0] == frame_idx:
                                _, tx, ty, tconf, tsize = _tp[-1]
                                st.session_state[pending_point_key] = (tx, ty)
                                st.session_state[pending_source_key] = "tracked"
                                if tsize:
                                    st.session_state[ai_radius_key] = tsize / 2
                                tracked_this_frame = True

            # WHOLE-FRAME AI FALLBACK: no prior confirmed position to
            # track from yet (first labeled frame of this clip, or the
            # tracker's gap ran out without finding the ball) — same
            # blind per-frame scan as before.
            if not tracked_this_frame:
                yolo_model = _load_yolo_model()
                if yolo_model is not None:
                    # BUG FOUND (2026-08-04, real coach report of near-total
                    # non-detection even on a clearly-visible ball): frame_rgb
                    # is RGB (converted for on-screen display). train_yolo.py
                    # trains on raw BGR frames (prepare_dataset.py writes
                    # cv2.imwrite(..., frame) with no color conversion), and
                    # ultralytics treats a raw numpy array the same way cv2
                    # itself does — BGR. Feeding it frame_rgb silently swapped
                    # red/blue for every prediction. Confirmed directly
                    # against 71 real labeled frames from a genuine training
                    # clip: BGR input hit the true ball position on 69/71
                    # frames (avg conf 0.467) vs only 59/71 for RGB (avg conf
                    # 0.476, but several frames lost enough confidence to fall
                    # below AI_PREFILL_CONF_THRESHOLD entirely, e.g. 0.41->0.14
                    # on one frame). Convert back to BGR for inference only —
                    # frame_rgb still drives the on-screen display, unchanged.
                    frame_bgr_for_model = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    results = yolo_model.predict(frame_bgr_for_model, conf=AI_PREFILL_CONF_THRESHOLD, verbose=False)
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        confs = boxes.conf.tolist()
                        best_i = confs.index(max(confs))
                        x1, y1, x2, y2 = boxes.xyxy[best_i].tolist()
                        st.session_state[pending_point_key] = ((x1 + x2) / 2, (y1 + y2) / 2)
                        st.session_state[pending_source_key] = "ai"
                        st.session_state[ai_radius_key] = max(x2 - x1, y2 - y1) / 2

    pending_point = st.session_state.get(pending_point_key)
    pending_source = st.session_state.get(pending_source_key)

    # PER-FRAME ADJUSTABLE RADIUS (2026-09-25, real coach request): this
    # used to be set ONCE on the clip's first click and then silently
    # reused for every later frame, regardless of source. Real problem
    # with that: a ball's TRUE apparent size shrinks as it travels away
    # from a fixed camera (perspective), so a radius calibrated on a
    # close, large first frame was systematically too big for a
    # far-down-the-pitch frame later in the same clip — meaning every
    # radius recorded after the first was honestly wrong, not just
    # imprecise, and that wrong size feeds straight into training
    # (prepare_dataset.py writes a YOLO box sized from this radius) and
    # into track_ball_from_seed's own size-trend consistency check.
    # Fix: no more one-time lock. `label_tool_radius` is now just the
    # LAST radius actually used (a rolling default, not a hard clip
    # value), and a per-frame AI/tracked size estimate — genuinely
    # per-frame, not carried over — takes priority when one exists,
    # since it reflects THIS frame's real apparent size. The coach can
    # always nudge it with the slider below; nothing here removes that.
    if pending_source in ("ai", "tracked") and ai_radius_key in st.session_state:
        radius = st.session_state[ai_radius_key]
    elif st.session_state.label_tool_radius is not None:
        radius = st.session_state.label_tool_radius
    else:
        radius = orig_w * DEFAULT_RADIUS_FRACTION

    marker_color = {"ai": (255, 165, 0), "tracked": (200, 180, 60)}.get(pending_source, (255, 60, 60))
    if pending_point is None:
        st.caption("Click the ball's center.")
    elif pending_source == "ai":
        st.caption("🤖 AI-suggested position (orange) — correct if it looks right, "
                   "or click elsewhere to fix it before confirming.")
    elif pending_source == "tracked":
        st.caption("🎯 Followed from your last confirmed click (teal) — correct if it "
                   "looks right, or click elsewhere to fix it before confirming.")
    else:
        st.caption("📍 Your click (red).")

    # ESCAPE HATCH (2026-08-02, real coach report of the image sometimes
    # not visually loading "fairly often," with the confirm/skip buttons
    # still clickable regardless): if the frame below looks blank, this
    # forces a fresh rerun instead of guessing whether to confirm or skip
    # on something you can't actually see.
    if st.button("🔄 If the frame below looks blank, click here to reload it"):
        st.rerun()

    display_img, scale = _display_image_with_marker(frame_rgb, pending_point, radius, marker_color)
    click = streamlit_image_coordinates(display_img, key=f"label_tool_click_{video_name}_{frame_idx}")

    if click is not None:
        new_point = (click["x"] / scale, click["y"] / scale)
        if pending_point != new_point:
            st.session_state[pending_point_key] = new_point
            st.session_state[pending_source_key] = "coach"
            st.rerun()

    # PER-FRAME RADIUS SLIDER (2026-09-25, replaces the old one-time
    # "first click on this clip, confirm size forever" screen — see the
    # radius-selection comment above for why locking it once was wrong).
    # Shown for every frame that has a point, not gated to the first —
    # the coach can leave it alone (it already carries a sensible
    # default forward) or shrink it as the ball gets further from the
    # camera. Key is unique per frame so each frame's slider genuinely
    # starts from ITS OWN best estimate (AI/tracked size, or the last
    # radius used) rather than reusing whatever the previous frame's
    # widget happened to be left at.
    if pending_point is not None:
        radius = st.slider(
            "Ball radius (pixels, original resolution) — shrink this as the ball "
            "moves further from the camera",
            min_value=2.0, max_value=orig_w * 0.1, value=float(radius),
            key=f"label_tool_radius_slider_{video_name}_{frame_idx}",
        )
        # Re-render below the clickable image above so the circle on
        # screen actually reflects any adjustment just made — the
        # clickable copy above was drawn before this slider could run.
        preview_img, _ = _display_image_with_marker(frame_rgb, pending_point, radius, marker_color)
        st.image(preview_img, caption="Preview at the radius above")

    col1, col2 = st.columns(2)
    with col1:
        confirm_disabled = pending_point is None
        if st.button("➡️ Confirm & next frame", disabled=confirm_disabled, use_container_width=True):
            x, y = pending_point
            # Notes distinguish "AI suggested this, coach accepted it as-is" from a
            # fully manual click — real signal for how often the pre-fill is
            # actually trustworthy, same reasoning as the auto-vs-confirmed
            # logging already built for release-point/foot-contact in the main
            # app. labeled_by stays "direct_click_v1" either way — downstream
            # (prepare_dataset.py) filters on that, not on this distinction.
            if pending_source == "ai":
                notes = ("AI-suggested position (V1 model pre-fill) accepted by the coach "
                         "without adjustment — no drawn marker ever existed on this video's pixels.")
            elif pending_source == "tracked":
                notes = ("Followed forward from the coach's last confirmed click on this clip "
                         "(track_ball_from_seed) and accepted without adjustment — no drawn "
                         "marker ever existed on this video's pixels.")
            else:
                notes = ("Directly clicked by the coach on the original, unmarked frame — "
                         "no drawn marker ever existed on this video's pixels.")
            client.table("ball_tracking_labels").upsert({
                "source_video_filename": video_name,
                "frame_index": frame_idx,
                "ball_x_px": x,
                "ball_y_px": y,
                "labeled_by": "direct_click_v1",
                "notes": notes,
            }, on_conflict="source_video_filename,frame_index").execute()
            _upsert_run(client, video_name, fps, orig_w, frame_rgb.shape[0], total_frames, frame_idx, x, y, radius)
            # Remembered so the NEXT frame's pre-fill can follow forward
            # from here instead of scanning blind — see the tracked
            # pre-fill block above. Diameter (2x radius) matches
            # track_ball_from_seed's own seed_size convention.
            st.session_state[last_confirmed_key] = (frame_idx, x, y, radius * 2)
            # Rolling default for the NEXT frame's slider, in case that
            # frame has no AI/tracked size estimate of its own to start
            # from — this is just a starting point the coach can still
            # adjust, never a re-imposed lock (see the radius-selection
            # comment above).
            st.session_state.label_tool_radius = radius
            st.session_state.label_tool_history.append(("confirm", frame_idx))
            st.session_state.label_tool_counts[video_name] = st.session_state.label_tool_counts.get(video_name, 0) + 1
            del st.session_state[pending_point_key]
            st.session_state.pop(pending_source_key, None)
            st.session_state.pop(ai_radius_key, None)
            st.session_state.label_tool_frame_ptr += 1
            _log(f"CONFIRMED '{video_name}' frame {frame_idx} (saved) — advancing to ptr "
                 f"{st.session_state.label_tool_frame_ptr}.")
            st.rerun()
    with col2:
        if st.button("↩️ Undo last", use_container_width=True, disabled=not st.session_state.label_tool_history):
            action, data = st.session_state.label_tool_history.pop()
            if action in ("confirm", "skip_hardneg"):
                # Both wrote a real ball_tracking_labels row (positive or
                # hard-negative) for exactly one frame — same undo: delete
                # that row, un-count it, step back one frame.
                client.table("ball_tracking_labels").delete().eq(
                    "source_video_filename", video_name
                ).eq("frame_index", data).execute()
                st.session_state.label_tool_counts[video_name] = max(
                    0, st.session_state.label_tool_counts.get(video_name, 1) - 1
                )
                st.session_state.label_tool_frame_ptr -= 1
            else:  # "skip" — data is how many steps that skip advanced by
                st.session_state.label_tool_frame_ptr -= data
            st.rerun()

    # SKIP, with a coach-chosen count — coach-requested feature: often
    # already knows from experience the ball won't be visible for a whole
    # stretch (bowler's run-up, follow-through) and doesn't want to click
    # "skip" one frame at a time through it. Defaults to 1 so the existing
    # single-skip behavior is unchanged unless the coach raises it. Stored
    # in history as ("skip", n) — not n separate entries — so "Undo last"
    # reverses the whole bulk skip as one action, not one frame at a time.
    skip_col1, skip_col2 = st.columns([2, 1])
    with skip_col1:
        skip_n = st.number_input(
            "Skip how many frames", min_value=1, max_value=100, value=1, key="label_tool_skip_n",
            help="E.g. skip 10 to jump past a stretch you already know the ball won't be visible in. "
                 "Clamped automatically if it would go past the end of this clip.",
        )
        # Hard-negative tagging only makes sense for a single, specific
        # frame — a bulk skip of N frames has no one frame to attach a
        # category to, so the picker is only offered when skip_n == 1.
        hard_neg_choice = HARD_NEGATIVE_CATEGORIES[0]
        if skip_n == 1:
            hard_neg_choice = st.selectbox(
                "Lookalike here? (optional)", HARD_NEGATIVE_CATEGORIES,
                key=f"label_tool_hardneg_main_{video_name}_{frame_idx}",
                help="Flags a glove/pad/helmet visible in THIS frame as a confirmed non-ball "
                     "example, instead of just passing over it unrecorded — helps the model "
                     "stop mistaking these for the ball.",
            )
    with skip_col2:
        st.write("")  # vertical alignment spacer so the button lines up with the number input
        if st.button("⏭️ No ball visible — skip", use_container_width=True):
            actual_skip = min(skip_n, len(sampled_indices) - ptr)
            if skip_n == 1 and hard_neg_choice != HARD_NEGATIVE_CATEGORIES[0]:
                _save_hard_negative(client, video_name, frame_idx, hard_neg_choice)
                st.session_state.label_tool_history.append(("skip_hardneg", frame_idx))
                st.session_state.label_tool_counts[video_name] = st.session_state.label_tool_counts.get(video_name, 0) + 1
            else:
                st.session_state.label_tool_history.append(("skip", actual_skip))
            if pending_point_key in st.session_state:
                del st.session_state[pending_point_key]
            st.session_state.pop(pending_source_key, None)
            st.session_state.pop(ai_radius_key, None)
            st.session_state.label_tool_frame_ptr += actual_skip
            st.rerun()


if __name__ == "__main__":
    main()
