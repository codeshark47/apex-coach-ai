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

Run locally (not deployed to Streamlit Cloud):
    streamlit run ball_tracking/label_tool.py
"""

import bisect
import datetime
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


def _load_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _display_image_with_marker(frame_rgb: np.ndarray, point, radius: float):
    """Downscaled preview for display, with a marker circle drawn at
    `point` (original-image pixel coords) if one exists yet. Returns the
    PIL image to show and the scale factor to convert a click on the
    DISPLAYED image back to original pixel coordinates."""
    img = Image.fromarray(frame_rgb)
    orig_w, orig_h = img.size
    scale = min(1.0, MAX_DISPLAY_WIDTH / orig_w)
    disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((disp_w, disp_h))

    if point is not None:
        draw = ImageDraw.Draw(img)
        dx, dy = point[0] * scale, point[1] * scale
        dr = max(3, radius * scale)
        draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], outline=(255, 60, 60), width=3)

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
    if st.session_state.get("label_tool_video_choice") not in video_names:
        _log(f"STALE SELECTION — '{st.session_state.get('label_tool_video_choice')}' not found "
             f"in the current {len(video_names)}-video list, clearing it so the picker falls "
             f"back to its default option.")
        st.session_state.pop("label_tool_video_choice", None)
    video_name = st.sidebar.selectbox(
        "Pick a video", video_names,
        format_func=lambda name: f"{name}  ({counts.get(name, 0)} frames already labeled)",
        key="label_tool_video_choice",
    )
    video_path = name_to_path[video_name]

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

    frame_rgb = _load_frame(video_path, frame_idx)
    if frame_rgb is None:
        st.warning("Could not read this frame — skipping.")
        st.session_state.label_tool_frame_ptr += 1
        st.rerun()
        return

    orig_w = frame_rgb.shape[1]
    radius = st.session_state.label_tool_radius or (orig_w * DEFAULT_RADIUS_FRACTION)

    pending_point_key = f"label_tool_pending_{video_name}_{frame_idx}"
    pending_point = st.session_state.get(pending_point_key)

    display_img, scale = _display_image_with_marker(frame_rgb, pending_point, radius)
    click = streamlit_image_coordinates(display_img, key=f"label_tool_click_{video_name}_{frame_idx}")

    if click is not None:
        new_point = (click["x"] / scale, click["y"] / scale)
        if pending_point != new_point:
            st.session_state[pending_point_key] = new_point
            st.rerun()

    # FIRST clip's radius calibration — one-time per video, right after the
    # first click, so the coach sets it by eye once instead of every frame.
    if st.session_state.label_tool_radius is None and pending_point is not None:
        st.info("First click on this clip — set the ball's approximate size, then confirm.")
        radius = st.slider(
            "Ball radius (pixels, original resolution)", min_value=2.0,
            max_value=orig_w * 0.1, value=float(radius), key="label_tool_radius_slider",
        )
        display_img, scale = _display_image_with_marker(frame_rgb, pending_point, radius)
        st.image(display_img)
        if st.button("✅ Confirm size for this whole clip"):
            st.session_state.label_tool_radius = radius
            st.rerun()
        return  # don't allow advancing until size is confirmed

    col1, col2 = st.columns(2)
    with col1:
        confirm_disabled = pending_point is None
        if st.button("➡️ Confirm & next frame", disabled=confirm_disabled, use_container_width=True):
            x, y = pending_point
            client.table("ball_tracking_labels").upsert({
                "source_video_filename": video_name,
                "frame_index": frame_idx,
                "ball_x_px": x,
                "ball_y_px": y,
                "labeled_by": "direct_click_v1",
                "notes": "Directly clicked by the coach on the original, unmarked frame — "
                         "no drawn marker ever existed on this video's pixels.",
            }, on_conflict="source_video_filename,frame_index").execute()
            _upsert_run(client, video_name, fps, orig_w, frame_rgb.shape[0], total_frames, frame_idx, x, y, radius)
            st.session_state.label_tool_history.append(("confirm", frame_idx))
            st.session_state.label_tool_counts[video_name] = st.session_state.label_tool_counts.get(video_name, 0) + 1
            del st.session_state[pending_point_key]
            st.session_state.label_tool_frame_ptr += 1
            _log(f"CONFIRMED '{video_name}' frame {frame_idx} (saved) — advancing to ptr "
                 f"{st.session_state.label_tool_frame_ptr}.")
            st.rerun()
    with col2:
        if st.button("↩️ Undo last", use_container_width=True, disabled=not st.session_state.label_tool_history):
            action, data = st.session_state.label_tool_history.pop()
            if action == "confirm":
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
    with skip_col2:
        st.write("")  # vertical alignment spacer so the button lines up with the number input
        if st.button("⏭️ No ball visible — skip", use_container_width=True):
            actual_skip = min(skip_n, len(sampled_indices) - ptr)
            st.session_state.label_tool_history.append(("skip", actual_skip))
            if pending_point_key in st.session_state:
                del st.session_state[pending_point_key]
            st.session_state.label_tool_frame_ptr += actual_skip
            st.rerun()


if __name__ == "__main__":
    main()
