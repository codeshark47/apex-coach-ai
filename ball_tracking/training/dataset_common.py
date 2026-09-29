"""
ball_tracking/training/dataset_common.py

Clip-discovery, compression-caching, and train/val split logic shared
between prepare_dataset.py (YOLO box-label format) and
prepare_tracknet_dataset.py (multi-frame heatmap format) — extracted
2026-09-29 so both dataset formats are built from IDENTICAL clips and
splits. This matters specifically because the two are meant to be
compared head-to-head later (validate_holdout.py, both architectures
against the same held-out clips) — if they silently diverged on which
clips count as train vs val, that comparison would be invalid without
either side knowing it.

Nothing in this file changes prepare_dataset.py's existing behavior —
this is a pure extraction, verified by checking prepare_dataset.py
still reports identical total_written/total_val_written counts before
and after the refactor.
"""

import os

from orchestrator import compress_video_file

VALID_LABELED_BY = "direct_click_v1"

# Where normalized copies of source videos are cached — see
# _normalized_video_path's own docstring for why this normalization
# must happen before ANY dataset (YOLO or tracknet) reads frames.
# Shared across both dataset builders so a clip compressed once by
# either script is reused by the other, not re-encoded twice.
COMPRESSED_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_compressed_cache")

# Set once a clip (ideally a different scene) has been labeled with the
# new tool — see prepare_dataset.py's original module docstring for the
# full history of why these specific clips were chosen.
VAL_CLIPS = {"PXL_20260801_040327130.mp4", "IMG_3082.MOV"}

SEARCH_DIRS = [
    "C:/Users/Shoaib/Downloads",
    "C:/Users/Shoaib/Downloads/for phase two",
]


def fetch_all(client, table, columns, filters=None):
    rows = []
    start = 0
    page_size = 1000
    while True:
        query = client.table(table).select(columns)
        for col, val in (filters or {}).items():
            query = query.eq(col, val)
        result = query.range(start, start + page_size - 1).execute()
        page = result.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


def find_video(filename):
    for d in SEARCH_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def normalized_video_path(original_path: str, filename: str) -> str:
    """
    Returns a path to a compressed, cached copy of original_path — every
    source video must go through the SAME compressor real coach uploads
    do (orchestrator.compress_video_file), so training images match the
    live inference regime instead of whatever raw format a source
    happened to be captured in (a real, confirmed train/inference
    mismatch found and fixed 2026-08-15). Reuses an existing cached copy
    rather than re-compressing every run.
    """
    os.makedirs(COMPRESSED_CACHE_DIR, exist_ok=True)
    clip_slug = "".join(c if c.isalnum() else "_" for c in filename.rsplit(".", 1)[0])
    cached_path = os.path.join(COMPRESSED_CACHE_DIR, f"{clip_slug}.mp4")
    if not os.path.exists(cached_path):
        # Real bug, pre-existing (not introduced by this extraction):
        # Windows console's cp1252 default encoding can't print a
        # filename containing emoji (confirmed 2026-09-29 -- a real
        # coach-downloaded clip's title, "158 Kph [emoji]...", crashed
        # this exact print). Encode defensively rather than assume every
        # future filename is plain ASCII.
        safe_filename = filename.encode("ascii", "replace").decode("ascii")
        print(f"  Compressing (first time only, cached for future runs): {safe_filename}")
        # max_fps=None: stored labels' frame_index values were captured
        # against the ORIGINAL video's own frame numbering (label_tool.py
        # reads native frame rate, no compression step). Resampling fps
        # here would shift which frame lands at which index — a real
        # risk, not theoretical (at least one labeled clip is ~120fps).
        compress_video_file(original_path, cached_path, max_fps=None)
    return cached_path
