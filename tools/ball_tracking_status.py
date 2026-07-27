"""
tools/ball_tracking_status.py

Reports progress toward the checkpoint for reassessing the ball-tracking
approach (CVAT/Label Studio annotation, training a YOLO nano/small model
+ ByteTrack + Kalman filter) — decided 2026-07-27 to defer both until
real data volume justifies the switch, rather than build that
infrastructure ahead of having anything to run it on. See the project
memory "ball-tracking data strategy" for the full reasoning.

Checkpoint: ~15-20 distinct clips / ~250-300 labeled frames in
ball_tracking_labels (Gemini's own proposed minimum was 20-30 sequences /
300-500 frames — this checkpoint is deliberately at the low end of that).

Usage:
    python tools/ball_tracking_status.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import profile_store as store

CLIP_CHECKPOINT = 15
FRAME_CHECKPOINT = 250


def _fetch_all_label_rows(client, page_size: int = 1000) -> list:
    """BUG FIX: an unpaginated select() silently truncates at Supabase's
    default 1000-row response cap — verified directly once real row count
    crossed 1000 (this table just did): the checkpoint report undercounted
    by over 100 rows with no error at all, just a wrong-but-plausible-
    looking number. Page through with .range() until a page comes back
    short, same pattern already used in tools/export_training_data.py."""
    rows = []
    start = 0
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
    return rows


def main():
    client = store.get_client()
    rows = _fetch_all_label_rows(client)

    total_frames = len(rows)
    distinct_clips = sorted(set(r["source_video_filename"] for r in rows))
    total_clips = len(distinct_clips)

    print(f"Labeled frames: {total_frames} (checkpoint: {FRAME_CHECKPOINT})")
    print(f"Distinct clips: {total_clips} (checkpoint: {CLIP_CHECKPOINT})")
    for name in distinct_clips:
        count = sum(1 for r in rows if r["source_video_filename"] == name)
        print(f"  - {name}: {count} frames")

    print()
    if total_clips >= CLIP_CHECKPOINT or total_frames >= FRAME_CHECKPOINT:
        print("CHECKPOINT REACHED — worth revisiting CVAT/Label Studio and a first "
              "YOLO fine-tune now that real data volume exists.")
    else:
        pct = max(total_clips / CLIP_CHECKPOINT, total_frames / FRAME_CHECKPOINT)
        print(f"Not yet at checkpoint (~{pct:.0%} of the way there by the closer metric) — "
              "keep collecting via the CapCut-circle + extract_circled_ball.py pipeline.")


if __name__ == "__main__":
    main()
