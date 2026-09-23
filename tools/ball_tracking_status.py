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

import datetime
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import profile_store as store

CLIP_CHECKPOINT = 15
FRAME_CHECKPOINT = 250

# STALENESS CHECK (2026-09-24, real confirmed bug, not a guess): a label's
# stored pixel position is only trustworthy if the video file it was
# clicked against hasn't changed since. Confirmed TWICE now, 10 days
# apart, on the exact same clip (WhatsApp Video 2026-08-14 at 5.12.16
# PM.mp4) — the file's mtime sits after its earliest label's created_at,
# and drawing the stored ground-truth point onto the CURRENT file's frame
# 306 lands in tree branches, nowhere near the bowler. validate_holdout.py
# silently returned 0/5 against this clip with no hint that the labels
# themselves, not the model or tracker, were the problem. This never had
# an automated check before — it was only ever caught by someone manually
# re-deriving it from a confusing validation failure. A buffer (not an
# exact-equality check) allows for a plain file copy/move, which can
# legitimately touch mtime without changing a single frame of content.
_STALENESS_BUFFER = datetime.timedelta(minutes=5)
# Only input/ is checked — the one directory this app's own tooling reads
# training/validation video from (label_tool.py, validate_holdout.py,
# prepare_dataset.py all resolve paths under here or accept an explicit
# path pointing here). Coaches' Downloads folders are a real source but
# aren't what any of THIS project's scripts read directly.
_LOCAL_VIDEO_SEARCH_DIRS = ["input"]


def _find_local_video(filename: str):
    for d in _LOCAL_VIDEO_SEARCH_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def _parse_created_at(value: str):
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


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
            .select("source_video_filename,notes,created_at")
            .range(start, start + page_size - 1)
            .execute()
        )
        page = result.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


def _report_staleness(rows: list):
    """For every distinct clip that still has a local file under input/,
    compares that file's mtime against the clip's EARLIEST label's
    created_at. A file modified after its own earliest label was created
    means the label's pixel position may no longer describe the current
    file's content at that frame — this doesn't invalidate the label FOR
    CERTAIN (a harmless re-save/copy can touch mtime too), but every such
    clip needs a real visual spot-check (draw the stored (x,y) onto that
    exact frame in the CURRENT file) before trusting any validation
    result against it, same discipline the 2026-09-13/2026-09-24
    incidents both required to actually diagnose."""
    by_clip = {}
    for r in rows:
        name = r["source_video_filename"]
        created = _parse_created_at(r.get("created_at"))
        if created is None:
            continue
        if name not in by_clip or created < by_clip[name]:
            by_clip[name] = created

    flagged = []
    checked = 0
    for name, earliest_created in sorted(by_clip.items()):
        local_path = _find_local_video(name)
        if local_path is None:
            continue
        checked += 1
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(local_path), tz=datetime.timezone.utc)
        if earliest_created.tzinfo is None:
            earliest_created = earliest_created.replace(tzinfo=datetime.timezone.utc)
        if mtime - earliest_created > _STALENESS_BUFFER:
            flagged.append((name, earliest_created, mtime))

    print(f"\nStaleness check: {checked} labeled clip(s) have a local file under input/ to check.")
    if flagged:
        print(f"WARNING: {len(flagged)} clip(s) modified AFTER their earliest label -- verify before trusting:")
        for name, created, mtime in flagged:
            print(f"  - {name}: labeled {created.isoformat()}, file modified {mtime.isoformat()}")
        print("  Before trusting any validation result on these: draw the stored (x,y) for a labeled")
        print("  frame onto that exact frame in the CURRENT file and look at it. Do not blame the")
        print("  model or tracker off a validation failure against a flagged clip without doing that.")
    elif checked:
        print("No staleness flags among locally-checkable clips.")


def main():
    client = store.get_client()
    all_rows = _fetch_all_label_rows(client)

    # Rows whose notes start with "FLAGGED" are known-bad (see
    # extract_circled_ball's false-positive-lock issue, found 2026-07-28) —
    # excluded from the checkpoint count so it reflects usable data, not
    # just raw row count. Still counted separately below so the flagged
    # volume itself is visible, not silently dropped.
    rows = [r for r in all_rows if not (r.get("notes") or "").startswith("FLAGGED")]
    flagged_rows = [r for r in all_rows if r not in rows]

    total_frames = len(rows)
    distinct_clips = sorted(set(r["source_video_filename"] for r in rows))
    total_clips = len(distinct_clips)

    print(f"Labeled frames: {total_frames} (checkpoint: {FRAME_CHECKPOINT})")
    print(f"Distinct clips: {total_clips} (checkpoint: {CLIP_CHECKPOINT})")
    for name in distinct_clips:
        count = sum(1 for r in rows if r["source_video_filename"] == name)
        print(f"  - {name}: {count} frames")

    if flagged_rows:
        flagged_clips = sorted(set(r["source_video_filename"] for r in flagged_rows))
        print(f"\nExcluded as flagged/unreliable: {len(flagged_rows)} frames across {len(flagged_clips)} clip(s):")
        for name in flagged_clips:
            print(f"  - {name}")

    print()
    if total_clips >= CLIP_CHECKPOINT or total_frames >= FRAME_CHECKPOINT:
        print("CHECKPOINT REACHED — worth revisiting CVAT/Label Studio and a first "
              "YOLO fine-tune now that real data volume exists.")
    else:
        pct = max(total_clips / CLIP_CHECKPOINT, total_frames / FRAME_CHECKPOINT)
        print(f"Not yet at checkpoint (~{pct:.0%} of the way there by the closer metric) — "
              "keep collecting via the CapCut-circle + extract_circled_ball.py pipeline.")

    _report_staleness(rows)


if __name__ == "__main__":
    main()
