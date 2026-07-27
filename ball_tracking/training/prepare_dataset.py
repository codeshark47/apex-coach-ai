"""
ball_tracking/training/prepare_dataset.py

Converts the verified ball_tracking_labels rows (real coach-circled
positions, already filtered for detection-radius consistency) into a
YOLO-format dataset: one image + one label .txt per labeled frame.

Box source: ball_tracking_labels only stores a center (x, y) — the
matching radius comes from ball_tracking_runs.raw_candidates for the
same (clip, frame), which IS the coach's actual drawn circle size. A
box built from [x-r, y-r, x+r, y+r] is an approximation (the true ball
edges aren't the same as the marker circle's edges), but it's a real,
non-fabricated bound derived from the coach's own annotation, not a
guessed constant.

SPLIT: by CLIP, not by random frame. Frames within one clip share the
same background/lighting, so a random-frame split would let the model
"see" near-duplicate scenes in both train and val and look like it
generalizes when it's really just memorizing.

VAL CLIP CHOICE — changed 2026-07-28: originally held out the
third-party Instagram clip (9.26.48 PM) as the toughest generalization
test. Verified directly (loaded the first training checkpoint and
compared its predictions against the stored labels frame by frame) that
a long stretch of THAT clip's labels are wrong — extract_marker_positions
locked onto a different reddish object on the left side of that scene
for ~80 consecutive frames and never found the coach's actual circle,
and the size-consistency filter didn't catch it because the false match
happened to be a similar size. Near-zero validation metrics through
epoch 22 of the first training run were at least partly an artifact of
scoring against these wrong labels, not necessarily the model failing to
generalize. Swapped to 12.34.34 AM.mp4, which got an exhaustive
multi-checkpoint visual verification earlier and came back 100% correct
— a held-out clip is only an honest test if its own labels are trusted.
The Instagram clip still needs a manual relabel/exclusion pass before
it's safe to use for training OR validation; it's currently just left
out of both until that happens.

Usage:
    python ball_tracking/training/prepare_dataset.py
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import profile_store as store

VAL_CLIPS = {"WhatsApp Video 2026-07-27 at 12.34.34 AM.mp4"}

# Confirmed (2026-07-28) to have a long stretch of mislabeled frames — see
# the module docstring. Excluded from BOTH splits, not just moved out of
# val, so its wrong labels don't corrupt training either. Remove from
# here once it's been manually relabeled or the bad frame range dropped.
EXCLUDED_CLIPS = {"WhatsApp Video 2026-07-27 at 9.26.48 PM.mp4"}

SEARCH_DIRS = [
    "C:/Users/Shoaib/Downloads",
    "C:/Users/Shoaib/Downloads/for phase two",
]

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def _fetch_all(client, table, columns):
    rows = []
    start = 0
    page_size = 1000
    while True:
        result = client.table(table).select(columns).range(start, start + page_size - 1).execute()
        page = result.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


def _find_video(filename):
    for d in SEARCH_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    client = store.get_client()

    labels = _fetch_all(client, "ball_tracking_labels", "source_video_filename,frame_index,ball_x_px,ball_y_px")
    runs = _fetch_all(client, "ball_tracking_runs", "source_video_filename,raw_candidates")
    radius_lookup = {}
    for r in runs:
        radius_lookup[r["source_video_filename"]] = {
            int(k): v["radius_px"] for k, v in (r["raw_candidates"] or {}).items()
        }

    by_clip = {}
    for row in labels:
        by_clip.setdefault(row["source_video_filename"], []).append(row)

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)

    total_written = 0
    for filename, rows in by_clip.items():
        if filename in EXCLUDED_CLIPS:
            print(f"EXCLUDED (known bad labels, see docstring): {filename}")
            continue
        split = "val" if filename in VAL_CLIPS else "train"
        video_path = _find_video(filename)
        if video_path is None:
            print(f"SKIP (video file not found): {filename}")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rows_by_frame = {r["frame_index"]: r for r in rows}
        clip_slug = "".join(c if c.isalnum() else "_" for c in filename.rsplit(".", 1)[0])

        idx = 0
        written_this_clip = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            row = rows_by_frame.get(idx)
            if row is not None:
                radius = radius_lookup.get(filename, {}).get(idx)
                if radius is not None:
                    x, y = row["ball_x_px"], row["ball_y_px"]
                    box_w, box_h = radius * 2, radius * 2
                    x_center_n = x / frame_w
                    y_center_n = y / frame_h
                    w_n = box_w / frame_w
                    h_n = box_h / frame_h

                    img_name = f"{clip_slug}_{idx}.jpg"
                    cv2.imwrite(os.path.join(OUT_ROOT, "images", split, img_name), frame)
                    with open(os.path.join(OUT_ROOT, "labels", split, img_name.replace(".jpg", ".txt")), "w") as f:
                        f.write(f"0 {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}\n")
                    written_this_clip += 1
                    total_written += 1
            idx += 1
        cap.release()
        print(f"{filename[:50]:50} -> {split:5} | {written_this_clip} frames written")

    yaml_path = os.path.join(OUT_ROOT, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {OUT_ROOT}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: ball\n"
        )

    print(f"\nTotal images written: {total_written}")
    print(f"Dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
