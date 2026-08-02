"""
ball_tracking/training/prepare_dataset.py

Converts verified ball_tracking_labels rows into a YOLO-format dataset:
one image + one label .txt per labeled frame.

ONLY uses rows with labeled_by='direct_click_v1' (see
ball_tracking/label_tool.py) — the earlier CapCut-circle workflow
(labeled_by='auto_extracted_from_marker_v1') was abandoned 2026-08-02
after discovering it poisoned the training images: the coach's drawn
circle is burned directly into the source video's pixels, so every
training image had a bright, solid-colored, consistently-shaped ring
sitting on the ball. An inpainting attempt removed the ring's color but
left a locally-smoothed patch the model learned to detect instead
(confirmed directly: a trained checkpoint's predicted box sat exactly
on the inpainting scar, not the ball, and metrics were unchanged before
and after inpainting). Filtering to direct_click_v1 at the query level
means old, compromised rows can never silently leak back into training
just because they're still sitting in the table — there is no
per-filename exclusion list to remember to update.

Box source: ball_tracking_labels only stores a center (x, y) — the
matching radius comes from ball_tracking_runs.raw_candidates for the
same (clip, frame) — for direct_click_v1 rows, this is the radius the
coach set once per clip in label_tool.py's size-calibration step, not a
guessed constant.

SPLIT: by CLIP, not by random frame. Frames within one clip share the
same background/lighting, so a random-frame split would let the model
"see" near-duplicate scenes in both train and val and look like it
generalizes when it's really just memorizing.

VAL_CLIPS: was "night time.mp4" (2026-08-02), changed same day after the
first training run scored near-zero on it — the coach confirmed the
ball wasn't clearly visible in that footage even to a human, so a low
score there reflects an ambiguous/noisy clip, not a real generalization
gap. Not a fair validation set. night_time.mp4 moved into training
instead (still useful as extra variety); PXL_20260801_040327130.mp4 is
now VAL_CLIPS — filename suggests a different phone/camera than the
rest, the best proxy for "different conditions" available in this
batch. Revisit once a clip from a different physical location or a
clearly-visible different lighting condition gets labeled.

Usage:
    python ball_tracking/training/prepare_dataset.py
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import profile_store as store

VALID_LABELED_BY = "direct_click_v1"

# Set once a clip (ideally a different scene) has been labeled with the
# new tool — see module docstring.
VAL_CLIPS = {"PXL_20260801_040327130.mp4"}

SEARCH_DIRS = [
    "C:/Users/Shoaib/Downloads",
    "C:/Users/Shoaib/Downloads/for phase two",
]

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def _fetch_all(client, table, columns, filters=None):
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


def _find_video(filename):
    for d in SEARCH_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    client = store.get_client()

    labels = _fetch_all(
        client, "ball_tracking_labels", "source_video_filename,frame_index,ball_x_px,ball_y_px",
        filters={"labeled_by": VALID_LABELED_BY},
    )
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
    total_val_written = 0
    for filename, rows in by_clip.items():
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
                    if split == "val":
                        total_val_written += 1
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
    if total_val_written == 0:
        print(
            "\nWARNING: no validation images written (VAL_CLIPS is empty or its clip(s) "
            "have no labels yet). Training will fail/be meaningless without a val set — "
            "set VAL_CLIPS to at least one labeled clip before running train_yolo.py."
        )


if __name__ == "__main__":
    main()
