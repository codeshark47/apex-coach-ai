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

HARD NEGATIVES: a row with ball_x_px/ball_y_px = NULL is a confirmed
"no ball, but a specific lookalike (glove/pad/helmet) is here" frame —
see label_tool.py's hard-negative picker, added 2026-08-04 to target the
false-positive pattern seen in ball_v1-6/v1-7 (model latching onto
gloves). These need no radius (there's no box to draw) and are written
as plain background images with NO matching .txt label file — the YOLO
format for "this image has zero objects," which is what teaches the
model these lookalikes are confirmed not-a-ball rather than merely
unseen.

TRAIN/INFERENCE CONSISTENCY (2026-08-15, real coach-driven finding):
source videos here were read at whatever native format they happened to
be in, while every live coach upload always gets forced through
orchestrator.compress_video_file (resolution capped, re-encoded to
H.264) before the app ever looks at it. Confirmed directly via ffprobe
on two of the coach's own clips: a raw native .MOV capture (HEVC,
1920x1080, ~8Mbps) and the SAME footage after WhatsApp's own
compression (H.264, roughly a third the resolution, a third the
bitrate) are measurably, drastically different files — independent of
which physical phone captured either one. Training on the raw regime
while every real inference happens on the compressed regime is a real
train/inference mismatch, not a device problem — this is what actually
explained a big chunk of the "iPhone clips score worse" pattern from
the 2026-08-14 device-bucket analysis, corrected here. Every source
video now gets run through the SAME compressor before frames are
extracted for training, cached under _compressed_cache/ so repeated
runs don't re-encode unchanged files. Label coordinates were captured
in the ORIGINAL video's pixel space (label_tool.py reads frames at
native resolution) — normalizing by the ORIGINAL frame's own width/
height (not the compressed video's) keeps the resulting fraction
correct regardless of resolution, since aspect ratio is preserved
end to end.

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
# EXTRACTED 2026-09-29 into dataset_common.py, so prepare_tracknet_dataset.py
# (a new, second dataset format for a motion-aware detector — see project
# memory) is built from IDENTICAL clips/splits, not a silently-diverged
# copy. Pure extraction, no behavior change — verified by confirming
# identical total_written/total_val_written counts before and after.
from ball_tracking.training.dataset_common import (
    VALID_LABELED_BY, VAL_CLIPS, SEARCH_DIRS,
    fetch_all as _fetch_all, find_video as _find_video,
    normalized_video_path as _normalized_video_path,
)

# ADDED IMG_3082.MOV (2026-08-14): the previous single-clip, 12-image
# validation set was too small to trust a metric swing of even one
# frame (each miss/hit moved recall by 8+ points) — v1-7's own log
# metrics bounced around based on essentially a coin flip's worth of
# samples. IMG_3082.MOV is a genuinely different device (iPhone .MOV vs
# the Pixel PXL_ clip already here), giving 46 total val images instead
# of 12 — same "different conditions" reasoning as the original pick.
# (VAL_CLIPS itself now lives in dataset_common.py — this comment stays
# here since it's the reasoning behind the actual clip choice, not the
# mechanics of the split.)

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


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
    total_hard_neg = 0
    for filename, rows in by_clip.items():
        # Windows console can't print some real filenames (emoji, etc.)
        # -- see dataset_common.normalized_video_path's own comment on
        # this exact crash, found 2026-09-29.
        safe_filename = filename.encode("ascii", "replace").decode("ascii")
        split = "val" if filename in VAL_CLIPS else "train"
        video_path = _find_video(filename)
        if video_path is None:
            print(f"SKIP (video file not found): {safe_filename}")
            continue

        # Label coordinates were captured against THIS (original) video's
        # own pixel dimensions — label_tool.py reads frames at native
        # resolution, before any compression. Must be probed BEFORE
        # swapping to the normalized copy below, or the fraction math
        # further down would silently use the wrong denominator.
        orig_cap = cv2.VideoCapture(video_path)
        orig_frame_w = int(orig_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_frame_h = int(orig_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_cap.release()

        # TRAIN/INFERENCE CONSISTENCY — see module docstring. The actual
        # training IMAGES come from the compressed copy, matching what a
        # live coach upload always looks like by the time the app (or
        # this same pipeline's own labeling tool, at click time) sees
        # it; only the LABEL fraction still needs the original's own
        # dimensions (see above).
        normalized_path = _normalized_video_path(video_path, filename)
        cap = cv2.VideoCapture(normalized_path)

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
                if row["ball_x_px"] is None:
                    # Confirmed hard negative (see module docstring) —
                    # background image, no label file.
                    img_name = f"{clip_slug}_{idx}.jpg"
                    cv2.imwrite(os.path.join(OUT_ROOT, "images", split, img_name), frame)
                    written_this_clip += 1
                    total_written += 1
                    total_hard_neg += 1
                    if split == "val":
                        total_val_written += 1
                else:
                    radius = radius_lookup.get(filename, {}).get(idx)
                    if radius is not None:
                        x, y = row["ball_x_px"], row["ball_y_px"]
                        box_w, box_h = radius * 2, radius * 2
                        # Normalized against the ORIGINAL video's own
                        # dimensions (not the compressed copy cap read
                        # from) — see the orig_frame_w/h comment above.
                        # Aspect ratio is preserved end to end, so this
                        # fraction stays correct regardless of the
                        # training image's actual resolution.
                        x_center_n = x / orig_frame_w
                        y_center_n = y / orig_frame_h
                        w_n = box_w / orig_frame_w
                        h_n = box_h / orig_frame_h

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
        print(f"{safe_filename[:50]:50} -> {split:5} | {written_this_clip} frames written")

    yaml_path = os.path.join(OUT_ROOT, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {OUT_ROOT}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: ball\n"
        )

    print(f"\nTotal images written: {total_written} (of which {total_hard_neg} hard-negative/background)")
    print(f"Dataset config: {yaml_path}")
    if total_val_written == 0:
        print(
            "\nWARNING: no validation images written (VAL_CLIPS is empty or its clip(s) "
            "have no labels yet). Training will fail/be meaningless without a val set — "
            "set VAL_CLIPS to at least one labeled clip before running train_yolo.py."
        )


if __name__ == "__main__":
    main()
