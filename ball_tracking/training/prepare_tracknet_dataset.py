"""
ball_tracking/training/prepare_tracknet_dataset.py

Builds a MULTI-FRAME, HEATMAP-target dataset for BallTrackNet-mini (see
project plan/memory) from the SAME real human-clicked labels
prepare_dataset.py already uses — no relabeling needed. Every real
label already collected over the past 2 months feeds this new
architecture directly.

WHY A SEPARATE FORMAT, NOT A YOLO BOX LABEL: this model takes 3
consecutive frames (not 1) and predicts a position via a per-pixel
heatmap (not a box) — a structurally different training target, built
specifically so the model can see the ball's own motion trail across
frames, which a single static frame cannot show regardless of how many
more single-frame examples it's given (see plan Context section for
the full 2-month history this addresses).

Uses dataset_common.py for clip discovery / compression-caching /
train-val split — IDENTICAL clips and splits to prepare_dataset.py, so
the two architectures can be compared head-to-head later on a fair
footing (validate_holdout.py, both against the same held-out clips).

ONLY 'direct_click_v1' labels (see dataset_common.VALID_LABELED_BY) --
same trust boundary as the existing YOLO dataset; the abandoned
CapCut-circle workflow is excluded here too.

TRAIN/INFERENCE CROP MATCH (the one thing that would silently break
everything if skipped): at real inference time, the tracker's crop is
centered on a PREDICTED position with real error, not the exact true
ball position. Training on crops centered exactly on the true position
would teach the model an unrealistic assumption ("the ball is always
dead-center") that real tracking can never satisfy. Every training
crop here is deliberately OFFSET and RADIUS-JITTERED (see
_jittered_crop_box) to match that real-world distribution -- this
mirrors the exact class of mismatch already found and fixed twice in
this project (BGR/RGB in label_tool.py, compression regime in
prepare_dataset.py's own history) and is the single easiest thing for
a future edit to accidentally "simplify" away, so don't.

Usage:
    python ball_tracking/training/prepare_tracknet_dataset.py
"""

import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import profile_store as store
from ball_tracking.training.dataset_common import (
    VALID_LABELED_BY, VAL_CLIPS,
    fetch_all, find_video, normalized_video_path,
)

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracknet_dataset")
INPUT_SIZE = 320  # see plan Stage 0: verified the ball survives crop+resize at this size
HEATMAP_SIGMA = 2.5  # px, in OUTPUT (320x320) space

# Matches track_ball_from_seed.py's own real search-radius range
# (search_radius_start=250 up to search_radius_cap=400/recovery_radius=500)
# -- training crops must span the same range real inference crops do.
CROP_RADIUS_RANGE = (250.0, 400.0)
CENTER_JITTER_RANGE = (30.0, 100.0)  # px offset magnitude from the true position

random.seed(0)  # reproducible dataset across re-runs, same spirit as train_yolo.py's fixed seed


def _jittered_crop_box(cx: float, cy: float, frame_w: int, frame_h: int):
    """
    A crop box that does NOT center exactly on (cx, cy) -- see module
    docstring's TRAIN/INFERENCE CROP MATCH section for why. Returns
    (x1, y1, x2, y2, actual_center_x, actual_center_y) where
    actual_center_x/y is where (cx, cy) really is have. Clamped to the
    frame, same pattern as track_ball_from_seed.py's own crop clamping.
    """
    radius = random.uniform(*CROP_RADIUS_RANGE)
    angle = random.uniform(0, 2 * np.pi)
    jitter_mag = random.uniform(*CENTER_JITTER_RANGE)
    box_cx = cx + jitter_mag * np.cos(angle)
    box_cy = cy + jitter_mag * np.sin(angle)

    x1 = int(max(0, box_cx - radius))
    y1 = int(max(0, box_cy - radius))
    x2 = int(min(frame_w, box_cx + radius))
    y2 = int(min(frame_h, box_cy + radius))
    return x1, y1, x2, y2


def _random_crop_box(frame_w: int, frame_h: int):
    """Hard-negative fallback with no real anchor point at all (see
    main()'s own comment on why) -- still uses the same real radius
    range so the model sees the same scale of content either way."""
    radius = random.uniform(*CROP_RADIUS_RANGE)
    cx = random.uniform(radius, max(radius, frame_w - radius))
    cy = random.uniform(radius, max(radius, frame_h - radius))
    x1 = int(max(0, cx - radius))
    y1 = int(max(0, cy - radius))
    x2 = int(min(frame_w, cx + radius))
    y2 = int(min(frame_h, cy + radius))
    return x1, y1, x2, y2


def _make_heatmap(size: int, x: float, y: float, sigma: float) -> np.ndarray:
    """Small Gaussian blob centered on (x, y) in a size x size output
    -- standard heatmap-keypoint-regression target (stacked-hourglass-
    style pose estimation), not TrackNetV2's own weighted-BCE variant --
    starting simple per the plan; only revisit if training collapses to
    all-zero output."""
    yy, xx = np.mgrid[0:size, 0:size]
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def _crop_gray_resized(frame_bgr: np.ndarray, box, size: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def main():
    client = store.get_client()

    labels = fetch_all(
        client, "ball_tracking_labels", "source_video_filename,frame_index,ball_x_px,ball_y_px",
        filters={"labeled_by": VALID_LABELED_BY},
    )
    by_clip = {}
    for row in labels:
        by_clip.setdefault(row["source_video_filename"], []).append(row)

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_ROOT, split, "frames"), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, split, "heatmaps"), exist_ok=True)

    manifest_rows = []
    total_written = 0
    total_hard_neg = 0

    for filename, rows in by_clip.items():
        safe_filename = filename.encode("ascii", "replace").decode("ascii")
        split = "val" if filename in VAL_CLIPS else "train"
        video_path = find_video(filename)
        if video_path is None:
            print(f"SKIP (video file not found): {safe_filename}")
            continue

        orig_cap = cv2.VideoCapture(video_path)
        orig_frame_w = int(orig_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_frame_h = int(orig_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_cap.release()

        normalized_path = normalized_video_path(video_path, filename)
        cap = cv2.VideoCapture(normalized_path)

        rows_by_frame = {r["frame_index"]: r for r in rows}
        # Real anchor for hard-negative crops (see _random_crop_box's
        # docstring): a hard-negative row has NO position of its own
        # (the coach skips with just a category tag, no click) -- the
        # nearest REAL ball position in the same clip is a much better
        # guess at where the confusable object actually is than a
        # uniformly random crop, since hard negatives are typically
        # spotted while the coach is looking right at the ball's actual
        # flight path. Falls back to a random crop if this clip has no
        # real positions at all.
        real_positions_sorted = sorted(
            (r["frame_index"], r["ball_x_px"], r["ball_y_px"])
            for r in rows if r["ball_x_px"] is not None
        )

        def _nearest_real_position(frame_idx):
            if not real_positions_sorted:
                return None
            best = min(real_positions_sorted, key=lambda p: abs(p[0] - frame_idx))
            return best[1], best[2]

        clip_slug = "".join(c if c.isalnum() else "_" for c in filename.rsplit(".", 1)[0])

        # Rolling buffer of the last 2 decoded (compressed-regime)
        # frames -- read once per frame regardless of whether it's
        # labeled, exactly like prepare_dataset.py's own frame loop, so
        # a labeled frame N always has real N-1/N-2 pixels on hand with
        # no extra seeking.
        prev1, prev2 = None, None  # prev1 = frame N-1, prev2 = frame N-2
        idx = 0
        written_this_clip = 0
        scale_x = INPUT_SIZE  # placeholder, real scale computed per-crop below

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            row = rows_by_frame.get(idx)
            if row is not None:
                # Pad by duplicating the earliest available frame for a
                # labeled frame at/near clip start (index 0 or 1) -- one
                # place only, per module docstring.
                f_n1 = prev1 if prev1 is not None else frame
                f_n2 = prev2 if prev2 is not None else f_n1

                if row["ball_x_px"] is None:
                    anchor = _nearest_real_position(idx)
                    if anchor is not None:
                        box = _jittered_crop_box(anchor[0], anchor[1], orig_frame_w, orig_frame_h)
                    else:
                        box = _random_crop_box(orig_frame_w, orig_frame_h)
                    heatmap = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
                    total_hard_neg += 1
                else:
                    x, y = row["ball_x_px"], row["ball_y_px"]
                    box = _jittered_crop_box(x, y, orig_frame_w, orig_frame_h)
                    x1, y1, x2, y2 = box
                    # Map the true position into the RESIZED output
                    # space for the heatmap target -- must use the same
                    # box this frame's crop actually used, not the
                    # nominal jitter values, since clamping at frame
                    # edges can shrink the box unevenly (see
                    # track_ball_from_seed.py's own identical clamping
                    # pattern).
                    out_x = (x - x1) * INPUT_SIZE / (x2 - x1)
                    out_y = (y - y1) * INPUT_SIZE / (y2 - y1)
                    heatmap = _make_heatmap(INPUT_SIZE, out_x, out_y, HEATMAP_SIGMA)

                stack = np.stack([
                    _crop_gray_resized(frame, box, INPUT_SIZE),
                    _crop_gray_resized(f_n1, box, INPUT_SIZE),
                    _crop_gray_resized(f_n2, box, INPUT_SIZE),
                ], axis=-1)  # H x W x 3 (current, N-1, N-2)

                example_name = f"{clip_slug}_{idx}"
                np.save(os.path.join(OUT_ROOT, split, "frames", f"{example_name}.npy"), stack)
                np.save(os.path.join(OUT_ROOT, split, "heatmaps", f"{example_name}.npy"), heatmap)
                manifest_rows.append((example_name, split, row["ball_x_px"] is not None))
                written_this_clip += 1
                total_written += 1

            prev2 = prev1
            prev1 = frame
            idx += 1
        cap.release()
        print(f"{safe_filename[:50]:50} -> {split:5} | {written_this_clip} examples written")

    manifest_path = os.path.join(OUT_ROOT, "manifest.csv")
    with open(manifest_path, "w") as f:
        f.write("example_name,split,has_ball\n")
        for name, split, has_ball in manifest_rows:
            f.write(f"{name},{split},{int(has_ball)}\n")

    print(f"\nTotal examples written: {total_written} (of which {total_hard_neg} hard-negative)")
    print(f"Manifest: {manifest_path}")
    val_count = sum(1 for _, split, _ in manifest_rows if split == "val")
    if val_count == 0:
        print(
            "\nWARNING: no validation examples written (VAL_CLIPS is empty or its clip(s) "
            "have no labels yet) -- training will be meaningless without a val set."
        )


if __name__ == "__main__":
    main()
