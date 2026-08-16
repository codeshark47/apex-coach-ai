"""
ball_tracking/validate_holdout.py

Evaluates track_ball_from_seed.py against real, human-labeled ground
truth for a clip that was deliberately NOT used to tune the tracker's
thresholds (see project memory for which clips WERE used: the primary
benchmark VID_20260411_092805.mp4, plus the "clip1/clip2/clip3" trio
from the 2026-08-16 tracker-fix session). Ground truth is pulled live
from `ball_tracking_labels` — never hardcoded — so this stays correct
as more frames get labeled.

This is a genuinely held-out check, not a re-run of the tuning clips:
its only job is to answer "do these thresholds generalize to a clip
they were never fit against," which is the real open question standing
between the current admin-only gate and any wider rollout (see the
"Still open, real next steps" / "How to apply" notes in the ball-
tracking project memory).

Usage:
    python ball_tracking/validate_holdout.py <video_path> [--model PATH] [--threshold PX]

If --model is omitted, auto-picks the newest training/runs/*/weights/best.pt
by modification time (same convention as label_tool.py's AI pre-fill).

The seed point is the FIRST real ground-truth frame found for this video
(by filename) rather than something hand-picked, so this script can be
re-pointed at any newly labeled held-out clip without editing code.
"""

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

import profile_store
from ball_tracking.track_ball_from_seed import track_ball_from_seed


def _latest_checkpoint() -> str:
    candidates = glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "runs", "*", "weights", "best.pt")
    )
    if not candidates:
        raise SystemExit(
            "No trained checkpoint found under ball_tracking/training/runs/*/weights/best.pt "
            "— pass --model explicitly."
        )
    return max(candidates, key=os.path.getmtime)


def _fetch_ground_truth(source_video_filename: str) -> dict:
    client = profile_store.get_client()
    rows = []
    start = 0
    page = 1000
    while True:
        resp = (
            client.table("ball_tracking_labels")
            .select("*")
            .eq("source_video_filename", source_video_filename)
            .range(start, start + page - 1)
            .execute()
        )
        batch = resp.data
        rows.extend(batch)
        if len(batch) < page:
            break
        start += page
    gt = {}
    for r in rows:
        x, y = r.get("ball_x_px"), r.get("ball_y_px")
        if x is not None and y is not None:
            gt[r["frame_index"]] = (float(x), float(y))
    return gt


def validate(video_path: str, model_path: str = None, threshold_px: float = 20.0,
             max_gap_frames: int = 40, seed_frame: int = None) -> dict:
    """
    Seeds from the first real ground-truth frame for this video's filename
    (or a specific one, via seed_frame, when a recording contains more
    than one labeled delivery and a particular segment needs testing) and
    tracks forward, then reports coverage + accuracy against every other
    real ground-truth frame reached. max_gap_frames caps how far past a
    real GT cluster's own span the tracker is allowed to run (avoids
    wasting time deep into a second, unrelated delivery later in the same
    recording — see the two-delivery clip this script was first built for).
    """
    filename = os.path.basename(video_path)
    gt = _fetch_ground_truth(filename)
    if not gt:
        raise SystemExit(f"No ground-truth labels found in ball_tracking_labels for '{filename}'.")

    frames = sorted(gt.keys())
    if seed_frame is None:
        seed_frame = frames[0]
    elif seed_frame not in gt:
        raise SystemExit(f"frame {seed_frame} has no ground-truth label for '{filename}'. "
                          f"Labeled frames: {frames}")
    seed_xy = gt[seed_frame]

    # Only track within max_gap_frames of the contiguous cluster the seed
    # belongs to — a recording with multiple separate deliveries (a real
    # gap of dozens of frames between labeled clusters) shouldn't have the
    # tracker wander through an unrelated, unlabeled stretch in between.
    # Walk forward from the seed's OWN position in the sorted frame list,
    # not from frames[0] — a non-default seed_frame can start mid-list.
    forward_frames = [f for f in frames if f >= seed_frame]
    cluster_end = seed_frame
    for f in forward_frames:
        if f - cluster_end > max_gap_frames:
            break
        cluster_end = f
    max_frames_forward = (cluster_end - seed_frame) + 10

    model_path = model_path or _latest_checkpoint()
    from ultralytics import YOLO
    model = YOLO(model_path)

    result = track_ball_from_seed(video_path, seed_frame, seed_xy, model,
                                   max_frames_forward=max_frames_forward)
    if result["status"] != "success":
        raise SystemExit(f"Tracking failed: {result.get('message')}")

    points = result["points"]
    by_frame = {p[0]: (p[1], p[2]) for p in points}
    last_frame = points[-1][0]

    checked = 0
    within_threshold = 0
    misses = []
    for f in frames:
        if f <= seed_frame or f > cluster_end:
            continue
        if f in by_frame:
            tx, ty = by_frame[f]
            gx, gy = gt[f]
            dist = math.hypot(tx - gx, ty - gy)
            checked += 1
            if dist < threshold_px:
                within_threshold += 1
            else:
                misses.append((f, round(dist, 1)))

    return {
        "video": filename,
        "model": model_path,
        "seed_frame": seed_frame,
        "seed_xy": seed_xy,
        "cluster_end": cluster_end,
        "reached_frame": last_frame,
        "span": last_frame - seed_frame,
        "checked": checked,
        "within_threshold": within_threshold,
        "threshold_px": threshold_px,
        "misses": misses,
        "points": points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path")
    parser.add_argument("--model", default=None, help="Path to a YOLO checkpoint; defaults to the newest trained run.")
    parser.add_argument("--threshold", type=float, default=20.0, help="Pixel distance counted as a hit (default 20).")
    parser.add_argument("--seed-frame", type=int, default=None,
                         help="Use a specific labeled frame as the seed, for recordings with more than one "
                              "labeled delivery (default: the first labeled frame found).")
    args = parser.parse_args()

    r = validate(args.video_path, model_path=args.model, threshold_px=args.threshold, seed_frame=args.seed_frame)
    print(f"video: {r['video']}")
    print(f"model: {r['model']}")
    print(f"seed: frame={r['seed_frame']} xy={r['seed_xy']}")
    print(f"labeled cluster spans up to frame {r['cluster_end']}")
    print(f"tracker reached frame {r['reached_frame']} (span={r['span']})")
    print(f"ground-truth check: {r['within_threshold']}/{r['checked']} within {r['threshold_px']}px")
    if r["misses"]:
        print(f"misses (frame, px_error): {r['misses']}")
