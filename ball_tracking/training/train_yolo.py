"""
ball_tracking/training/train_yolo.py

First real fine-tune of YOLOv8-nano (pretrained COCO weights) on the
direct-click dataset (see prepare_dataset.py) — the CapCut-circled
workflow this replaced poisoned every training image with a burned-in
ring; this dataset has none of that. CPU-only — no GPU available in
this environment, so epoch count is kept modest for a first pass rather
than assuming GPU-scale training time.

This is explicitly a FIRST EXPERIMENT, not a production model: 16 clips
/ 336 frames (2026-08-02) is real signal but not broad scene diversity
yet (see the project memory on ball-tracking strategy). First run held
out "night time.mp4" and scored near-zero on it — turned out the ball
wasn't clearly visible in that footage even to the coach labeling it,
so that was a noisy/unfair test, not a real finding about night
performance. That clip moved into training; the held-out clip is now
PXL_20260801_040327130.mp4 (a different phone/camera from the rest,
the best available proxy for "different conditions" in this batch).
"""

import os

from ultralytics import YOLO

DATASET_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "dataset.yaml")


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATASET_YAML,
        epochs=40,
        imgsz=640,
        batch=8,
        patience=15,
        project=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
        name="ball_v1",
        device="cpu",
        workers=2,
        verbose=True,
    )


if __name__ == "__main__":
    main()
