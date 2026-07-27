"""
ball_tracking/training/train_yolo.py

First experimental fine-tune of YOLOv8-nano (pretrained COCO weights) on
the real coach-circled ball dataset (see prepare_dataset.py). CPU-only —
no GPU available in this environment, so epoch count is kept modest for
a first pass rather than assuming GPU-scale training time.

This is explicitly a FIRST EXPERIMENT, not a production model: 14 clips
is real signal but not broad scene diversity yet (see the project memory
on ball-tracking strategy). The one held-out validation clip is a
different camera/bowler/net entirely — its results are the honest
answer to "does this generalize at all," not just "did it memorize
training frames."
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
