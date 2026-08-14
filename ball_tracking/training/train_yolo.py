"""
ball_tracking/training/train_yolo.py

Fine-tune of YOLOv8-nano on the direct-click dataset (see
prepare_dataset.py) — the CapCut-circled workflow this replaced
poisoned every training image with a burned-in ring; this dataset has
none of that. CPU-only — no GPU available in this environment, so
epoch count is kept modest rather than assuming GPU-scale training time.

WARM-START FROM THE PREVIOUS CHECKPOINT (2026-08-14, real coach ask:
"the AI should learn from human clicks"): every run used to start over
from generic COCO weights (yolov8n.pt), throwing away everything the
previous run already learned about what a cricket ball looks like —
each label→retrain cycle was independent, not cumulative. Now loads the
most recently trained checkpoint (by file modification time, same
selection logic label_tool.py already uses for its own AI pre-fill) and
fine-tunes FROM there when one exists, falling back to yolov8n.pt only
on a fresh checkout with no prior run at all. This is what actually
makes the label→retrain loop compound instead of restarting blind every
time, and should converge faster per cycle too, since it isn't
relearning "what is a ball" from zero each time — just refining it on
the newly labeled batch.

This is explicitly an EARLY-STAGE model, not a finished product: see the
project memory on ball-tracking strategy for the real, current accuracy
picture and known gaps (recall varies a lot by camera/device — see
ball_tracking_status.py and the dataset's VAL_CLIPS for the latest
honest numbers, not a number hardcoded here that will go stale).
"""

import glob
import os

from ultralytics import YOLO

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(TRAINING_DIR, "dataset", "dataset.yaml")


def _latest_checkpoint():
    """Most recently trained best.pt under runs/*/weights/, or None if
    this is a fresh checkout with no prior run yet."""
    candidates = glob.glob(os.path.join(TRAINING_DIR, "runs", "*", "weights", "best.pt"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main():
    start_from = _latest_checkpoint() or "yolov8n.pt"
    print(f"Starting from: {start_from}")
    model = YOLO(start_from)
    model.train(
        data=DATASET_YAML,
        epochs=40,
        imgsz=640,
        batch=8,
        patience=15,
        project=os.path.join(TRAINING_DIR, "runs"),
        name="ball_v1",
        device="cpu",
        workers=2,
        verbose=True,
    )


if __name__ == "__main__":
    main()
