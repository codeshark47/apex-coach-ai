"""
ball_tracking/tracknet_adapter.py

Thin adapter so track_ball_from_seed.py and label_tool.py's AI pre-fill
can use BallTrackNetMini exactly like an ultralytics YOLO object --
both call sites only ever touch `.predict(crop, conf=..., verbose=...)`
-> `results[0].boxes` with `.xyxy[i]`, `.conf[i]`, `len(boxes)`. This
class duck-types that exact surface so NEITHER call site needs to know
or care which architecture it's holding.

USES_TEMPORAL_CONTEXT = True is the one marker track_ball_from_seed.py
checks (see that file's own small, additive change) to decide whether
to pass the last 2 real frames' crops alongside the current one -- an
ordinary ultralytics YOLO object has no such attribute, so that file's
behavior is completely unchanged when a YOLO checkpoint is in use.
"""

import glob
import os

import cv2
import numpy as np
import torch

from ball_tracking.training.tracknet_model import BallTrackNetMini, INPUT_SIZE

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training")


def _latest_checkpoint():
    """Same 'pick newest by mtime' convention already used for YOLO
    checkpoints (label_tool.py's _load_yolo_model, validate_holdout.py's
    _latest_checkpoint)."""
    candidates = glob.glob(os.path.join(TRAINING_DIR, "runs_tracknet", "*", "best.pt"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


class _FakeBoxes:
    """Duck-types ultralytics' Boxes object for exactly the 3 things
    every call site reads: .xyxy[i], .conf[i], len(boxes)."""

    def __init__(self, xyxy_list, conf_list):
        self.xyxy = [torch.tensor(b, dtype=torch.float32) for b in xyxy_list]
        self.conf = [torch.tensor(c, dtype=torch.float32) for c in conf_list]

    def __len__(self):
        return len(self.xyxy)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class BallTrackNetAdapter:
    USES_TEMPORAL_CONTEXT = True

    def __init__(self, checkpoint_path: str = None):
        checkpoint_path = checkpoint_path or _latest_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No BallTrackNetMini checkpoint found under "
                "ball_tracking/training/runs_tracknet/*/best.pt -- train one first "
                "(train_tracknet.py) or pass checkpoint_path explicitly."
            )
        self.checkpoint_path = checkpoint_path
        self.model = BallTrackNetMini()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()

    def _prep_gray(self, crop_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

    def predict(self, crop, conf: float = 0.02, verbose: bool = False, prev_crops: list = None):
        """
        crop: current-frame crop (BGR, same convention as the real YOLO
        call sites already use -- see track_ball_from_seed.py's own
        comment on why BGR matters here).
        prev_crops: [prev1_crop, prev2_crop] from the SAME field of view
        one and two frames back, or None/containing Nones when no
        tracked history exists yet (e.g. label_tool.py's whole-frame
        fallback with nothing to seed from) -- pads by duplicating the
        current crop, same convention as prepare_tracknet_dataset.py's
        own clip-start padding, so training and inference treat a
        missing-history frame identically.
        """
        crop_h, crop_w = crop.shape[:2]
        prev_crops = prev_crops or [None, None]
        prev1 = prev_crops[0] if prev_crops[0] is not None else crop
        prev2 = prev_crops[1] if len(prev_crops) > 1 and prev_crops[1] is not None else prev1

        stack = np.stack([
            self._prep_gray(crop),
            self._prep_gray(prev1),
            self._prep_gray(prev2),
        ], axis=0)  # 3 x INPUT_SIZE x INPUT_SIZE, matches training's CHW order
        tensor = torch.from_numpy(stack).float().unsqueeze(0) / 255.0  # 1x3xHxW

        with torch.no_grad():
            heatmap = self.model(tensor)[0, 0].numpy()  # INPUT_SIZE x INPUT_SIZE

        peak_val = float(heatmap.max())
        if peak_val < conf:
            return [_FakeResult(_FakeBoxes([], []))]

        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # Box size from the REAL spatial extent of the heatmap blob
        # above half-max around the peak, not a constant -- a constant
        # would silently turn track_ball_from_seed.py's size-trend-
        # consistency check into a no-op (size_velocity would compute to
        # ~0 every frame regardless of what's actually detected).
        above_half = heatmap >= (peak_val * 0.5)
        ys, xs = np.nonzero(above_half)
        blob_w = float(xs.max() - xs.min() + 1) if len(xs) else 4.0
        blob_h = float(ys.max() - ys.min() + 1) if len(ys) else 4.0

        # Map peak + blob size from INPUT_SIZE space back to the crop's
        # own real pixel space.
        scale_x = crop_w / INPUT_SIZE
        scale_y = crop_h / INPUT_SIZE
        cx = peak_x * scale_x
        cy = peak_y * scale_y
        half_w = (blob_w * scale_x) / 2.0
        half_h = (blob_h * scale_y) / 2.0

        box = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
        return [_FakeResult(_FakeBoxes([box], [peak_val]))]
