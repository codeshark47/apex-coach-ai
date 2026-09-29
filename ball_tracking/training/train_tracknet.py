"""
ball_tracking/training/train_tracknet.py

Trains BallTrackNetMini (tracknet_model.py) on the multi-frame heatmap
dataset built by prepare_tracknet_dataset.py. Plain PyTorch loop --
torch/torchvision are already installed via ultralytics's own
dependency chain (confirmed directly: torch==2.13.0+cpu,
torchvision==0.28.0+cpu), no new dependency needed.

CPU-ONLY, same constraint as train_yolo.py (device="cpu" there) --
this is exactly why the model itself is kept small (see
tracknet_model.py's own docstring: ~489K params vs YOLOv8n's ~3.2M).
Per the project plan: measure real per-epoch time on the first few
epochs before committing to a full run, don't assume it'll be fast
just because the model is smaller.

WARM-START, same convention as train_yolo.py's own "compounds instead
of restarting" design (real coach ask, 2026-08-14): loads the most
recently trained runs_tracknet/*/best.pt if one exists, so a later
labeling round doesn't throw away everything a previous round already
learned. Falls back to random init only on a genuinely fresh run.

Usage:
    python ball_tracking/training/train_tracknet.py [--epochs N] [--batch-size N]
"""

import argparse
import csv
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tracknet_model import BallTrackNetMini, INPUT_SIZE

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(TRAINING_DIR, "tracknet_dataset")
RUNS_ROOT = os.path.join(TRAINING_DIR, "runs_tracknet")


class BallHeatmapDataset(Dataset):
    """Reads the .npy frame-stacks + heatmap targets prepare_tracknet_dataset.py
    wrote, for one split (train/val), via its manifest.csv."""

    def __init__(self, split: str):
        manifest_path = os.path.join(DATASET_ROOT, "manifest.csv")
        self.examples = []
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.examples.append(row["example_name"])
        self.split = split

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        name = self.examples[idx]
        stack = np.load(os.path.join(DATASET_ROOT, self.split, "frames", f"{name}.npy"))  # H x W x 3, uint8
        heatmap = np.load(os.path.join(DATASET_ROOT, self.split, "heatmaps", f"{name}.npy"))  # H x W, float32

        # HWC uint8 -> CHW float32 [0,1] -- standard normalization, matches
        # what tracknet_adapter.py's inference path must also do (same
        # preprocessing on both sides, checked directly when the adapter
        # is written, not assumed to match).
        stack_t = torch.from_numpy(stack).permute(2, 0, 1).float() / 255.0
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).float()  # 1 x H x W
        return stack_t, heatmap_t


def _latest_checkpoint():
    candidates = glob.glob(os.path.join(RUNS_ROOT, "*", "best.pt"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--name", type=str, default=None, help="Run name under runs_tracknet/ (default: auto-numbered).")
    args = parser.parse_args()

    train_ds = BallHeatmapDataset("train")
    val_ds = BallHeatmapDataset("val")
    print(f"Train examples: {len(train_ds)}, val examples: {len(val_ds)}")
    if len(val_ds) == 0:
        raise SystemExit("No validation examples -- run prepare_tracknet_dataset.py first "
                          "and confirm VAL_CLIPS has labeled data.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = BallTrackNetMini().to(device)

    start_from = _latest_checkpoint()
    if start_from:
        print(f"Warm-starting from: {start_from}")
        model.load_state_dict(torch.load(start_from, map_location=device))
    else:
        print("No prior checkpoint found -- starting from random init.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    run_name = args.name or f"tracknet_v{len(glob.glob(os.path.join(RUNS_ROOT, 'tracknet_v*'))) + 1}"
    run_dir = os.path.join(RUNS_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")

    best_val_loss = float("inf")
    patience = 15
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        for stacks, heatmaps in train_loader:
            stacks, heatmaps = stacks.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            preds = model(stacks)
            loss = criterion(preds, heatmaps)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * stacks.size(0)
        train_loss = train_loss_sum / max(1, len(train_ds))

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for stacks, heatmaps in val_loader:
                stacks, heatmaps = stacks.to(device), heatmaps.to(device)
                preds = model(stacks)
                val_loss_sum += criterion(preds, heatmaps).item() * stacks.size(0)
        val_loss = val_loss_sum / max(1, len(val_ds))

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  ({epoch_time:.1f}s)")

        torch.save(model.state_dict(), last_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"No improvement for {patience} epochs -- stopping early.")
                break

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
