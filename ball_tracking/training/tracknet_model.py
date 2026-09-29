"""
ball_tracking/training/tracknet_model.py

BallTrackNetMini: a small U-Net-style encoder-decoder that predicts a
per-pixel heatmap of the ball's position from 3 stacked consecutive
grayscale frames (current, N-1, N-2) -- see the project plan for the
full reasoning (2-month YOLO whack-a-mole history, why a scaled-down
TrackNet-style model rather than published TrackNetV2 or a
stacked-channel YOLO variant).

SINGLE SOURCE OF TRUTH: both train_tracknet.py and tracknet_adapter.py
import THIS class -- never redefine the architecture in either place.
A drifted second definition (even a seemingly-trivial one) would load
mismatched weights silently, the same class of bug this project has
already been burned by more than once (BGR/RGB, compression regime).

Much shallower than TrackNetV2's own VGG16-scale encoder, deliberately
-- this model only ever needs to answer "where's the ball in this
small crop," not "where in a full broadcast frame," since the seeded
tracker's own physics-based local search already solves the second
problem. Grayscale input (not RGB) -- color hasn't been the
differentiator in 2 months of YOLO fixes; stacking grayscale frames is
what actually encodes motion, at 1/3 the channel cost of stacking RGB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 320  # must match prepare_tracknet_dataset.py's own INPUT_SIZE


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BallTrackNetMini(nn.Module):
    """
    Input:  (B, 3, 320, 320) -- 3 stacked grayscale frames.
    Output: (B, 1, 320, 320) -- sigmoid heatmap, [0, 1].

    Encoder: 3 downsample stages (16 -> 32 -> 64 channels).
    Bottleneck: 128 channels.
    Decoder: bilinear upsample + skip concat, mirroring the encoder.
    """

    def __init__(self):
        super().__init__()
        self.enc1 = _conv_block(3, 16)
        self.enc2 = _conv_block(16, 32)
        self.enc3 = _conv_block(32, 64)
        self.bottleneck = _conv_block(64, 128)

        self.dec3 = _conv_block(128 + 64, 64)
        self.dec2 = _conv_block(64 + 32, 32)
        self.dec1 = _conv_block(32 + 16, 16)

        self.head = nn.Conv2d(16, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip_a = self.enc1(x)                       # 16 x 320 x 320
        x = self.pool(skip_a)                        # 16 x 160 x 160
        skip_b = self.enc2(x)                         # 32 x 160 x 160
        x = self.pool(skip_b)                          # 32 x 80 x 80
        skip_c = self.enc3(x)                           # 64 x 80 x 80
        x = self.pool(skip_c)                            # 64 x 40 x 40

        x = self.bottleneck(x)                            # 128 x 40 x 40

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 128x80x80
        x = self.dec3(torch.cat([x, skip_c], dim=1))       # 64 x 80 x 80

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 64x160x160
        x = self.dec2(torch.cat([x, skip_b], dim=1))       # 32 x 160 x 160

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 32x320x320
        x = self.dec1(torch.cat([x, skip_a], dim=1))       # 16 x 320 x 320

        return torch.sigmoid(self.head(x))                 # 1 x 320 x 320


if __name__ == "__main__":
    # Quick sanity check: real param count and a forward-pass shape
    # check -- see plan's own note not to trust the estimated ~0.5-1M
    # param count without measuring it.
    model = BallTrackNetMini()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BallTrackNetMini parameter count: {n_params:,}")
    dummy = torch.zeros(2, 3, INPUT_SIZE, INPUT_SIZE)
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)} (expected (2, 1, {INPUT_SIZE}, {INPUT_SIZE}))")
