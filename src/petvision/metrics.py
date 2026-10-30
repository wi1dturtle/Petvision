"""Segmentation metrics for binary masks.

IoU  = |P ∩ G| / |P ∪ G|
Dice = 2|P ∩ G| / (|P| + |G|)      (Dice = 2 IoU / (1 + IoU) per image)

We report two averages:
  * mean per-image IoU/Dice (every photo counts equally; what a user experiences)
  * dataset-level ("micro") IoU: sum of intersections / sum of unions over all pixels
"""
from __future__ import annotations

import numpy as np
import torch


def binary_stats(pred: torch.Tensor, target: torch.Tensor):
    """pred, target: bool/0-1 tensors of shape (B, H, W). Returns per-image inter, union, |P|, |G|."""
    pred = pred.bool().flatten(1)
    target = target.bool().flatten(1)
    inter = (pred & target).sum(1).double()
    union = (pred | target).sum(1).double()
    return inter, union, pred.sum(1).double(), target.sum(1).double()


def iou_dice_from_stats(inter, union, p, g, eps=1e-7):
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (p + g + eps)
    return iou, dice


class MetricTracker:
    def __init__(self):
        self.ious, self.dices, self.accs = [], [], []
        self.inter = self.union = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        inter, union, p, g = binary_stats(pred, target)
        iou, dice = iou_dice_from_stats(inter, union, p, g)
        acc = (pred.bool() == target.bool()).flatten(1).float().mean(1)
        self.ious += iou.tolist()
        self.dices += dice.tolist()
        self.accs += acc.tolist()
        self.inter += inter.sum().item()
        self.union += union.sum().item()
        return iou

    def compute(self):
        return {
            "miou": float(np.mean(self.ious)),
            "mdice": float(np.mean(self.dices)),
            "pixel_acc": float(np.mean(self.accs)),
            "micro_iou": float(self.inter / max(self.union, 1)),
            "n": len(self.ious),
        }


def boundary_band(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Pixels within `width` px of the mask contour (simple dilation - erosion, no scipy needed)."""
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * width + 1
    dil = torch.nn.functional.max_pool2d(m, k, 1, width)
    ero = -torch.nn.functional.max_pool2d(-m, k, 1, width)
    return ((dil - ero) > 0)[0, 0].numpy()
