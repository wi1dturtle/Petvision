"""Dataset loading, splitting, caching and augmentation.

Oxford-IIIT Pet trimaps use: 1 = pet, 2 = background, 3 = boundary (unclear).
We turn this into a binary mask: foreground = pet OR boundary.
(The boundary band is part of the animal outline, mostly fur edges.)
"""
from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRIMAP_PET, TRIMAP_BG, TRIMAP_BOUNDARY = 1, 2, 3


# ----------------------------------------------------------------------------
# Index / splits
# ----------------------------------------------------------------------------
def read_split_file(path: Path):
    """Return list of dicts: name, class_id (1..37), species (1=cat, 2=dog), breed."""
    items = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, cid, species, _ = line.split()
            breed = name.rsplit("_", 1)[0]
            items.append({"name": name, "class_id": int(cid),
                          "species": "cat" if species == "1" else "dog",
                          "breed": breed})
    return items


def make_splits(data_root: str | Path, val_frac: float = 0.1, seed: int = 42):
    """Official trainval -> train/val (stratified by breed); official test kept as test."""
    root = Path(data_root)
    trainval = read_split_file(root / "annotations" / "trainval.txt")
    test = read_split_file(root / "annotations" / "test.txt")
    by_breed = defaultdict(list)
    for it in trainval:
        by_breed[it["breed"]].append(it)
    rng = random.Random(seed)
    train, val = [], []
    for breed in sorted(by_breed):
        lst = by_breed[breed]
        rng.shuffle(lst)
        n_val = max(1, round(len(lst) * val_frac))
        val += lst[:n_val]
        train += lst[n_val:]
    return {"train": train, "val": val, "test": test}


def load_image(path) -> Image.Image:
    # some files are grayscale / RGBA / CMYK, so always convert
    return Image.open(path).convert("RGB")


def trimap_to_binary(trimap: np.ndarray) -> np.ndarray:
    return ((trimap == TRIMAP_PET) | (trimap == TRIMAP_BOUNDARY)).astype(np.uint8)


# ----------------------------------------------------------------------------
# Cache: resized uint8 arrays so CPU training does not decode JPEGs every epoch
# ----------------------------------------------------------------------------
def build_cache(data_root, split_items, size: int, out_path: Path):
    """Resize every image to size x size (bilinear) and trimap (nearest)."""
    root = Path(data_root)
    n = len(split_items)
    imgs = np.zeros((n, size, size, 3), dtype=np.uint8)
    masks = np.zeros((n, size, size), dtype=np.uint8)  # raw trimap values
    for i, it in enumerate(split_items):
        im = load_image(root / "images" / f"{it['name']}.jpg").resize((size, size), Image.BILINEAR)
        tm = Image.open(root / "annotations" / "trimaps" / f"{it['name']}.png").resize((size, size), Image.NEAREST)
        imgs[i] = np.asarray(im)
        masks[i] = np.asarray(tm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, images=imgs, trimaps=masks,
             names=np.array([it["name"] for it in split_items]))
    return out_path


def get_cache(data_root, split: str, size: int, splits=None):
    cache = Path(data_root) / "cache" / f"{split}_{size}.npz"
    if not cache.exists():
        splits = splits or make_splits(data_root)
        print(f"[cache] building {cache} ({len(splits[split])} images)")
        build_cache(data_root, splits[split], size, cache)
    d = np.load(cache)
    return d["images"], d["trimaps"], d["names"]


class CachedPetDataset(torch.utils.data.Dataset):
    """Returns (float image CHW normalized, float mask 1HW, name)."""

    def __init__(self, data_root, split, size):
        self.images, self.trimaps, self.names = get_cache(data_root, split, size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(trimap_to_binary(self.trimaps[i])).float()[None]
        return normalize(img), mask, str(self.names[i])


def normalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype).view(-1, 1, 1)
    return (img - mean) / std


# ----------------------------------------------------------------------------
# Batch-level augmentation (vectorised, cheap on CPU)
# ----------------------------------------------------------------------------
def augment_batch(x: torch.Tensor, y: torch.Tensor, p_flip=0.5, scale=(0.75, 1.0),
                  max_rot_deg=15.0, color=0.2):
    """x: normalized images B3HW, y: masks B1HW. Random flip, scale/translate/rotate, brightness/contrast."""
    b = x.shape[0]
    dev = x.device
    # random affine: zoom in (s<1 = crop), small rotation, translation inside valid range
    s = torch.empty(b, device=dev).uniform_(*scale)
    ang = torch.empty(b, device=dev).uniform_(-max_rot_deg, max_rot_deg) * math.pi / 180
    tx = (torch.rand(b, device=dev) * 2 - 1) * (1 - s)
    ty = (torch.rand(b, device=dev) * 2 - 1) * (1 - s)
    flip = torch.where(torch.rand(b, device=dev) < p_flip, -1.0, 1.0)
    cos, sin = torch.cos(ang) * s, torch.sin(ang) * s
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1),
                         torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y = F.grid_sample(y, grid, mode="nearest", padding_mode="reflection", align_corners=False)
    # brightness / contrast jitter (applied in normalized space, roughly equivalent)
    if color > 0:
        c = torch.empty(b, 1, 1, 1, device=dev).uniform_(1 - color, 1 + color)
        br = torch.empty(b, 1, 1, 1, device=dev).uniform_(-color, color)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * c + mean + br
    return x, y


def preprocess_pil(img: Image.Image, size: int) -> torch.Tensor:
    """PIL image -> normalized 1x3xSxS tensor (same squash-resize used in training)."""
    arr = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR))
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return normalize(t)[None]


def worker_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
