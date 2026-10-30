"""Fast checks that do not need the dataset or the trained weights.   Run:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petvision.data import augment_batch, trimap_to_binary  # noqa: E402
from petvision.inference import cutout, overlay  # noqa: E402
from petvision.metrics import MetricTracker  # noqa: E402
from petvision.model import build_model  # noqa: E402


def test_trimap_mapping():
    tri = np.array([[1, 2, 3]], dtype=np.uint8)
    assert trimap_to_binary(tri).tolist() == [[1, 0, 1]]


def test_metrics_known_values():
    p = torch.tensor([[[1, 1, 0, 0]]])
    g = torch.tensor([[[1, 0, 0, 0]]])
    tr = MetricTracker()
    tr.update(p, g)
    m = tr.compute()
    assert abs(m["miou"] - 0.5) < 1e-6          # |P∩G|=1, |P∪G|=2
    assert abs(m["mdice"] - 2 / 3) < 1e-6       # 2*1 / (2+1)
    assert abs(m["pixel_acc"] - 0.75) < 1e-6


def test_model_shapes_and_aug():
    m = build_model(pretrained=False).eval()
    x = torch.randn(2, 3, 128, 128)
    y = (torch.rand(2, 1, 128, 128) > 0.5).float()
    xa, ya = augment_batch(x, y)
    assert xa.shape == x.shape and ya.shape == y.shape
    assert set(torch.unique(ya).tolist()) <= {0.0, 1.0}  # masks stay binary after nearest warping
    with torch.no_grad():
        assert m(x).shape == (2, 1, 128, 128)


def test_visual_outputs():
    im = Image.new("RGB", (40, 30), (200, 100, 50))
    prob = np.zeros((30, 40), dtype=np.float32)
    prob[10:20, 10:30] = 1.0
    assert cutout(im, prob).mode == "RGBA"
    assert np.asarray(cutout(im, prob))[15, 15, 3] == 255
    assert np.asarray(cutout(im, prob))[0, 0, 3] == 0
    assert overlay(im, prob > 0.5).size == im.size
