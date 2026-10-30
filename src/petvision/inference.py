"""Inference wrapper used by the CLI, the API and the Gradio demo.

Two backends with the same interface:
  * "torch": the PyTorch checkpoint (weights/petvision_unet_mnv3.pt)
  * "onnx" : the exported / quantized ONNX model run by onnxruntime
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import IMAGENET_MEAN, IMAGENET_STD


def _to_input(images: list[Image.Image], size: int) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = np.stack([np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
                    for im in images])
    arr = (arr - mean) / std
    return arr.transpose(0, 3, 1, 2).copy()  # NCHW


class Segmenter:
    def __init__(self, model_path: str | Path, backend: str | None = None, size: int | None = None,
                 threads: int | None = None, tta: bool = False):
        model_path = Path(model_path)
        self.backend = backend or ("onnx" if model_path.suffix == ".onnx" else "torch")
        self.tta = tta
        if self.backend == "torch":
            from .model import load_checkpoint
            if threads:
                torch.set_num_threads(threads)
            self.model, cfg = load_checkpoint(model_path)
            self.size = size or cfg["size"]
        else:
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            meta_path = model_path.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            self.size = size or meta.get("size") or self.sess.get_inputs()[0].shape[-1]

    # -- raw logits at model resolution -------------------------------------------------
    def _logits(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            with torch.no_grad():
                t = torch.from_numpy(x)
                out = self.model(t)
                if self.tta:
                    out = 0.5 * (out + torch.flip(self.model(torch.flip(t, [3])), [3]))
                return out.numpy()
        out = self.sess.run(None, {self.input_name: x})[0]
        if self.tta:
            out_f = self.sess.run(None, {self.input_name: x[..., ::-1].copy()})[0][..., ::-1]
            out = 0.5 * (out + out_f)
        return out

    def predict_proba(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Foreground probability maps (H x W float32) at each image's ORIGINAL resolution."""
        x = _to_input(images, self.size)
        logits = torch.from_numpy(np.ascontiguousarray(self._logits(x)))
        probs = []
        for i, im in enumerate(images):
            w, h = im.size
            up = F.interpolate(logits[i:i + 1], size=(h, w), mode="bilinear", align_corners=False)
            probs.append(torch.sigmoid(up)[0, 0].numpy())
        return probs

    def predict_mask(self, image: Image.Image, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba([image])[0] > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------------------
# Visual outputs
# ---------------------------------------------------------------------------------------
def overlay(image: Image.Image, mask: np.ndarray, color=(255, 64, 160), alpha=0.5) -> Image.Image:
    img = np.asarray(image.convert("RGB")).astype(np.float32)
    m = mask.astype(bool)
    img[m] = img[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(img.astype(np.uint8))


def cutout(image: Image.Image, prob: np.ndarray, soft: bool = True, threshold: float = 0.5) -> Image.Image:
    """RGBA image with the background made transparent. Soft = use probability as alpha (smoother fur edges)."""
    rgba = image.convert("RGBA")
    if soft:
        a = np.clip((prob - 0.3) / 0.4, 0, 1)  # map [0.3, 0.7] -> [0, 1], crisp but not jagged
    else:
        a = (prob > threshold).astype(np.float32)
    arr = np.asarray(rgba).copy()
    arr[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask * 255).astype(np.uint8), "L")


def replace_background(image: Image.Image, prob: np.ndarray, bg_color=(30, 144, 255)) -> Image.Image:
    a = np.clip((prob - 0.3) / 0.4, 0, 1)[..., None]
    img = np.asarray(image.convert("RGB")).astype(np.float32)
    bg = np.ones_like(img) * np.array(bg_color, dtype=np.float32)
    return Image.fromarray((img * a + bg * (1 - a)).astype(np.uint8))
