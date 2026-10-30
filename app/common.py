"""Shared model loading for the API and the Gradio demo."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from petvision.inference import Segmenter  # noqa: E402

# First existing file wins. Override with:  PETVISION_MODEL=path/to/model.(pt|onnx)
# Default = ONNX FP32: same accuracy as PyTorch, ~2.5x faster on CPU (see REPORT.md, section 6).
# INT8 is 3.6x smaller but loses ~1.3 IoU points, so it is the choice only when size matters.
CANDIDATES = [
    os.environ.get("PETVISION_MODEL", ""),
    ROOT / "weights" / "petvision_unet_mnv3_fp32.onnx",
    ROOT / "weights" / "petvision_unet_mnv3_int8.onnx",
    ROOT / "weights" / "petvision_unet_mnv3.pt",
]


def load_segmenter() -> Segmenter:
    for c in CANDIDATES:
        if c and Path(c).exists():
            print(f"[petvision] loading model {c}")
            return Segmenter(c)
    raise FileNotFoundError("No model found in weights/. See README section 'Getting the trained model'.")
