"""Copy a few test images to assets/examples/ and render a demo figure
(input | mask | overlay | background removed) for the README / slides.

    python scripts/make_demo_figure.py --model weights/petvision_unet_mnv3_int8.onnx
"""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from petvision.inference import Segmenter, cutout, overlay  # noqa: E402

# all from the official TEST split (never seen in training)
DEFAULT = ["Bengal_196", "Sphynx_21", "beagle_199", "pug_197", "Ragdoll_20", "samoyed_196"]


def checker(h, w, s=16):
    yy, xx = np.mgrid[:h, :w]
    c = (((yy // s) + (xx // s)) % 2).astype(np.float32)
    return (200 + 40 * c)[..., None].repeat(3, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/petvision_unet_mnv3_int8.onnx")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--names", nargs="+", default=DEFAULT)
    ap.add_argument("--out", default="results/demo/demo_grid.png")
    args = ap.parse_args()

    ex_dir = ROOT / "assets" / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    seg = Segmenter(args.model)
    n = len(args.names)
    fig, axes = plt.subplots(n, 4, figsize=(12, 2.9 * n))
    for r, name in enumerate(args.names):
        src = Path(args.data_root) / "images" / f"{name}.jpg"
        im = Image.open(src).convert("RGB")
        im.thumbnail((640, 640))
        im.save(ex_dir / f"{name}.jpg", quality=90)
        prob = seg.predict_proba([im])[0]
        mask = (prob > 0.5).astype(np.uint8)
        cut = np.asarray(cutout(im, prob)).astype(np.float32)
        a = cut[..., 3:] / 255
        comp = cut[..., :3] * a + checker(*cut.shape[:2]) * (1 - a)
        for c, (img, t) in enumerate([(im, name), (mask, "predicted mask"), (overlay(im, mask), "overlay"),
                                      (comp.astype(np.uint8), "background removed")]):
            axes[r, c].imshow(img, cmap="gray" if c == 1 else None)
            axes[r, c].set_title(t, fontsize=9)
            axes[r, c].axis("off")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=90)
    print("saved", args.out)


if __name__ == "__main__":
    main()
