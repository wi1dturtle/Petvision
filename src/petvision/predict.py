"""Run the model on your own images.

    python -m petvision.predict --model weights/petvision_unet_mnv3.pt --input my_cat.jpg --out outputs/
    python -m petvision.predict --model weights/petvision_unet_mnv3_int8.onnx --input photos_folder/ --out outputs/

For every image it writes:  <name>_mask.png, <name>_overlay.jpg, <name>_cutout.png (transparent background)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from .inference import Segmenter, cutout, mask_image, overlay

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/petvision_unet_mnv3.pt")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    inp, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in inp.iterdir() if p.suffix.lower() in EXTS) if inp.is_dir() else [inp]
    seg = Segmenter(args.model, tta=args.tta)
    for p in files:
        im = Image.open(p).convert("RGB")
        t0 = time.perf_counter()
        prob = seg.predict_proba([im])[0]
        dt = (time.perf_counter() - t0) * 1000
        mask = (prob > args.threshold).astype("uint8")
        mask_image(mask).save(out / f"{p.stem}_mask.png")
        overlay(im, mask).save(out / f"{p.stem}_overlay.jpg", quality=92)
        cutout(im, prob).save(out / f"{p.stem}_cutout.png")
        print(f"{p.name}: foreground {mask.mean() * 100:.1f}% of pixels, {dt:.0f} ms")


if __name__ == "__main__":
    main()
