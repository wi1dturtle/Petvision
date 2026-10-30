"""Evaluate a model on the official Oxford-IIIT Pet TEST split, at original image resolution.

Outputs (in --out):
  metrics.json          overall + per-species metrics
  per_image.csv         IoU / Dice for every test image (for error analysis)
  per_breed.csv         mean IoU per breed
  worst_cases.png       the lowest-IoU predictions
  best_cases.png        examples of good predictions

Example:
    python -m petvision.evaluate --model weights/petvision_unet_mnv3.pt --out results/eval_final
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .data import TRIMAP_BOUNDARY, load_image, make_splits, trimap_to_binary
from .inference import Segmenter
from .metrics import MetricTracker, boundary_band


def evaluate_segmenter(seg: Segmenter, data_root, items, batch_size=16, threshold=0.5, keep_preds=False):
    root = Path(data_root)
    tr_all, tr_core = MetricTracker(), MetricTracker()
    rows, preds = [], {}
    t_model = 0.0
    for b in range(0, len(items), batch_size):
        chunk = items[b:b + batch_size]
        ims = [load_image(root / "images" / f"{it['name']}.jpg") for it in chunk]
        t0 = time.perf_counter()
        probs = seg.predict_proba(ims)
        t_model += time.perf_counter() - t0
        for it, im, prob in zip(chunk, ims, probs):
            tri = np.asarray(Image.open(root / "annotations" / "trimaps" / f"{it['name']}.png"))
            gt = trimap_to_binary(tri)
            pred = (prob > threshold).astype(np.uint8)
            iou = tr_all.update(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None]).item()
            # "core" metric: ignore pixels the annotators marked as uncertain boundary
            keep = tri != TRIMAP_BOUNDARY
            tr_core.update(torch.from_numpy(pred * keep)[None], torch.from_numpy(gt * keep)[None])
            band = boundary_band(gt, width=5)
            err = pred != gt
            rows.append({"name": it["name"], "breed": it["breed"], "species": it["species"],
                         "iou": iou, "dice": tr_all.dices[-1],
                         "fg_frac": float(gt.mean()), "pred_fg_frac": float(pred.mean()),
                         "err_frac": float(err.mean()),
                         "err_in_boundary_band": float((err & band).sum() / max(err.sum(), 1)),
                         "h": im.size[1], "w": im.size[0]})
            if keep_preds:
                preds[it["name"]] = pred
    df = pd.DataFrame(rows)
    res = tr_all.compute()
    res["miou_ignore_boundary"] = tr_core.compute()["miou"]
    res["model_time_per_img_ms"] = 1000 * t_model / len(items)
    for sp in ("cat", "dog"):
        d = df[df.species == sp]
        res[f"miou_{sp}"] = float(d.iou.mean())
        res[f"mdice_{sp}"] = float(d.dice.mean())
    res["frac_images_iou_below_0.8"] = float((df.iou < 0.8).mean())
    res["frac_images_iou_below_0.5"] = float((df.iou < 0.5).mean())
    res["mean_err_share_in_boundary_band"] = float(df.err_in_boundary_band.mean())
    return res, df, preds


def case_grid(df, preds, data_root, out_path, title, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(data_root)
    n = min(n, len(df))
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.4 * n))
    for r, (_, row) in enumerate(df.head(n).iterrows()):
        im = load_image(root / "images" / f"{row['name']}.jpg")
        gt = trimap_to_binary(np.asarray(Image.open(root / "annotations" / "trimaps" / f"{row['name']}.png")))
        pr = preds[row["name"]]
        for c, (img, t) in enumerate([(im, row["name"]), (gt, "ground truth"),
                                      (pr, f"pred IoU={row['iou']:.2f}")]):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray" if c else None)
            ax.set_title(t, fontsize=8)
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt checkpoint or .onnx file")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max-images", type=int, default=0, help="evaluate a fixed subset (0 = all)")
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = make_splits(args.data_root)[args.split]
    if args.max_images:
        # fixed, breed-balanced subset: every k-th image of the sorted list
        step = max(1, len(items) // args.max_images)
        items = items[::step][:args.max_images]
    seg = Segmenter(args.model, tta=args.tta)
    res, df, preds = evaluate_segmenter(seg, args.data_root, items, threshold=args.threshold,
                                        keep_preds=not args.no_figures)
    res |= {"model": str(args.model), "split": args.split, "tta": args.tta, "input_size": seg.size}
    print(json.dumps(res, indent=1))
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    df.sort_values("iou").to_csv(out / "per_image.csv", index=False)
    df.groupby(["species", "breed"]).agg(miou=("iou", "mean"), mdice=("dice", "mean"), n=("iou", "size")) \
        .sort_values("miou").to_csv(out / "per_breed.csv")
    if not args.no_figures:
        d = df.sort_values("iou")
        case_grid(d, preds, args.data_root, out / "worst_cases.png", "Lowest-IoU test images")
        case_grid(d.iloc[len(d) // 2::max(1, len(d) // 16)], preds, args.data_root, out / "typical_cases.png",
                  "Typical test images (around the median IoU and above)")


if __name__ == "__main__":
    main()
