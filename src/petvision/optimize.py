"""Optimization: export to ONNX, quantize to INT8, and compare speed / size / accuracy.

Variants compared (same test images, same metric code):
  1. pytorch_fp32 : the trained checkpoint run in eager PyTorch
  2. onnx_fp32    : same weights exported to ONNX, run with onnxruntime (graph optimizations on)
  3. onnx_int8    : static post-training quantization (QDQ, per-channel weights) calibrated on train images

Example:
    python -m petvision.optimize --model weights/petvision_unet_mnv3.pt --out-dir weights --eval-images 1000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import get_cache, normalize, make_splits
from .evaluate import evaluate_segmenter
from .inference import Segmenter
from .model import load_checkpoint


def export_onnx(model, size, path: Path):
    dummy = torch.randn(1, 3, size, size)
    kw = dict(input_names=["image"], output_names=["logits"], opset_version=17,
              dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, dummy, str(path), **kw)
    return path


class CalibReader:
    """Feeds N normalized training images to the ORT static-quantization calibrator."""

    def __init__(self, data_root, size, n=128, input_name="image"):
        imgs, _, _ = get_cache(data_root, "train", size)
        idx = np.linspace(0, len(imgs) - 1, n).astype(int)
        self.batches = []
        for i in idx:
            t = torch.from_numpy(imgs[i]).permute(2, 0, 1).float() / 255.0
            self.batches.append({input_name: normalize(t)[None].numpy()})
        self.it = iter(self.batches)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.batches)


def quantize_int8(fp32_path: Path, int8_path: Path, data_root, size, n_calib=128):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = fp32_path.with_name(fp32_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(int8_path), CalibReader(data_root, size, n_calib),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"CalibPercentile": 99.99})
    pre.unlink(missing_ok=True)
    return int8_path


def benchmark(seg: Segmenter, size, runs=50, warmup=10, batch=1):
    x = np.random.randn(batch, 3, size, size).astype(np.float32)
    for _ in range(warmup):
        seg._logits(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        seg._logits(x)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000 / batch
    return {"latency_ms_mean": float(ts.mean()), "latency_ms_p50": float(np.median(ts)),
            "latency_ms_p95": float(np.percentile(ts, 95)), "throughput_img_s": float(1000 / ts.mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--results", default="results/optimization")
    ap.add_argument("--eval-images", type=int, default=1000)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args(argv)

    out_dir, res_dir = Path(args.out_dir), Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_checkpoint(args.model)
    size = cfg["size"]
    stem = Path(args.model).stem
    fp32 = export_onnx(model, size, out_dir / f"{stem}_fp32.onnx")
    int8 = quantize_int8(fp32, out_dir / f"{stem}_int8.onnx", args.data_root, size)
    for p in (fp32, int8):
        p.with_suffix(".json").write_text(json.dumps({"size": size, "source": str(args.model)}))

    variants = {"pytorch_fp32": Path(args.model), "onnx_fp32": fp32, "onnx_int8": int8}
    items = make_splits(args.data_root)["test"]
    step = max(1, len(items) // args.eval_images)
    items = items[::step][:args.eval_images]

    rows = []
    for name, path in variants.items():
        row = {"variant": name, "file": str(path), "size_mb": path.stat().st_size / 2**20}
        for th in args.threads:
            seg = Segmenter(path, threads=th)
            b = benchmark(seg, size)
            row |= {f"{k}_{th}t": v for k, v in b.items()}
        seg = Segmenter(path, threads=max(args.threads))
        m, _, _ = evaluate_segmenter(seg, args.data_root, items)
        row |= {"test_miou": m["miou"], "test_mdice": m["mdice"], "test_pixel_acc": m["pixel_acc"],
                "n_eval": m["n"]}
        print(json.dumps(row))
        rows.append(row)

    (res_dir / "optimization.json").write_text(json.dumps(rows, indent=1))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "optimization.csv", index=False)
    print(df.drop(columns=["file"]).to_string())


if __name__ == "__main__":
    main()
