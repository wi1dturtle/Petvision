"""INT8 quantization ablation: which calibration setting loses the least accuracy?

    python scripts/quant_ablation.py --fp32 weights/petvision_unet_mnv3_fp32.onnx --eval-images 1000
"""
import argparse
import json
import sys
from pathlib import Path

import onnx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from petvision.data import make_splits  # noqa: E402
from petvision.evaluate import evaluate_segmenter  # noqa: E402
from petvision.inference import Segmenter  # noqa: E402
from petvision.optimize import CalibReader, benchmark  # noqa: E402


def main():
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="weights/petvision_unet_mnv3_fp32.onnx")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--eval-images", type=int, default=1000)
    ap.add_argument("--out", default="results/optimization")
    args = ap.parse_args()
    out = Path(args.out)
    tmp = out / "quant_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    pre = tmp / "pre.onnx"
    quant_pre_process(args.fp32, str(pre), skip_symbolic_shape=True)

    # nodes of the segmentation head + last decoder block (most sensitive: they produce the fine mask)
    nodes = [n.name for n in onnx.load(str(pre)).graph.node if n.op_type == "Conv"]
    tail = nodes[-3:]

    settings = {
        "minmax": dict(calibrate_method=CalibrationMethod.MinMax),
        "percentile_99.99": dict(calibrate_method=CalibrationMethod.Percentile,
                                 extra_options={"CalibPercentile": 99.99}),
        "entropy": dict(calibrate_method=CalibrationMethod.Entropy),
        "percentile_99.99_keep_last3_fp32": dict(calibrate_method=CalibrationMethod.Percentile,
                                                 extra_options={"CalibPercentile": 99.99},
                                                 nodes_to_exclude=tail),
    }
    items = make_splits(args.data_root)["test"]
    step = max(1, len(items) // args.eval_images)
    items = items[::step][:args.eval_images]

    rows = []
    for name, kw in settings.items():
        path = tmp / f"{name}.onnx"
        quantize_static(str(pre), str(path), CalibReader(args.data_root, args.size, 128),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, **kw)
        seg = Segmenter(path, size=args.size, threads=1)
        lat = benchmark(seg, args.size)["latency_ms_mean"]
        m, _, _ = evaluate_segmenter(seg, args.data_root, items)
        row = {"setting": name, "test_miou": m["miou"], "test_mdice": m["mdice"],
               "latency_ms_1t": lat, "size_mb": path.stat().st_size / 2**20}
        print(json.dumps(row))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "quant_ablation.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
