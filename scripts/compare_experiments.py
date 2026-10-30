"""Collect all runs in runs/, evaluate each best checkpoint on the same test subset,
measure its CPU latency, and make the comparison table + plots used in the report.

    python scripts/compare_experiments.py --eval-images 1000
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from petvision.data import make_splits  # noqa: E402
from petvision.evaluate import evaluate_segmenter  # noqa: E402
from petvision.inference import Segmenter  # noqa: E402
from petvision.optimize import benchmark  # noqa: E402

NICE = {"scratch": "From scratch", "frozen": "Frozen encoder", "finetune": "Full fine-tune",
        "discrim": "Fine-tune, encoder LR x0.1", "two_stage": "Two-stage (freeze 2 ep, then x0.1)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results/experiments")
    ap.add_argument("--eval-images", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = make_splits(args.data_root)["test"]
    step = max(1, len(items) // args.eval_images)
    items = items[::step][:args.eval_images]

    cache_path = out / "experiments.json"
    done = {r["run"]: r for r in json.loads(cache_path.read_text())} if cache_path.exists() else {}
    rows, histories = [], {}
    for summ in sorted(Path(args.runs).glob("*/summary.json")):
        run = summ.parent.name
        s = json.loads(summ.read_text())
        hist = json.loads((summ.parent / "history.json").read_text())
        histories[run] = hist
        if run in done:
            rows.append(done[run])
            continue
        cfg = s["config"]
        seg = Segmenter(summ.parent / "best.pt", threads=args.threads)
        m, _, _ = evaluate_segmenter(seg, args.data_root, items)
        lat = benchmark(seg, cfg["size"], runs=30)
        row = {"run": run, "strategy": cfg["strategy"], "size": cfg["size"], "epochs": cfg["epochs"],
               "trainable_params_M": hist[-1]["trainable_params"] / 1e6,
               "median_epoch_time_min": float(np.median([h["epoch_time_s"] for h in hist])) / 60,
               "best_val_miou": s["best_val_miou"], "best_epoch": s["best_epoch"],
               "test_miou": m["miou"], "test_mdice": m["mdice"], "test_pixel_acc": m["pixel_acc"],
               "test_miou_ignore_boundary": m["miou_ignore_boundary"],
               "latency_ms_bs1": lat["latency_ms_mean"], "n_test": m["n"]}
        print(json.dumps(row))
        rows.append(row)
        cache_path.write_text(json.dumps(rows, indent=1))
    cache_path.write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows)
    df.to_csv(out / "experiments.csv", index=False)

    def md_table(d, cols):
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in d.iterrows():
            lines.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                                           for c in cols) + " |")
        return "\n".join(lines)

    cols = ["run", "strategy", "size", "trainable_params_M", "median_epoch_time_min", "best_val_miou",
            "test_miou", "test_mdice", "test_miou_ignore_boundary", "latency_ms_bs1"]
    (out / "experiments.md").write_text(md_table(df.sort_values(["size", "test_miou"]), cols) + "\n")
    print(df[cols].to_string())

    # --- plots ------------------------------------------------------------------------
    strat = {k: v for k, v in histories.items() if k.startswith("strategy_")}
    if strat:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for run, h in strat.items():
            name = NICE.get(run.split("_", 1)[1].rsplit("_", 1)[0], run)
            ax[0].plot([r["epoch"] for r in h], [r["val_miou"] for r in h], marker="o", label=name)
            ax[1].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], marker="o", label=name)
        ax[0].set_title("Validation mIoU (128 px)")
        ax[0].set_xlabel("epoch")
        ax[1].set_title("Training loss (BCE + Dice)")
        ax[1].set_xlabel("epoch")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=.3)
        ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "strategies_curves.png", dpi=110)
        plt.close(fig)

    best_strategy = df[df.run.str.startswith("strategy_")].sort_values("best_val_miou").iloc[-1]["strategy"] \
        if strat else None
    # same 6-epoch budget only (the 15-epoch "long_" runs are a separate experiment)
    res = df[(df.strategy == best_strategy) & ~df.run.str.startswith("long_")].sort_values("size")
    if len(res) > 1:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.plot(res.latency_ms_bs1, res.test_miou, marker="o")
        for _, r in res.iterrows():
            ax.annotate(f"{r['size']} px", (r.latency_ms_bs1, r.test_miou), textcoords="offset points",
                        xytext=(6, -10))
        ax.set_xlabel("CPU latency per image, ms (batch 1)")
        ax.set_ylabel("test mIoU")
        ax.set_title(f"Resolution trade-off ({best_strategy})")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "resolution_tradeoff.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
