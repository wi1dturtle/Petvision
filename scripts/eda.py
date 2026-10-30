"""Exploratory data analysis for Oxford-IIIT Pet (segmentation view).

    python scripts/eda.py --data-root data --out results/eda
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from petvision.data import make_splits, trimap_to_binary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results/eda")
    args = ap.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = make_splits(root)
    rows = []
    for split, items in splits.items():
        for it in items:
            im = Image.open(root / "images" / f"{it['name']}.jpg")
            tri = np.asarray(Image.open(root / "annotations" / "trimaps" / f"{it['name']}.png"))
            rows.append({**it, "split": split, "w": im.size[0], "h": im.size[1], "mode": im.mode,
                         "pet_frac": float((tri == 1).mean()), "boundary_frac": float((tri == 3).mean()),
                         "fg_frac": float(trimap_to_binary(tri).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "dataset_index.csv", index=False)

    n_imgs_on_disk = len(list((root / "images").glob("*.jpg")))
    stats = {
        "images_on_disk": n_imgs_on_disk,
        "images_with_split_labels": len(df),
        "split_sizes": df.split.value_counts().to_dict(),
        "species_counts": df.species.value_counts().to_dict(),
        "n_breeds": int(df.breed.nunique()),
        "images_per_breed_min_max": [int(df.breed.value_counts().min()), int(df.breed.value_counts().max())],
        "image_modes": dict(Counter(df["mode"])),
        "width_median": float(df.w.median()), "height_median": float(df.h.median()),
        "width_min_max": [int(df.w.min()), int(df.w.max())], "height_min_max": [int(df.h.min()), int(df.h.max())],
        "aspect_ratio_median": float((df.w / df.h).median()),
        "fg_frac_mean": float(df.fg_frac.mean()), "fg_frac_p5_p95": [float(df.fg_frac.quantile(.05)),
                                                                     float(df.fg_frac.quantile(.95))],
        "boundary_frac_mean": float(df.boundary_frac.mean()),
        "boundary_share_of_foreground": float(df.boundary_frac.sum() / df.fg_frac.sum()),
    }
    (out / "eda_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    # figure 1: distributions
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].hist(df.fg_frac, bins=40, color="#4c72b0")
    ax[0].set_title("Foreground share per image")
    ax[0].set_xlabel("fraction of pixels that are pet")
    ax[1].scatter(df.w, df.h, s=2, alpha=0.3)
    ax[1].set_title("Image sizes")
    ax[1].set_xlabel("width")
    ax[1].set_ylabel("height")
    ax[1].set_xlim(0, 1000)
    ax[1].set_ylim(0, 1000)
    bc = df.groupby(["species", "breed"]).size().reset_index(name="n").sort_values(["species", "n"])
    ax[2].bar(range(len(bc)), bc.n, color=["#dd8452" if s == "cat" else "#4c72b0" for s in bc.species])
    ax[2].set_title("Images per breed (orange = cat, blue = dog)")
    ax[2].set_xticks([])
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=100)
    plt.close(fig)

    # figure 2: sample images with trimaps
    sample = df[df.split == "train"].sample(6, random_state=0)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for i, (_, r) in enumerate(sample.iterrows()):
        axes[0, i].imshow(Image.open(root / "images" / f"{r['name']}.jpg").convert("RGB"))
        axes[0, i].set_title(r["breed"], fontsize=8)
        axes[1, i].imshow(np.asarray(Image.open(root / "annotations" / "trimaps" / f"{r['name']}.png")),
                          cmap="viridis", vmin=1, vmax=3)
        axes[1, i].set_title("trimap: 1 pet / 2 bg / 3 edge", fontsize=8)
        for a in axes[:, i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
