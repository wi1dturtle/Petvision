"""Train / fine-tune a pet segmentation model.

Example:
    python -m petvision.train --strategy discrim --size 160 --epochs 8 --out runs/discrim_160
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import CachedPetDataset, augment_batch, worker_seed
from .metrics import MetricTracker
from .model import (DEFAULT_ENCODER, STRATEGIES, build_model, count_params, make_param_groups,
                    save_checkpoint, set_encoder_trainable, train_mode)


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def seg_loss(logits, target, kind="bce_dice"):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if kind == "bce":
        return bce
    return bce + dice_loss(logits, target)


@torch.no_grad()
def evaluate_cached(model, loader, device):
    model.eval()
    tr = MetricTracker()
    loss_sum, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += seg_loss(logits, y).item() * len(x)
        n += len(x)
        tr.update(logits[:, 0] > 0, y[:, 0] > 0.5)
    out = tr.compute()
    out["loss"] = loss_sum / n
    return out


def lr_lambda_factory(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def build_optimizer(model, lr, strategy, wd):
    groups = make_param_groups(model, lr, strategy["encoder_lr_mult"])
    return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--strategy", default="discrim", choices=list(STRATEGIES))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--arch", default="unet", choices=["unet", "fpn"])
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--loss", default="bce_dice", choices=["bce", "bce_dice"])
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--max-train", type=int, default=0, help="use only N training images (0 = all)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/exp")
    args = ap.parse_args(argv)

    worker_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    strat = STRATEGIES[args.strategy]
    train_ds = CachedPetDataset(args.data_root, "train", args.size)
    val_ds = CachedPetDataset(args.data_root, "val", args.size)
    if args.max_train:
        train_ds = torch.utils.data.Subset(train_ds, range(min(args.max_train, len(train_ds))))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                           num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(args.encoder, pretrained=strat["pretrained"], arch=args.arch).to(device)
    freeze_now = strat["freeze"] or strat["warmup_epochs"] > 0
    set_encoder_trainable(model, not freeze_now)
    opt = build_optimizer(model, args.lr, strat, args.wd)

    steps_per_epoch = len(train_dl)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total, warmup_steps=steps_per_epoch // 2))

    cfg = vars(args) | {"strategy_cfg": strat, "params_total": count_params(model), "device": device}
    print(json.dumps(cfg, indent=1))
    history, best = [], -1.0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        if strat["warmup_epochs"] and epoch == strat["warmup_epochs"] + 1:
            # two-stage: unfreeze encoder, rebuild optimizer, keep the remaining cosine schedule
            print("[two_stage] unfreezing encoder")
            set_encoder_trainable(model, True)
            step_now = sched.last_epoch
            opt = build_optimizer(model, args.lr, strat, args.wd)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total, steps_per_epoch // 2))
            for _ in range(step_now):
                sched.step()
        train_mode(model)
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            if not args.no_aug:
                x, y = augment_batch(x, y)
            logits = model(x)
            loss = seg_loss(logits, y, args.loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(x)
            n += len(x)
        train_time = time.time() - t0
        val = evaluate_cached(model, val_dl, device)
        rec = {"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val["loss"], "val_miou": val["miou"],
               "val_mdice": val["mdice"], "val_pixel_acc": val["pixel_acc"], "epoch_time_s": train_time,
               "trainable_params": count_params(model, trainable_only=True)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if val["miou"] > best:
            best = val["miou"]
            save_checkpoint(out / "best.pt", model, cfg, rec)
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=1)

    summary = {"best_val_miou": best, "best_epoch": max(history, key=lambda r: r["val_miou"])["epoch"],
               "total_train_time_s": time.time() - t_start, "config": cfg}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("done", json.dumps({k: v for k, v in summary.items() if k != "config"}))


if __name__ == "__main__":
    main()
