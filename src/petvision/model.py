"""Model construction and transfer-learning strategies."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# timm encoder through smp's "tu-" prefix -> ImageNet weights come from the Hugging Face hub
DEFAULT_ENCODER = "tu-mobilenetv3_large_100"


def build_model(encoder: str = DEFAULT_ENCODER, pretrained: bool = True, arch: str = "unet") -> nn.Module:
    if encoder.startswith("tu-"):
        weights = True if pretrained else None  # smp convention for timm encoders
    else:
        weights = "imagenet" if pretrained else None
    kwargs = dict(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1)
    if arch == "unet":
        return smp.Unet(decoder_channels=(128, 64, 32, 16, 8), **kwargs)
    if arch == "fpn":
        return smp.FPN(**kwargs)
    raise ValueError(f"unknown arch {arch}")


def set_encoder_trainable(model: nn.Module, trainable: bool):
    for p in model.encoder.parameters():
        p.requires_grad = trainable
    # frozen encoder -> keep BatchNorm statistics frozen too (eval mode)
    model._encoder_frozen = not trainable


def train_mode(model: nn.Module):
    model.train()
    if getattr(model, "_encoder_frozen", False):
        model.encoder.eval()


def make_param_groups(model: nn.Module, lr: float, encoder_lr_mult: float):
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": rest, "lr": lr, "name": "decoder"}]
    if enc:
        groups.append({"params": enc, "lr": lr * encoder_lr_mult, "name": "encoder"})
    return groups


def count_params(model: nn.Module, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


# Transfer-learning strategies compared in the experiments.
#   scratch     : random init, everything trained
#   frozen      : ImageNet encoder frozen, only decoder + head trained
#   finetune    : ImageNet encoder, everything trained with the same LR
#   discrim     : ImageNet encoder, encoder LR = 0.1 x decoder LR (discriminative LRs)
#   two_stage   : frozen encoder for the first `warmup_epochs`, then unfreeze with discriminative LR
STRATEGIES = {
    "scratch":   dict(pretrained=False, freeze=False, encoder_lr_mult=1.0, warmup_epochs=0),
    "frozen":    dict(pretrained=True, freeze=True, encoder_lr_mult=0.0, warmup_epochs=0),
    "finetune":  dict(pretrained=True, freeze=False, encoder_lr_mult=1.0, warmup_epochs=0),
    "discrim":   dict(pretrained=True, freeze=False, encoder_lr_mult=0.1, warmup_epochs=0),
    "two_stage": dict(pretrained=True, freeze=False, encoder_lr_mult=0.1, warmup_epochs=2),
}


def save_checkpoint(path, model, cfg: dict, metrics: dict | None = None):
    torch.save({"state_dict": model.state_dict(), "config": cfg, "metrics": metrics or {}}, path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg.get("encoder", DEFAULT_ENCODER), pretrained=False, arch=cfg.get("arch", "unet"))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg
