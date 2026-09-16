"""Loading a pretrained Volt backbone into the interactive model.

The released checkpoints (``weights/hf/Volt_experiments/...``) are DDP saves of a
full segmentor:

    state_dict: { "module.backbone.<...>": ..., "module.seg_heads.ScanNet.<...>": ... }

so the backbone lives under ``module.backbone.`` and everything else (the
semantic heads, optimizer, EMA, ...) is irrelevant here.  ``VoltInteractive`` also
owns a transformer / prompt encoder / mask decoder that the checkpoint has never
seen, so loading is deliberately restricted to ``backbone.*`` and never
``strict=True``.

They are multi-dataset "joint" runs (ScanNet + ScanNet200 + ScanNet++ +
ARKitScenes), which is why several seg heads are present.

Backbone configs implied by the weights (both have head_dim 64, matching Volt's
hardcoded ``freq_split=(12, 12, 8)``):

    joint_training_small : embed_dim 384, depth 12, num_heads 6,  up_mlp_dim 128
    joint_training_base  : embed_dim 768, depth 12, num_heads 12, up_mlp_dim 256
"""

import os

import torch

CHECKPOINT_PATHS = {
    "small": "weights/hf/Volt_experiments/joint_training_small/scannet/model/model_last.pth",
    "base": "weights/hf/Volt_experiments/joint_training_base/scannet/model/model_last.pth",
}

# the backbone config each released checkpoint requires; these are not free choices
BACKBONE_CONFIG = {
    "small": dict(embed_dim=384, depth=12, num_heads=6, up_mlp_dim=128),
    "base": dict(embed_dim=768, depth=12, num_heads=12, up_mlp_dim=256),
}


def backbone_config(variant: str = "small", **overrides) -> dict:
    """A ``Volt`` config dict for the released ``variant`` weights."""
    if variant not in BACKBONE_CONFIG:
        raise KeyError(f"unknown variant {variant!r}; have {list(BACKBONE_CONFIG)}")
    cfg = dict(
        type="Volt",
        in_channels=6,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.0,
        stride=5,
        kernel_size=5,
        increase_drop_path=False,
        **BACKBONE_CONFIG[variant],
    )
    cfg.update(overrides)
    return cfg


def strip_ddp_prefix(state_dict: dict) -> dict:
    """``module.foo`` -> ``foo`` for every key (DDP saves)."""
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def load_volt_backbone(
    model,
    path: str | None = None,
    variant: str = "small",
    use_ema: bool = False,
    verbose: bool = True,
):
    """Copy ``backbone.*`` weights from a released checkppoint into ``model``.

    Args:
        model: a ``VoltInteractive`` instance.
        path: checkpoint path; defaults to the released ``variant`` checkpoint.
        variant: ``"small"`` or ``"base"``, used to resolve the default path.
        use_ema: use ``ema_state_dict`` instead of ``state_dict``.  The EMA copy is
            usually the better starting point for a frozen backbone.

    Returns:
        the list of backbone keys that were *not* found in the checkpoint.
    """
    path = path or CHECKPOINT_PATHS[variant]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"checkpoint {path} not found. Note that `weights/*` is gitignored, so "
            f"it does not show up in a plain `fd`/`git status`."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if use_ema and ckpt.get("ema_state_dict"):
        state = ckpt["ema_state_dict"]
        source = "ema_state_dict"
    else:
        state = ckpt["state_dict"]
        source = "state_dict"
    state = strip_ddp_prefix(state)

    backbone = {
        k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")
    }
    if not backbone:
        raise KeyError(f"no 'backbone.*' keys in {path}")

    target = model.backbone
    missing, unexpected = target.load_state_dict(backbone, strict=False)
    if verbose:
        print(
            f"[checkpoint] {path}\n"
            f"             source={source} epoch={ckpt.get('epoch')} "
            f"metric={ckpt.get('best_metric_value')}\n"
            f"             loaded {len(backbone)} backbone tensors, "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if unexpected:
            print(f"             unexpected: {unexpected[:5]}")
    return list(missing)
