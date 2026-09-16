"""Measure interactive-head training throughput and emit a scene subset.

The subset size has to come from a measurement, not from a guess: the dominant
cost is not obvious (data loading with GridSample, the frozen 12-layer backbone
forward, or the head itself all matter), and it depends on batch size, AMP and
worker count.

This script:
  1. builds the exact dataset/transform/dataloader from the training config,
  2. times real forward+backward steps (including data loading),
  3. reports seconds/step, scenes/hour and peak GPU memory,
  4. writes a deterministic scene-level subset for the requested time budget,
  5. prints the ``epoch`` value to paste into the config.

Usage:
    python scripts/make_interactive_subset.py --hours 6 --batch-size 8
    python scripts/make_interactive_subset.py --hours 6 --dry-run   # measure only
"""

import argparse
import random
import time
from functools import partial
from pathlib import Path

import torch

from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import point_collate_fn
from pointcept.engines.defaults import default_config_parser
from pointcept.models.builder import build_model
from pointcept.models.volt.interactive.checkpoint import load_volt_backbone

CONFIG = "configs/scannet/insseg-volt-interactive-0-base.py"
SUBSET_DIR = "data/scannet/tasks/insseg_interactive"


def build_loader(cfg, batch_size, num_worker, shuffle=True):
    dataset = build_dataset(cfg.data.train)
    return dataset, torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        collate_fn=partial(point_collate_fn, mix_prob=0.0),
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_worker > 0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-worker", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12, help="timed steps")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{SUBSET_DIR}/subset.txt")
    ap.add_argument(
        "--subset-size", type=int, default=None,
        help="write exactly this many scenes instead of sizing from --hours",
    )
    ap.add_argument("--dry-run", action="store_true", help="measure only, write nothing")
    args = ap.parse_args()

    cfg = default_config_parser(args.config, None)
    # the config references a subset file that this script is about to create;
    # measure on the full split, then write the subset
    cfg.data.train.lr_file = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  config={args.config}")
    print(f"batch_size={args.batch_size}  num_worker={args.num_worker}  amp={cfg.enable_amp}")

    # ---- model ----------------------------------------------------------
    model = build_model(cfg.model).to(device)
    load_volt_backbone(model, path=cfg.weight)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"trainable params: {n_train / 1e6:.2f} M")

    opt = torch.optim.AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.enable_amp)

    # ---- data -----------------------------------------------------------
    dataset, loader = build_loader(cfg, args.batch_size, args.num_worker)
    print(f"dataset: {len(dataset)} scenes")
    it = iter(loader)

    def one_step():
        input_dict = next(it)
        for k in input_dict:
            if isinstance(input_dict[k], torch.Tensor):
                input_dict[k] = input_dict[k].cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda", enabled=cfg.enable_amp, dtype=torch.float16
        ):
            out = model(input_dict)
            loss = out["loss"]
        if cfg.enable_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad or 1e9)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad or 1e9)
            opt.step()
        return loss.detach().item(), int(input_dict["offset"].numel())

    # ---- warmup + timing ------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses, scenes = [], 0
    for _ in range(args.steps):
        loss, n = one_step()
        losses.append(loss)
        scenes += n
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sec_per_step = elapsed / args.steps
    scenes_per_sec = scenes / elapsed
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"\nmeasured: {sec_per_step:.3f} s/step  ->  {scenes_per_sec * 3600:,.0f} scenes/hour"
        f"  ({scenes_per_sec * 3600 / args.batch_size:,.0f} steps/hour)"
    )
    print(f"peak GPU memory: {peak:.2f} GiB")
    print(f"loss (warmup-ish, not converged): {sum(losses) / len(losses):.4f}")

    total_scenes = len(dataset)
    budget_steps = int(args.hours * 3600 / sec_per_step)
    budget_scenes = int(budget_steps * args.batch_size)
    print(
        f"\n{args.hours:.1f} h budget -> {budget_steps:,} steps "
        f"= {budget_scenes:,} samples"
    )

    subset_size = (
        min(total_scenes, args.subset_size)
        if args.subset_size is not None
        else min(total_scenes, budget_scenes)
    )
    if args.subset_size is not None:
        steps_for_subset = int(args.subset_size / args.batch_size * sec_per_step)
        print(
            f"\nrequested subset size {args.subset_size} scenes"
            f" -> 1 epoch = {args.subset_size // args.batch_size} steps"
            f" = {steps_for_subset / 60:.1f} min at the measured rate"
        )
    # how many times each scene is seen, and a round epoch count
    epochs_for_subset = budget_scenes / subset_size
    suggested_epoch = max(1, int(round(epochs_for_subset)))
    steps_per_epoch = max(1, subset_size // args.batch_size)
    print(
        f"subset: {subset_size} scenes  ->  {steps_per_epoch} steps/epoch\n"
        f"        {epochs_for_subset:.2f} passes over the subset fit the budget\n"
        f"        => set epoch = {suggested_epoch} in {args.config} "
        f"(= {suggested_epoch * steps_per_epoch:,} steps)"
    )
    if subset_size == total_scenes:
        print("        (the whole train split fits -- no subset needed)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    # ---- write the subset ----------------------------------------------
    names = sorted(
        {Path(p).name for p in dataset.data_list}
    ) if hasattr(dataset, "data_list") else []
    if not names:
        print("\ncould not read dataset.data_list; not writing a subset file")
        return
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(names, min(subset_size, len(names))))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chosen) + "\n")
    print(f"\nwrote {len(chosen)} scene names to {out}")
    print("the config already points at this file via lr_file")


if __name__ == "__main__":
    main()
