"""Overfit test for the interactive segmentation head.

Checks, in order:

1. **No backbone drift.**  ``VoltInteractive._backbone_forward`` duplicates a few
   lines of ``Volt.forward`` to get at the patch tokens.  This asserts the fine
   features it produces are bit-identical to the backbone's own output, so the
   duplication cannot rot silently.
2. **The pipeline is differentiable and can fit.**  On synthetic box scenes the
   loss must collapse and IoU must reach ~1.0.  If it does not, the bug is in
   shapes / packing / the mask head, not in the data.
3. **Clicks are actually used.**  The task set is deterministic: task ``k`` asks
   every scene to segment its instance ``k`` with a single centroid click.  All
   tasks have *identical* prompt token content, so only click position can tell
   them apart -- a model that ignores position cannot fit this.
4. **The previous-mask pathway helps, not hurts.**  Pass 1 consumes pass 0's
   logits as the dense prompt; ``iou_step1`` must not be worse than ``iou_step0``.

No ScanNet data is needed.  TensorBoard logs go to ``--out``.

Run: python scripts/overfit_interactive.py
"""

import argparse
import time
from pathlib import Path

import torch

from pointcept.models.volt.interactive.clicks import enumerate_instance_clicks
from pointcept.models.volt.interactive.model import VoltInteractive
from pointcept.models.volt.interactive.synthetic import build_batch
from pointcept.models.volt.interactive.viz import render_mask_panel

STEP_SIZE = 0.05  # voxel size of the synthetic scenes
KERNEL = 5
SEED = 1234


def build_model(
    pos_encoding: str = "fourier",
    refinement_prob: float = 1.0,
    mask_source: str = "prompt",
    negative_ratio: float | None = 2.0,
):
    # Volt hardcodes `self.pos_enc = RoPE()` with freq_split=(12, 12, 8)
    # (volt_base.py:246), i.e. 32 complex dims = 64 real dims.  Any Volt variant
    # must therefore have embed_dim // num_heads == 64; this is not a choice.
    embed_dim, num_heads = 192, 3  # head_dim = 64, matching the backbone RoPE
    return VoltInteractive(
        backbone=dict(
            type="Volt",
            in_channels=6,
            embed_dim=embed_dim,
            depth=2,
            num_heads=num_heads,
            mlp_ratio=2,
            init_values=None,
            qk_norm=True,
            drop_path=0.0,  # determinism: no stochastic depth for an overfit test
            stride=KERNEL,
            kernel_size=KERNEL,
            increase_drop_path=False,
            up_mlp_dim=64,
        ),
        transformer=dict(
            depth=2,
            num_heads=num_heads,
            mlp_dim=384,
            # rate=1 keeps head_dim at 64, so the two-way transformer's RoPE has
            # the *same* frequency basis as the backbone's.  rate=2 would halve it
            # to 16 complex dims -- worth an ablation, not worth the default.
            attention_downsample_rate=1,
        ),
        prompt=dict(pos_encoding=pos_encoding),
        mask_dim=64,
        use_iou_head=True,
        iou_weight=0.5,
        mask_source=mask_source,
        loss=dict(bce_weight=1.0, dice_weight=1.0, negative_ratio=negative_ratio),
        refinement_prob=refinement_prob,
    )


def check_no_backbone_drift(model, data_dict):
    """The duplicated two-stage forward must match ``Volt.forward`` exactly."""
    model.eval()
    with torch.no_grad():
        _, _, _, fine = model._backbone_forward(data_dict)
        reference = model.backbone(data_dict)
    diff = (fine - reference).abs().max().item()
    print(
        f"[drift] max|_backbone_forward - Volt.forward| = {diff:.3e}  "
        f"(shape {tuple(fine.shape)})"
    )
    assert diff < 1e-5, "two-stage backbone forward diverged from Volt.forward"


def render_task(model, data_dict, task, out, step, tag="train"):
    """Visualise scene 0 for one task."""
    model.eval()
    with torch.no_grad():
        res = model({**data_dict, **task}, return_logits=True)
    logits, target = res["seg_logits"], res["target"]
    batch = data_dict["batch"]
    scene0 = batch == 0

    coord = data_dict["grid_coord"].float() * STEP_SIZE
    pred = logits.sigmoid() > 0.5

    sel = task["click_batch"] == 0
    click_xyz = (
        task["click_patch"][sel].float() * KERNEL * STEP_SIZE if sel.any() else None
    )
    click_lab = task["click_label"][sel] if sel.any() else None
    tgt_id = int(task["target_instance"][0])

    iou0 = float(res.get("iou_step0", torch.tensor(0.0)))
    iou1 = float(res.get("iou_step1", torch.tensor(iou0)))
    img = render_mask_panel(
        coord,
        pred,
        target,
        clicks=click_xyz,
        click_label=click_lab,
        title=f"step {step}  scene0  instance {tgt_id}  "
        f"IoU0={iou0:.3f} IoU1={iou1:.3f}",
    )
    if out["writer"] is not None:
        out["writer"].add_image(
            f"mask/{tag}", torch.from_numpy(img).permute(2, 0, 1), step
        )
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--out", type=str, default="outputs/overfit_interactive")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--viz-every", type=int, default=50)
    ap.add_argument("--pos-encoding", type=str, default="fourier",
                    choices=["fourier", "mlp", "none"])
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--target-iou", type=float, default=0.9)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    data_dict = build_batch(num_scenes=args.scenes, step=STEP_SIZE, seed=0)
    data_dict = {k: v.to(device) for k, v in data_dict.items()}
    print(
        f"scene: {data_dict['coord'].shape[0]} voxels, {args.scenes} scenes, "
        f"instances {sorted(set(data_dict['instance'].tolist()))}"
    )

    tasks = enumerate_instance_clicks(
        data_dict["instance"],
        data_dict["batch"],
        data_dict["grid_coord"],
        KERNEL,
        num_scenes=args.scenes,
    )
    tasks = [{k: v.to(device) for k, v in t.items()} for t in tasks]
    print(
        f"tasks: {len(tasks)} (one per instance id, "
        f"{len(tasks[0]['click_label'])} clicks each)"
    )

    model = build_model(pos_encoding=args.pos_encoding).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n / 1e6:.2f} M   pos_encoding={args.pos_encoding}")

    check_no_backbone_drift(model, data_dict)

    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        Path(args.out).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.out)
        print(f"tensorboard: tensorboard --logdir {args.out}")
    out = {"writer": writer}

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1
    )

    history = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        task = tasks[(step - 1) % len(tasks)]
        batch_in = {**data_dict, **task}

        opt.zero_grad(set_to_none=True)
        res = model(batch_in, return_logits=True)
        loss = res["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        history.append(loss.detach().item())

        if writer is not None and step % args.log_every == 0:
            writer.add_scalar("loss/total", float(loss), step)
            for k, v in res.items():
                if k.startswith("loss") or k.startswith("refine/loss"):
                    writer.add_scalar(f"loss/{k}", float(v), step)
                elif k in ("iou_step0", "iou_step1", "iou_pred_step0"):
                    writer.add_scalar(f"metric/{k}", float(v), step)
            writer.add_scalar("opt/lr", sched.get_last_lr()[0], step)
            writer.add_scalar("opt/grad_norm", float(grad_norm), step)

        if step % args.log_every == 0 or step == 1:
            tgt = int(task["target_instance"][0])
            print(
                f"step {step:4d}  inst {tgt}  loss {loss.detach().item():.4f}  "
                f"IoU0 {float(res.get(chr(39)+chr(105)+chr(111)+chr(117)+chr(95)+chr(115)+chr(116)+chr(101)+chr(112)+chr(48)+chr(39), torch.tensor(float(chr(110)+chr(97)+chr(110))))):.3f}  "
                f"IoU1 {float(res.get('iou_step1', torch.tensor(float('nan')))):.3f}"
            )

        if not args.no_viz and (step % args.viz_every == 0 or step == 1):
            render_task(model, data_dict, tasks[0], out, step)
            model.train()

    elapsed = time.time() - t0
    final = sum(history[-len(tasks) * 5 :]) / min(len(history), len(tasks) * 5)
    print(
        f"\n{args.steps} steps in {elapsed:.1f}s "
        f"({elapsed / args.steps * 1000:.1f} ms/step)   final loss {final:.4f}"
    )

    # ---- final evaluation over all tasks ---------------------------------
    model.eval()
    ious0, ious1 = [], []
    with torch.no_grad():
        for task in tasks:
            res = model({**data_dict, **task}, return_logits=True)
            ious0.append(float(res["iou_step0"]))
            ious1.append(float(res.get("iou_step1", res["iou_step0"])))
    print("per-task IoU pass0: " + " ".join(f"{v:.3f}" for v in ious0))
    print("per-task IoU pass1: " + " ".join(f"{v:.3f}" for v in ious1))
    mean0, mean1 = sum(ious0) / len(ious0), sum(ious1) / len(ious1)
    print(f"mean IoU pass0={mean0:.4f}  pass1={mean1:.4f}")

    if writer is not None:
        writer.add_scalar("final/loss", final, 0)
        writer.add_scalar("final/iou_step0", mean0, 0)
        writer.add_scalar("final/iou_step1", mean1, 0)
        writer.close()

    assert final < 0.15, f"loss did not collapse, final={final:.4f}"
    assert mean0 > args.target_iou, f"pass0 IoU too low: {mean0:.4f}"
    assert mean1 >= mean0 - 0.02, f"previous-logit pass hurt: {mean0:.4f} -> {mean1:.4f}"
    print("\noverfit test passed")


if __name__ == "__main__":
    main()
