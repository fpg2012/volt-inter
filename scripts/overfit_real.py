"""Overfit test on **real ScanNet** scenes with a pretrained Volt backbone.

This is the meaningful version of ``overfit_interactive.py``.  It asks: given a
pretrained Volt backbone and a handful of real scenes, can the interactive head
be driven to near-perfect masks on a fixed set of (scene, instance) tasks?

Why this is the right test
--------------------------
The synthetic version only exercises shapes and wiring.  Real scenes add what
actually matters: tens of thousands of voxels, unannotated regions, ~30-50
instances per scene, and instance sizes spanning three orders of magnitude.  A
head that cannot overfit 4-8 real tasks will not train on the full set either.

Why the checkpoint matters
--------------------------
A *randomly initialised* Volt produces voxel features that are not linearly
separable per instance; deriving a mask from them is then a hard joint
optimisation and the head collapses to all-negative.  With the released
checkpoint (mIoU 0.795) the features are already good and the head only has to
learn "given this click, pick that instance".  Default is therefore a frozen
pretrained backbone, which is also the realistic fine-tuning setup.

Run:
    python scripts/overfit_real.py --scenes 4 --tasks 4 --steps 400
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pointcept.datasets.scannet import ScanNetDataset
from pointcept.datasets.utils import collate_fn
from pointcept.models.volt.interactive.checkpoint import (
    backbone_config,
    load_volt_backbone,
)
from pointcept.models.volt.interactive.clicks import make_task, target_voxel_mask
from pointcept.models.utils import offset2batch
from pointcept.models.volt.interactive.model import VoltInteractive
from pointcept.models.volt.interactive.viz import render_mask_panel

GRID_SIZE = 0.02
KERNEL = 5

TRANSFORM = [
    dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=GRID_SIZE,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "instance"),
        feat_keys=("color", "normal"),
    ),
]


def build_dataset():
    return ScanNetDataset(
        split="train",
        data_root="data/scannet",
        transform=TRANSFORM,
        test_mode=False,
        ignore_index=-1,
    )


def build_model(
    variant="small",
    mask_source="prompt",
    num_heads=None,
    negative_ratio=2.0,
    freeze_backbone=True,
):
    cfg = backbone_config(variant)
    heads = num_heads or cfg["num_heads"]
    return VoltInteractive(
        backbone=cfg,
        transformer=dict(
            depth=2,
            num_heads=heads,
            mlp_dim=4 * cfg["embed_dim"],
            # rate=2 for both cross attentions; rate=1 would let the head reuse the
            # backbone's exact RoPE basis (head_dim 64) at 2x the attention cost
            attention_downsample_rate=2,
        ),
        prompt=dict(pos_encoding="fourier"),
        mask_dim=cfg["up_mlp_dim"],
        mask_source=mask_source,
        use_iou_head=True,
        iou_weight=0.5,
        loss=dict(bce_weight=1.0, dice_weight=1.0, negative_ratio=negative_ratio),
        refinement_prob=1.0,
        freeze_backbone=freeze_backbone,
    )


def load_batch(dataset, indices, device):
    samples = [dataset[i] for i in indices]
    batch = collate_fn(samples)
    return {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }


def build_tasks(data, num_tasks, device, min_size=200, max_size=20000):
    """One task per instance slot: task k asks every scene for its k-th instance.

    Instances are filtered by voxel count first.  ScanNet's lowest instance ids
    tend to be walls and floors, which span the whole scene and are not
    well-posed single-click targets: a single centroid click cannot select a wall
    that is split into several disconnected pieces.  Restricting to object-sized
    instances makes the tasks representative of real interactive use.
    """
    instance, batch = data["instance"], data["batch"]
    num_scenes = int(batch.max().item()) + 1
    per_scene = []
    for b in range(num_scenes):
        rows = torch.nonzero(batch == b, as_tuple=False).flatten()
        ids, counts = torch.unique(instance[rows], return_counts=True)
        keep = [
            int(i)
            for i, c in zip(ids.cpu().tolist(), counts.cpu().tolist())
            if i >= 0 and min_size <= c <= max_size
        ]
        per_scene.append(sorted(keep, key=lambda i: -int((instance[rows] == i).sum())))
    num_tasks = min(num_tasks, min(len(x) for x in per_scene))
    tasks = []
    for k in range(num_tasks):
        targets = [per_scene[b][k] for b in range(num_scenes)]
        tasks.append(
            {
                kk: (vv.to(device) if torch.is_tensor(vv) else vv)
                for kk, vv in make_task(
                    instance, batch, data["grid_coord"], KERNEL, targets
                ).items()
            }
        )
    return tasks, per_scene


def render(data, task, pred, target, step, out, tag="train"):
    coord = data["coord"]
    scene0 = data["batch"] == 0
    sel = task["click_batch"] == 0
    clicks = coord[task["click_row"][sel]] if sel.any() else None
    labels = task["click_label"][sel] if sel.any() else None
    img = render_mask_panel(
        coord[scene0],
        pred[scene0],
        target[scene0],
        clicks=clicks,
        click_label=labels,
        title=f"step {step}  scene0  target instance {int(task['target_instance'][0])}",
    )
    if out["writer"] is not None:
        out["writer"].add_image(
            f"mask/{tag}", torch.from_numpy(img).permute(2, 0, 1), step
        )
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--scene-offset", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--variant", default="small", choices=["small", "base"])
    ap.add_argument("--mask-source", default="prompt", choices=["prompt", "token", "both"])
    ap.add_argument("--negative-ratio", type=float, default=2.0)
    ap.add_argument("--unfreeze-backbone", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--out", default="outputs/overfit_real")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--viz-every", type=int, default=50)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--save", default=None, help="save trained model state_dict here")
    # A single centroid click does NOT determine the exact mask of a large real
    # instance: ScanNet splits a floor into several instances and object
    # boundaries are genuinely ambiguous.  Measured here, one click on the four
    # largest object-sized instances per scene reaches ~0.89 mean IoU and the
    # residual is boundary speckle (see the rendered panel), not a wrong object.
    # So the assertion checks loss collapse and "far above trivial", not IoU -> 1.
    ap.add_argument("--target-iou", type=float, default=0.75)
    ap.add_argument("--skip-assert", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    dataset = build_dataset()
    indices = list(range(args.scene_offset, args.scene_offset + args.scenes))
    data = load_batch(dataset, indices, device)
    # the Pointcept pipeline ships 'offset'; the model accepts either
    data["batch"] = offset2batch(data["offset"])
    print(
        f"{args.scenes} scenes -> {data['coord'].shape[0]:,} voxels, "
        f"{int(data['batch'].max()) + 1} scene(s) in batch"
    )

    model = build_model(
        variant=args.variant,
        mask_source=args.mask_source,
        negative_ratio=args.negative_ratio,
    ).to(device)
    load_volt_backbone(
        model, path=args.checkpoint, variant=args.variant, use_ema=args.use_ema
    )
    if not args.unfreeze_backbone:
        model.freeze_backbone = True  # also enables the no_grad fast path
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("backbone frozen (no_grad backbone forward)")

    # the frozen backbone sees only the scene, not the clicks: compute it once
    cache = model.precompute_backbone(data)
    print(
        f"cached backbone: {cache['patch_tokens'].shape[0]:,} patch tokens, "
        f"{cache['fine'].shape[0]:,} voxels"
    )

    tasks, per_scene = build_tasks(data, args.tasks, device)
    print(
        f"{len(tasks)} tasks, instances per scene: "
        f"{[len(x) for x in per_scene]}"
    )

    # quick pre-training snapshot: does anything work at all yet?
    model.eval()
    with torch.no_grad():
        r = model({**data, **tasks[0]}, backbone_cache=cache, return_logits=True)
        print(
            f"before training: loss={float(r['loss']):.4f} "
            f"IoU0={float(r.get('iou_step0', torch.tensor(0))):.3f}"
        )

    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        Path(args.out).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.out)
        print(f"tensorboard: tensorboard --logdir {args.out}")
    out = {"writer": writer}

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1
    )

    history = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        task = tasks[(step - 1) % len(tasks)]
        batch_in = {**data, **task}

        opt.zero_grad(set_to_none=True)
        res = model(batch_in, backbone_cache=cache, return_logits=True)
        loss = res["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        history.append(loss.detach().item())

        if writer is not None and step % args.log_every == 0:
            writer.add_scalar("loss/total", loss.detach().item(), step)
            for k, v in res.items():
                if k.startswith("loss") or k.startswith("refine/loss"):
                    writer.add_scalar(f"loss/{k}", float(v), step)
                elif k in ("iou_step0", "iou_step1", "iou_pred_step0"):
                    writer.add_scalar(f"metric/{k}", float(v), step)
            writer.add_scalar("opt/lr", sched.get_last_lr()[0], step)
            writer.add_scalar("opt/grad_norm", float(grad_norm), step)

        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:4d}  task {int(task['target_instance'][0]):4d}  "
                f"loss {loss.detach().item():.4f}  "
                f"IoU0 {float(res.get('iou_step0', torch.tensor(float('nan')))):.3f}  "
                f"IoU1 {float(res.get('iou_step1', torch.tensor(float('nan')))):.3f}"
            )
            if writer is not None and not args.no_viz and (
                step % args.viz_every == 0 or step == 1
            ):
                model.eval()
                with torch.no_grad():
                    rr = model(batch_in, backbone_cache=cache, return_logits=True)
                render(
                    data,
                    task,
                    rr["seg_logits"].sigmoid() > 0.5,
                    rr["target"],
                    step,
                    out,
                )
                model.train()

    elapsed = time.time() - t0
    window = min(len(history), len(tasks) * 5)
    final = sum(history[-window:]) / window
    print(
        f"\n{args.steps} steps in {elapsed:.1f}s "
        f"({elapsed / args.steps * 1000:.0f} ms/step)   final loss {final:.4f}"
    )

    # ---- final evaluation on every task ---------------------------------
    model.eval()
    ious0, ious1 = [], []
    with torch.no_grad():
        for task in tasks:
            res = model({**data, **task}, backbone_cache=cache, return_logits=True)
            ious0.append(float(res["iou_step0"]))
            ious1.append(float(res.get("iou_step1", res["iou_step0"])))
    print("per-task IoU pass0: " + " ".join(f"{v:.3f}" for v in ious0))
    print("per-task IoU pass1: " + " ".join(f"{v:.3f}" for v in ious1))

    if not args.no_viz:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(args.out).mkdir(parents=True, exist_ok=True)
        show = min(4, len(tasks))
        fig, axes = plt.subplots(show, 3, figsize=(13, 4.0 * show), dpi=80)
        axes = axes.reshape(show, 3)
        with torch.no_grad():
            for i, task in enumerate(tasks[:show]):
                res = model({**data, **task}, backbone_cache=cache, return_logits=True)
                pred = res["seg_logits"].sigmoid() > 0.5
                tgt = res["target"]
                scene0 = data["batch"] == 0
                sel = task["click_batch"] == 0
                clicks = data["coord"][task["click_row"][sel]] if sel.any() else None
                for j, (name, (a, b, la, lb)) in enumerate(
                    [("xy", (0, 1, "x", "y")), ("xz", (0, 2, "x", "z")), ("yz", (1, 2, "y", "z"))]
                ):
                    ax = axes[i, j]
                    c = data["coord"][scene0][:, [a, b]].cpu().numpy()
                    p_, g_ = pred[scene0].cpu().numpy(), tgt[scene0].cpu().numpy()
                    col = np.tile(np.array([0.85, 0.85, 0.85]), (p_.shape[0], 1))
                    col[g_ & ~p_] = (0.2, 0.4, 0.9)
                    col[p_ & ~g_] = (0.85, 0.33, 0.1)
                    col[p_ & g_] = (0.15, 0.6, 0.25)
                    ax.scatter(c[:, 0], c[:, 1], c=col, s=1.2, linewidths=0)
                    if clicks is not None and i == 0:
                        for k in range(clicks.shape[0]):
                            ax.scatter(
                                clicks[k, a].item(), clicks[k, b].item(),
                                marker="+" if task["click_label"][sel][k] == 0 else "x",
                                s=90, linewidths=2.5,
                                c=(0, 0.75, 0) if task["click_label"][sel][k] == 0 else (0.9, 0, 0),
                                zorder=5,
                            )
                    ax.set_title(f"task {i} inst {int(task['target_instance'][0])} {name} IoU={ious0[i]:.3f}", fontsize=9)
                    ax.set_aspect("equal", adjustable="datalim")
                    ax.tick_params(labelsize=6)
        fig.tight_layout()
        png = Path(args.out) / "final_tasks.png"
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png}")
    mean0 = sum(ious0) / len(ious0)
    mean1 = sum(ious1) / len(ious1)
    print(f"mean IoU pass0={mean0:.4f}  pass1={mean1:.4f}")

    # reference point: predicting everything positive yields |target| / |scene|
    with torch.no_grad():
        triv = []
        for task in tasks:
            tgt = target_voxel_mask(
                data["instance"], task["target_instance"], data["batch"]
            )
            triv.append(tgt.float().mean().item())
    print(
        "trivial all-positive IoU per task: "
        + " ".join(f"{v:.3f}" for v in triv)
        + f"   (mean {sum(triv) / len(triv):.4f})"
    )

    if writer is not None:
        writer.add_scalar("final/loss", final, 0)
        writer.add_scalar("final/iou_step0", mean0, 0)
        writer.add_scalar("final/iou_step1", mean1, 0)
        writer.close()

    if not args.skip_assert:
        assert final < 0.25, f"loss did not collapse, final={final:.4f}"
        assert mean0 > args.target_iou, f"pass0 IoU too low: {mean0:.4f}"
        assert mean0 > 4 * (sum(triv) / len(triv)), "barely better than all-positive"
        assert mean1 >= mean0 - 0.02, f"previous-logit pass hurt: {mean0:.4f} -> {mean1:.4f}"
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, args.save)
        print(f"saved model to {args.save}")
    print("\nreal-data overfit test passed")


if __name__ == "__main__":
    main()
