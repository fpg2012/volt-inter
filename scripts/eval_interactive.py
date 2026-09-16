"""Interactive evaluation protocol: IoU@k and NoC@t on ScanNet val.

The semantic-segmentation evaluator cannot be used here: this model predicts the
binary mask of one clicked instance, and the metric that matters is how quickly
the mask converges as clicks accumulate.

Protocol (the one used by the interactive-3D-segmentation literature, e.g.
InterObject3D / Easy3D):

  * click 1 is a positive click at the GT voxel nearest the instance centroid --
    the click a user makes when aiming at an object,
  * after each round, if the mask is still wrong, add one click on a misclassified
    voxel: a positive click on a false negative, a negative click on a false
    positive (whichever error is larger),
  * clicks accumulate; the model is re-run with the full click set each round,
  * IoU@k is the IoU after k clicks, NoC@t is the first k reaching IoU >= t
    (capped at --max-clicks).

Reported numbers are averaged over instances, and split into small / medium /
large instances -- a single click on an entire wall behaves nothing like a single
click on a chair, and averaging them hides that.

Usage:
    python scripts/eval_interactive.py --checkpoint exp/insseg-volt-interactive-0-base/model/model_last.pth
"""

import argparse
import json
from pathlib import Path

import torch

from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import collate_fn
from pointcept.models.builder import build_model
from pointcept.models.utils import offset2batch
from pointcept.models.volt.interactive.checkpoint import load_volt_backbone

GRID_SIZE = 0.02
KERNEL = 5


def build_model_from_cfg(cfg, device, checkpoint, ema=False):
    model = build_model(cfg.model).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("ema_state_dict", state["state_dict"]) if ema else state["state_dict"]
        state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded {checkpoint}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        # fall back to the released backbone only
        load_volt_backbone(model, variant="small")
    model.eval()
    model.freeze_backbone = True
    return model


@torch.no_grad()
def evaluate_scene(model, data, cache, target_ids, max_clicks, iou_thresholds, seed=0):
    """Run the click protocol for one scene's instances. Returns per-instance records."""
    gen = torch.Generator(device=data["instance"].device).manual_seed(seed)
    device = data["instance"].device
    scene = int(data["batch"].max().item()) + 1
    out = []

    for target in target_ids:
        tgt = data["instance"] == target
        if not tgt.any():
            continue
        rows = torch.nonzero(tgt, as_tuple=False).flatten()
        centroid = data["grid_coord"][rows].float().mean(0, keepdim=True)
        first = rows[torch.argsort((data["grid_coord"][rows].float() - centroid).pow(2).sum(-1))[0]]

        click_rows = [int(first)]
        click_labels = [0]  # 0 = positive, 1 = negative
        ious = []
        noc = {t: None for t in iou_thresholds}

        for k in range(max_clicks):
            task = {
                "click_patch": data["grid_coord"][
                    torch.tensor(click_rows, device=device)
                ]
                // KERNEL,
                "click_label": torch.tensor(click_labels, device=device),
                "click_batch": torch.zeros(len(click_rows), dtype=torch.long, device=device),
                "target_instance": torch.tensor(
                    [target] * scene, dtype=data["instance"].dtype, device=device
                ),
            }
            res = model({**data, **task}, backbone_cache=cache, return_logits=True)
            logits = res["seg_logits"]
            pred = logits.sigmoid() > 0.5
            inter = (pred & tgt).sum().item()
            union = (pred | tgt).sum().item()
            iou = inter / union if union else 1.0
            ious.append(iou)
            for t in iou_thresholds:
                if noc[t] is None and iou >= t:
                    noc[t] = k + 1

            fp = torch.nonzero(pred & ~tgt, as_tuple=False).flatten()
            fn = torch.nonzero(~pred & tgt, as_tuple=False).flatten()
            if k + 1 >= max_clicks or fp.numel() + fn.numel() == 0:
                break
            if fn.numel() >= fp.numel():
                pick = fn[torch.randint(fn.numel(), (1,), generator=gen, device=device).item()]
                click_labels.append(0)
            else:
                pick = fp[torch.randint(fp.numel(), (1,), generator=gen, device=device).item()]
                click_labels.append(1)
            click_rows.append(int(pick))

        out.append(
            dict(
                instance=int(target),
                n_voxels=int(tgt.sum().item()),
                iou=ious,
                noc=noc,
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scannet/insseg-volt-interactive-0-base.py")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenes", type=int, default=20, help="number of val scenes")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--instances-per-scene", type=int, default=5)
    ap.add_argument("--min-voxels", type=int, default=200)
    ap.add_argument("--max-clicks", type=int, default=10)
    ap.add_argument("--out", default="outputs/eval_interactive.json")
    args = ap.parse_args()

    from pointcept.engines.defaults import default_config_parser

    cfg = default_config_parser(args.config, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_from_cfg(cfg, device, args.checkpoint, args.ema)

    # an evaluation transform: no augmentation, just voxelise
    cfg.data.train.transform = [
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
    cfg.data.train.split = args.split
    cfg.data.train.lr_file = None
    dataset = build_dataset(cfg.data.train)

    thresholds = (0.5, 0.8, 0.9)
    records = []
    for i in range(args.offset, min(len(dataset), args.offset + args.scenes)):
        sample = dataset[i]
        batch = collate_fn([sample])
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        data["batch"] = offset2batch(data["offset"])
        cache = model.precompute_backbone(data)

        inst, counts = torch.unique(data["instance"], return_counts=True)
        keep = [
            int(x)
            for x, c in zip(inst.cpu().tolist(), counts.cpu().tolist())
            if x >= 0 and c >= args.min_voxels
        ]
        keep = sorted(keep, key=lambda x: -int((data["instance"] == x).sum()))[
            : args.instances_per_scene
        ]
        if not keep:
            continue
        recs = evaluate_scene(
            model, data, cache, keep, args.max_clicks, thresholds, seed=i
        )
        for r in recs:
            r["scene"] = i
        records += recs
        if (i - args.offset + 1) % 5 == 0:
            print(f"  scored {i - args.offset + 1}/{args.scenes} scenes, {len(records)} instances")

    if not records:
        print("no instances scored")
        return

    def bucket(n):
        return "small" if n < 2000 else ("medium" if n < 10000 else "large")

    def final_iou(r):
        return r["iou"][-1]

    print(f"\n{len(records)} instances over {args.scenes} scenes")
    summary = {}
    for name, subset in [
        ("all", records),
        ("small  (<2k voxels)", [r for r in records if bucket(r["n_voxels"]) == "small"]),
        ("medium (2k-10k)", [r for r in records if bucket(r["n_voxels"]) == "medium"]),
        ("large  (>10k)", [r for r in records if bucket(r["n_voxels"]) == "large"]),
    ]:
        if not subset:
            continue
        maxk = max(len(r["iou"]) for r in subset)
        iou_at = {}
        for k in range(maxk):
            # Instances that converged early stop producing clicks; carry their
            # final IoU forward instead of dropping them.  Dropping them would
            # leave only the hard instances in later columns and make IoU@k look
            # like it decreases with more clicks.
            vals = [r["iou"][min(k, len(r["iou"]) - 1)] for r in subset]
            iou_at[k + 1] = sum(vals) / len(vals)

        # NoC must be averaged over ALL instances, not only the successes: the
        # mean over successes is ~1 by construction and says nothing.  Instances
        # that never reach the threshold are charged the click budget.
        noc, success = {}, {}
        for t in thresholds:
            reached = [r["noc"][t] for r in subset]
            success[t] = sum(v is not None for v in reached) / len(reached)
            capped = [v if v is not None else maxk + 1 for v in reached]
            noc[t] = sum(capped) / len(capped)

        summary[name] = dict(n=len(subset), iou_at=iou_at, noc=noc, success=success)
        print(f"\n{name}: n={len(subset)}")
        print("  IoU@k: " + "  ".join(f"@{k}={v:.3f}" for k, v in iou_at.items()))
        print(
            "  NoC@t: "
            + "  ".join(f"@{int(t * 100)}={noc[t]:.2f}" for t in sorted(noc))
            + "   (unreached charged " + str(maxk + 1) + ")"
        )
        print(
            "  converged@t: "
            + "  ".join(f"@{int(t * 100)}={success[t] * 100:.0f}%" for t in sorted(success))
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(summary=summary, records=records), indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
