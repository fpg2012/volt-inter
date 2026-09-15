#!/usr/bin/env python
"""Evaluate Volt on ScanNet val with a single GPU (chunk-free, whole-scene forward)."""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.models.default import DefaultSegmentorV2
from pointcept.datasets.transform import GridSample, Compose

SCANNET20 = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refridgerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
]


class CenterShift:
    def __call__(self, data_dict):
        c = data_dict["coord"].mean(0)
        data_dict["coord"] = data_dict["coord"] - c
        return data_dict


# NOTE(local): default checkpoint path below is machine-specific to this workstation
# (/workspace/lab/Volt). Override with --ckpt on other machines.
DEFAULT_CKPT = (
    "/workspace/lab/Volt/weights/hf/Volt_experiments/"
    "joint_training_small/scannet/model/model_last.pth"
)


def build_model(device, ckpt=DEFAULT_CKPT):
    model = DefaultSegmentorV2(
        num_classes=(20, 200, 100, 185),
        backbone_out_channels=128,
        conditions=("ScanNet", "ScanNet200", "ScanNet++", "ARKitScenes"),
        backbone=dict(
            type="Volt", in_channels=6, embed_dim=384, depth=12, num_heads=6,
            mlp_ratio=4, init_values=None, qk_norm=True, drop_path=0.0,
            stride=5, kernel_size=5, increase_drop_path=False, up_mlp_dim=128,
        ),
        criteria=[],
    ).to(device).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/scannet/val")
    ap.add_argument("--n", type=int, default=50, help="number of val scenes")
    ap.add_argument("--grid", type=float, default=0.02)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="model_last.pth path")
    args = ap.parse_args()

    device = "cuda"
    model = build_model(device, args.ckpt)
    scenes = sorted(glob.glob(os.path.join(args.data_root, "scene*")))[: args.n]
    assert scenes, f"no scenes under {args.data_root}"
    print(f"evaluating {len(scenes)} scenes, grid={args.grid}")

    gs = GridSample(grid_size=args.grid, hash_type="fnv", mode="train",
                    return_grid_coord=True, return_inverse=True)

    iou_per = np.zeros(20, dtype=np.float64)
    seen_per = np.zeros(20, dtype=np.int64)
    t_fwd = 0.0
    for i, p in enumerate(scenes):
        coord = np.load(os.path.join(p, "coord.npy")).astype(np.float32)
        color = np.load(os.path.join(p, "color.npy")).astype(np.float32)
        normal = np.load(os.path.join(p, "normal.npy")).astype(np.float32)
        seg = np.load(os.path.join(p, "segment20.npy")).reshape(-1).astype(np.int64)

        d = CenterShift()(dict(coord=coord, color=color, normal=normal,
                               segment=seg, instance=seg))
        d = gs(d)
        # note: grid sample picks one point per voxel; use its segment label
        grid = d["grid_coord"]
        seg_v = d["segment"].astype(np.int64)
        c = d["coord"]
        col = d["color"]
        nrm = d["normal"]

        feat = np.concatenate([col / 255.0, nrm], axis=1).astype(np.float32)
        data = dict(
            grid_coord=torch.from_numpy(grid).to(device).int(),
            feat=torch.from_numpy(feat).to(device),
            batch=torch.zeros(len(grid), dtype=torch.long, device=device),
            condition=["ScanNet"],
        )
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(data)["seg_logits"]
        torch.cuda.synchronize(); t_fwd += time.perf_counter() - t0
        pred = logits.argmax(-1).cpu().numpy().astype(np.int64)

        valid = seg_v >= 0
        for k in range(20):
            inter = ((pred == k) & (seg_v == k) & valid).sum()
            union = (((pred == k) | (seg_v == k)) & valid).sum()
            if union > 0:
                seen_per[k] += 1
                iou_per[k] += inter / union
        if (i + 1) % 10 == 0:
            done = iou_per / np.maximum(seen_per, 1)
            print(f"[{i+1}/{len(scenes)}] mIoU(so far)={done[seen_per>0].mean()*100:.1f}")

    miou = iou_per / np.maximum(seen_per, 1)
    print("\n=== per-class IoU ===")
    for k in range(20):
        if seen_per[k]:
            print(f"  {SCANNET20[k]:20s} {miou[k]*100:5.1f}")
    print(f"\nmIoU: {miou[seen_per>0].mean()*100:.1f} over {len(scenes)} scenes")
    print(f"avg forward: {t_fwd/len(scenes)*1000:.0f} ms/scene, "
          f"mem {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")


if __name__ == "__main__":
    main()
