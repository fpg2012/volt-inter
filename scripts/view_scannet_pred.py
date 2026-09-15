#!/usr/bin/env python
"""Visualize Volt ScanNet prediction vs ground truth with viser.

Run: .venv/bin/python scripts/view_scannet_pred.py [scene_dir] [--port 8090]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.datasets.transform import GridSample
from scripts.eval_scannet import build_model, SCANNET20

PALETTE = np.array([
    [174,199,232],[152,223,138],[31,119,180],[255,187,120],[188,189,34],
    [140,86,75],[255,152,150],[214,39,40],[197,176,213],[148,103,189],
    [196,156,148],[23,190,207],[247,182,210],[219,219,141],[255,127,14],
    [158,218,229],[44,160,44],[112,128,144],[227,119,134],[82,84,163],
], dtype=np.float32) / 255.0


class CenterShift:
    def __call__(self, data_dict):
        data_dict["coord"] = data_dict["coord"] - data_dict["coord"].mean(0)
        return data_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default=None)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--grid", type=float, default=0.02)
    args = ap.parse_args()

    scene = args.scene or sorted(glob.glob("data/scannet/val/scene*"))[0]

    coord = np.load(f"{scene}/coord.npy").astype(np.float32)
    color = np.load(f"{scene}/color.npy").astype(np.float32)
    normal = np.load(f"{scene}/normal.npy").astype(np.float32)
    seg = np.load(f"{scene}/segment20.npy").reshape(-1).astype(np.int64)

    gs = GridSample(grid_size=args.grid, hash_type="fnv", mode="train",
                    return_grid_coord=True)
    d = CenterShift()(dict(coord=coord, color=color, normal=normal, segment=seg))
    d = gs(d)

    device = "cuda"
    model = build_model(device)
    data = dict(
        grid_coord=torch.from_numpy(d["grid_coord"]).to(device).int(),
        feat=torch.from_numpy(np.concatenate(
            [d["color"] / 255.0, d["normal"]], axis=1).astype(np.float32)).to(device),
        batch=torch.zeros(len(d["grid_coord"]), dtype=torch.long, device=device),
        condition=["ScanNet"],
    )
    with torch.no_grad():
        pred = model(data)["seg_logits"].argmax(-1).cpu().numpy().astype(np.int64)
    gt = d["segment"].astype(np.int64)
    pts = d["coord"].astype(np.float32)
    print(f"scene {scene}: {len(pts)} voxels, pred acc "
          f"{(pred == gt)[gt >= 0].mean() * 100:.1f}%")

    import viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    # pred: original position; gt: shifted along x for side-by-side
    server.scene.add_point_cloud("pred", points=pts,
                                 colors=PALETTE[pred], point_size=0.02)
    server.scene.add_point_cloud("gt", points=pts + np.array([pts[:, 0].max() * 1.2 + 2, 0, 0]),
                                 colors=PALETTE[np.where(gt >= 0, gt, 0)], point_size=0.02)

    print(f"left=prediction, right=ground truth -> http://localhost:{args.port}")
    print("Ctrl-C to stop")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
