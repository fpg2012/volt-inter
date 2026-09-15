#!/usr/bin/env python
"""Visualize the Volt segmentation result ply with viser.

Run:  .venv/bin/python scripts/view_result.py [path.ply] [port]
Then open the printed http://localhost:<port> in your browser.
"""
import argparse
import time

import numpy as np
import trimesh
import viser


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", nargs="?", default="outputs/garden_volt_s.ply")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    mesh = trimesh.load(args.ply, process=False)
    pts = np.asarray(mesh.vertices, dtype=np.float32)
    # color stored in vertex_colors (uint8) or vertex colors attribute
    col = None
    if hasattr(mesh, "visual") and mesh.visual is not None:
        vis = mesh.visual
        if hasattr(vis, "vertex_colors") and vis.vertex_colors is not None:
            col = np.asarray(vis.vertex_colors)[:, :3] / 255.0
    if col is None:
        col = np.full_like(pts, 0.8)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_point_cloud(
        "volt_result",
        points=pts,
        colors=col,
        point_size=0.01,
    )
    server.scene.add_frame("origin", wxyz=(1.0, 0.0, 0.0, 0.0), position=(0, 0, 0))

    print(f"Loaded {len(pts)} points -> {args.ply}")
    print(f"Open http://localhost:{args.port} in a browser")
    print("Ctrl-C to stop")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
