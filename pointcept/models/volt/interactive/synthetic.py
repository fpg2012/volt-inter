"""Synthetic scenes for the overfit test.

No ScanNet data is needed: a scene is a floor plus a few separate boxes, which is
enough to check that the whole interactive pipeline is differentiable and can be
driven to zero loss on a handful of scenes.

Why boxes and not Gaussian blobs: after voxelisation a blob's "instance" is
ambiguous, and an overfit test that passes on an ill-defined target proves
nothing.  Boxes stay separated by construction, so the target mask is exactly the
box's voxels.
"""

import numpy as np
import torch


def _box_surface(corner, size, step):
    """Voxel coordinates on the surface of an axis-aligned box."""
    lo = np.array(corner, dtype=np.float64)
    hi = lo + np.array(size, dtype=np.float64)
    axes = []
    for axis in range(3):
        coords = np.arange(lo[axis], hi[axis] + 1e-9, step)
        for fixed in (lo[axis], hi[axis]):
            others = [np.arange(lo[a], hi[a] + 1e-9, step) for a in range(3) if a != axis]
            grid = np.meshgrid(*others, indexing="ij")
            pts = np.zeros((grid[0].size, 3))
            k = 0
            for a in range(3):
                if a == axis:
                    pts[:, a] = fixed
                else:
                    pts[:, a] = grid[k].reshape(-1)
                    k += 1
            axes.append(pts)
    return np.concatenate(axes, 0)


def make_scene(step=0.05, extent=2.0, box_size=0.5, rng=None):
    """One synthetic scene.

    Returns:
        coord: ``[N, 3]`` float positions.
        instance: ``[N]`` int instance ids.  Floor is 0, boxes are 1..K.
    """
    rng = rng or np.random.default_rng(0)
    pts = [(_box_surface((0.0, 0.0, 0.0), (extent, extent, step), step), 0)]

    for i, (cx, cy) in enumerate(rng.uniform(0.3, extent - box_size - 0.3, (3, 2))):
        pts.append(
            (
                _box_surface((cx, cy, step), (box_size, box_size, box_size), step),
                i + 1,
            )
        )

    coord = np.concatenate([p for p, _ in pts], 0)
    instance = np.concatenate(
        [np.full(p.shape[0], i, dtype=np.int64) for p, i in pts], 0
    )

    # voxelise: one point per unique voxel cell
    grid = np.floor(coord / step + 1e-6).astype(np.int64)
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    return coord[unique_idx], instance[unique_idx]


def build_batch(num_scenes=2, step=0.05, seed=0):
    """Voxelised batch in the layout the Volt backbone and the head expect."""
    coords, instances = [], []
    for s in range(num_scenes):
        c, inst = make_scene(step=step, rng=np.random.default_rng(seed + s))
        coords.append(c)
        instances.append(inst)

    coord = np.concatenate(coords, 0).astype(np.float32)
    instance = np.concatenate(instances, 0).astype(np.int64)

    # shift so that the minimum voxel coordinate is exactly 0, which is what
    # GridSample guarantees and what the tokenizer's floor-division relies on
    grid_coord = np.floor(coord / step + 1e-6).astype(np.int64)
    grid_coord -= grid_coord.min(0)

    batch = np.concatenate(
        [np.full(c.shape[0], s, dtype=np.int64) for s, c in enumerate(coords)]
    )

    # 6 input channels: centred position + a constant normal-ish term.  The
    # overfit test only needs *some* usable features.
    centre = coord.mean(0, keepdims=True)
    feat = np.concatenate([(coord - centre), np.ones_like(coord) * 0.5], 1).astype(
        np.float32
    )

    to = lambda x, d=None: torch.as_tensor(np.asarray(x), dtype=d)  # noqa: E731
    return dict(
        coord=to(coord),
        grid_coord=to(grid_coord),
        feat=to(feat, torch.float32),
        batch=to(batch),
        instance=to(instance),
    )


def patch_coord_of_voxel(grid_coord: torch.Tensor, kernel_size: int):
    """Kept for the visualiser: voxel coord -> patch coord."""
    return grid_coord // kernel_size
