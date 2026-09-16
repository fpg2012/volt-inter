"""Interactive click simulation.

Clicks are sampled **on voxels**, never in a continuous world frame: a click is
identified with a voxel index and its prompt position is that voxel's patch
coordinate, ``grid_coord // kernel_size``.  This matters because the backbone's
``grid_coord`` has already been shifted by ``min_coord`` inside
``GridSample`` (see ``pointcept/datasets/transform.py``), so the absolute
coordinate frame of ``coord`` and of ``grid_coord`` differ.  Working through
voxel indices sidesteps the whole question and guarantees the prompt lands in
exactly the same frame the backbone's RoPE was built from.

This is also what makes the port cheap at inference time: a real click in world
space is converted by finding the nearest voxel, then taking that voxel's patch
coordinate.
"""

import torch

# Voxels with this instance id carry no annotation (see ScanNetDataset).
IGNORE_INSTANCE = -1

# 0 = positive click, 1 = negative click
NEGATIVE = 1
POSITIVE = 0


def voxel_to_patch(grid_coord: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """``[N, 3]`` voxel grid coords -> ``[N, 3]`` patch grid coords.

    Mirrors ``Tokenizer.forward`` (``volt_base.py``), which does the same floor
    division to decide which patch a voxel belongs to.
    """
    return grid_coord // kernel_size


def _target_voxel_mask(instance: torch.Tensor, target: int) -> torch.Tensor:
    return instance == target


def sample_clicks(
    instance: torch.Tensor,
    batch: torch.Tensor,
    grid_coord: torch.Tensor,
    kernel_size: int,
    num_click_range=(1, 4),
    neg_ratio=0.5,
    generator: torch.Generator | None = None,
):
    """Sample a prompt set per scene.

    Every scene gets at least one positive click, and the first positive click is
    the instance voxel closest to that instance's centroid -- the "obvious" click
    a user would make, and the setting most interactive papers report.

    Returns:
        click_patch:  ``[P, 3]`` integer patch coords, packed over all scenes.
        click_label:  ``[P]`` 0 = positive, 1 = negative.
        click_batch:  ``[P]`` scene id per click.
        target_instance: ``[B]`` the instance each scene is asked to segment.
            ``-1`` when the scene has no annotated instance (loss must skip it).
    """
    B = int(batch.max().item()) + 1
    click_patch, click_label, click_batch, targets = [], [], [], []

    for b in range(B):
        rows = torch.nonzero(batch == b, as_tuple=False).flatten()
        inst = instance[rows]
        coords = grid_coord[rows]

        present = torch.unique(inst[inst >= 0])
        if present.numel() == 0:
            targets.append(-1)
            continue

        target = present[
            torch.randint(
                present.numel(), (1,), generator=generator, device=instance.device
            ).item()
        ]
        targets.append(int(target))

        tgt_rows = rows[inst == target]
        # first positive click: the voxel nearest the instance centroid
        centroid = grid_coord[tgt_rows].float().mean(dim=0, keepdim=True)
        order = torch.argsort(
            (grid_coord[tgt_rows].float() - centroid).pow(2).sum(-1)
        )
        tgt_rows = tgt_rows[order]

        n_pos = int(
            torch.randint(
                num_click_range[0],
                num_click_range[1] + 1,
                (1,),
                generator=generator,
                device=instance.device,
            ).item()
        )
        n_pos = min(n_pos, tgt_rows.numel())
        n_neg = int(round(n_pos * neg_ratio))

        chosen = [(int(tgt_rows[0]), POSITIVE)]
        # remaining positives: random instance voxels
        if n_pos > 1 and tgt_rows.numel() > 1:
            pick = torch.randperm(
                tgt_rows.numel() - 1, generator=generator, device=instance.device
            )[: n_pos - 1]
            chosen += [(int(tgt_rows[i + 1]), POSITIVE) for i in pick]

        # negatives: voxels outside the target instance
        neg_pool = rows[inst != target]
        if n_neg > 0 and neg_pool.numel() > 0:
            pick = torch.randint(
                neg_pool.numel(),
                (min(n_neg, neg_pool.numel()),),
                generator=generator,
                device=instance.device,
            )
            chosen += [(int(neg_pool[i]), NEGATIVE) for i in pick]

        # deduplicate identical patch coords, keeping the earlier (ordered) click
        seen = set()
        for row, label in chosen:
            patch = tuple((grid_coord[row] // kernel_size).tolist())
            if patch in seen:
                continue
            seen.add(patch)
            click_patch.append(grid_coord[row] // kernel_size)
            click_label.append(label)
            click_batch.append(b)

    if not click_patch:
        empty = grid_coord.new_zeros((0, 3))
        return (
            empty,
            grid_coord.new_zeros((0,), dtype=torch.long),
            grid_coord.new_zeros((0,), dtype=torch.long),
            torch.tensor(targets, device=instance.device, dtype=instance.dtype),
        )

    return (
        torch.stack(click_patch),
        torch.tensor(click_label, device=instance.device, dtype=torch.long),
        torch.tensor(click_batch, device=instance.device, dtype=torch.long),
        torch.tensor(targets, device=instance.device, dtype=instance.dtype),
    )


def centroid_click_row(instance, batch, grid_coord, target: int, scene: int):
    """Row index of the instance voxel nearest its centroid, or ``None``.

    This is the click a user would make when aiming at an object, and it is what
    interactive papers report when they simulate clicks on GT.
    """
    rows = torch.nonzero((batch == scene) & (instance == target), as_tuple=False)
    if rows.numel() == 0:
        return None
    rows = rows.flatten()
    centroid = grid_coord[rows].float().mean(dim=0, keepdim=True)
    order = torch.argsort((grid_coord[rows].float() - centroid).pow(2).sum(-1))
    return int(rows[order[0]])


def enumerate_instance_clicks(
    instance: torch.Tensor,
    batch: torch.Tensor,
    grid_coord: torch.Tensor,
    kernel_size: int,
    num_scenes: int | None = None,
    add_negative: bool = True,
):
    """A deterministic, low-variance prompt set: one task per instance id.

    Task ``k`` asks every scene in the batch to segment its instance ``k``, using a
    centroid click (plus one negative click on a different instance).  Instance ids
    that do not occur in every scene are skipped, so the batch stays homogeneous
    and the loss needs no per-scene masking.

    Used by the overfit test.  Every task uses a single positive click, so the
    prompt token content is identical across tasks and only the *position*
    differs -- a model that ignores click position cannot tell them apart and the
    test fails.  That is exactly the failure this is meant to catch.

    Returns a list of dicts ready to be merged into ``data_dict``.
    """
    num_scenes = num_scenes if num_scenes is not None else int(batch.max().item()) + 1
    per_scene = [
        set(
            torch.unique(instance[(batch == b) & (instance >= 0)])
            .cpu()
            .tolist()
        )
        for b in range(num_scenes)
    ]
    common = sorted(set.intersection(*per_scene)) if per_scene else []

    tasks = []
    for target in common:
        patch, label, scene = [], [], []
        for b in range(num_scenes):
            row = centroid_click_row(instance, batch, grid_coord, target, b)
            if row is None:
                continue
            patch.append(grid_coord[row] // kernel_size)
            label.append(POSITIVE)
            scene.append(b)
            if add_negative:
                others = torch.unique(
                    instance[(batch == b) & (instance >= 0) & (instance != target)]
                )
                if others.numel():
                    neg_row = centroid_click_row(
                        instance, batch, grid_coord, int(others[0]), b
                    )
                    if neg_row is not None:
                        patch.append(grid_coord[neg_row] // kernel_size)
                        label.append(NEGATIVE)
                        scene.append(b)
        if not patch:
            continue
        tasks.append(
            dict(
                click_patch=torch.stack(patch),
                click_label=torch.tensor(label, dtype=torch.long),
                click_batch=torch.tensor(scene, dtype=torch.long),
                target_instance=torch.full(
                    (num_scenes,),
                    target,
                    dtype=instance.dtype,
                    device=instance.device,
                ),
            )
        )
    return tasks


def make_task(
    instance: torch.Tensor,
    batch: torch.Tensor,
    grid_coord: torch.Tensor,
    kernel_size: int,
    targets,
    num_scenes: int | None = None,
    add_negative: bool = True,
):
    """Per-scene target instance ids -> a click/target dict for ``data_dict``.

    Unlike ``enumerate_instance_clicks`` this does not require the same instance
    id to exist in every scene, which is what real ScanNet needs: each scene has
    its own arbitrary set of instance ids.

    Args:
        targets: one instance id per scene.  If an id is absent from its scene,
            that scene falls back to its smallest available id; ``-1`` means
            "no target", and the scene contributes no clicks and an all-False
            target mask.

    Returns a dict with ``click_patch`` / ``click_label`` / ``click_batch`` /
    ``target_instance``, plus ``click_row`` (voxel indices) purely so that
    visualisation can read positions straight out of ``coord`` and avoid having
    to reason about which coordinate frame it is in.
    """
    num_scenes = num_scenes if num_scenes is not None else int(batch.max().item()) + 1
    targets = list(targets)
    assert len(targets) == num_scenes

    patch, label, scene, rows = [], [], [], []
    resolved = []
    for b in range(num_scenes):
        present = torch.unique(instance[(batch == b) & (instance >= 0)])
        if present.numel() == 0 or targets[b] < 0:
            resolved.append(-1)
            continue
        target = int(targets[b])
        if target not in set(present.cpu().tolist()):
            target = int(present[0])
        resolved.append(target)

        row = centroid_click_row(instance, batch, grid_coord, target, b)
        if row is None:
            resolved[-1] = -1
            continue
        patch.append(grid_coord[row] // kernel_size)
        label.append(POSITIVE)
        scene.append(b)
        rows.append(row)

        if add_negative:
            others = torch.unique(
                instance[(batch == b) & (instance >= 0) & (instance != target)]
            )
            if others.numel():
                neg_row = centroid_click_row(
                    instance, batch, grid_coord, int(others[0]), b
                )
                if neg_row is not None:
                    patch.append(grid_coord[neg_row] // kernel_size)
                    label.append(NEGATIVE)
                    scene.append(b)
                    rows.append(neg_row)

    empty = patch[0].new_zeros((0, 3)) if patch else grid_coord.new_zeros((0, 3))
    return dict(
        click_patch=torch.stack(patch) if patch else empty,
        click_label=torch.tensor(label, dtype=torch.long, device=instance.device),
        click_batch=torch.tensor(scene, dtype=torch.long, device=instance.device),
        click_row=torch.tensor(rows, dtype=torch.long, device=instance.device),
        target_instance=torch.tensor(
            resolved, dtype=instance.dtype, device=instance.device
        ),
    )


def target_voxel_mask(instance: torch.Tensor, target_instance: torch.Tensor, batch: torch.Tensor):
    """``[N]`` float target mask for the binary interactive loss."""
    valid = batch < target_instance.numel()
    per_voxel_target = torch.zeros_like(batch)
    per_voxel_target[valid] = target_instance[batch[valid]]
    return (instance == per_voxel_target) & (per_voxel_target >= 0)
