"""Binary losses for interactive segmentation.

The task is class-agnostic: predict the mask of the clicked instance.  Both
losses operate on raw logits over the whole point set of a scene, with no
``ignore_index`` handling -- ``clicks.target_voxel_mask`` already guarantees
that scenes without an annotated target are excluded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target.float())


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    batch: torch.Tensor | None = None,
    eps: float = 1.0,
) -> torch.Tensor:
    """Soft Dice, computed per scene and then averaged.

    ``eps`` follows SAM (1.0, not a tiny value): with a single-instance target
    the mask is often small, and a tiny eps makes the loss explode on empty-ish
    predictions.
    """
    prob = logits.sigmoid().flatten()
    target = target.float().flatten()

    if batch is not None:
        # scatter into per-scene numerator/denominator
        B = int(batch.max().item()) + 1
        num = prob.new_zeros(B).index_add_(0, batch, prob * target)
        den = prob.new_zeros(B).index_add_(0, batch, prob + target)
        return (1 - (2 * num + eps) / (den + eps)).mean()

    return 1 - (2 * (prob * target).sum() + eps) / ((prob + target).sum() + eps)


class InteractiveLoss(nn.Module):
    """``bce_weight * BCE + dice_weight * soft Dice``.

    ``negative_ratio`` enables class-balanced sampling: per scene, only
    ``negative_ratio * n_positive`` negatives (chosen at random) take part in the
    loss.  This is not a detail.  A whole-scene BCE on a single-instance target
    has roughly 10% positives, and from a generic initialisation the model
    collapses to predicting everything negative; measured on synthetic scenes the
    loss sat flat at 1.20 and IoU stayed at 0.000 indefinitely.  With balanced
    sampling the same setup starts learning immediately.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        negative_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.negative_ratio = negative_ratio

    def _balanced_subset(self, target, batch, num_scenes):
        """Per-scene positive voxels plus a bounded random sample of negatives."""
        idx = []
        for b in range(num_scenes):
            rows = torch.nonzero(batch == b, as_tuple=False).flatten()
            pos = rows[target[rows]]
            neg = rows[~target[rows]]
            keep = int(max(pos.numel() * self.negative_ratio, 1))
            if neg.numel() > keep:
                perm = torch.randperm(neg.numel(), device=neg.device)[:keep]
                neg = neg[perm]
            if pos.numel() == 0:
                # keep the scene in the batch so per-scene Dice stays defined
                pos = rows[:1]
            idx.append(torch.cat([pos, neg]))
        return torch.cat(idx)

    def forward(self, logits, target, batch):
        if self.negative_ratio is not None:
            num_scenes = int(batch.max().item()) + 1
            idx = self._balanced_subset(target, batch, num_scenes)
            logits, target, batch = logits[idx], target[idx], batch[idx]

        out = {}
        total = logits.new_zeros(())
        if self.bce_weight:
            bce = sigmoid_ce_loss(logits, target)
            out["loss_bce"] = bce
            total = total + self.bce_weight * bce
        if self.dice_weight:
            dice = dice_loss(logits, target, batch)
            out["loss_dice"] = dice
            total = total + self.dice_weight * dice
        out["loss"] = total
        return out


def iou_from_logits(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5):
    """Plain mask IoU at a fixed threshold (used by the overfit test)."""
    pred = logits.sigmoid() > thresh
    target = target.bool()
    inter = (pred & target).sum().float()
    union = (pred | target).sum().float()
    return (inter / union.clamp(min=1)).item()
