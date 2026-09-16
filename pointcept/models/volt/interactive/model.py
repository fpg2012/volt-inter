"""Volt + interactive segmentation head.

The backbone is run in two stages so that the patch tokens produced by the
transformer blocks stay accessible:

    tokenizer -> blocks -> [coarse patch tokens] -> Detokenizer -> [fine voxel features]

``Volt.forward`` (``volt_base.py``) only exposes the final fine features, and it
is left untouched, so ``_backbone_forward`` below repeats those ~8 lines.  The
overfit script asserts that the fine features reconstructed this way are
identical to ``Volt.forward``'s output, so the duplication cannot drift silently.

Token counts that matter:
  * fine voxels  N ~ 10^4 .. 10^6  (grid_size 0.02 m)
  * patch tokens M ~ 10^4 .. 10^5  (kernel_size 5 -> 0.1 m patches)

The two-way transformer only ever runs at patch resolution; the mask is taken by
dot product at voxel resolution.  That keeps masks fine-grained without paying
attention cost at voxel resolution, which is the whole reason the port is
affordable.
"""

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch

from .attention import TwoWayTransformer, packed_seqlens
from .clicks import sample_clicks, target_voxel_mask
from .losses import InteractiveLoss
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


def scene_iou(logits, target, batch, num_scenes, thresh=0.5):
    """Per-scene IoU at a fixed threshold; ``0`` for scenes with no target.

    Accumulated in float32 even when ``logits`` is half precision under autocast:
    voxel counts per scene reach 10^5, where fp16 accumulation loses too much.
    """
    pred = logits.sigmoid() > thresh
    src_i = (pred & target).float()
    src_u = (pred | target).float()
    inter = src_i.new_zeros(num_scenes).index_add_(0, batch.long(), src_i)
    union = src_u.new_zeros(num_scenes).index_add_(0, batch.long(), src_u)
    return inter / union.clamp(min=1), union > 0


# keys that override click simulation; used by evaluation, where the clicks come
# from a user (or from a benchmark protocol) rather than from GT sampling
CLICK_KEYS = ("click_patch", "click_label", "click_batch", "target_instance")


@MODELS.register_module()
class VoltInteractive(nn.Module):
    """Class-agnostic interactive segmentation on top of a Volt backbone.

    Args:
        backbone: Volt config dict.
        transformer: two-way transformer config (``depth``, ``num_heads``,
            ``mlp_dim``, ``attention_downsample_rate``, ``rope_theta``).
            ``embedding_dim`` is taken from the backbone so prompt tokens and
            patch tokens live in one space.
        prompt: ``PromptEncoder`` config.
        mask_dim: width of the mask embedding used by the dot-product mask head.
        mask_source: where the mask embedding comes from -- ``"prompt"`` (the
            positive click tokens, refined by the transformer; recommended),
            ``"token"`` (SAM2's learned mask token), or ``"both"``.
        use_iou_head: keep SAM2's IoU prediction head (cheap auxiliary task, and
            useful for early stopping / best-of-k selection).
        freeze_backbone: train only the interactive head.
        loss: ``InteractiveLoss`` config.
        refinement_prob: probability of running a second decoder pass that
            consumes the first pass's logits as the dense prompt.  This is what
            teaches the "previous mask" pathway; 0 disables it.
        clicks: click sampling config (``num_click_range``, ``neg_ratio``).
        fine_dim: channel count of the backbone's voxel features.  Inferred from
            the backbone's Detokenizer when ``None``.
    """

    def __init__(
        self,
        backbone,
        transformer=None,
        prompt=None,
        mask_dim: int = 128,
        mask_source: str = "prompt",
        use_iou_head: bool = True,
        iou_weight: float = 0.5,
        freeze_backbone: bool = False,
        loss=None,
        refinement_prob: float = 0.5,
        clicks=None,
        fine_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = build_model(backbone)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # the token width is whatever the backbone's attention actually uses
        embed_dim = self.backbone.blocks[0].attn.qkv.in_features
        # channel count of the features the Detokenizer emits
        if fine_dim is None:
            fine_dim = self.backbone.decoder.unembed.out_channels

        transformer = dict(transformer or {})
        attention_downsample_rate = transformer.pop("attention_downsample_rate", 2)
        self.transformer = TwoWayTransformer(
            embedding_dim=embed_dim,
            attention_downsample_rate=attention_downsample_rate,
            **transformer,
        )

        # The raw transformer-block output is unnormalised: measured std 13.3 on
        # real ScanNet with the released checkpoint, against 0.13 for a randomly
        # initialised backbone.  Fed straight in as keys/values that saturates the
        # two-way attention (logits ~|q||k| ~ 100) and swamps the dense prompt from
        # the previous mask (std ~0.24).  The backbone's own decoder begins with a
        # BatchNorm for precisely this reason, so normalise here too.
        self.patch_norm = nn.LayerNorm(embed_dim)

        self.prompt_encoder = PromptEncoder(embed_dim, **(prompt or {}))
        self.mask_decoder = MaskDecoder(
            transformer_dim=embed_dim,
            transformer=self.transformer,
            fine_dim=fine_dim,
            mask_dim=mask_dim,
            mask_source=mask_source,
            use_iou_head=use_iou_head,
        )

        self.criteria = InteractiveLoss(**(loss or {}))
        self.use_iou_head = use_iou_head
        self.iou_weight = iou_weight
        self.refinement_prob = refinement_prob
        self.click_cfg = dict(clicks or {})

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode.

        ``requires_grad=False`` is not enough.  Volt's decoder contains
        BatchNorm1d, so a train-mode backbone derives its features from batch
        statistics -- i.e. from whichever other scenes happen to share the batch
        -- while inference uses running statistics.  Pinning eval mode is the
        correct treatment and costs nothing.

        Honest caveat on magnitude: with real batches (~240k voxels over 12
        scenes) the batch statistics are already very close to the running ones.
        Measured, the two modes agree to cosine 0.9875 on the fine features, so
        this is hygiene rather than a fix for a large error.  An earlier version
        of this comment blamed a 0.88 -> 0.10 evaluation gap on BatchNorm; that
        was wrong, the cause was the non-persistent buffer in
        ``PositionEmbeddingRandom3D`` (see prompt_encoder.py).
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    def _backbone_forward(self, data_dict):
        """Two-stage backbone forward; returns patch tokens and fine features.

        Mirrors ``Volt.forward``; any change there must be mirrored here.  The
        overfit script checks the fine features agree with ``Volt.forward``.

        With a frozen backbone this runs under ``no_grad``: none of its outputs
        need to be differentiated (they are constants as far as the head is
        concerned), which skips 12 transformer layers worth of saved activations.
        The eval-mode requirement is handled by ``train()`` above.
        """
        import contextlib

        ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with ctx:
            backbone = self.backbone
            grid_coord = data_dict["grid_coord"]
            feat = data_dict["feat"]
            batch = data_dict["batch"]

            indices = torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous()

            tokens, coarse_indices, inverse, offset_id = backbone.tokenizer(feat, indices)
            cu_seqlens, max_seqlen = backbone.compute_seqlens(coarse_indices[:, 0])
            freqs_cis = backbone.pos_enc.compute_axial_cis_efficient(
                coarse_indices[:, 1:]
            )
            for blk in backbone.blocks:
                tokens = blk(tokens, freqs_cis, cu_seqlens, max_seqlen)

            fine = backbone.decoder(tokens, inverse, offset_id)
        return tokens, coarse_indices, inverse, fine

    def _decode(
        self,
        patch_tokens,
        prompt_content,
        click_patch,
        click_label,
        click_batch,
        num_scenes,
        patch_coord,
        cu_patches,
        max_patches,
        fine,
        voxel_batch,
        dense_prompt,
        click_pos_encoding=None,
    ):
        return self.mask_decoder(
            patch_tokens=patch_tokens,
            click_patch=click_patch,
            click_label=click_label,
            prompt_content=prompt_content,
            click_batch=click_batch,
            num_scenes=num_scenes,
            patch_coord=patch_coord,
            cu_patches=cu_patches,
            max_patches=max_patches,
            fine_features=fine,
            voxel_batch=voxel_batch,
            dense_prompt=dense_prompt,
            click_pos_encoding=click_pos_encoding,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _get_batch(data_dict):
        """The Pointcept pipeline delivers ``offset``; the synthetic harness and
        the tests use ``batch``.  Accept both."""
        if "batch" in data_dict:
            return data_dict["batch"]
        if "offset" in data_dict:
            return offset2batch(data_dict["offset"])
        raise KeyError("data_dict needs either 'batch' or 'offset'")

    def precompute_backbone(self, data_dict):
        """Run the frozen backbone once and cache its outputs.

        Patch tokens and fine voxel features depend only on the scene, never on the
        clicks.  With a frozen backbone they are therefore constants that can be
        computed once and reused across every task, click set and refinement step.
        Without this, real-data overfitting re-runs 12 transformer layers over
        hundreds of thousands of voxels on every training step, which dominates the
        step time entirely.

        The backbone is forced into eval mode here.  Volt's decoder contains
        BatchNorm1d, so a cache built in train mode would hold batch statistics
        while everything downstream uses running statistics.  As in ``train()``
        this is correct hygiene rather than a large effect (measured cosine 0.9875
        between the two modes), but the cache must at least be built in the same
        mode it is consumed in, or the whole point of caching is lost.
        """
        if not self.freeze_backbone:
            raise RuntimeError(
                "precompute_backbone is only valid for a frozen backbone; "
                "caching features whose parameters still receive gradients "
                "would detach the backbone from the loss."
            )
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            batch = self._get_batch(data_dict)
            with torch.no_grad():
                tokens, coarse_indices, inverse, fine = self._backbone_forward(
                    {**data_dict, "batch": batch}
                )
        finally:
            if was_training:
                self.backbone.train()
        return dict(
            patch_tokens=tokens,
            coarse_indices=coarse_indices,
            inverse=inverse,
            fine=fine,
        )

    def forward(self, data_dict, backbone_cache=None, return_logits: bool = False):
        """
        Args:
            backbone_cache: optional output of ``precompute_backbone``, to skip the
                (frozen) backbone entirely.
            return_logits: also return the non-scalar tensors (``seg_logits``,
                ``target``, the click tensors).  It defaults to False because
                Pointcept's ``InformationWriter`` calls ``.item()`` on *every* key
                of the output dict, so multi-element tensors would crash training.
        """
        instance = data_dict["instance"]
        batch = self._get_batch(data_dict)
        data_dict = {**data_dict, "batch": batch}
        num_scenes = int(batch.max().item()) + 1

        if backbone_cache is None:
            patch_tokens, coarse_indices, inverse, fine = self._backbone_forward(
                data_dict
            )
        else:
            patch_tokens = backbone_cache["patch_tokens"]
            coarse_indices = backbone_cache["coarse_indices"]
            inverse = backbone_cache["inverse"]
            fine = backbone_cache["fine"]

        patch_tokens = self.patch_norm(patch_tokens)
        patch_batch = coarse_indices[:, 0].long()
        patch_coord = coarse_indices[:, 1:].long()
        cu_patches, max_patches = packed_seqlens(patch_batch, num_seqs=num_scenes)
        num_patches = patch_tokens.shape[0]

        if all(k in data_dict for k in CLICK_KEYS):
            # evaluation path: clicks are given, not simulated
            click_patch = data_dict["click_patch"].to(batch.device)
            click_label = data_dict["click_label"].to(batch.device)
            click_batch = data_dict["click_batch"].to(batch.device)
            target_instance = data_dict["target_instance"].to(batch.device)
        else:
            click_patch, click_label, click_batch, target_instance = sample_clicks(
                instance,
                batch,
                data_dict["grid_coord"],
                kernel_size=self.backbone.tokenizer.kernel_size,
                generator=data_dict.get("click_generator"),
                **self.click_cfg,
            )
        # Fourier-encode clicks relative to the batch's patch extent, so the
        # encoding resolution adapts to the scene rather than to a constant.
        coord_scale = patch_coord.max().item() + 1
        prompt_content = self.prompt_encoder(click_patch, click_label, coord_scale)
        # raw position encoding, kept separate so the mask embedding always has
        # a direct path to the click location (see MaskDecoder.forward)
        click_pos_encoding = self.prompt_encoder.position_encoding(
            click_patch, coord_scale
        )
        target = target_voxel_mask(instance, target_instance, batch)

        info = {
            "num_patches": torch.tensor(float(num_patches), device=batch.device),
            "num_clicks": torch.tensor(float(click_patch.shape[0]), device=batch.device),
        }

        # ---- pass 0: no previous mask --------------------------------
        dense0 = self.prompt_encoder.no_prev_embedding(patch_tokens)
        logits0, iou0 = self._decode(
            patch_tokens,
            prompt_content,
            click_patch,
            click_label,
            click_batch,
            num_scenes,
            patch_coord,
            cu_patches,
            max_patches,
            fine,
            batch,
            dense0,
            click_pos_encoding,
        )
        losses = self.criteria(logits0, target, batch)
        total = losses["loss"]
        logits_used = logits0

        # SAM2 trains the IoU head with MSE against the real mask IoU.  Without
        # this the head's parameters receive no gradient at all.
        iou0_scene, has_target = scene_iou(logits0.detach(), target, batch, num_scenes)
        if self.use_iou_head and iou0 is not None and has_target.any():
            iou_loss = torch.nn.functional.mse_loss(
                iou0[has_target].float(), iou0_scene[has_target]
            )
            losses["loss_iou"] = iou_loss
            total = total + self.iou_weight * iou_loss

        # ---- pass 1: previous mask as dense prompt --------------------
        use_refine = self.refinement_prob > 0 and (
            not self.training or torch.rand(1).item() < self.refinement_prob
        )
        if use_refine:
            dense1 = self.prompt_encoder.dense_prompt_from_logits(
                logits0.detach(), inverse, num_patches
            )
            logits1, iou1 = self._decode(
                patch_tokens,
                prompt_content,
                click_patch,
                click_label,
                click_batch,
                num_scenes,
                patch_coord,
                cu_patches,
                max_patches,
                fine,
                batch,
                dense1,
                click_pos_encoding,
            )
            for k, v in self.criteria(logits1, target, batch).items():
                losses[f"refine/{k}"] = v
            total = total + losses[f"refine/loss"]
            if self.use_iou_head and iou1 is not None:
                iou1_scene, _ = scene_iou(logits1.detach(), target, batch, num_scenes)
                iou_loss1 = torch.nn.functional.mse_loss(
                    iou1[has_target].float(), iou1_scene[has_target]
                )
                losses["refine/loss_iou"] = iou_loss1
                total = total + self.iou_weight * iou_loss1
            logits_used = logits1

        losses["loss"] = total

        # ---- diagnostics ---------------------------------------------
        if has_target.any():
            info["iou_step0"] = iou0_scene[has_target].mean()
        if use_refine and has_target.any():
            iou1_scene, _ = scene_iou(logits_used, target, batch, num_scenes)
            info["iou_step1"] = iou1_scene[has_target].mean()
        if iou0 is not None:
            info["iou_pred_step0"] = iou0.mean()

        # seg_logits / target / clicks are only returned on request: see the
        # docstring on ``return_logits``.
        out = {**info, **losses}
        if return_logits:
            out.update(
                seg_logits=logits_used,
                target=target,
                click_patch=click_patch,
                click_label=click_label,
                click_batch=click_batch,
                click_row=data_dict.get("click_row"),
            )
        return out
