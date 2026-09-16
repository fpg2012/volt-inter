"""Mask decoder, ported from SAM2's ``MaskDecoder`` to point cloud tokens.

Kept from SAM2 (shape-agnostic, ported as-is):
  * the learned output tokens (mask / IoU),
  * the hypernetwork MLP that turns the mask token into a mask embedding,
  * the mask2former-style dot product ``mask_embed . pixel_embed``,
  * the IoU prediction head.

Dropped on purpose:
  * ``num_multimask_outputs`` and the stability-based multimask selection
    (``_get_stability_scores``, ``_dynamic_multimask_via_stability``).
    A single granularity is wanted here, so there is exactly one mask token.
    Re-enabling later only needs ``num_mask_tokens > 1`` plus the selection
    logic; the shape of this module already allows it.
  * ``obj_score_token`` / ``pred_obj_scores`` (SAM2's video occlusion state,
    no analogue in single-shot refinement).
  * ``output_upscaling`` (``ConvTranspose2d``) and ``high_res_features`` (FPN).
    Volt's own ``Decoder``/``Detokenizer`` already produces fine voxel features,
    so the dot product is taken directly against those instead of upsampling a
    coarse mask grid.  This is what keeps masks fine-grained while the two-way
    transformer only ever runs at patch resolution.

One deviation in token order: SAM2 puts the IoU token first, this puts the mask
token first, so the mask token is always at index 0 of an output-token block.
"""

import torch
import torch.nn as nn

from .attention import MLP, TwoWayTransformer, packed_seqlens


class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: TwoWayTransformer,
        fine_dim: int,
        mask_dim: int = 128,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_iou_head: bool = True,
        mask_source: str = "prompt",
        prompt_summary: bool = True,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.mask_dim = mask_dim
        self.use_iou_head = use_iou_head
        self.prompt_summary = prompt_summary
        assert mask_source in ("prompt", "token", "both"), mask_source
        self.mask_source = mask_source

        # fuses [transformer-refined click | raw click content | raw position
        # encoding] into the mask embedding
        self.prompt_fuse = nn.Linear(3 * transformer_dim, transformer_dim)

        # a single mask token: no multimask disambiguation
        self.mask_token = nn.Embedding(1, transformer_dim)
        self.num_out_tokens = 1
        if use_iou_head:
            self.iou_token = nn.Embedding(1, transformer_dim)
            self.num_out_tokens += 1

        self.output_hypernetwork_mlp = MLP(
            transformer_dim, transformer_dim, mask_dim, num_layers=3, activation=nn.GELU
        )
        self.pixel_proj = nn.Linear(fine_dim, mask_dim)

        if use_iou_head:
            self.iou_prediction_head = MLP(
                transformer_dim, iou_head_hidden_dim, 1, iou_head_depth
            )

    # ------------------------------------------------------------------
    def build_queries(
        self,
        prompt_content: torch.Tensor,
        click_patch: torch.Tensor,
        click_batch: torch.Tensor,
        num_scenes: int,
    ):
        """Pack ``[output tokens | click tokens]`` per scene.

        Output tokens are position-less.  That is expressed with
        ``pos_mask=False``, which ``apply_rotary`` turns into an identity
        rotation -- this replaces SAM2's ``skip_first_layer_pe``.

        Returns ``(queries, cu_q, max_q, q_coord, q_pos_mask, cu_click)``.
        """
        device = prompt_content.device
        out_tokens = [self.mask_token.weight]
        if self.use_iou_head:
            out_tokens.append(self.iou_token.weight)
        out_tokens = torch.cat(out_tokens, 0)  # [num_out_tokens, D]

        if click_batch.numel():
            cu_click, _ = packed_seqlens(click_batch, num_seqs=num_scenes)
        else:
            cu_click = torch.zeros(num_scenes + 1, dtype=torch.int32, device=device)
        click_counts = torch.diff(cu_click).long()

        queries, coords, mask = [], [], []
        for b in range(num_scenes):
            queries.append(out_tokens)
            coords.append(
                torch.zeros(
                    self.num_out_tokens, 3, dtype=torch.long, device=device
                )
            )
            mask.append(
                torch.zeros(self.num_out_tokens, dtype=torch.bool, device=device)
            )
            ps, pe = int(cu_click[b]), int(cu_click[b + 1])
            if pe > ps:
                queries.append(prompt_content[ps:pe])
                coords.append(click_patch[ps:pe])
                mask.append(torch.ones(pe - ps, dtype=torch.bool, device=device))

        queries = torch.cat(queries, 0)
        cu_q, max_q = packed_seqlens(
            torch.repeat_interleave(
                torch.arange(num_scenes, device=device),
                click_counts + self.num_out_tokens,
            )
        )
        return (
            queries,
            cu_q,
            max_q,
            torch.cat(coords, 0),
            torch.cat(mask, 0),
            cu_click,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        patch_tokens: torch.Tensor,
        click_patch: torch.Tensor,
        click_label: torch.Tensor,
        prompt_content: torch.Tensor,
        click_batch: torch.Tensor,
        num_scenes: int,
        patch_coord: torch.Tensor,
        cu_patches: torch.Tensor,
        max_patches: int,
        fine_features: torch.Tensor,
        voxel_batch: torch.Tensor,
        dense_prompt: torch.Tensor,
        click_pos_encoding: torch.Tensor | None = None,
    ):
        """Returns per-voxel logits ``[N]`` and predicted IoU ``[B]`` (or None)."""
        queries, cu_q, max_q, q_coord, q_pos_mask, cu_click = self.build_queries(
            prompt_content, click_patch, click_batch, num_scenes
        )

        # dense prompt (pooled previous mask logits) is added to the patch tokens
        hs, _ = self.transformer(
            patch_tokens + dense_prompt,
            queries,
            cu_q,
            cu_patches,
            max_q,
            max_patches,
            q_coord,
            patch_coord,
            q_pos_mask,
        )

        # output tokens start each scene's query block
        starts = cu_q[:-1].long()
        mask_token_out = hs[starts]  # [B, D]

        # Per-scene row bookkeeping.  ``is_prompt`` selects the click rows of the
        # packed query tensor; they are in the same order as ``prompt_content``.
        row_scene = torch.repeat_interleave(
            torch.arange(num_scenes, device=hs.device),
            torch.diff(cu_q).long(),
        )
        offset_in_scene = torch.arange(hs.shape[0], device=hs.device) - cu_q[
            row_scene
        ].long()
        is_prompt = offset_in_scene >= self.num_out_tokens
        prompt_rows = torch.nonzero(is_prompt, as_tuple=False).flatten()
        click_scene = row_scene[prompt_rows]

        def scene_mean(src, keep):
            sel_scene = click_scene[keep]
            # dtype-explicit: under autocast ``src`` may be half while fresh
            # ``torch.ones`` would be float32, and index_add_ requires a match
            counts = src.new_zeros(num_scenes).index_add_(
                0,
                sel_scene,
                torch.ones(int(keep.sum()), device=src.device, dtype=src.dtype),
            )
            summed = src.new_zeros(num_scenes, src.shape[-1]).index_add_(
                0, sel_scene, src[keep]
            )
            return summed / counts.clamp(min=1).unsqueeze(-1)

        if self.mask_source in ("prompt", "both") and click_label.numel():
            # The mask embedding is read off the *positive* click tokens, with the
            # raw click content and the raw position encoding concatenated in.
            #
            # Why not the learned mask token alone: RoPE rotates q/k, never v, and
            # the mask token carries no position, so its only route to the click
            # location is via the attention weights.  Measured, the transformer's
            # output for a click has pairwise cosine similarity 0.76-0.82 across
            # clicks on *different* objects (0.00 for the raw position encoding),
            # i.e. the residual stream washes the position out.  Deriving the mask
            # embedding from the click tokens, with an explicit position path, fits
            # the 4-task synthetic benchmark at IoU 1.000; the learned-token route
            # does not train at all.  SAM can afford the learned token because it
            # is trained on ~1B masks.
            keep = click_label.long() == 0
            ctx = scene_mean(hs[prompt_rows], keep)
            raw = scene_mean(prompt_content, keep)
            parts = [ctx, raw]
            if click_pos_encoding is not None:
                parts.append(scene_mean(click_pos_encoding, keep))
            else:
                parts.append(torch.zeros_like(ctx))
            prompt_repr = self.prompt_fuse(torch.cat(parts, dim=-1))
            if self.mask_source == "both":
                prompt_repr = prompt_repr + mask_token_out
            mask_token_out = prompt_repr
        elif self.prompt_summary and prompt_rows.numel() > 0:
            # "token" mode with the prompt-summary shortcut bolted on
            keep_all = torch.ones(
                prompt_rows.numel(), dtype=torch.bool, device=hs.device
            )
            mask_token_out = mask_token_out + scene_mean(hs[prompt_rows], keep_all)

        # LayerNorm normalises away scale, and the fused output feeds a 3-layer MLP;
        # running it in fp32 keeps ``prompt_fuse`` numerically safe under autocast.
        mask_token_out = mask_token_out.float()

        iou_pred = None
        if self.use_iou_head:
            iou_pred = self.iou_prediction_head(hs[starts + 1]).squeeze(-1)

        # mask2former-style: one mask embedding per scene, dotted with every voxel
        hyper_in = self.output_hypernetwork_mlp(mask_token_out)  # [B, mask_dim]
        pixel_embed = self.pixel_proj(fine_features)  # [N, mask_dim]
        logits = (pixel_embed * hyper_in[voxel_batch.long()]).sum(-1)

        return logits, iou_pred
