"""Prompt encoder for click prompts (3-D points, no rays).

Deliberate differences from SAM2's ``PromptEncoder``:

* **Position enters twice, and it has to.**  RoPE supplies the
  translation-invariant *relative*-position bias in q/k, but RoPE does not rotate
  v, so a prompt token built from polarity alone carries no position into the
  value path.  Measured: clicks on two different objects produced a mean logit
  difference of ~1e-3, i.e. the click was ignored.  ``PositionEmbeddingRandom3D``
  (SAM2's random Fourier features, made 3-D) therefore also puts absolute position
  into the content.  ``pos_encoding="none"`` keeps the broken variant around as an
  ablation.
* **Polarity gets its own embedding, not a different rotation.**  Encoding
  polarity as a rotation angle would break the property that a click and a patch
  interact only through their offset (see ``attention.py``).
* **No ``not_a_point`` embedding.**  SAM2 needs it because its prompts live in a
  dense ``B x N`` tensor padded to a fixed length.  Packed varlen tokens have no
  padding.
* **No box prompts.**  A box in 3-D would have to be an oriented box, which is a
  separate feature with its own sampling questions.
* **The previous mask is pooled to patch level, not convolved.**  SAM2 runs a
  strided ``Conv2d`` stack over a 2-D mask.  Here the mask is a per-voxel logit
  vector while the two-way transformer consumes patch tokens, so the logits are
  scattered to patches with ``index_add_`` and lifted with an MLP.
"""

import torch
import torch.nn as nn

from .attention import MLP

POSITIVE = 0
NEGATIVE = 1
NUM_POINT_TYPES = 2


class PositionEmbeddingRandom3D(nn.Module):
    """Random Fourier features for a 3-D point, ported from SAM2's
    ``PositionEmbeddingRandom`` (Apache-2.0, Meta Platforms).

    Why this exists at all, given that the two-way attention already applies RoPE:

        RoPE rotates q and k, never v.  A prompt token whose value carries no
        position therefore reaches downstream tokens only through the attention
        *weights*, and a single click can modulate at most one scalar weight per
        layer.  Measured effect of a click on object A versus object B: a mean
        logit difference of ~1e-3, i.e. the click is effectively ignored.

    So position has to enter the *content* as well.  RoPE supplies the
    translation-invariant relative-position bias in q/k; this module supplies
    well-scaled absolute position in v.  SAM2 uses only this (no RoPE).

    The Gaussian matrix is a fixed buffer and is never trained.

    Args:
        num_pos_feats: output width is ``2 * num_pos_feats``.
        The coordinate scale is passed at call time, not fixed here.  A magic
        constant that must match the data is a footgun: with patch coords in
        ``0..7`` (small synthetic scene) a hardcoded ``coord_scale=64`` compresses
        them into a ~0.05 rad wedge of the Fourier basis, the encoding becomes
        almost constant across the scene, and the click silently stops carrying
        position again.  Callers pass the batch's own patch extent.
    """

    def __init__(self, num_pos_feats: int) -> None:
        super().__init__()
        # PERSISTENT on purpose.  This is a random but *fixed* projection that the
        # head depends on: if it is not saved, every reconstructed model gets a
        # different one and the trained head becomes meaningless.  The failure is
        # silent -- load_state_dict reports missing=0/unexpected=0 because a
        # non-persistent buffer is outside the comparison entirely.  Measured, it
        # took evaluation IoU from ~0.55 to ~0.00 on otherwise identical weights.
        # SAM2's PositionEmbeddingRandom registers this buffer as persistent too.
        self.register_buffer("gaussian_matrix", torch.randn((3, num_pos_feats)))

    def forward(
        self, coords: torch.Tensor, coord_scale: float | torch.Tensor
    ) -> torch.Tensor:
        """``[P, 3]`` integer patch coords -> ``[P, 2 * num_pos_feats]``."""
        x = coords.float() / coord_scale
        x = 2 * x - 1
        x = x @ self.gaussian_matrix
        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class PromptEncoder(nn.Module):
    """Click polarity embeddings + dense prompt from the previous mask logits.

    Args:
        embed_dim: token width, must match the two-way transformer.
        use_coord_content: also append an MLP of the (normalised) patch coords to
            the token content.  RoPE already carries position, so this is off by
            default; it is an ablation ("is relative position enough?").
        coord_scale: divisor for patch coords when ``use_coord_content`` is on.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_encoding: str = "fourier",
    ) -> None:
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.pos_encoding = pos_encoding

        self.type_emb = nn.Embedding(NUM_POINT_TYPES, embed_dim)
        if pos_encoding == "fourier":
            # PositionEmbeddingRandom3D(embed_dim // 2) emits embed_dim (sin + cos)
            self.coord_pe = PositionEmbeddingRandom3D(embed_dim // 2)
            self.fuse = nn.Linear(2 * embed_dim, embed_dim)
        elif pos_encoding == "mlp":
            self.coord_pe = MLP(3, embed_dim, embed_dim, num_layers=2)
            self.fuse = nn.Linear(2 * embed_dim, embed_dim)
        elif pos_encoding == "none":
            # the ablation: RoPE only.  Measured to leave the click ignored.
            self.coord_pe = None
            self.fuse = None
        else:
            raise ValueError(f"unknown pos_encoding: {pos_encoding}")

        # dense prompt: pooled previous logit (1 channel) -> token space
        self.prev_logit_mlp = MLP(
            1, embed_dim, embed_dim, num_layers=2, activation=nn.GELU
        )
        # learned "no previous mask" embedding, used on the first iteration
        self.no_prev_embed = nn.Embedding(1, embed_dim)

    def forward(
        self,
        click_patch: torch.Tensor,
        click_label: torch.Tensor,
        coord_scale: float | torch.Tensor,
    ) -> torch.Tensor:
        """``[P, 3]`` patch coords + ``[P]`` polarity -> ``[P, embed_dim]``.

        Args:
            coord_scale: patch-coordinate extent of the batch (patch extent, not
                metres).  Determines the resolution of the Fourier encoding; pass
                ``patch_coord.max() + 1`` so it adapts to the scene instead of
                relying on a constant that has to match the data.
        """
        pos = self.position_encoding(click_patch, coord_scale)
        if pos is None:
            return self.type_emb(click_label)
        # Fuse rather than add.  Adding a per-element N(0, 1) type embedding to the
        # position encoding dilutes the position: measured pairwise cosine
        # similarity between clicks on different objects is then 0.61-0.68 (0.00
        # for the raw encoding).  A linear fusion can scale the position channel
        # up, and its gradient is direct.
        return self.fuse(torch.cat([self.type_emb(click_label), pos], dim=-1))

    def position_encoding(
        self, click_patch: torch.Tensor, coord_scale: float | torch.Tensor
    ) -> torch.Tensor | None:
        """Raw positional content of a click, or ``None`` in the ``"none"`` mode."""
        if self.pos_encoding == "fourier":
            return self.coord_pe(click_patch, coord_scale)
        if self.pos_encoding == "mlp":
            return self.coord_pe(click_patch.float() / coord_scale)
        return None

    def dense_prompt_from_logits(
        self,
        prev_logits: torch.Tensor,
        inverse: torch.Tensor,
        num_patches: int,
    ) -> torch.Tensor:
        """Pool voxel-level previous logits to patch tokens and lift to token space.

        This replaces SAM2's ``_embed_masks`` (a strided ``Conv2d`` stack over a
        2-D mask).  Here the previous mask lives on *voxels* while the two-way
        transformer consumes *patch* tokens, so the logits are mean-pooled into
        patches with ``index_add_`` and then lifted with an MLP.

        Args:
            prev_logits: ``[N]`` previous mask logits, one per input voxel, in the
                same order as the backbone input (which is the order the mask head
                emits).
            inverse: ``[N]`` voxel -> patch index, as returned by the tokenizer.
            num_patches: number of patch tokens ``M``.

        Returns:
            ``[M, embed_dim]``, to be added to the patch tokens.
        """
        pooled = prev_logits.new_zeros(num_patches)
        counts = prev_logits.new_zeros(num_patches)
        pooled.index_add_(0, inverse, prev_logits)
        counts.index_add_(0, inverse, torch.ones_like(prev_logits))
        pooled = pooled / counts.clamp(min=1)
        return self.prev_logit_mlp(pooled.unsqueeze(-1))

    def no_prev_embedding(self, like: torch.Tensor) -> torch.Tensor:
        """Constant dense prompt used on the first iteration (no history)."""
        return self.no_prev_embed.weight.expand_as(like.view(-1, self.embed_dim)).view_as(like)
