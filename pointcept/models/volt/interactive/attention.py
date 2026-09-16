"""Flash-attention based building blocks for the interactive segmentation head.

Ported from SAM2's ``sam2/modeling/sam/transformer.py`` (Apache-2.0,
Copyright (c) Meta Platforms, Inc. and affiliates) and adapted to
variable-length point cloud token sequences:

1. All token tensors are *packed* 1-D sequences of shape ``[N_total, C]``
   (the flash-attn varlen convention) instead of ``[B, N, C]`` or ``[B, C, H, W]``.
2. Additive positional encodings are replaced by Volt's axial 3-D RoPE
   (see ``pointcept/models/volt/volt_base.py:RoPE``).

Note on RoPE vs SAM2's positional encoding
-------------------------------------------
SAM2 adds ``image_pe`` / ``query_pe`` to the *token* vectors, so one positional
encoding of size ``embedding_dim`` serves every attention in the block regardless
of ``attention_downsample_rate``.

RoPE rotates ``q``/``k`` *inside* the attention, so its frequency budget is tied
to ``head_dim = internal_dim // num_heads``.  Since the two-way block mixes
``downsample_rate=1`` (self-attention) and ``downsample_rate=2`` (both
cross-attentions), the block needs **one RoPE per distinct head_dim**.  This is
handled by ``RoPEBank``: attentions receive *positions* rather than precomputed
frequencies, and each builds frequencies in its own basis.

Consequence worth knowing: with ``attention_downsample_rate=2`` the cross
attention gets half the frequency bands of the backbone.  Setting the rate to 1
and ``embedding_dim // num_heads == 64`` lets the head reuse the backbone's
frequency basis exactly, which is why that configuration is worth an ablation.

Original SAM2 two-way block layout (unchanged):
    (1) self-attention over the sparse tokens
    (2) cross-attention sparse -> dense (token to patch)
    (3) MLP
    (4) cross-attention dense -> sparse (patch to token)
"""

import flash_attn
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.volt.volt_base import RoPE


class MLP(nn.Module):
    """Ported verbatim from SAM2's ``sam2_utils.py``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: type = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


def apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply axial RoPE to a packed token tensor.

    Args:
        x: ``[N, num_heads, head_dim]``.
        freqs_cis: complex tensor broadcastable to ``[N, 1, head_dim // 2]``.
            Rows whose frequency is ``1 + 0j`` are left untouched, which is how
            position-less tokens (the learned output tokens) are handled.
    """
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
    return x_out.type_as(x).contiguous()


def freq_split_for_head_dim(head_dim: int) -> tuple:
    """Split the complex dimensions of one head over the x/y/z axes.

    Mirrors Volt's backbone ratio ``(12, 12, 8)`` for 64 real dims, i.e.
    ``3 / 8, 3 / 8, 2 / 8`` of the available complex dimensions.
    """
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    c = head_dim // 2
    n_x = round(c * 3 / 8)
    n_y = round(c * 3 / 8)
    n_z = c - n_x - n_y
    assert n_z > 0, f"head_dim {head_dim} too small for 3-D axial RoPE"
    return (n_x, n_y, n_z)


def make_rope(
    head_dim: int,
    theta: float = 100.0,
    max_grid_size: tuple = (1024, 1024, 512),
) -> RoPE:
    """Build a ``volt_base.RoPE`` whose frequency budget matches ``head_dim``."""
    rope = RoPE(
        theta=theta,
        freq_split=freq_split_for_head_dim(head_dim),
        max_grid_size=max_grid_size,
    )
    # RoPE keeps freq_split as a local; expose it so downstream code (and the
    # head_dim consistency check) does not have to reverse-engineer the caches.
    rope.freq_split = freq_split_for_head_dim(head_dim)
    return rope


def rope_complex_dims(rope: nn.Module) -> int:
    """Number of complex dimensions a RoPE instance produces per head.

    Read from the precomputed cache tables so that instances built elsewhere
    (e.g. the backbone's own ``RoPE``) are handled too.
    """
    if hasattr(rope, "freq_split"):
        return sum(rope.freq_split)
    return sum(
        getattr(rope, f"cis_cache_{axis}").shape[-1] for axis in ("x", "y", "z")
    )


def build_axial_freqs(
    rope: nn.Module,
    grid_coord: torch.Tensor,
    pos_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build RoPE frequencies for a packed token sequence.

    Args:
        rope: a ``volt_base.RoPE`` instance.
        grid_coord: ``[N_total, 3]`` integer patch coordinates for every token.
            Values at unpositioned rows are ignored (but must index the cache).
        pos_mask: optional ``[N_total]`` boolean, ``True`` where the token has a
            spatial position.  Unpositioned tokens get an identity rotation,
            which makes ``apply_rotary`` a no-op on them -- this replaces SAM2's
            ``skip_first_layer_pe``.

    Returns:
        complex tensor of shape ``[N_total, 1, head_dim // 2]``.
    """
    grid_coord = grid_coord.long().clamp(min=0)
    cis = rope.compute_axial_cis_efficient(grid_coord)  # [1, N_total, C_cis]
    cis = cis.squeeze(0)
    if pos_mask is not None:
        cis = torch.where(pos_mask.unsqueeze(-1), cis, torch.ones_like(cis))
    return cis.unsqueeze(1)


def flash_dtype(x: torch.Tensor) -> torch.dtype:
    """flash-attn only accepts fp16/bf16; follow the incoming dtype when it is
    already half precision, otherwise fall back to fp16 (as Volt does)."""
    return torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float16


def packed_seqlens(batch: torch.Tensor, num_seqs: int | None = None) -> tuple:
    """``[N_total]`` packed token scene ids -> ``cu_seqlens``, ``max_seqlen``.

    Same convention as ``Volt.compute_seqlens`` (``volt_base.py:266``), which
    adds 1 so that the first cumulative sum entry is 0.

    ``num_seqs`` matters for *click* sequences: a scene may contribute no clicks
    at all (no annotated instance), and without ``minlength`` bincount would
    return a short tensor and every downstream ``cu[b]`` lookup would be off by
    the number of empty scenes.
    """
    minlength = num_seqs + 1 if num_seqs is not None else 0
    counts = torch.bincount(batch + 1, minlength=minlength)
    cu_seqlens = torch.cumsum(counts, dim=0, dtype=torch.int32)
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    return cu_seqlens, int(seqlens.max().item()) if seqlens.numel() else 0


class FlashAttention(nn.Module):
    """SAM2's ``Attention`` (with ``downsample_rate`` / ``kv_in_dim``) rewritten
    on top of ``flash_attn_varlen_func``.

    ``use_flash=False`` switches to a per-sequence ``scaled_dot_product_attention``
    loop.  That path makes the module runnable on CPU for debugging and is the
    numerical reference used by the equivalence test.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int | None = None,
        use_flash: bool = True,
        rope: RoPE | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.head_dim = self.internal_dim // num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        if rope is not None:
            c = rope_complex_dims(rope)
            assert c * 2 == self.head_dim, (
                f"RoPE has {c} complex dims ({2 * c} real) but head_dim is "
                f"{self.head_dim}; each mismatch needs its own RoPE instance"
            )
        self.rope = rope

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout
        self.use_flash = use_flash

    def _project(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        return proj(x).view(-1, self.num_heads, self.head_dim)

    def _rope(self, x, coord, pos_mask):
        if self.rope is None or coord is None:
            return x
        return apply_rotary(x, build_axial_freqs(self.rope, coord, pos_mask))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        q_coord: torch.Tensor | None = None,
        k_coord: torch.Tensor | None = None,
        q_pos_mask: torch.Tensor | None = None,
        k_pos_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._rope(self._project(q, self.q_proj), q_coord, q_pos_mask)
        k = self._rope(self._project(k, self.k_proj), k_coord, k_pos_mask)
        v = self._project(v, self.v_proj)

        dtype = q.dtype
        dropout_p = self.dropout_p if self.training else 0.0

        if self.use_flash:
            out = flash_attn.flash_attn_varlen_func(
                q.to(flash_dtype(q)),
                k.to(flash_dtype(k)),
                v.to(flash_dtype(v)),
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=dropout_p,
            )
            out = out.reshape(-1, self.internal_dim)
        else:
            # Reference path: one SDPA call per packed sequence.
            chunks = []
            for b in range(len(cu_seqlens_q) - 1):
                qs, qe = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
                ks, ke = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
                # [N, H, D] -> [1, H, N, D]
                chunk = F.scaled_dot_product_attention(
                    q[qs:qe].transpose(0, 1).unsqueeze(0),
                    k[ks:ke].transpose(0, 1).unsqueeze(0),
                    v[ks:ke].transpose(0, 1).unsqueeze(0),
                    dropout_p=dropout_p,
                )
                chunks.append(chunk.squeeze(0).transpose(0, 1))
            out = torch.cat(chunks, dim=0).reshape(-1, self.internal_dim)

        return self.out_proj(out.to(dtype))


class TwoWayAttentionBlock(nn.Module):
    """Ported from SAM2.

    ``skip_first_layer_pe`` is gone: with RoPE applied inside attention there is
    no separate PE to skip, because a position-less token is expressed as an
    identity rotation (``1 + 0j``).

    ``rope_self`` and ``rope_cross`` must have head_dim
    ``embedding_dim // num_heads`` and
    ``embedding_dim // (attention_downsample_rate * num_heads)`` respectively.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type = nn.ReLU,
        attention_downsample_rate: int = 2,
        use_flash: bool = True,
        rope_self: RoPE | None = None,
        rope_cross: RoPE | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = FlashAttention(
            embedding_dim, num_heads, use_flash=use_flash, rope=rope_self
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = FlashAttention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            use_flash=use_flash,
            rope=rope_cross,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = FlashAttention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            use_flash=use_flash,
            rope=rope_cross,
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        q_coord: torch.Tensor | None = None,
        k_coord: torch.Tensor | None = None,
        q_pos_mask: torch.Tensor | None = None,
    ):
        # (1) self-attention over the sparse tokens
        queries = self.norm1(
            queries
            + self.self_attn(
                queries,
                queries,
                queries,
                cu_seqlens_q,
                cu_seqlens_q,
                max_seqlen_q,
                max_seqlen_q,
                q_coord,
                q_coord,
                q_pos_mask,
                q_pos_mask,
            )
        )

        # (2) cross-attention, tokens attending to the dense patch tokens
        queries = self.norm2(
            queries
            + self.cross_attn_token_to_image(
                queries,
                keys,
                keys,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                q_coord,
                k_coord,
                q_pos_mask,
                None,
            )
        )

        # (3) MLP
        queries = self.norm3(queries + self.mlp(queries))

        # (4) cross-attention, dense patch tokens attending to the tokens.
        #     The dense tokens are updated here, so their K/V cannot be cached
        #     across clicks.  Fine: the decoder is only 2 layers deep.
        keys = self.norm4(
            keys
            + self.cross_attn_image_to_token(
                keys,
                queries,
                queries,
                cu_seqlens_k,
                cu_seqlens_q,
                max_seqlen_k,
                max_seqlen_q,
                k_coord,
                q_coord,
                None,
                q_pos_mask,
            )
        )

        return queries, keys


class TwoWayTransformer(nn.Module):
    """Ported from SAM2.  The dense feature map is replaced by a packed token
    sequence, so the ``flatten(2).permute(0, 2, 1)`` dance is gone and the
    batch dimension is carried by ``cu_seqlens`` instead.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type = nn.ReLU,
        attention_downsample_rate: int = 2,
        use_flash: bool = True,
        rope_theta: float = 100.0,
        rope_self: RoPE | None = None,
        rope_cross: RoPE | None = None,
        max_grid_size: tuple = (1024, 1024, 512),
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.use_flash = use_flash

        # one RoPE per distinct head_dim in the block (see module docstring)
        if rope_self is None:
            rope_self = make_rope(
                embedding_dim // num_heads, rope_theta, max_grid_size
            )
        if rope_cross is None:
            rope_cross = make_rope(
                embedding_dim // (attention_downsample_rate * num_heads),
                rope_theta,
                max_grid_size,
            )
        self.rope_self = rope_self
        self.rope_cross = rope_cross

        self.layers = nn.ModuleList(
            [
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    use_flash=use_flash,
                    rope_self=rope_self,
                    rope_cross=rope_cross,
                )
                for _ in range(depth)
            ]
        )

        self.final_attn_token_to_image = FlashAttention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            use_flash=use_flash,
            rope=rope_cross,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,
        queries: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        q_coord: torch.Tensor | None = None,
        k_coord: torch.Tensor | None = None,
        q_pos_mask: torch.Tensor | None = None,
    ):
        keys = image_tokens
        for layer in self.layers:
            queries, keys = layer(
                queries,
                keys,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                q_coord,
                k_coord,
                q_pos_mask,
            )

        queries = self.norm_final_attn(
            queries
            + self.final_attn_token_to_image(
                queries,
                keys,
                keys,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                q_coord,
                k_coord,
                q_pos_mask,
                None,
            )
        )

        return queries, keys
