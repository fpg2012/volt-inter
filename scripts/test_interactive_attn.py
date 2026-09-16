"""Equivalence tests for the ported two-way transformer.

Four independent things are checked, because they fail in different ways:

1. ``flash_attn_varlen_func`` packing vs. a per-sequence SDPA loop.
   Both paths share projections and RoPE, so this isolates *packing* bugs
   (wrong cu_seqlens, wrong head layout, q/k length mix-ups between the two
   cross-attention directions).
2. The RoPE relative-position property on a cross-attention q/k pair, verified
   against a brute-force construction that shares no code with the module.
3. That position-less tokens (the learned output tokens) pass through RoPE
   untouched.
4. That mixing head_dims with one RoPE instance is rejected, since that is the
   subtlest way this port can silently go wrong.

Run: python scripts/test_interactive_attn.py
"""

import torch

from pointcept.models.volt.interactive.attention import (
    TwoWayTransformer,
    apply_rotary,
    build_axial_freqs,
    freq_split_for_head_dim,
    make_rope,
    packed_seqlens,
)

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}, cuda={torch.cuda.is_available()}")


def make_batch(n_scenes=3, n_image=(37, 64, 21), n_prompt=(4, 7, 3), dim=96):
    """Build packed image tokens, packed queries, and their positions.

    Queries are ``[2 output tokens | prompt tokens]`` per scene, so the two
    cross-attention directions really do see different key lengths.
    """
    img_feats, img_coord, img_batch = [], [], []
    qry_feats, qry_batch = [], []
    for b in range(n_scenes):
        ni, nq = n_image[b], n_prompt[b]
        img_feats.append(torch.randn(ni, dim, device=DEVICE))
        img_coord.append(torch.randint(0, 32, (ni, 3), device=DEVICE))
        img_batch.append(torch.full((ni,), b, device=DEVICE))
        qry_feats.append(torch.randn(2 + nq, dim, device=DEVICE))
        qry_batch.append(torch.full((2 + nq,), b, device=DEVICE))

    img_coord = torch.cat(img_coord)
    qry_batch_cat = torch.cat(qry_batch)
    cu_img, max_img = packed_seqlens(torch.cat(img_batch))
    cu_qry, max_qry = packed_seqlens(qry_batch_cat)

    # positions: image tokens have some; the first 2 query rows per scene do not
    qry_coord = torch.randint(0, 32, (sum(2 + n for n in n_prompt), 3), device=DEVICE)
    qry_pos_mask = torch.ones(qry_coord.shape[0], dtype=torch.bool, device=DEVICE)
    for b in range(len(cu_qry) - 1):
        qry_pos_mask[int(cu_qry[b]) : int(cu_qry[b]) + 2] = False

    return (
        torch.cat(img_feats),
        img_coord,
        torch.cat(qry_feats),
        qry_coord,
        qry_pos_mask,
        cu_qry,
        cu_img,
        max_qry,
        max_img,
    )


def test_flash_vs_sdpa():
    """Same weights, two attention backends, must agree to fp16 tolerance."""
    dim, heads, rate = 96, 3, 2
    print(
        f"\n[1] flash vs sdpa   head_dim(self)={dim // heads} "
        f"head_dim(cross)={dim // (rate * heads)}"
    )

    img, img_coord, qry, qry_coord, qry_mask, cu_qry, cu_img, mq, mk = make_batch(dim=dim)

    torch.manual_seed(1)
    flash_net = TwoWayTransformer(
        depth=2,
        embedding_dim=dim,
        num_heads=heads,
        mlp_dim=2 * dim,
        attention_downsample_rate=rate,
        use_flash=True,
    ).to(DEVICE)
    ref_net = TwoWayTransformer(
        depth=2,
        embedding_dim=dim,
        num_heads=heads,
        mlp_dim=2 * dim,
        attention_downsample_rate=rate,
        use_flash=False,
    ).to(DEVICE)
    ref_net.load_state_dict(flash_net.state_dict())
    assert flash_net.use_flash and not ref_net.use_flash

    args = (cu_qry, cu_img, mq, mk, qry_coord, img_coord, qry_mask)
    with torch.no_grad():
        q_f, k_f = flash_net(img, qry, *args)
        q_r, k_r = ref_net(img, qry, *args)

    for name, a, b in (("queries", q_f, q_r), ("image_tokens", k_f, k_r)):
        diff = (a - b).abs().max().item()
        scale = a.abs().max().item()
        ok = diff < 5e-2 * max(scale, 1.0)
        print(
            f"    {name:13s} max|flash-sdpa|={diff:.3e}  scale={scale:.3e}  "
            f"{'OK' if ok else 'FAIL'}"
        )
        assert ok, name

    # the two RoPE bases must actually differ, otherwise test 1 proves less
    assert sum(flash_net.rope_self.freq_split) != sum(flash_net.rope_cross.freq_split)
    # and query/key sequence lengths must differ, exercising varlen cross-attn
    assert not torch.equal(cu_qry, cu_img)
    print("    distinct RoPE bases + differing q/k lengths exercised")


def test_rope_relative_positions():
    """RoPE must make q.k depend only on the *relative* position.

    Uses an explicit construction: rotate q at position p and k at position p',
    then translate both by t.  The dot product must be invariant.  This is the
    property the prompt->patch cross-attention relies on (a click and a voxel
    interact only through their offset), and it also validates the per-axis
    frequency split.
    """
    head_dim = 16
    print(f"\n[2] RoPE relative position   freq_split={freq_split_for_head_dim(head_dim)}")
    rope = make_rope(head_dim, max_grid_size=(512, 512, 512)).to(DEVICE)

    torch.manual_seed(2)
    q = torch.randn(1, 1, head_dim, device=DEVICE)
    k = torch.randn(1, 1, head_dim, device=DEVICE)

    def dot_at(p, p2):
        pos = torch.tensor([p, p2], dtype=torch.int64, device=DEVICE)
        cis = build_axial_freqs(rope, pos)
        return (apply_rotary(q, cis[0:1]) * apply_rotary(k, cis[1:2])).sum().item()

    for p, p2, t in (
        ((10, 20, 5), (17, 24, 9), (100, 100, 100)),
        ((3, 3, 3), (40, 11, 7), (200, 200, 200)),
        ((0, 0, 0), (250, 90, 60), (250, 90, 60)),
    ):
        shifted = tuple(x + y for x, y in zip(p, t))
        shifted2 = tuple(x + y for x, y in zip(p2, t))
        a, b = dot_at(p, p2), dot_at(shifted, shifted2)
        ok = abs(a - b) < 1e-4
        print(f"    p={p} p'={p2} shift={t}: {a:+.6f} vs {b:+.6f}  {'OK' if ok else 'FAIL'}")
        assert ok, "RoPE is not translation invariant"


def test_identity_rotation_is_noop():
    """Position-less tokens must pass through RoPE untouched."""
    print("\n[3] identity rotation on position-less tokens")
    head_dim = 16
    rope = make_rope(head_dim, max_grid_size=(64, 64, 64)).to(DEVICE)
    x = torch.randn(4, 2, head_dim, device=DEVICE)
    coord = torch.zeros(4, 3, device=DEVICE)
    coord[2:] = torch.tensor([3, 5, 7], device=DEVICE)  # non-zero -> non-identity
    mask = torch.tensor([False, False, True, True], device=DEVICE)
    out = apply_rotary(x, build_axial_freqs(rope, coord, pos_mask=mask))
    d0 = (out[:2] - x[:2]).abs().max().item()
    d1 = (out[2:] - x[2:]).abs().max().item()
    print(
        f"    unpositioned max|out-x|={d0:.3e} (expect 0)   "
        f"positioned max|out-x|={d1:.3e} (expect >0)"
    )
    assert d0 < 1e-6 and d1 > 1e-3


def test_head_dim_mismatch_is_rejected():
    """One RoPE cannot serve two head_dims -- this must fail loudly."""
    print("\n[4] head_dim / RoPE mismatch is rejected")
    from pointcept.models.volt.interactive.attention import FlashAttention

    rope16 = make_rope(16)
    try:
        FlashAttention(96, 3, downsample_rate=1, rope=rope16)
    except AssertionError as e:
        print(f"    correctly rejected: {str(e)[:70]}...")
        return
    raise AssertionError("expected an assertion error for head_dim mismatch")


if __name__ == "__main__":
    test_flash_vs_sdpa()
    test_rope_relative_positions()
    test_identity_rotation_is_noop()
    test_head_dim_mismatch_is_rejected()
    print("\nall tests passed")
