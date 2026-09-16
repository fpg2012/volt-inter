"""Low-cost training visualisation.

No Open3D / no window: a point set is rendered to a PNG buffer with matplotlib's
Agg backend and handed to TensorBoard's ``add_image``.  Three orthographic
projections (xy / xz / yz) are drawn side by side, which is enough to see whether
the predicted mask is drifting, splitting, or latching onto a neighbouring
object, and whether the clicks landed where they should.

Cost is one matplotlib figure per logged step and is intended to be run every
N steps, not every step.
"""

import io

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# colour scheme: agreement / false positive / false negative / ignored background
C_BG = (0.82, 0.82, 0.82)
C_BOTH = (0.15, 0.60, 0.25)
C_PRED = (0.85, 0.33, 0.10)
C_GT = (0.20, 0.40, 0.85)
C_POS_CLICK = (0.0, 0.75, 0.0)
C_NEG_CLICK = (0.85, 0.0, 0.0)

VIEWS = {
    "xy": (0, 1, "x", "y"),
    "xz": (0, 2, "x", "z"),
    "yz": (1, 2, "y", "z"),
}


def _to_numpy(x, dtype=None):
    """Uniform tensor/array -> numpy, moving off GPU (visualisation only)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(dtype) if dtype is not None else x


def _colors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    c = np.tile(np.array(C_BG), (pred.shape[0], 1))
    c[gt & ~pred] = C_GT
    c[pred & ~gt] = C_PRED
    c[pred & gt] = C_BOTH
    return c


def render_mask_panel(
    coord,
    pred,
    gt,
    clicks=None,
    click_label=None,
    title: str = "",
    point_size: float = 2.0,
    dpi: int = 90,
) -> np.ndarray:
    """Render three orthographic projections as an ``[H, W, 3]`` uint8 image.

    Args:
        coord: ``[N, 3]`` float positions.
        pred: ``[N]`` bool predicted mask.
        gt: ``[N]`` bool ground-truth mask.
        clicks: optional ``[P, 3]`` float click positions (same frame as coord).
        click_label: optional ``[P]`` 0 = positive, 1 = negative.
    """
    coord = _to_numpy(coord)
    pred = _to_numpy(pred, bool)
    gt = _to_numpy(gt, bool)

    colors = _colors(pred, gt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=dpi)
    for ax, (name, (i, j, li, lj)) in zip(axes, VIEWS.items()):
        ax.scatter(coord[:, i], coord[:, j], c=colors, s=point_size, linewidths=0)
        if clicks is not None and len(clicks):
            ck = _to_numpy(clicks)
            labels = _to_numpy(click_label)
            for k in range(ck.shape[0]):
                pos = bool(labels[k] == 0)
                ax.scatter(
                    ck[k, i],
                    ck[k, j],
                    marker="+" if pos else "x",
                    s=70,
                    linewidths=2.0,
                    c=C_POS_CLICK if pos else C_NEG_CLICK,
                    zorder=5,
                )
        inter = int((pred & gt).sum())
        union = int((pred | gt).sum())
        iou = inter / union if union else 1.0
        ax.set_title(f"{name}   IoU={iou:.3f}", fontsize=9)
        ax.set_xlabel(li, fontsize=8)
        ax.set_ylabel(lj, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)[:, :, :3]


def render_click_progress(curves: dict, title: str = "interactive refinement") -> np.ndarray:
    """Plot IoU-vs-step curves in the style of the NoC/IoU@k protocol."""
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=100)
    for name, values in curves.items():
        ax.plot(range(1, len(values) + 1), values, marker="o", label=name, linewidth=1.5)
    ax.set_xlabel("refinement step")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)[:, :, :3]
