# Interactive segmentation on ScanNet: class-agnostic, click-prompted instance masks.
#
# This file is intentionally not a semantic-segmentation config:
#   * the target is a binary mask of ONE clicked instance, not a class map
#   * clicks are simulated on GT voxels during training (see clicks.py)
#   * the backbone is frozen, so the run fits a single 12 GB GPU in a few hours
#
# Run:
#   python tools/train.py --config-file configs/scannet/insseg-volt-interactive-0-base.py --num-gpus 1
#
# TensorBoard (the trainer writes one automatically):
#   tensorboard --logdir exp/insseg-volt-interactive-0-base
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_200,
)

_base_ = ["../_base_/default_runtime.py"]

# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
# Sizing measured with scripts/make_interactive_subset.py on a single RTX 4070
# (12 GB), frozen backbone, AMP:
#
#   batch   s/step   scenes/hour   peak GPU
#     8      0.761      37,867       6.00 GiB
#    12      0.839-0.892  48-51k      7.05-7.48 GiB   <- best throughput
#    16      1.195      48,189       8.74 GiB
#
# batch 12 is ~28% faster than batch 8 and batch 16 is no faster than 12.
#
# 6 h at batch 12 = 24,200 steps = 290k samples.  The full train split is 1201
# scenes -> 100 steps/epoch, so the WHOLE SPLIT fits in 6 h, and a subset would
# only reduce data diversity (make_interactive_subset.py will happily suggest
# "epoch = 1030" for a 300-scene subset, which is 1030 passes over 300 scenes --
# strictly worse than 180 passes over all 1201).
#
# epoch=180 = 18,000 steps.  At the measured 0.89 s/step that is 4.5 h, leaving
# ~35% headroom for sustained-run slowdown (page cache, thermal, other GPU users).
epoch = 180
eval_epoch = 180
evaluate = False  # no semantic-seg evaluator applies; see scripts/eval_interactive.py
test_only = False

batch_size = 12
num_worker = 10

# Mixing blends coordinates and features of two scenes.  The collator does remap
# instance ids for it, but for a click-prompted task the mixed geometry is not
# something clicks can be sampled from meaningfully, so it stays off.
mix_prob = 0

enable_amp = True
amp_dtype = "float16"
clip_grad = 1.0
empty_cache = False
use_ema = False
sync_bn = False
enable_wandb = False  # set True if you want wandb in addition to TensorBoard
find_unused_parameters = True

# ---------------------------------------------------------------------------
# pretrained backbone
# ---------------------------------------------------------------------------
# NOTE: do NOT use weights/volt-small-scannet.pth -- that file is corrupt (its
# zip central directory cannot be read) despite being referenced by the existing
# insseg-spformer-volt-*-base.py configs.  These two HuggingFace runs load cleanly:
#   joint_training_small : embed_dim 384, depth 12, num_heads 6,  up_mlp_dim 128
#   joint_training_base  : embed_dim 768, depth 12, num_heads 12, up_mlp_dim 256
# This config uses "small"; for "base" swap the backbone block below for
# pointcept.models.volt.interactive.checkpoint.backbone_config("base") values.
weight = "weights/hf/Volt_experiments/joint_training_small/scannet/model/model_last.pth"

# --- backbone: must match the checkpoint exactly, these are not free choices ---
# head_dim is fixed at 64 by Volt itself: pos_enc = RoPE() is hardcoded with
# freq_split=(12, 12, 8) in volt_base.py, i.e. 32 complex = 64 real dims, so
# embed_dim // num_heads must be 64.
# drop_path is 0 because the backbone is frozen: it is a feature extractor here,
# and stochastic depth would only inject noise into its (cached) features.
backbone = dict(
    type="Volt",
    in_channels=6,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    init_values=None,  # no LayerScale parameters in the checkpoint
    qk_norm=True,  # q_norm/k_norm exist in the checkpoint
    drop_path=0.0,
    stride=5,
    kernel_size=5,
    increase_drop_path=False,
    up_mlp_dim=128,
)

model = dict(
    type="VoltInteractive",
    backbone=backbone,
    # The two-way transformer only ever runs at patch resolution (~7k tokens per
    # scene); the mask is taken by dot product at voxel resolution, so masks stay
    # fine without paying attention cost at voxel resolution.
    transformer=dict(
        depth=2,
        num_heads=6,
        mlp_dim=1536,
        # rate=2 halves the cross-attention head_dim to 32, so the head uses a
        # RoPE basis of its own.  rate=1 would keep head_dim 64 and let it reuse
        # the backbone's exact basis at 2x the attention cost -- an ablation.
        attention_downsample_rate=2,
    ),
    # Position enters twice on purpose: RoPE (relative, inside q/k) plus this
    # Fourier encoding (absolute, inside the value path).  RoPE alone does not
    # rotate v, which measurably leaves the click ignored.
    prompt=dict(pos_encoding="fourier"),
    mask_dim=128,
    # "prompt" reads the mask embedding off the positive click tokens.  "token"
    # is SAM2's learned mask token, which does not train at this scale.
    mask_source="prompt",
    use_iou_head=True,
    iou_weight=0.5,
    freeze_backbone=True,
    # Whole-scene BCE on a single-instance target is ~10% positive and collapses
    # to all-negative; balanced sampling is what makes it train.
    loss=dict(bce_weight=1.0, dice_weight=1.0, negative_ratio=2.0),
    # probability of a second decoder pass that consumes pass 0's logits
    refinement_prob=1.0,
    clicks=dict(num_click_range=(1, 4), neg_ratio=0.5),
)

# ---------------------------------------------------------------------------
# optimisation: only the interactive head is trained (~6.5 M params)
# ---------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

# Scene-level subset.  NOT used by default: the whole train split fits the time
# budget (see above), and using fewer scenes only reduces data diversity.
#
# Subset files live under data/ and are therefore gitignored; generate one with
# (deterministic, seed 0):
#   python scripts/make_interactive_subset.py --subset-size 300 --batch-size 12
# At batch 12 a 300-scene subset is 25 steps/epoch, so `epoch = 20` is a ~7 minute
# smoke run and `epoch = 180` an ~80 minute one.  The script measures the step
# time and prints the matching epoch.
#
lr_file = None  # e.g. "data/scannet/tasks/insseg_interactive/subset.txt"

# Transforms are lighter than the semantic-segmentation recipe on purpose.  The
# backbone is frozen, so expensive feature-space augmentation (ElasticDistortion)
# buys little while dominating CPU time.  RandomScale/RandomRotate/RandomFlip keep
# the head robust to placement, and they are applied to coordinates only, which is
# exactly what the click sampler needs to stay consistent (clicks are sampled from
# grid_coord after these transforms).
train_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "instance"),
        feat_keys=("color", "normal"),
    ),
]

data = dict(
    num_classes=200,
    ignore_index=-1,
    names=CLASS_LABELS_200,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        lr_file=lr_file,
        transform=train_transform,
        test_mode=False,
    ),
)

# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------
# Deliberately no SemSegEvaluator / PreciseEvaluator: they consume ``seg_logits``
# as a class map, which this model does not produce.  The trainer still writes
# TensorBoard scalars for every scalar entry of the model output (loss, loss_bce,
# refine/loss, iou_step0, iou_pred_step0, num_clicks, ...).
hooks = [
    dict(
        type="CheckpointLoader",
        # the released checkpoints are DDP saves with a ``module.`` prefix; this
        # maps ``module.backbone.*`` onto the model's ``backbone.*`` and leaves
        # the multi-dataset seg heads to be ignored (strict=False)
        keywords="module.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

# Tester: not applicable to a promptable model, see scripts/eval_interactive.py
test = dict(type="SemSegTester", verbose=False)
