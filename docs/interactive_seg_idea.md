# 交互式分割：设计思路

基于 Volt backbone，做一个类 SAM 的交互式分割头：用 two-way transformer 做 image/prompt 双向融合，再上采样并由类 Mask2Former 的解码方式输出 logits，其中上一轮的 mask logit 也作为输入反馈回去（类似 SAM 的 mask memory），以支持迭代式 refinement。

Prompt 用 3D 点而不是射线。射线虽然更贴近真实交互（屏幕点击本质是一条射线，且不要求点云先被解析成点），但在训练数据构造和**负点击**的表达上都更麻烦——点提示下"负样本"是一个确定的 3D 点，而射线的正负是逐视角定义的，实现和评测都复杂得多，所以先不做。

Prompt encoder 复用 Volt backbone 的轴向 3D RoPE 来编码点击位置。注意正/负点击的区分不应该是"旋转不同角度"，而应该用独立的 learned type embedding——RoPE 的旋转角语义上是位置（模型读的是 q·k 的相对角度），把正负编码进角度会破坏相对位置的解释，也容易把 backbone 预训练好的位置编码带偏。

---

# 实现笔记与实测结论

以下都是在本机上跑出来验证过的，数字来自 `scripts/overfit_*.py` 和 `scripts/make_interactive_subset.py`。

## 代码位置

```
pointcept/models/volt/interactive/
├── attention.py       # SAM2 two-way transformer → varlen + 轴向 3D RoPE
├── prompt_encoder.py  # 点击 content（type embedding + Fourier PE）、prev logit 池化
├── mask_decoder.py    # 单 mask token、hypernetwork、点积出 mask、IoU 头
├── clicks.py          # 点击模拟（含确定性 task 集）
├── losses.py          # BCE + Dice + 类别平衡采样
├── model.py           # VoltInteractive：两段式 backbone + 上述组件
├── checkpoint.py      # 加载官方 Volt 权重
├── synthetic.py       # 合成盒体场景（合成 overfit 用）
└── viz.py             # matplotlib 三板投影，无 GUI
```

## 设计决策（都被消融验证过）

**1. 位置必须同时进入 q/k 和 value。**
RoPE 只旋转 q/k，不旋转 v。若 prompt token 只有 type embedding，点击位置只能通过**一个标量注意力权重**影响下游，实测不同物体的点击产生的 logit 差异只有 ~1e-3（logit 尺度 0.2），即点击被忽略。
⇒ 最终方案：RoPE（相对位置，在 q/k 里）+ SAM 的 random Fourier PE（绝对位置，在 content/v 里）。`pos_encoding="none"` 保留为消融，会失败。

**2. mask embedding 从正点击 token 导出，而不是 SAM 的 learned mask token。**
实测 transformer 输出给点击 token 的表示，在不同物体之间的 pairwise 余弦相似度是 0.76–0.82（原始位置编码是 0.00）——残差流把位置洗掉了。用 learned mask token 的路线在合成 4 任务基准上**完全训不起来**（IoU 0）；换成从正点击 token 取 mask embedding 后 IoU = 1.000。
SAM 能用纯 learned mask token 是因为它训了 ~1B 个 mask，这个 head 没有。

**3. 类别平衡的损失采样是必需项，不是调优项。**
单实例目标约占场景 10% 正类。整场景 BCE 从随机初始化出发会 collapse 到全负：实测 loss 平在 1.20、IoU 恒为 0.000，永远不会自己走出来。`negative_ratio` 打开后立刻开始收敛。
⇒ `loss.negative_ratio=2.0`，即每个场景取 2× 正类数的随机负类参与损失。

**4. 注意力只在 patch 分辨率跑，mask 在体素分辨率取。**
`logits = pixel_proj(fine_voxel) · hyper_in`。two-way transformer 只看 ~7k patch token/场景；细粒度由 Volt 的 `Decoder`/`Detokenizer` 提供。这是 head 能负担得起的原因。

**5. `attention_downsample_rate` 决定 RoPE 频率基。**
two-way block 里 self-attn 是 rate=1、两个 cross-attn 是 rate=2，head_dim 不同（64 vs 32），所以**一个 block 需要两个 RoPE 实例**。SAM2 没这个问题，因为它的 PE 加在 token 上、与 head 无关。
rate=2 更省算力；rate=1 且 `embed_dim // num_heads == 64` 时能复用 backbone 的**同一套频率基**（局部性先验可直接迁移），代价是 2× 注意力算力。

## 踩过的坑（都是静默失败，值得记下来）

**A. Volt 硬编码 head_dim = 64。**
`volt_base.py:246` 是 `self.pos_enc = RoPE()`，默认 `freq_split=(12,12,8)` = 32 复数 = 64 实数。所以任何 Volt 变体必须满足 `embed_dim // num_heads == 64`。两个官方 checkpoint 都符合。

**B. `weights/volt-small-scannet.pth` 是损坏的**（zip central directory 读不出来），但 `configs/scannet/insseg-spformer-volt-*-base.py` 正引用它。可用的权重在 `weights/hf/Volt_experiments/`。

**C. 非 persistent 的随机 buffer 会静默破坏 save/load。**
`PositionEmbeddingRandom3D.gaussian_matrix` 是随机初始化的固定投影。我一开始写了 `persistent=False` → 不进 `state_dict` → 每次重建模型都是另一个随机矩阵 → 训练好的 head 完全失效。
最阴险的地方是 `load_state_dict` 报 `missing=0 unexpected=0`，因为非 persistent buffer 根本不在比对范围内。实测同一权重在进程内 IoU 0.85、重新加载后 0.00。
（SAM2 原版这个 buffer 是默认 persistent 的。）

**D. 冻结 backbone 时 BatchNorm 必须钉在 eval。**
作用量很小（实测两种模式在 fine 特征上余弦一致度 0.9875，因为 24 万体素下 batch 统计量已接近 running stats），但属于必须做对的事：否则特征会依赖同 batch 的其他场景，且 `precompute_backbone` 的缓存必须和消费它的模式一致。

**E. `InformationWriter` 会对输出 dict 的每个 key 调 `.item()`。**
所以训练模式下返回的 dict 里不能有多元素张量。张量输出放在 `return_logits=True` 后面（Pointcept 的 trainer 不传这个参数）。

**F. `fd` 默认遵守 `.gitignore`，而本仓库 `.gitignore` 里有 `data/*` 和 `weights/*`。**
不加 `-u` / `--no-ignore` 会看不到数据集和权重，从而误判"没有数据"。

**G. `packed_seqlens` 需要 `minlength`。**
某个场景可能一个点击都没有（没有标注实例）。不用 `minlength=B+1` 的话 `bincount` 返回的长度会偏短，之后所有 `cu_seqlens[b]` 索引都会错位。

## 评测协议的统计口径

写 `eval_interactive.py` 时踩到两个统计陷阱：

* **NoC 不能只对成功的实例取均值**——那样恒等于 1，没有意义。正确做法是对**所有**实例取均值，未达标的按点击预算计。
* **提前收敛的实例要 carry forward 其最终 IoU**。若在 IoU@k 的后期列里丢掉它们，就只剩难实例，IoU@k 会看起来随点击增多而下降。

## 实测数字

单张 RTX 4070 (12 GB)，冻结 backbone，AMP：

| batch | s/step | scenes/hour | 峰值显存 |
|---|---|---|---|
| 8 | 0.761 | 37,867 | 6.00 GiB |
| **12** | **0.839–0.892** | **48–51k** | **7.05–7.48 GiB** |
| 16 | 1.195 | 48,189 | 8.74 GiB |

ScanNet 在 grid 0.02m / kernel 5 下（30 个随机场景）：voxels median 10.7 万、max 24.5 万；**patch token median 6,974、p90 11,063、max 13,829**。

**6h 预算下全集 1201 场景就够了**（batch 12，约 180 epoch），用子集只会减少数据多样性——在 300 场景子集上凑满 6h 需要跑 1030 遍。

真实数据 overfit（3 场景 × 4 个任务，400 步）：mean IoU **0.8797**，平凡全正基线 0.0754。评测脚本同权重得 IoU@1 = 0.873，两者自洽；NoC@80 = 1.50（92% 收敛），NoC@90 = 4.58（42% 收敛）。

## 尚未解决

`refine`（prev logit 反馈）这条通路目前**几乎没有效果**：pass1 与 pass0 的 IoU 差异在 0.001 量级。原因是 dense prompt 只加到 patch token 上，而 mask embedding 是从点击 token 导出的，prev logit 对它的影响是间接的。合理的修法是把池化后的 prev logit 也拼进 `prompt_fuse` 的输入，给 mask memory 一条直接通路。
