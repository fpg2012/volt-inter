---
name: scannet-dataset
description: 本机 Volt 项目里 ScanNet/ScanNet200 数据集的来源（HuggingFace Pointcept/scannet-compressed）、压缩包与解压位置、Pointcept npy 格式、split 划分、类别常量、以及在训练/评估/可视化脚本中的基本用法。当需要处理 data/scannet、写 ScanNet 相关配置、加载点云样本或排查数据问题时使用。
---

# ScanNet 数据集（Volt 本地版）

本项目的 ScanNet 是 **Pointcept 预处理过的 Pointcept-format 版本**，不是 ScanNet
官网的原始 `_vh_clean_2.ply`。每个场景已经是一个目录，里面是 `coord/color/normal/
segment20/segment200/instance` 等 `.npy` 文件，可直接被 `ScanNetDataset` 读取。

## 1. 位置与来源

| 说明 | 路径 |
|---|---|
| 解压后可直接训练/评估的数据 | `/workspace/lab/Volt/data/scannet/` （约 12G） |
| 原始压缩包 + HF 下载缓存 | `/workspace/lab/Volt/data/scannet_archive/` （约 5.6G） |
| 压缩包本体 | `data/scannet_archive/scannet.tar.gz` （5,978,743,303 B） |
| HF 下载元数据 | `data/scannet_archive/.cache/huggingface/` |

**来源**：HuggingFace 数据集 **`Pointcept/scannet-compressed`**（gated，需同意
ScanNet 使用条款）。已确认：

- `~/.cache/huggingface/hub/datasets--Pointcept--scannet-compressed/refs/main`
  = `8d4fab97cd37889fc8edec5dd09ab0afa4c84a87`，与
  `data/scannet_archive/.cache/huggingface/trees/8d4fab97….json` 的 tree hash 一致；
- LFS 对象 sha256 = `0da5939fc1897c0393169049747a8629db4c80ae8d6420d0f48bfb55e0c56a83`；
- `data/scannet_archive/README.md` 是该 HF dataset card（ScanNet TOS）。

> 和官方原始数据的区别：ScanNet 官网（https://www.scan-net.org/，
> 数据实际托管在 kaldir.vc.in.tum.de）给的是原始 mesh/标注；本项目用的是
> Pointcept 已经跑完 `preprocess_scannet.py` 的成品。**不要**再去官网下原始数据，
> 除非要重新做预处理或生成 superpoint。

### 重新下载 / 重新解压

```bash
# 需先在 https://huggingface.co/datasets/Pointcept/scannet-compressed 同意条款
hf download Pointcept/scannet-compressed --repo-type dataset \
    --local-dir data/scannet_archive
# 压缩包内路径是 ./train ./val ./test ./tasks，直接解到 data/scannet
tar -xzf data/scannet_archive/scannet.tar.gz -C data/scannet
```

## 2. 目录结构

```
data/scannet/
├── train/                 # 1201 个场景目录，sceneXXXX_YY/
├── val/                   # 312 个场景
├── test/                  # 100 个场景（有标注，可本地评估）
└── tasks/                 # Data-Efficient 子集划分
    ├── points/            # points20 / points50 / points100 / points200（按采样点数）
    └── scenes/            # 1.txt / 5.txt / 10.txt / 20.txt（按场景数）
```

单个场景目录内容（size 与点数成正比，示例 `scene0000_00`，81369 点）：

```
scene0000_00/
├── coord.npy       (N, 3) float32   坐标（米）
├── color.npy       (N, 3) uint8     RGB 0-255
├── normal.npy      (N, 3) float32   法向
├── segment20.npy   (N,)  int64      ScanNet20 标签，-1 = ignore/unlabeled
├── segment200.npy  (N,) int64      ScanNet200 标签，-1 = ignore
└── instance.npy    (N,)  int64      实例 id，-1 = 无实例
```

- 顶点顺序一一对应：所有 npy 的第 0 维长度相同（= 点数）。
- `ScanNetDataset` 只加载 `VALID_ASSETS = [coord, color, normal, segment20,
  instance, superpoint]`，`ScanNet200Dataset` 用 `segment200`。
- `superpoint.npy`（实例分割用）**不在这份数据里**，需要自己对原始 PLY 跑
  `preprocess_superpoints.py` 生成（见下）。
- 类别常量在 `pointcept/datasets/preprocessing/scannet/meta_data/
  scannet200_constants.py`：`VALID_CLASS_IDS_20`（20 类）、
  `VALID_CLASS_IDS_200`（200 类）；类别名列表见
  `configs/scannet/semseg-volt-base.py` 的 `data.names`。
- split 名单：`meta_data/scannetv2_{train,val,test}.txt`（v2，1201/312/100，
  与本目录数量一致）；`scannetv1_*` 是旧版。

## 3. 从原始数据重新生成（一般不需要）

只有重新做预处理或要 superpoint 时才需要 ScanNet 原始数据 `${RAW_SCANNET_DIR}`：

```bash
# 语义分割标签（生成上面的 npy）
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py \
    --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
# 实例分割 superpoint（README.md:139）
python pointcept/datasets/preprocessing/scannet/preprocess_superpoints.py \
    --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```

`preprocess_superpoints.py` 会遍历 raw 的 `*_vh_clean_2.ply`，用 `pointseg.segment_mesh`
生成 `superpoint.npy`，并按 train/val/test 写入输出目录。

## 4. 基本用法

### 4.1 直接用 Dataset 类

```python
from pointcept.datasets import build_dataset   # 或 configs 里注册的 ScanNetDataset

cfg = dict(
    type="ScanNetDataset",       # 200 类用 "ScanNet200Dataset"
    split="val",
    data_root="data/scannet",
    transform=[
        dict(type="CenterShift", apply_z=True),
        dict(type="NormalizeColor"),
    ],
)
ds = build_dataset(cfg)
d = ds[0]        # dict: coord/color/normal/segment/instance/name/split ...
print(d["coord"].shape, d["segment"].shape)
```

- `__getitem__` 返回的 `segment` 由 `segment20` 或 `segment200` 重命名而来
  （`ScanNet200Dataset` 用 200）。缺失时填 `-1`。
- 训练时的完整 transform / 增强写在 `configs/scannet/semseg-volt-base.py`
  （`data.train/val/test`），GridSample `grid_size=0.02` 是项目统一体素大小。
- 200 类配置在 `configs/scannet200/`，`data_root` 同样指向 `data/scannet`
  （同一份数据，只是换 label 文件）。

### 4.2 训练 / 测试

```bash
# README.md 的用法：-d 选 dataset 子目录名，-c 选 configs/<dataset>/<config>.py
sh scripts/train.sh -g 4 -d scannet    -c semseg-volt-base -n semseg-volt-base
sh scripts/train.sh -g 4 -d scannet200 -c semseg-volt-base -n semseg-volt-base
sh scripts/test.sh  -g 1 -d scannet    -c semseg-volt-base -n <exp_name>
```

`scripts/train.sh` 会把 `configs/<DATASET>/<CONFIG>.py` 里的 `data_root` 当作相对
仓库根目录的路径，所以必须从仓库根目录运行。

### 4.3 本项目自带的单场景评估 / 可视化脚本

这两个是本地改造的、不需要分布式 runner 的脚本：

```bash
# 单 GPU 整场景前向，算 ScanNet20 的 mIoU（默认 50 个 val 场景）
.venv/bin/python scripts/eval_scannet.py --n 50
.venv/bin/python scripts/eval_scannet.py --ckpt weights/volt-small-scannet.pth

# viser 可视化：左=预测，右=GT，浏览器打开 http://localhost:8090
.venv/bin/python scripts/view_scannet_pred.py                 # 默认第一个 val 场景
.venv/bin/python scripts/view_scannet_pred.py data/scannet/val/scene0704_00 --port 8090
```

脚本内部直接读 `coord/color/normal/segment20.npy`，做 `CenterShift` + `GridSample`，
`feat = concat(color/255, normal)`，交给带 `condition=["ScanNet"]` 的
`DefaultSegmentorV2`。默认 checkpoint 是
`weights/hf/Volt_experiments/joint_training_small/scannet/model/model_last.pth`
（`--ckpt` 可覆盖）。

### 4.4 权重来源（HF）

```bash
mkdir -p weights
curl -L -o weights/volt-base-scannet.pth  https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_base/scannet/model/model_last.pth
curl -L -o weights/volt-small-scannet.pth https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_small/scannet/model/model_last.pth
# 200 类把 scannet 换成 scannet200
```

## 5. 常见坑

- **标签是 `-1` 表示 ignore**，不是 0；算 mIoU/训练 loss 时 `ignore_index=-1`。
- `segment20.npy` 和 `segment200.npy` 的类别 id 是**两套不同映射**，不要混用；
  `ScanNetDataset` 读 20，`ScanNet200Dataset` 读 200。
- 本数据是**预处理后的 voxel/mesh 顶点级**数据，不是原始帧；点数与 PLY 顶点数一致。
- `superpoint.npy` 缺失时，实例分割配置（`insseg-*`）无法直接跑，要先做 §3 的
  superpoint 预处理。
- `test/` 这 100 个场景在本数据里**带标注**，可以本地评估；官方 benchmark 的 test
  是不公开标签的，注意区分。
- `data/scannet` 与压缩包都在 `.gitignore` 的 `data/*` 下（只保留 `.gitkeep`），
  **不要提交**。
- HF dataset 是 gated 的，token 存在 `~/.cache/huggingface/stored_tokens`（条目名
  `scannet`）里；换机器要重新同意条款并配置 token。
