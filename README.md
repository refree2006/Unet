---

# U-Net on Oxford-IIIT Pet *(Learning Notes)*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)]()

## 1) Overview

This repo is my **hands-on practice** of training/inferencing a tiny **U-Net** on the **Oxford-IIIT Pet** dataset.
It aims at **quick experiments & reproducible notes**, not a polished framework.

* Minimal training script: `train_unet_localpets.py`
* One-file inference: `predict_unet_localpets.py`
* Utilities: `peek_pt.py`, `download_data.py`
* Logs in `train_log.txt`

## 2) Dataset

* **Oxford-IIIT Pet** (commonly referred to as Oxford Pets).
* Task: binary/trimap masks for pet segmentation.
* Put images & masks under a structure like:

```
data/
  images/
    xxx.jpg
  masks/
    xxx.png
```

You can tweak paths in the scripts if needed.

## 3) Environment

```bash
pip install -r requirements.txt

# Choose ONE (according to your machine)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA
# or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # CPU
```

## 4) Quick Start

**Train**

```bash
python train_unet_localpets.py \
  --data-root ./data \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-3 \
  --save-dir ./out
```

**Inference**

```bash
python predict_unet_localpets.py \
  --data-root ./data \
  --ckpt ./out/unet_best.pt \
  --save-dir ./pred_out
```

**Peek a checkpoint (shape only)**

```bash
python peek_pt.py ./out/unet_best.pt
```

## 5) Repo Layout

```
UNET/
├─ out/            # checkpoints
├─ pred_out/       # predictions
├─ train_unet_localpets.py
├─ predict_unet_localpets.py
├─ peek_pt.py
├─ download_data.py
├─ train_log.txt
├─ requirements.txt
└─ README.md
```

## 6) Notes & Limits

* Focus is on **readability** and **small-scale experiments**.
* No heavy tricks (AMP/mixed precision, DDP, etc.)—add as you wish.

## 7) Roadmap (personal)

* [ ] Clean metric utilities (mIoU / Dice).
* [ ] Simple data augmentation switches.
* [ ] Tiny validation dashboard.

---

# 使用 U-Net 进行 Oxford-IIIT Pet 分割（学习记录）

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)]()

## 1）项目概述

这是我在 **Oxford-IIIT Pet（Oxford Pets）** 数据集上练习 **U-Net** 的小仓库，侧重**可复现的个人实验记录**，不追求工程化。

* 训练脚本：`train_unet_localpets.py`
* 推理脚本：`predict_unet_localpets.py`
* 小工具：`peek_pt.py`、`download_data.py`
* 训练日志：`train_log.txt`
* `out/`（权重）和 `pred_out/`（预测结果）**已加入 .gitignore**

## 2）数据集

* **Oxford-IIIT Pet**（常称 Oxford Pets），任务是宠物分割（包含二值/trimap mask）。
* 建议目录结构：

```
data/
  images/
    xxx.jpg
  masks/
    xxx.png
```

如路径不同，可在脚本中自行修改。

## 3）环境安装

```bash
pip install -r requirements.txt

# 根据环境任选其一
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 环境
# 或
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # 仅 CPU
```

## 4）快速运行

**训练**

```bash
python train_unet_localpets.py \
  --data-root ./data \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-3 \
  --save-dir ./out
```

**推理**

```bash
python predict_unet_localpets.py \
  --data-root ./data \
  --ckpt ./out/unet_best.pt \
  --save-dir ./pred_out
```

**查看权重结构**

```bash
python peek_pt.py ./out/unet_best.pt
```

## 5）仓库结构

```
UNET/
├─ out/            # 保存权重
├─ pred_out/       # 保存预测图
├─ train_unet_localpets.py
├─ predict_unet_localpets.py
├─ peek_pt.py
├─ download_data.py
├─ train_log.txt
├─ requirements.txt
└─ README.md
```

## 6）说明与限制

* 代码以**简单可读**为主，适合**小规模实验**。
* 暂未加入 AMP、DDP 等工程特性，后续按需补充。

## 7）个人待办

* [ ] mIoU / Dice 等指标工具。
* [ ] 基础数据增强开关。
* [ ] 简易验证可视化面板。

---