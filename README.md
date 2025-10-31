🧩 项目介绍
项目概述

本项目实现了一个最小可复现的 U-Net 图像分割流水线，面向 Oxford-IIIT Pet 数据集的宠物前景分割任务。目标是提供一套干净、易读、可快速上手的参考实现，便于课程作业、模型对比与进一步研究扩展。

为什么做这个

许多开源仓库较为臃肿，初学者难以下手；

希望用几十行到几百行的代码，把数据下载→数据加载→训练→推理→可视化串成一条清晰的路径；

作为后续研究（例如更复杂的编码器/损失函数/数据增强/多任务学习）的稳定起点。

方法与设计

模型：经典 U-Net（Encoder-Decoder + Skip Connections），结构简单、收敛稳定、对边界细节友好。

任务：宠物二类/多类分割（前景/背景，或包含边界/trimap 的扩展）。

损失：默认 BCEWithLogitsLoss / CrossEntropyLoss，可选 Dice / Focal / BCE+Dice。

增强：Resize、随机翻转、色彩抖动（可选）；掩码插值统一用 NEAREST 以避免类别混叠。

日志：训练过程关键指标与学习率写入 train_log.txt，便于快速回溯。

输出：训练权重与可视化结果默认写入 out/、pred_out/（已在 .gitignore 忽略）。

数据与预处理

数据集：Oxford-IIIT Pet（37 个品种，约 7.3k 张图像，含分类标签、像素级分割、trimap）。

一键下载：支持 torchvision.datasets.OxfordIIITPet 自动下载与缓存到 data/oxford-pet/。

目录示例：

data/oxford-pet/
├─ images/
├─ annotations/
│  ├─ trimaps/
│  └─ xmls/
├─ trainval.txt
└─ test.txt

训练配置（示例）

优化器：Adam / AdamW（lr=1e-3 起步）

批大小：8~16（视显存而定）

训练轮次：50~100

学习率调度：Cosine / Step（可选）

示例命令（Windows PowerShell）

python train_unet_localpets.py ^
  --epochs 50 ^
  --lr 1e-3 ^
  --batch-size 8 ^
  --save-dir out


示例命令（Linux / macOS）

python train_unet_localpets.py \
  --epochs 50 \
  --lr 1e-3 \
  --batch-size 8 \
  --save-dir out

推理与可视化
# Windows
python predict_unet_localpets.py ^
  --ckpt path\to\unet_best.pt ^
  --input-dir path\to\images ^
  --out-dir pred_out

# Linux / macOS
python predict_unet_localpets.py \
  --ckpt path/to/unet_best.pt \
  --input-dir path/to/images \
  --out-dir pred_out


建议在 README 中展示 2~3 张输入图像/预测掩码/叠加效果的对比图，便于直观感受模型质量。
（将图片放在 images/ 目录，然后用 ![](/images/demo.png) 引用）

结果（占位，待你替换真实指标）
指标	TrainVal	Test
Dice	0.87	0.85
IoU	0.79	0.77

评估脚本可扩展为：mIoU、Dice、边界 F1（在 trimap 区域加权）。

已知局限 & 下一步

局限：基础 U-Net 对复杂边界、毛发细节可能较弱；对远景/遮挡鲁棒性一般。

改进方向：

更强的编码器（ResNet/MobileNet/ViT-Encoder）；

更适配的损失（Dice/Focal/Tversky/Boundary Loss）；

边界增强（使用 trimap 的边界加权或双头预测）；

轻量化与部署（ONNX/TensorRT/量化蒸馏）。

项目结构（示例）
UNET/
├─ train_unet_localpets.py
├─ train_unet_pets.py
├─ predict_unet_localpets.py
├─ peek_pt.py
├─ train_log.txt
├─ requirements.txt
├─ download_data.py
├─ out/        # 训练输出（忽略）
└─ pred_out/   # 推理输出（忽略）

运行方法（摘要）
pip install -r requirements.txt
# 选一个安装 PyTorch：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU
python download_data.py
python train_unet_localpets.py --epochs 50 --lr 1e-3 --batch-size 8 --save-dir out

许可与引用

代码许可：MIT（可按需更改）

数据集引用（Oxford-IIIT Pet）：

@inproceedings{parkhi2012catsdogs,
  title={Cats and dogs},
  author={Parkhi, Omkar M and Vedaldi, Andrea and Zisserman, Andrew and Jawahar, CV},
  booktitle={CVPR},
  year={2012}
}

🧩 Project Overview (English)
Summary

This repository provides a minimal, reproducible U-Net pipeline for foreground segmentation on the Oxford-IIIT Pet dataset. It aims to be clean, readable, and easy to start with, serving as a stable baseline for coursework, comparisons, and further research.

Motivation

Many repos are over-engineered for beginners;

We want a small yet complete path from download → dataloading → training → inference → visualization;

Acts as a solid starting point for extending encoders, losses, augmentations, or multi-task settings.

Method & Design

Model: classic U-Net (encoder-decoder with skip connections), stable and boundary-friendly.

Task: binary / multi-class pet segmentation (foreground vs background, or trimap-aware variants).

Loss: BCEWithLogits / CrossEntropy, optional Dice / Focal / BCE+Dice.

Augmentations: resize, flips, color jitter (optional); use NEAREST for mask resizing to avoid class mixing.

Logging: key metrics & LR into train_log.txt.

Outputs: training results in out/, predictions in pred_out/ (both ignored by git).

Data & Preprocessing

Dataset: Oxford-IIIT Pet (37 breeds, ~7.3k images; classification labels, pixel-wise segmentation, trimap).

One-line download via torchvision.datasets.OxfordIIITPet into data/oxford-pet/.

See the directory layout in the Chinese section above.

Training Config (example)

Optimizer: Adam / AdamW (start with lr=1e-3)

Batch size: 8~16 (GPU-dependent)

Epochs: 50~100

LR schedule: cosine / step (optional)

Commands

# Linux / macOS
python train_unet_localpets.py \
  --epochs 50 --lr 1e-3 --batch-size 8 --save-dir out
# Windows (PowerShell)
python train_unet_localpets.py ^
  --epochs 50 ^
  --lr 1e-3 ^
  --batch-size 8 ^
  --save-dir out

Inference & Visualization
# Linux / macOS
python predict_unet_localpets.py \
  --ckpt path/to/unet_best.pt \
  --input-dir path/to/images \
  --out-dir pred_out

# Windows (PowerShell)
python predict_unet_localpets.py ^
  --ckpt path\to\unet_best.pt ^
  --input-dir path\to\images ^
  --out-dir pred_out


Show a few image / mask / overlay triplets in the README for intuitive comparison.

Results (placeholder)
Metric	TrainVal	Test
Dice	0.87	0.85
IoU	0.79	0.77
Limitations & Future Work

Limitations: plain U-Net may struggle with hairy boundaries and occlusions.

Next: stronger encoders (ResNet/MobileNet/ViT), boundary losses, trimap-aware weighting, lightweight deployment (ONNX/TensorRT/quantization/distillation).

How to Run (recap)
pip install -r requirements.txt
# choose ONE for PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU
python download_data.py
python train_unet_localpets.py --epochs 50 --lr 1e-3 --batch-size 8 --save-dir out

License & Citation

Code: MIT (feel free to adjust)

Dataset (Oxford-IIIT Pet): see the BibTeX in the Chinese section.