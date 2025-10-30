# U-Net (LocalPets)

> Minimal U-Net training / inference scripts for the LocalPets dataset (toy example).

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Highlights
- Tiny, readable training script: `train_unet_localpets.py`
- One-file inference: `predict_unet_localpets.py`
- Logs in `train_log.txt`；输出目录 `out/`、预测目录 `pred_out/`（已被 .gitignore 忽略）

## 📦 Install
```bash
pip install -r requirements.txt
# choose ONE of:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA
# or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU
