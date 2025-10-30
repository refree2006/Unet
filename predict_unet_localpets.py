import argparse, os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# -------- U-Net（与训练保持一致） --------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__(); self.pool = nn.MaxPool2d(2); self.conv = DoubleConv(in_c, out_c)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c//2, 2, stride=2)
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh, dw = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([x2, x1], 1))

class UNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.d1  = Down(64, 128)
        self.d2  = Down(128, 256)
        self.d3  = Down(256, 512)
        self.d4  = Down(512, 1024)
        self.u1  = Up(1024, 512)
        self.u2  = Up(512, 256)
        self.u3  = Up(256, 128)
        self.u4  = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)
    def forward(self, x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
        x=self.u1(x5,x4); x=self.u2(x,x3); x=self.u3(x,x2); x=self.u4(x,x1)
        return self.outc(x)

def overlay_mask(img_pil, mask_bool, alpha=120):
    """用红色把前景区域叠加到原图上"""
    r,g,b = 255,0,0
    overlay = Image.new("RGBA", img_pil.size, (0,0,0,0))
    arr = np.zeros((img_pil.size[1], img_pil.size[0], 4), dtype=np.uint8)
    arr[mask_bool] = [r,g,b,alpha]
    overlay = Image.fromarray(arr, mode="RGBA")
    return Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="out/unet_best.pt")
    ap.add_argument("--input",   type=str, required=True, help="待分割图片路径")
    ap.add_argument("--size",    type=int, default=256,   help="推理时的长宽（需与训练一致）")
    ap.add_argument("--output",  type=str, default="out")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(2).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state); model.eval()

    # 读取与预处理
    in_path = Path(args.input)
    orig = Image.open(in_path).convert("RGB")
    tfm = transforms.Compose([transforms.Resize((args.size,args.size)), transforms.ToTensor()])
    x = tfm(orig).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        logits = model(x)                  # [1,2,H,W]
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # 0/1

    # 保存结果
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray(pred*255)          # 0/255 可视化
    mask_path = outdir / f"{in_path.stem}_mask.png"
    mask_img.save(mask_path)

    vis = overlay_mask(Image.open(in_path).convert("RGB").resize((args.size,args.size)),
                       pred.astype(bool))
    overlay_path = outdir / f"{in_path.stem}_overlay.png"
    vis.save(overlay_path)

    print("Saved:", mask_path, "and", overlay_path)

if __name__ == "__main__":
    main()
