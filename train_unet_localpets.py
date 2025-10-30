import os, argparse, random
import numpy as np 
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ---------- U-Net (简版) ----------
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
        super().__init__(); self.up = nn.ConvTranspose2d(in_c, in_c//2, 2, stride=2); self.conv = DoubleConv(in_c, out_c)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh, dw = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([x2, x1], 1))

class UNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.d1  = Down(64, 128)
        self.d2  = Down(128, 256)
        self.d3  = Down(256, 512)
        self.d4  = Down(512, 1024)      # ← 底部通道改到 1024

        self.u1  = Up(1024, 512)        # ← 与 x4(512) 对齐，拼接后=512+512=1024
        self.u2  = Up(512, 256)         # ← 与 x3(256)
        self.u3  = Up(256, 128)         # ← 与 x2(128)
        self.u4  = Up(128, 64)          # ← 与 x1(64)

        self.outc = nn.Conv2d(64, n_classes, 1)
    def forward(self,x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
        x=self.u1(x5,x4); x=self.u2(x,x3); x=self.u3(x,x2); x=self.u4(x,x1); return self.outc(x)

# ---------- 直接读本地 images/ 与 annotations/trimaps/ ----------
class LocalPetSeg(Dataset):
    def __init__(self, root, split="train", size=256, seed=42):
        self.root = Path(root)
        imgs = sorted([p for p in (self.root/"images").glob("*.jpg")])
        # 匹配到同名的 trimap
        pairs = []
        for im in imgs:
            m = (self.root/"annotations/trimaps"/(im.stem + ".png"))
            if m.exists(): pairs.append((im,m))
        if not pairs: raise RuntimeError("没有找到任何 (image, trimap) 对，检查路径是否为含 images/ 和 annotations/trimaps/ 的那层。")
        random.Random(seed).shuffle(pairs)
        k = int(0.8*len(pairs))
        self.items = pairs[:k] if split=="train" else pairs[k:]
        self.timg = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
        self.tmsk = transforms.Compose([transforms.Resize((size,size), interpolation=transforms.InterpolationMode.NEAREST)])
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        ip, mp = self.items[i]
        img = self.timg(Image.open(ip).convert("RGB"))
        m = self.tmsk(Image.open(mp))          
        m = torch.from_numpy(np.array(m, dtype=np.uint8))
        if m.ndim == 3:                          
            m = m.squeeze(0)
        mask = (m == 1).long()                 
        return img, mask

def iou_score(logits, y):
    pred = logits.argmax(1)
    inter = ((pred==1) & (y==1)).sum().item()
    union = ((pred==1) | (y==1)).sum().item()
    return inter/(union+1e-6)

def train(args):
    # 选择GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = LocalPetSeg(args.data, "train", args.size); va = LocalPetSeg(args.data, "val", args.size)
    # 把单张图片一次打包一成一个batch，shuffle=True：每轮训练前，把所有图片的顺序打乱。num_workers=2：这是同时开几个工人线程去读数据。
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # UNet(2)：实例化一个 U-Net 模型，2 表示输出类别数（背景 + 前景）。.to(device)：把模型搬到设备上（cuda 就是显卡，cpu 就是处理器）。这样后面输入数据的时候，模型和数据都在同一个硬件上，能正常计算。
    # torch.optim.AdamW：一种优化器（训练时更新模型参数的规则），比传统 Adam 多了权重衰减（weight_decay），有助于防止过拟合。
    # model.parameters()：把模型的所有可训练参数传给优化器。
    # lr=args.lr：学习率（一次更新走多大步），来自命令行参数。weight_decay=1e-4：权重衰减系数，相当于对模型参数加一个轻微的 L2 正则化。
    model = UNet(2).to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 交叉熵损失（Cross Entropy Loss），用来衡量模型预测的分类概率分布与真实标签的差距。
    # best = 0.0用来记录目前验证集上得到的最佳 IoU 值（交并比）。
    ce = nn.CrossEntropyLoss(); best=0.0
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs+1):
        model.train(); run=0.0
        for x,y in tqdm(tl, desc=f"Epoch {ep}/{args.epochs}"):
            x,y = x.to(device), y.to(device); opt.zero_grad(); logit = model(x); loss = ce(logit,y); loss.backward(); opt.step()
            run += loss.item()*x.size(0)
        tr_loss = run/len(tr)

        model.eval(); vloss=0.0; viou=0.0
        with torch.no_grad():
            for i,(x,y) in enumerate(vl):
                x,y = x.to(device), y.to(device); logit = model(x)
                vloss += ce(logit,y).item()*x.size(0); viou += iou_score(logit,y)*x.size(0)
                if i==0:
                    pred = logit.argmax(1).float().unsqueeze(1)
                    vis = torch.cat([x[:4], pred[:4].repeat(1,3,1,1)], 0)
                    save_image(vis.cpu(), out/f"epoch{ep:02d}_preview.png", nrow=4)
        v_loss = vloss/len(va); v_iou = viou/len(va)
        print(f"[Epoch {ep}] train_loss={tr_loss:.4f}  val_loss={v_loss:.4f}  val_iou={v_iou:.4f}")
        if v_iou>best: best=v_iou; torch.save(model.state_dict(), out/"unet_best.pt"); print(f"new best {best:.4f}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="指向含 images/ 与 annotations/ 的根目录")
    ap.add_argument("--out",  type=str, default="out")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    print("Using data dir:", args.data)
    train(args)
