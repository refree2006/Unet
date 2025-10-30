import os, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.utils import save_image
from tqdm import tqdm

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
        dh, dw = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.d1  = Down(64, 128)
        self.d2  = Down(128, 256)
        self.d3  = Down(256, 512)
        self.d4  = Down(512, 512)
        self.u1  = Up(512, 256)
        self.u2  = Up(256, 128)
        self.u3  = Up(128, 64)
        self.u4  = Up(64, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.d1(x1); x3 = self.d2(x2); x4 = self.d3(x3); x5 = self.d4(x4)
        x = self.u1(x5, x4); x = self.u2(x, x3); x = self.u3(x, x2); x = self.u4(x, x1)
        return self.outc(x)

class PetSeg(torch.utils.data.Dataset):
    def __init__(self, root, split="trainval", size=256):
        self.ds = OxfordIIITPet(root=root, split="trainval", target_types="segmentation", download=True)
        n = len(self.ds); k = int(0.8*n)
        self.idxs = range(0, k) if split=="trainval" else range(k, n)
        self.timg = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        self.tmask = transforms.Compose([transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST)])
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        img, mask = self.ds[self.idxs[i]]
        img = self.timg(img)
        mask = self.tmask(mask)
        mask = torch.as_tensor(mask, dtype=torch.int64).squeeze(0)  # 0边界/1前景/2背景
        mask = (mask == 1).long()  # 前景=1 其他=0
        return img, mask

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = PetSeg(args.data, "trainval", args.size)
    val_set   = PetSeg(args.data, "val", args.size)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(n_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()

    def iou_score(logits, y):
        pred = logits.argmax(1)
        inter = ((pred==1) & (y==1)).sum().item()
        union = ((pred==1) | (y==1)).sum().item()
        return inter / (union + 1e-6)

    best = 0.0; outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs+1):
        model.train(); run=0.0
        for x,y in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); logit = model(x); loss = ce(logit, y); loss.backward(); opt.step()
            run += loss.item()*x.size(0)
        tr_loss = run/len(train_loader.dataset)

        model.eval(); vloss=0.0; viou=0.0
        with torch.no_grad():
            for i,(x,y) in enumerate(val_loader):
                x,y = x.to(device), y.to(device)
                logit = model(x)
                vloss += ce(logit,y).item()*x.size(0)
                viou  += iou_score(logit,y)*x.size(0)
                if i==0:
                    pred = logit.argmax(1).float().unsqueeze(1)
                    vis = torch.cat([x[:4], pred[:4].repeat(1,3,1,1)], dim=0)
                    save_image(vis.cpu(), outdir/f"epoch{ep:02d}_preview.png", nrow=4)
        v_loss = vloss/len(val_loader.dataset); v_iou = viou/len(val_loader.dataset)
        print(f"[Epoch {ep}] train_loss={tr_loss:.4f}  val_loss={v_loss:.4f}  val_iou={v_iou:.4f}")
        if v_iou > best:
            best = v_iou
            torch.save(model.state_dict(), outdir/"unet_best.pt")
            print(f"  ✓ new best IoU={best:.4f}, saved.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(Path.home()/ "datasets/oxford-pet"))
    p.add_argument("--out",  type=str, default="out")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    print("Using data dir:", args.data)
    train(args)
