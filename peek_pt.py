import torch
from collections import OrderedDict

path = r"\UNET\out\unet_best.pt"

# 尝试 TorchScript
try:
    ts = torch.jit.load(path, map_location="cpu")
    print("这是 TorchScript 模型："); print(ts)
    raise SystemExit
except Exception:
    pass

obj = torch.load(path, map_location="cpu", weights_only=True)
print("torch.load 类型：", type(obj))

if isinstance(obj, dict) and ("state_dict" in obj or "model" in obj):
    sd = obj.get("state_dict", obj.get("model"))
elif isinstance(obj, dict):
    sd = obj
else:
    sd = None

if sd is None:
    print("不是常见的 state_dict/checkpoint，可能是 TorchScript。")
else:
    sd = OrderedDict((k.replace("module.",""), v) for k,v in sd.items())
    for i,(k,v) in enumerate(sd.items()):
        print(f"{i:03d} {k:40s} {tuple(v.shape)}")
        if i>=20: break
    print("Total parameters:", sum(v.numel() for v in sd.values()))
