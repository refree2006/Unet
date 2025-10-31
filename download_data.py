from torchvision.datasets import OxfordIIITPet

root = "data/oxford-pet"
OxfordIIITPet(root=root, split="trainval", target_types=("segmentation",), download=True)
OxfordIIITPet(root=root, split="test",      target_types=("segmentation",), download=True)
print("✅ Oxford-IIIT Pet 下载/检查完成：", root)
