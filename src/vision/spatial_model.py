# src/vision/spatial_model.py
import os
import torch
import timm

CKPT_PATH = "checkpoints/spatial_convnext_tiny_video.pt"

class SpatialNet(torch.nn.Module):
    def __init__(self, arch='convnext_tiny', num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

def load_spatial_model(device='cuda'):
    m = SpatialNet().to(device)          # <-- FP32 model (DO NOT call .half())
    if os.path.exists(CKPT_PATH):
        try:
            ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(CKPT_PATH, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            m.backbone.load_state_dict(ckpt["model"])
        else:
            m.backbone.load_state_dict(ckpt)
        print(f"✅ Loaded trained weights from {CKPT_PATH}")
    else:
        print("⚠️ Using ImageNet weights (no trained weights found).")
    m.eval()
    m.float()                              # <-- extra safety: force FP32
    return m
