import os, json, torch, timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

CKPT = "checkpoints/spatial_convnext_tiny_video.pt"
VAL_ROOT = "data/frames_val"  # has real/ and fake/

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

def load_model(device):
    m = timm.create_model('convnext_tiny', pretrained=True, num_classes=1)
    ckpt = torch.load(CKPT, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt: ckpt = ckpt["model"]
    m.load_state_dict(ckpt); m.to(device).eval()
    return m

@torch.inference_mode()
def collect_probs(device):
    ds = datasets.ImageFolder(VAL_ROOT, transform=tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    y_true, y_prob = [], []
    model = load_model(device)
    for x, y in tqdm(loader, desc="Valid eval"):
        x = x.to(device, dtype=torch.float32)
        p = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
        y_prob.extend(p.tolist()); y_true.extend(y.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_prob)

def find_best_threshold(y_true, y_prob):
    # Youden’s J statistic on ROC-like grid
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_j, best = 0.5, -1, None
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred==1)&(y_true==1)).sum()
        tn = ((pred==0)&(y_true==0)).sum()
        fp = ((pred==1)&(y_true==0)).sum()
        fn = ((pred==0)&(y_true==1)).sum()
        tpr = tp / max(tp+fn,1)
        fpr = fp / max(fp+tn,1)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
            best = dict(tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn),tpr=float(tpr),fpr=float(fpr))
    return best_t, best

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_true, y_prob = collect_probs(device)
    thr, stats = find_best_threshold(y_true, y_prob)
    acc = ((y_prob>=thr).astype(int) == y_true).mean()
    os.makedirs("configs", exist_ok=True)
    json.dump({"threshold": float(thr), "val_acc": float(acc), "stats": stats},
              open("configs/calibration.json","w"), indent=2)
    print(f"Suggested threshold = {thr:.2f}, val_acc={acc:.3f}")
