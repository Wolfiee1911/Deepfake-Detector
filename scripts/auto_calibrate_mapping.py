# scripts/auto_calibrate_mapping.py
import os, json, glob, random, torch, timm
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

CKPT = "checkpoints/spatial_convnext_tiny_video.pt"
VAL_ROOT = "data/frames_val"   # expects frames_val/real and frames_val/fake
OUT  = "configs/calibration.json"
N_PER_CLASS = 1000             # speed vs accuracy; adjust if you want

tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def load_model(device="cuda"):
    m = timm.create_model('convnext_tiny', pretrained=False, num_classes=1)
    ckpt = torch.load(CKPT, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    m.load_state_dict(ckpt)
    m.eval().to(device).float()
    return m

def sample_paths(root, cls, n):
    pats = glob.glob(os.path.join(root, cls, "*.jpg"))
    random.shuffle(pats)
    return pats[:min(n, len(pats))]

def collect_logits(model, device):
    real = sample_paths(VAL_ROOT, "real", N_PER_CLASS)
    fake = sample_paths(VAL_ROOT, "fake", N_PER_CLASS)
    xs, ys = [], []
    with torch.inference_mode():
        for y, paths in [(0, fake), (1, real)]:  # y=1 real, y=0 fake (ground truth)
            for p in tqdm(paths, desc=f"Scoring {'REAL' if y==1 else 'FAKE'}"):
                img = Image.open(p).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device, dtype=torch.float32)
                logit = float(model(x).item())
                xs.append(logit)
                ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)

def probs_from_logits(logits, interpret="REAL"):
    # model sigmoid(logit) interpreted as p(REAL) or p(FAKE)
    p_pos = 1.0 / (1.0 + np.exp(-logits))
    if interpret.upper() == "REAL":
        return 1.0 - p_pos   # p_fake = 1 - p_real
    else:
        return p_pos         # p_fake = p_fake

def best_threshold(p_fake, y_true):
    thrs = np.linspace(0.05, 0.95, 181)
    best_acc, best_t = -1.0, 0.5
    for t in thrs:
        pred = (p_fake >= t).astype(np.int64)  # 1=fake, 0=real
        acc = (pred == (1 - y_true)).mean()    # y_true: 1=real → 1 - y_true = 0 (real), 1 (fake) OK
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, float(best_acc)

def main():
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir(VAL_ROOT):
        raise SystemExit(f"{VAL_ROOT} not found. You need frames_val/real and frames_val/fake.")

    model = load_model(device)
    logits, y_true = collect_logits(model, device)

    p_fake_REAL = probs_from_logits(logits, "REAL")
    t_REAL, acc_REAL = best_threshold(p_fake_REAL, y_true)

    p_fake_FAKE = probs_from_logits(logits, "FAKE")
    t_FAKE, acc_FAKE = best_threshold(p_fake_FAKE, y_true)

    if acc_FAKE > acc_REAL:
        choice = "FAKE"
        threshold = t_FAKE
        acc = acc_FAKE
    else:
        choice = "REAL"
        threshold = t_REAL
        acc = acc_REAL

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"threshold": threshold, "assume_output": choice, "val_acc_estimate": acc},
              open(OUT, "w"), indent=2)
    print(f"[OK] Wrote {OUT}")
    print(f"  assume_output: {choice} (sigmoid interprets as p({choice}))")
    print(f"  threshold:     {threshold:.3f}")
    print(f"  est val acc:   {acc:.3f}")

if __name__ == "__main__":
    main()
