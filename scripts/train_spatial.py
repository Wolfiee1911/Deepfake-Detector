import os
import random
import torch
import torch.multiprocessing as mp
import timm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ------------ CONFIG ------------
TRAIN_ROOT = "data/frames_train"   # has real/ and fake/
VAL_ROOT   = "data/frames_val"     # has real/ and fake/
IMG_SIZE = 224
BATCH_TRAIN = 32
BATCH_VAL   = 64
EPOCHS = 5
LR = 2e-4
WD = 1e-4
NUM_WORKERS = 2          # multiprocessing ON (Windows-safe with __main__ guard)
SEED = 42
# --------------------------------

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    s = SEED + worker_id
    random.seed(s)
    torch.manual_seed(s)

def build_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(TRAIN_ROOT, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_ROOT,   transform=val_tf)

    print(f"[INFO] Train images: {len(train_ds)}  |  Val images: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_TRAIN, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_VAL, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    return train_loader, val_loader

def evaluate(model, val_loader, device, autocast_ctx):
    model.eval()
    total = correct = 0
    with torch.inference_mode():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)          # keep FP32; AMP will cast
            y = y.to(device, non_blocking=True)
            with autocast_ctx:
                p = torch.sigmoid(model(x)).squeeze(1)
            pred = (p > 0.5).long()
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)

def main():
    # Windows needs spawn
    mp.set_start_method("spawn", force=True)

    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_dataloaders()

    # IMPORTANT: keep model in FP32 for AMP
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=1)
    model = model.to(device)

    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # AMP autocast & scaler (pick the variant your torch supports)
    try:
        autocast_ctx = torch.amp.autocast('cuda')
        scaler = torch.amp.GradScaler('cuda')
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast()
        scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x = x.to(device, non_blocking=True)                 # keep FP32 tensors
            y = y.float().unsqueeze(1).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(x)
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item() * x.size(0)

        val_acc = evaluate(model, val_loader, device, autocast_ctx)
        print(f"Epoch {epoch+1}: train_loss={run_loss/len(train_loader.dataset):.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/spatial_convnext_tiny_video.pt')
            print(f"✅ Saved best weights (val_acc={best_acc:.3f}) -> checkpoints/spatial_convnext_tiny_video.pt")

    print("Training complete. Best val_acc:", best_acc)

if __name__ == "__main__":
    main()
