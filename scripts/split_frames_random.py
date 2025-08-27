import os, glob, random, shutil
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="data/frames", help="Source folder with real/ and fake/")
parser.add_argument("--out_train", default="data/frames_train", help="Output train folder")
parser.add_argument("--out_val", default="data/frames_val", help="Output val folder")
parser.add_argument("--ratio", type=float, default=0.80, help="Train ratio (0-1)")
parser.add_argument("--copy", action="store_true", help="Copy instead of move (uses more disk)")
parser.add_argument("--clean", action="store_true", help="Clean target folders before splitting")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

random.seed(args.seed)

src_real = Path(args.src) / "real"
src_fake = Path(args.src) / "fake"
if not src_real.exists() or not src_fake.exists():
    raise SystemExit(f"Could not find {src_real} and/or {src_fake}. Make sure your frames are in data/frames/real and data/frames/fake.")

train_real = Path(args.out_train) / "real"
train_fake = Path(args.out_train) / "fake"
val_real   = Path(args.out_val) / "real"
val_fake   = Path(args.out_val) / "fake"
for p in [train_real, train_fake, val_real, val_fake]:
    p.mkdir(parents=True, exist_ok=True)

def maybe_clean(folder: Path):
    if args.clean and folder.exists():
        for f in folder.glob("*"):
            try:
                f.unlink()
            except IsADirectoryError:
                shutil.rmtree(f)

# optional: clean old contents
for p in [train_real, train_fake, val_real, val_fake]:
    maybe_clean(p)

def split_and_transfer(src_dir: Path, dst_train: Path, dst_val: Path):
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        imgs.extend(glob.glob(str(src_dir / ext)))
    imgs = list(map(Path, imgs))
    if not imgs:
        print(f"[WARN] No images found in {src_dir}")
        return 0,0

    random.shuffle(imgs)
    cut = int(len(imgs) * args.ratio)
    tr, va = imgs[:cut], imgs[cut:]

    mover = shutil.copy2 if args.copy else shutil.move
    moved_tr = moved_va = 0
    for s in tr:
        d = dst_train / s.name
        if not d.exists():
            mover(str(s), str(d))
            moved_tr += 1
    for s in va:
        d = dst_val / s.name
        if not d.exists():
            mover(str(s), str(d))
            moved_va += 1
    return moved_tr, moved_va

tr_r, va_r = split_and_transfer(src_real, train_real, val_real)
tr_f, va_f = split_and_transfer(src_fake, train_fake, val_fake)

print(f"REAL -> train:{tr_r}  val:{va_r}")
print(f"FAKE -> train:{tr_f}  val:{va_f}")
print(f"Done. Train frames: {args.out_train}\\{{real,fake}}  |  Val frames: {args.out_val}\\{{real,fake}}")
print("Note: Use --copy to keep originals in data/frames, or run without it to move files (save disk).")