# quick_check.py
import os
from pathlib import Path

root = Path("data")
train = root/"train"
val = root/"val"

def count_dir(p):
    return len([f for f in p.glob("*.jpg")]) if p.exists() else 0

print("Checking data directories...\n")
all_ok = True
for split in ("train","val"):
    print(f"== {split} ==")
    for i in range(10):
        p = Path("data")/split/str(i)
        c = count_dir(p)
        print(f"  class {i}: {c}")
    print("")
# advice
min_train = min([count_dir(Path("data")/"train"/str(i)) for i in range(10)]) if (Path("data")/"train").exists() else 0
print("Recommendation:")
if min_train >= 200:
    print("  Good: each class has >=200 images. Proceed to training with train.py.")
    print("  Command -> python train.py --train-dir data/train --val-dir data/val --img-size 64 --batch 32 --epochs 25")
else:
    print(f"  Some classes have fewer than 200 images (min={min_train}). Capture more data for under-represented classes.")
    print("  Command -> python collect.py --target 300 --append --preview")
print("\nIf any class has zero images, run collect.py and capture at least 100 images per missing class.")
