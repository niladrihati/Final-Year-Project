#!/usr/bin/env python3
"""
collect.py

Capture labeled hand images using hand_wrapper.get_hand_image() and hand_wrapper.get_label().
Saves raw captures to data/raw/<label>/ and then splits into data/train/ and data/val/.

Usage:
    python collect.py --target 300 --val-split 0.15
"""

import os
import argparse
import time
import random
import csv
import shutil
from pathlib import Path
from typing import List
import cv2
import numpy as np

# Import the wrapper you already have
from hand_wrapper import get_hand_image, get_label

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Collect labeled hand images (0-9).")
parser.add_argument("--out", type=str, default="data", help="Output root directory.")
parser.add_argument("--target", type=int, default=300, help="Target images per class (0-9).")
parser.add_argument("--val-split", type=float, default=0.15, help="Fraction to allocate to validation.")
parser.add_argument("--classes", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated class labels.")
parser.add_argument("--append", action="store_true", help="Append to existing data/raw instead of overwriting.")
parser.add_argument("--preview", action="store_true", help="Show preview window while capturing.")
args = parser.parse_args()

OUT_ROOT = Path(args.out)
RAW_DIR = OUT_ROOT / "raw"
TRAIN_DIR = OUT_ROOT / "train"
VAL_DIR = OUT_ROOT / "val"
MANIFEST_CSV = OUT_ROOT / "manifest.csv"
TARGET_PER_CLASS = args.target
VAL_SPLIT = float(args.val_split)
CLASS_LABELS: List[str] = [c.strip() for c in args.classes.split(",")]

# ---------- Prepare directories ----------
def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for lbl in CLASS_LABELS:
        (RAW_DIR / lbl).mkdir(parents=True, exist_ok=True)
    if not args.append:
        # do not remove if append requested
        pass

ensure_dirs()

# ---------- Current counts ----------
def current_counts():
    counts = {}
    for lbl in CLASS_LABELS:
        p = RAW_DIR / lbl
        counts[lbl] = len(list(p.glob("*.jpg")))
    return counts

# ---------- Save helper ----------
def unique_filename(label: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    rnd = random.randint(1000, 9999)
    return f"{label}_{ts}_{rnd}.jpg"

def save_image_rgb(img_rgb: np.ndarray, path: Path):
    # hand_wrapper returns RGB uint8. OpenCV writes BGR.
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if img_rgb.ndim == 3 and img_rgb.shape[-1] == 3 else img_rgb
    cv2.imwrite(str(path), bgr)

# ---------- Capture loop ----------
print(f"Starting capture. Target per class: {TARGET_PER_CLASS}. Validation split: {VAL_SPLIT}")
print("Instructions:")
print(" - Press numeric key 0-9 in the labeling GUI to label the shown frame.")
print(" - In labeling GUI press 'q' to cancel current labeling attempt.")
print(" - Use Ctrl+C to stop the entire script anytime (data saved).")
print("If --preview is set this script will also show the captured frame in a window.")

try:
    while True:
        counts = current_counts()
        # check if all classes reached target
        done = all(counts[lbl] >= TARGET_PER_CLASS for lbl in CLASS_LABELS)
        print("Counts:", ", ".join(f"{lbl}:{counts[lbl]}" for lbl in CLASS_LABELS))
        if done:
            print("Target reached for all classes.")
            break

        # Get image (blocking). Keep trying if frame fails.
        try:
            frame = get_hand_image(timeout=5.0)
        except Exception as e:
            print("Warning: failed to capture frame:", e)
            continue

        # Optional preview
        if args.preview:
            try:
                disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 and frame.shape[-1] == 3 else frame
                cv2.imshow("Preview - press any key to continue", disp)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
            except Exception:
                pass

        # Ask user for label interactively
        lbl = get_label()
        if lbl == -1:
            print("Labeling cancelled. Continuing loop.")
            continue
        lbl = str(int(lbl))
        if lbl not in CLASS_LABELS:
            print(f"Label {lbl} not in allowed classes {CLASS_LABELS}. Skipping.")
            continue

        if counts[lbl] >= TARGET_PER_CLASS:
            print(f"Class {lbl} already reached target ({counts[lbl]}). Skipping save.")
            continue

        fname = unique_filename(lbl)
        out_path = RAW_DIR / lbl / fname
        save_image_rgb(frame, out_path)
        print(f"Saved {out_path}  (class {lbl})")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nInterrupted by user. Proceeding to split existing data.")

# ---------- Split into train/val ----------
print("Splitting into train/val ...")
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

manifest_rows = []
for lbl in CLASS_LABELS:
    src_folder = RAW_DIR / lbl
    all_files = sorted([p for p in src_folder.glob("*.jpg")])
    if len(all_files) == 0:
        print(f"Warning: no files for class {lbl} in {src_folder}")
        continue
    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * VAL_SPLIT))
    val_files = set(all_files[:n_val])
    train_files = all_files[n_val:]

    # create class dirs
    (TRAIN_DIR / lbl).mkdir(parents=True, exist_ok=True)
    (VAL_DIR / lbl).mkdir(parents=True, exist_ok=True)

    # Move files to train/val
    for p in train_files:
        dst = TRAIN_DIR / lbl / p.name
        if dst.exists():
            dst = TRAIN_DIR / lbl / f"dup_{int(time.time()*1000)}_{p.name}"
        shutil.copy2(p, dst)
        manifest_rows.append((str(dst.relative_to(OUT_ROOT)), lbl, "train"))

    for p in val_files:
        dst = VAL_DIR / lbl / p.name
        if dst.exists():
            dst = VAL_DIR / lbl / f"dup_{int(time.time()*1000)}_{p.name}"
        shutil.copy2(p, dst)
        manifest_rows.append((str(dst.relative_to(OUT_ROOT)), lbl, "val"))

# ---------- Write manifest ----------
OUT_ROOT.mkdir(parents=True, exist_ok=True)
with open(MANIFEST_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label", "split"])
    for row in manifest_rows:
        writer.writerow(row)

print(f"Done. Train dir: {TRAIN_DIR}, Val dir: {VAL_DIR}")
print(f"Manifest csv: {MANIFEST_CSV}")
print("Summary counts (train / val):")
for lbl in CLASS_LABELS:
    tcount = len(list((TRAIN_DIR / lbl).glob("*.jpg"))) if (TRAIN_DIR / lbl).exists() else 0
    vcount = len(list((VAL_DIR / lbl).glob("*.jpg"))) if (VAL_DIR / lbl).exists() else 0
    print(f"  class {lbl}: train={tcount}, val={vcount}")

print("You can now run: python train.py --train-dir data/train --val-dir data/val")

