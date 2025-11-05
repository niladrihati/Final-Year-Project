#!/usr/bin/env python3
"""
convert_tflite.py

Convert a Keras .h5 model to:
 - float16 quantized TFLite
 - full integer (int8) quantized TFLite using a representative dataset

Usage:
    python convert_tflite.py --model digit_cnn.h5 --train-dir data/train \
        --img-size 64 --out-prefix digit_cnn_tflite --repr-samples 300

Notes:
 - Representative dataset should match model preprocessing (resize, /255.0).
 - Requires TensorFlow 2.x.
"""

import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description="Convert Keras model to TFLite (float16 & int8).")
parser.add_argument("--model", type=str, required=True, help="Path to Keras model (.h5 or SavedModel dir).")
parser.add_argument("--train-dir", type=str, default="data/train", help="Training images (used for representative dataset).")
parser.add_argument("--img-size", type=int, default=64, help="Model input size (square).")
parser.add_argument("--out-prefix", type=str, default="digit_cnn", help="Prefix for output .tflite files.")
parser.add_argument("--repr-samples", type=int, default=300, help="Number of representative samples to use for int8 quant.")
parser.add_argument("--batch", type=int, default=1, help="Batch size for representative dataset generator.")
args = parser.parse_args()

MODEL_PATH = args.model
TRAIN_DIR = args.train_dir
IMG_SIZE = (args.img_size, args.img_size)
OUT_PREFIX = args.out_prefix
REPR_SAMPLES = int(args.repr_samples)
BATCH = int(args.batch)

print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded. Input shape:", model.input_shape)

# ---------- Helper: build representative dataset tf.data.Dataset ----------
def build_rep_dataset(train_dir, img_size=IMG_SIZE, batch_size=BATCH, max_samples=REPR_SAMPLES):
    """
    Returns a tf.data.Dataset that yields batches of preprocessed images (float32, [0,1]).
    This dataset will be used by the representative dataset generator.
    """
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels=None,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    # normalize to match training preprocessing (/255.0)
    ds = ds.map(lambda x: tf.cast(x, tf.float32) / 255.0)
    # limit samples
    if max_samples is not None:
        ds = ds.take(max_samples // batch_size if batch_size > 0 else max_samples)
    return ds

rep_ds = build_rep_dataset(TRAIN_DIR, IMG_SIZE, BATCH, REPR_SAMPLES)

# Representative generator for converter (yields input as list of ndarrays)
def representative_gen():
    """
    Yields representative samples as required by TFLite converter.
    Each yield must be a list or tuple of input arrays matching model input signature.
    """
    count = 0
    for batch in rep_ds:
        # batch is a float32 tensor in [0,1], shape (B,H,W,C)
        arr = batch.numpy()
        # If model expects a different input range modify here (e.g., *255)
        for i in range(arr.shape[0]):
            yield [np.expand_dims(arr[i].astype(np.float32), axis=0)]
            count += 1
            if count >= REPR_SAMPLES:
                return

# ---------- 1) Save float16 quantized TFLite ----------
print("Converting -> float16 quantized TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# float16 quantization
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
fp16_path = f"{OUT_PREFIX}_float16.tflite"
with open(fp16_path, "wb") as f:
    f.write(tflite_fp16)
print("Saved float16 TFLite:", fp16_path)

# ---------- 2) Full integer (int8) quantization ----------
print("Converting -> full integer (INT8) quantized TFLite (may be slower).")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Use representative dataset for full integer quantization
converter.representative_dataset = representative_gen
# Ensure target ops are INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set input and output types to int8 for full integer quantization.
# Note: If you prefer uint8 for some hardware, set tf.uint8 instead.
try:
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
except Exception:
    # older TF versions may not have these attributes; continue anyway
    pass

try:
    tflite_int8 = converter.convert()
    int8_path = f"{OUT_PREFIX}_int8.tflite"
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)
    print("Saved int8 TFLite:", int8_path)
except Exception as e:
    print("Full int8 conversion failed:", e)
    print("You can try with fewer representative samples or check model ops unsupported by INT8.")

# ---------- 3) Optional: save a dynamic-range (no representative) quantized model ----------
print("Converting -> dynamic-range (weight-only) quantized TFLite (smaller, no representative).")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dr = converter.convert()
dr_path = f"{OUT_PREFIX}_dynamic_range.tflite"
with open(dr_path, "wb") as f:
    f.write(tflite_dr)
print("Saved dynamic-range TFLite:", dr_path)

print("Conversion complete. Files generated (if conversion succeeded):")
for p in [fp16_path, int8_path if 'int8_path' in locals() else None, dr_path]:
    if p and Path(p).exists():
        print(" -", p)
