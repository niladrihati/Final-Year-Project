#!/usr/bin/env python3
"""
train.py

Train a compact digit classifier (0-9) with TensorFlow 2.x.
Usage:
    python train.py --train-dir data/train --val-dir data/val --epochs 25 --batch 32
"""

import argparse
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import time

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Train digit CNN (0-9).")
parser.add_argument("--train-dir", type=str, default="data/train")
parser.add_argument("--val-dir", type=str, default="data/val")
parser.add_argument("--img-size", type=int, default=64, help="Square image size (H=W).")
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--model-out", type=str, default="digit_cnn.h5")
parser.add_argument("--history-out", type=str, default="train_history.json")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

TRAIN_DIR = Path(args.train_dir)
VAL_DIR = Path(args.val_dir)
IMG_SIZE = (args.img_size, args.img_size)
BATCH = args.batch
EPOCHS = args.epochs
LR = args.lr
MODEL_OUT = args.model_out
HISTORY_OUT = args.history_out

tf.random.set_seed(args.seed)

# ---------- Data pipeline ----------
AUTOTUNE = tf.data.AUTOTUNE

def build_datasets(train_dir: Path, val_dir: Path, img_size=IMG_SIZE, batch=BATCH):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(train_dir),
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch,
        shuffle=True,
        seed=args.seed,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(val_dir),
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch,
        shuffle=False,
    )

    # Augmentation and normalization (applied on-the-fly)
    data_augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.08),
    ], name="data_augmentation")

    normalization = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")

    def prepare(ds, training=False):
        ds = ds.map(lambda x, y: (tf.image.resize(x, img_size), y), num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.map(lambda x, y: (data_augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
        return ds.cache().prefetch(AUTOTUNE)

    return prepare(train_ds, training=True), prepare(val_ds, training=False), train_ds.class_names

train_ds, val_ds, CLASS_NAMES = build_datasets(TRAIN_DIR, VAL_DIR)

NUM_CLASSES = len(CLASS_NAMES)
print("Classes:", CLASS_NAMES)

# ---------- Model ----------
def build_model(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES):
    inputs = tf.keras.Input(shape=input_shape)
    # Block 1
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)  # small conv to learn low-level edges
    x = tf.keras.layers.MaxPool2D()(x)
    # Block 2
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)      # increase channels for features
    x = tf.keras.layers.MaxPool2D()(x)
    # Block 3
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)      # richer features before pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)                             # aggressive dimensionality reduction
    x = tf.keras.layers.Dense(64, activation="relu")(x)                         # compact dense for classification
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="digit_cnn")
    return model

model = build_model()
model.summary()

# ---------- Compile ----------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------- Callbacks ----------
timestamp = time.strftime("%Y%m%d-%H%M%S")
ckpt_path = f"{Path(MODEL_OUT).stem}_{timestamp}.h5"
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

# ---------- Train ----------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Save final model (best weights likely restored by EarlyStopping)
model.save(MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")

# Save training history
hist_out = {
    "params": {
        "img_size": IMG_SIZE,
        "batch": BATCH,
        "epochs": EPOCHS,
        "lr": LR,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES
    },
    "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
}
with open(HISTORY_OUT, "w") as f:
    json.dump(hist_out, f, indent=2)
print(f"Saved training history to {HISTORY_OUT}")

# ---------- Simple evaluation on validation set ----------
# Collect predictions and labels (works for moderate-sized val sets)
y_true = []
y_pred = []
y_prob = []

for x_batch, y_batch in val_ds:
    probs = model.predict(x_batch, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(y_batch.numpy().tolist())
    y_pred.extend(preds.tolist())
    y_prob.extend(np.max(probs, axis=1).tolist())

y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred, dtype=int)
y_prob = np.array(y_prob, dtype=float)

acc = (y_true == y_pred).mean() if y_true.size > 0 else 0.0
print(f"Validation accuracy: {acc:.4f}  ({y_true.size} samples)")

# Confusion matrix (printed compactly)
if y_true.size > 0:
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    # show per-class precision/recall simple compute
    tp = np.diag(cm).astype(np.float32)
    per_class_support = cm.sum(axis=1).astype(np.float32)
    pred_support = cm.sum(axis=0).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_support > 0, tp / pred_support, 0.0)
        recall = np.where(per_class_support > 0, tp / per_class_support, 0.0)
    for i, cname in enumerate(CLASS_NAMES):
        print(f"Class {cname}: support={int(per_class_support[i])}, precision={precision[i]:.3f}, recall={recall[i]:.3f}")

print("Training complete.")
print(f"Best checkpoint saved as {ckpt_path} (ModelCheckpoint). Final exported model: {MODEL_OUT}")

# ---------- Quick inference helper (example) ----------
def infer_image_np(img_np):
    """
    Accepts HxWxC RGB uint8 numpy array. Returns (label:int, confidence:float).
    """
    img = tf.image.resize(img_np, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    probs = model.predict(img, verbose=0)[0]
    label = int(np.argmax(probs))
    conf = float(np.max(probs))
    return label, conf

# Example: not executed unless user calls
if __name__ == "__main__":
    # quick sanity check: take one batch from val and infer
    for x_batch, y_batch in val_ds.take(1):
        lab, conf = infer_image_np(x_batch[0].numpy())
        print("Sample infer -> label:", lab, "conf:", conf)
