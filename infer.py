#!/usr/bin/env python3
"""
infer.py

Loads the trained TensorFlow model and performs real-time inference
using the hand camera module to predict digits (0–9).
Provides both text and optional voice output.

Usage:
    python infer.py --model digit_cnn.h5 --speak True
"""

import argparse
import numpy as np
import tensorflow as tf
import cv2
import time
import pyttsx3
from hand_wrapper import get_hand_image

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Run real-time digit inference from hand module.")
parser.add_argument("--model", type=str, default="digit_cnn.h5", help="Path to trained model file (.h5).")
parser.add_argument("--img-size", type=int, default=64, help="Model input image size.")
parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold to accept prediction.")
parser.add_argument("--speak", type=str, default="False", help="Speak predictions aloud (True/False).")
parser.add_argument("--interval", type=float, default=1.0, help="Seconds between inferences.")
args = parser.parse_args()

MODEL_PATH = args.model
IMG_SIZE = (args.img_size, args.img_size)
CONF_THRESH = float(args.threshold)
SPEAK = args.speak.lower() == "true"
INTERVAL = args.interval

# ---------- Model load ----------
print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ---------- TTS setup ----------
engine = None
if SPEAK:
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        print("Voice output enabled.")
    except Exception as e:
        print("Warning: could not initialize voice engine:", e)
        SPEAK = False

# ---------- Prediction helper ----------
def preprocess(img_np: np.ndarray) -> np.ndarray:
    """Resize and normalize RGB image for model input."""
    img = cv2.resize(img_np, IMG_SIZE)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_digit(img_np: np.ndarray):
    """Return (pred_label:int, confidence:float, top2:list)."""
    x = preprocess(img_np)
    probs = model.predict(x, verbose=0)[0]
    top2_idx = probs.argsort()[-2:][::-1]
    top2 = [(int(i), float(probs[i])) for i in top2_idx]
    label, conf = top2[0]
    if conf < CONF_THRESH:
        return -1, conf, top2
    return label, conf, top2

def speak_text(text: str):
    if engine:
        engine.say(text)
        engine.runAndWait()

# ---------- Real-time inference loop ----------
print("Starting live inference.")
print("Press Ctrl+C to stop.")

try:
    while True:
        frame = None
        try:
            frame = get_hand_image(timeout=5.0)
        except Exception as e:
            print("Capture failed:", e)
            continue

        if frame is None:
            print("No frame captured.")
            continue

        label, conf, top2 = predict_digit(frame)
        if label == -1:
            msg = f"Uncertain ({conf:.2f})"
        else:
            msg = f"Predicted: {label} (conf {conf:.2f})"

        # print text output
        print(msg)

        # display frame (optional)
        disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Digit Inference - press q to quit", disp)

        # optional voice output
        if SPEAK and label != -1:
            speak_text(str(label))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("User requested exit.")
            break

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    cv2.destroyAllWindows()
    if engine:
        engine.stop()
    print("Session ended.")
