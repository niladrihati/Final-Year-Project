"""
hand_wrapper.py

Provides two runtime APIs used by the training/inference scripts:

def get_hand_image() -> np.ndarray
    - Returns a single HxWxC uint8 RGB image captured from the hand module.

def get_label() -> int
    - Interactive helper used during data collection.
    - Displays live frame and waits for a numeric key 0-9 to be pressed.
    - Returns the pressed integer label.
    - Press 'q' to cancel and return -1.

This wrapper attempts to import HandDetector from `hand_module`.
If that fails it will try to load a module from the uploaded file path
(/mnt/data/2a535ba5-b32b-4d32-b115-168efb3c7348.py).
"""

import time
import numpy as np
import cv2
import importlib
import importlib.util
import sys
from typing import Optional

# Try importing a HandDetector from common module names
HAND_MODULE_PATHS = [
    "hand_module",  # expected name
    "/mnt/data/2a535ba5-b32b-4d32-b115-168efb3c7348.py",  # uploaded fallback
]

HandDetector = None

for path in HAND_MODULE_PATHS:
    try:
        if path.endswith(".py"):
            # load by file path
            spec = importlib.util.spec_from_file_location("hand_module_local", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
        else:
            mod = importlib.import_module(path)
        # attempt to fetch a HandDetector class/object
        if hasattr(mod, "HandDetector"):
            HandDetector = getattr(mod, "HandDetector")
            break
        # fallback: find any class with 'Hand' in name
        for attr_name in dir(mod):
            if "hand" in attr_name.lower() and attr_name.lower().startswith("hand"):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type):
                    HandDetector = attr
                    break
        if HandDetector:
            break
    except Exception:
        continue

if HandDetector is None:
    raise ImportError(
        "Could not import HandDetector from 'hand_module'. "
        "Place a module named hand_module.py exposing a HandDetector class "
        "or adjust HAND_MODULE_PATHS."
    )

# Instantiate detector (best-effort). Accept both parameterless or common constructors.
try:
    detector = HandDetector()
except TypeError:
    # try common alternative signatures
    try:
        detector = HandDetector(device=0)
    except Exception:
        detector = None

# If detector is None we will use OpenCV fallback
USE_FALLBACK_OPENCV = detector is None


def _cv2_bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_hand_image(timeout: Optional[float] = None) -> np.ndarray:
    """
    Capture a single image from the hand module and return an RGB uint8 numpy array.
    timeout: optional seconds to wait before raising RuntimeError (None = block until frame)
    """
    start = time.time()
    if not USE_FALLBACK_OPENCV:
        # Try using the provided detector's API. Common method names: get_frame, read, capture
        for method in ("get_frame", "read", "capture", "get_image", "get_hand_image"):
            try:
                fn = getattr(detector, method)
                frame = fn()
                # if method returns (ok, frame) handle that
                if isinstance(frame, tuple) and len(frame) == 2 and isinstance(frame[0], bool):
                    _, frame = frame
                if frame is None:
                    raise RuntimeError("HandDetector returned None frame")
                return _cv2_bgr_to_rgb(frame)
            except Exception:
                continue
        # fallback: if detector has 'run' or 'start' and produces attribute 'last_frame'
        try:
            if hasattr(detector, "start"):
                detector.start()
                # give it a moment
                time.sleep(0.1)
            if hasattr(detector, "last_frame"):
                frame = getattr(detector, "last_frame")
                if frame is not None:
                    return _cv2_bgr_to_rgb(frame)
        except Exception:
            pass

    # Fallback: direct OpenCV capture from default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.open(0)
    while True:
        if timeout and (time.time() - start) > timeout:
            cap.release()
            raise RuntimeError("Timeout while getting camera frame")
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        cap.release()
        return _cv2_bgr_to_rgb(frame)


def get_label() -> int:
    """
    Interactive labelling helper.
    Shows live camera feed (overlay) and waits for a numeric key 0-9.
    Returns int label pressed, or -1 if user pressed 'q' to cancel.
    """
    print("Labeling mode: press numeric key 0-9 to label current frame. Press 'q' to cancel.")
    # Prefer to use detector's preview if available
    window_name = "Labeling - press 0-9 (q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cap = None

    try:
        if not USE_FALLBACK_OPENCV:
            # try to use detector preview loop if available
            if hasattr(detector, "start"):
                try:
                    detector.start()
                except Exception:
                    pass
            # live loop - try to pull frames via get_hand_image wrapper
            while True:
                frame = None
                try:
                    frame = get_hand_image(timeout=2.0)
                except Exception:
                    pass
                if frame is None:
                    # fallback to opencv capture if detector fails temporarily
                    if cap is None:
                        cap = cv2.VideoCapture(0)
                    ret, bgr = cap.read()
                    if not ret:
                        continue
                    frame = _cv2_bgr_to_rgb(bgr)
                # convert RGB->BGR for OpenCV display
                disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv2.destroyWindow(window_name)
                    if cap:
                        cap.release()
                    return -1
                # numeric keys 48-57 correspond to '0'-'9'
                if 48 <= key <= 57:
                    label = key - 48
                    cv2.destroyWindow(window_name)
                    if cap:
                        cap.release()
                    return int(label)
        else:
            # pure OpenCV fallback interactive capture
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.open(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cap.release()
                    cv2.destroyWindow(window_name)
                    return -1
                if 48 <= key <= 57:
                    label = key - 48
                    cap.release()
                    cv2.destroyWindow(window_name)
                    return int(label)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


# Simple test when run directly
if __name__ == "__main__":
    print("Testing get_hand_image() ...")
    try:
        img = get_hand_image(timeout=5.0)
        print(f"Captured image shape: {img.shape}, dtype: {img.dtype}")
        # show for a moment
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Captured", bgr)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Failed to capture image:", e)

    lbl = get_label()
    print("Label returned:", lbl)
