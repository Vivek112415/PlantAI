"""
image_utils.py
All OpenCV / NumPy / Pillow image-handling code lives here:
 - loading a file from disk
 - resizing + normalizing for the CNN
 - building a Tkinter-displayable preview thumbnail
"""
import cv2
import numpy as np
from PIL import Image, ImageTk

from config import IMG_SIZE


def load_image_bgr(path):
    """Read an image file with OpenCV. Returns a BGR numpy array or None."""
    data = cv2.imread(path)
    return data


def preprocess_for_model(path, target_size=IMG_SIZE):
    """
    Full pipeline used before feeding an image to the CNN:
      1. read with OpenCV
      2. convert BGR -> RGB
      3. resize
      4. normalize to [0, 1]
      5. add batch dimension
    Returns a numpy array shaped (1, H, W, 3), or raises ValueError.
    """
    img = load_image_bgr(path)
    if img is None:
        raise ValueError(f"Could not read image at: {path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)


def make_preview_thumbnail(path, max_size=(380, 380)):
    """
    Returns a Pillow-based ImageTk.PhotoImage suitable for display in a
    Tkinter Label/Canvas, preserving aspect ratio.
    """
    pil_img = Image.open(path).convert("RGB")
    pil_img.thumbnail(max_size, Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


def make_small_thumbnail(path, size=(64, 64)):
    """Small square-ish thumbnail, used in history rows / reports."""
    pil_img = Image.open(path).convert("RGB")
    pil_img.thumbnail(size, Image.LANCZOS)
    return pil_img


def image_to_base64(path, max_size=(300, 300)):
    """Encode a resized image as base64 for embedding into HTML reports."""
    import io
    import base64
    pil_img = Image.open(path).convert("RGB")
    pil_img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
