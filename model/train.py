"""
train.py
Trains the plant-recognition model from a folder-per-class image dataset,
using MobileNetV2 transfer learning (see model/model_builder.py).

Expected dataset layout:

    dataset/
        Tulsi/
            img1.jpg
            img2.jpg
            ...
        Neem/
            img1.jpg
            ...
        ... one folder per plant. Empty folders are fine — they're
        automatically skipped (see note below).

Run from the project root:
    python -m model.train
or
    python model/train.py

Outputs (all saved into model/):
    plant_model.h5           - the trained Keras model
    class_indices.json       - maps class index -> plant name (used by predict.py)
    training_history.json    - epoch-by-epoch accuracy/loss (for the AI Model tab)

Notes on two easy-to-hit gotchas this script now handles automatically:

  1. .jfif files (common when saving images straight from a browser/
     Google Images) are silently ignored by Keras's image loader, which
     only recognizes .jpg/.jpeg/.png/.bmp/.ppm/.tif/.tiff. This script
     renames any *.jfif files it finds to *.jpg first (same bytes —
     JFIF *is* JPEG data, just a different file extension), so they
     actually get used.

  2. Only folders that contain at least MIN_IMAGES_PER_CLASS images are
     used as classes. Previously, *every* folder under dataset/ counted
     as a class even if empty, which silently forced the model to
     "guess" among many classes it had never seen a single image for —
     this is the #1 cause of very low confidence scores (e.g. ~7% with
     15 classes is just 1/15, i.e. random guessing).
"""
import os
import sys
import json

# allow running this file directly (python model/train.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from config import (
    DATASET_DIR, MODEL_PATH, CLASS_INDEX_PATH, TRAINING_HISTORY_PATH,
    IMG_SIZE, BATCH_SIZE, DEFAULT_EPOCHS,
)
from model.model_builder import build_transfer_model

MIN_IMAGES_PER_CLASS = 2
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff")


def _normalize_jfif_files(dataset_dir):
    """Renames *.jfif -> *.jpg in-place (JFIF is JPEG data under a
    different extension) so Keras's loader actually picks them up."""
    renamed = 0
    for root, _dirs, files in os.walk(dataset_dir):
        for name in files:
            if name.lower().endswith(".jfif"):
                src = os.path.join(root, name)
                dst = os.path.join(root, os.path.splitext(name)[0] + ".jpg")
                if os.path.exists(dst):
                    dst = os.path.join(
                        root, os.path.splitext(name)[0] + "_jfif.jpg")
                os.rename(src, dst)
                renamed += 1
    if renamed:
        print(f"[i] Renamed {renamed} .jfif file(s) to .jpg so they can be used for training.")


def _scan_classes(dataset_dir):
    """
    Returns (valid_class_names, report) where report is a list of
    (name, image_count, included: bool) for every folder found, so the
    user can see exactly what was used and what was skipped.
    """
    if not os.path.isdir(dataset_dir):
        return [], []

    report = []
    valid = []
    for name in sorted(os.listdir(dataset_dir)):
        full = os.path.join(dataset_dir, name)
        if not os.path.isdir(full):
            continue
        count = sum(
            1 for f in os.listdir(full)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        included = count >= MIN_IMAGES_PER_CLASS
        report.append((name, count, included))
        if included:
            valid.append(name)
    return valid, report


def _print_scan_report(report):
    print("\nDataset scan:")
    for name, count, included in report:
        tag = "OK" if included else "SKIPPED (not enough images)"
        print(f"  - {name:<20s} {count:>3d} image(s)  [{tag}]")
    print()


def train(epochs=DEFAULT_EPOCHS, dataset_dir=DATASET_DIR, fine_tune=False):
    _normalize_jfif_files(dataset_dir)
    valid_classes, report = _scan_classes(dataset_dir)
    _print_scan_report(report)

    if len(valid_classes) < 2:
        print(
            "[!] Not enough trained-ready classes found.\n"
            f"    Each plant folder needs at least {MIN_IMAGES_PER_CLASS} images, "
            "and at least 2 such folders are needed.\n"
            f"    Add more images under {dataset_dir}/<PlantName>/ and re-run.\n"
        )
        return

    # Keep batch_size sane for small datasets
    smallest_class_count = min(
        c for _n, c, included in report if included
    )
    effective_batch_size = max(1, min(BATCH_SIZE, smallest_class_count))

    # 80/20 train/validation split with light augmentation for a small dataset
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
    )

    common_args = dict(
        directory=dataset_dir,
        classes=valid_classes,          # <-- only the folders that actually have images
        target_size=IMG_SIZE,
        batch_size=effective_batch_size,
        class_mode="categorical",
    )

    train_gen = datagen.flow_from_directory(subset="training", **common_args)
    val_gen = datagen.flow_from_directory(subset="validation", **common_args)

    print(f"Training on {len(valid_classes)} classes: {', '.join(valid_classes)}\n")

    model = build_transfer_model(
        input_shape=IMG_SIZE + (3,),
        num_classes=len(valid_classes),
        fine_tune=fine_tune,
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )

    # class_indices maps name -> index; invert it for prediction lookup
    class_indices = {v: k for k, v in train_gen.class_indices.items()}
    with open(CLASS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)

    # Save history for the GUI's "AI Model" tab
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=2)

    model.save(MODEL_PATH)
    print(f"\n[OK] Model saved to {MODEL_PATH}")
    print(f"[OK] Class map saved to {CLASS_INDEX_PATH}")


if __name__ == "__main__":
    train()