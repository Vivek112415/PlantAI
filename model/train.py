"""
train.py
Trains the plant-recognition CNN from a folder-per-class image dataset.

Expected dataset layout (create this yourself, it is NOT included):

    dataset/
        Tulsi/
            img1.jpg
            img2.jpg
            ...
        Neem/
            img1.jpg
            ...
        Aloe Vera/
            ...
        ... one folder per plant, 50-300+ images each is a good start.

Run from the project root:
    python -m model.train
or
    python model/train.py

Outputs (all saved into model/):
    plant_model.h5           - the trained Keras model
    class_indices.json       - maps class index -> plant name (used by predict.py)
    training_history.json    - epoch-by-epoch accuracy/loss (for the Stats tab)
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
from model.model_builder import build_cnn


def count_classes(dataset_dir):
    if not os.path.isdir(dataset_dir):
        return 0
    return len([d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))])


def train(epochs=DEFAULT_EPOCHS, dataset_dir=DATASET_DIR):
    num_classes = count_classes(dataset_dir)
    if num_classes < 2:
        print(
            "\n[!] Not enough classes found in the dataset folder.\n"
            f"    Expected structure: {dataset_dir}/<PlantName>/*.jpg\n"
            "    Please add at least 2 plant folders with images and re-run.\n"
        )
        return

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

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    model = build_cnn(input_shape=IMG_SIZE + (3,), num_classes=num_classes)
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

    # Save history for the GUI's "AI Performance" tab
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=2)

    model.save(MODEL_PATH)
    print(f"\n[OK] Model saved to {MODEL_PATH}")
    print(f"[OK] Class map saved to {CLASS_INDEX_PATH}")


if __name__ == "__main__":
    train()
