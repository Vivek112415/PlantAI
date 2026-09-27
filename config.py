"""
config.py
Central configuration: paths, image settings, and app constants.
Keeping everything here means every other module stays clean.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

PLANTS_DB_PATH = os.path.join(DATA_DIR, "plants_data.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
FAVORITES_PATH = os.path.join(DATA_DIR, "favorites.json")

MODEL_PATH = os.path.join(MODEL_DIR, "plant_model.h5")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_indices.json")
TRAINING_HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

IMG_SIZE = (128, 128)        # width, height fed to the CNN
CHANNELS = 3
BATCH_SIZE = 16
DEFAULT_EPOCHS = 25

APP_TITLE = "PlantSense - Medicinal Plant Recognition System"

APP_MIN_SIZE = (1080, 680)

# Ensure runtime folders exist even on a fresh checkout
for _d in (DATA_DIR, MODEL_DIR, DATASET_DIR, REPORTS_DIR, ASSETS_DIR):
    os.makedirs(_d, exist_ok=True)
