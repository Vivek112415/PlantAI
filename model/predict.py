"""
predict.py
Thin wrapper around the trained Keras model used by the GUI.

Designed to fail *gracefully*: if no model has been trained yet
(model/plant_model.h5 missing), the app should still fully function
for browsing/searching the plant library - it just can't recognize
images yet. That state is reported via `PlantPredictor.available`.
"""
import os
import json

from config import MODEL_PATH, CLASS_INDEX_PATH, IMG_SIZE
from utils.image_utils import preprocess_for_model


class PlantPredictor:
    def __init__(self):
        self.model = None
        self.class_map = {}
        self.available = False
        self._load()

    def _load(self):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASS_INDEX_PATH)):
            return
        try:
            # Imported lazily: TensorFlow is slow to import, and we don't
            # want to pay that cost if there's no model to load anyway.
            from tensorflow.keras.models import load_model
            self.model = load_model(MODEL_PATH)
            with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.class_map = {int(k): v for k, v in raw.items()}
            self.available = True
        except Exception as exc:  # noqa: BLE001 - surfaced to the GUI as text
            self.available = False
            self.load_error = str(exc)

    def predict(self, image_path):
        """
        Returns a dict:
            {
              "plant": str,
              "confidence": float (0-100),
              "top3": [(plant, confidence), ...]
            }
        Raises RuntimeError if no trained model is available, and
        ValueError if the image can't be read.
        """
        if not self.available:
            raise RuntimeError(
                "No trained model found. Run `python model/train.py` "
                "with a populated dataset/ folder first."
            )

        batch = preprocess_for_model(image_path, target_size=IMG_SIZE)
        probs = self.model.predict(batch, verbose=0)[0]

        ranked = sorted(
            ((self.class_map.get(i, f"class_{i}"), float(p) * 100.0)
             for i, p in enumerate(probs)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        top_plant, top_conf = ranked[0]
        return {
            "plant": top_plant,
            "confidence": top_conf,
            "top3": ranked[:3],
        }
