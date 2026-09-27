"""
database.py
Handles all local, file-based persistence:
 - plants_data.json  -> read-only reference knowledge base
 - history.json       -> every recognition ever made
 - favorites.json      -> plant names the user starred

No external DB / API needed, everything is plain JSON so the whole
project stays dependency-light and fully offline.
"""
import json
import os
from datetime import datetime
from collections import Counter

from config import PLANTS_DB_PATH, HISTORY_PATH, FAVORITES_PATH


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class PlantDatabase:
    """Single access point for all app data."""

    def __init__(self):
        self.plants = _load_json(PLANTS_DB_PATH, {})
        self.history = _load_json(HISTORY_PATH, [])
        self.favorites = set(_load_json(FAVORITES_PATH, []))

    # ---------- Plant knowledge ----------
    def get_plant(self, name):
        return self.plants.get(name)

    def all_plant_names(self):
        return sorted(self.plants.keys())

    def all_categories(self):
        cats = {info.get("category", "Other") for info in self.plants.values()}
        return ["All"] + sorted(cats)

    def plants_in_category(self, category):
        if category == "All":
            return self.all_plant_names()
        return sorted(
            name for name, info in self.plants.items()
            if info.get("category", "Other") == category
        )

    def search(self, query):
        """Case-insensitive search across name, scientific name, and uses."""
        query = (query or "").strip().lower()
        if not query:
            return self.all_plant_names()
        results = []
        for name, info in self.plants.items():
            haystack = " ".join([
                name,
                info.get("scientific_name", ""),
                info.get("family", ""),
                " ".join(info.get("traditional_uses", [])),
            ]).lower()
            if query in haystack:
                results.append(name)
        return sorted(results)

    # ---------- History ----------
    def add_history(self, image_name, image_path, predicted_plant, confidence):
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": image_name,
            "image_path": image_path,
            "predicted_plant": predicted_plant,
            "confidence": round(float(confidence), 2),
        }
        self.history.append(record)
        _save_json(HISTORY_PATH, self.history)
        return record

    def get_history(self):
        return list(reversed(self.history))  # newest first

    def clear_history(self):
        self.history = []
        _save_json(HISTORY_PATH, self.history)

    def export_history_csv(self, csv_path):
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Image", "Predicted Plant", "Confidence (%)"])
            for r in self.history:
                writer.writerow([r["timestamp"], r["image_name"],
                                  r["predicted_plant"], r["confidence"]])

    # ---------- Favorites ----------
    def toggle_favorite(self, plant_name):
        if plant_name in self.favorites:
            self.favorites.remove(plant_name)
            added = False
        else:
            self.favorites.add(plant_name)
            added = True
        _save_json(FAVORITES_PATH, sorted(self.favorites))
        return added

    def is_favorite(self, plant_name):
        return plant_name in self.favorites

    def get_favorites(self):
        return sorted(self.favorites)

    # ---------- Statistics ----------
    def get_statistics(self):
        total = len(self.history)
        plant_counts = Counter(r["predicted_plant"] for r in self.history)
        most_common = plant_counts.most_common(5)
        avg_conf = (
            round(sum(r["confidence"] for r in self.history) / total, 2)
            if total else 0.0
        )
        category_counts = Counter()
        for r in self.history:
            info = self.plants.get(r["predicted_plant"], {})
            category_counts[info.get("category", "Other")] += 1

        return {
            "total_predictions": total,
            "unique_plants_identified": len(plant_counts),
            "average_confidence": avg_conf,
            "most_common": most_common,
            "category_counts": dict(category_counts),
            "total_known_plants": len(self.plants),
            "total_favorites": len(self.favorites),
        }
