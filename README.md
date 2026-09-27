# \U0001F33F PlantAI — Medicinal Plant Recognition System

A fully offline, Python desktop application that identifies medicinal
plants/leaves from a photo using a CNN (TensorFlow/Keras), shows
traditionally reported uses and safety precautions from a local
knowledge base, and keeps a history + statistics of recognitions.
No internet connection or API key is required at runtime.

> ⚠️ **Important disclaimer:** AI identification of a plant does **not**
> confirm that it is safe to consume or suitable for treating any
> disease. All "health benefit" information shown by this app comes
> from a locally stored, user-editable reference file
> (`data/plants_data.json`) and is for general educational purposes
> only — it is **not** medical advice.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📷 Image Upload | Select a plant/leaf photo (jpg/png/bmp/webp) |
| 🤖 AI Recognition | CNN model (Keras) classifies the plant |
| 📊 Confidence Score | Shows prediction confidence % + top-3 alternatives |
| 🌱 Plant Details | Common name, scientific name, family, category, description |
| 💚 Health Benefits | Traditionally reported uses, from local dataset |
| ⚠️ Safety Info | Precautions / warnings per plant |
| 🔍 Plant Search | Search by name, scientific name, family, or use |
| 🗂️ Category Filter | Filter the library by plant type (Herb, Tree, Shrub…) |
| 📚 Plant Library | Browse all 15 supported plants, even without a trained model |
| ⭐ Favorites | Star plants for quick access later |
| 📜 Recognition History | Every prediction is timestamped and saved locally |
| 📤 CSV Export | Export history to a spreadsheet-friendly CSV |
| 💾 Save Report | Save a `.txt` + styled `.html` report per recognition |
| 📈 Statistics | Total predictions, most-recognized plants, category breakdown |
| 🤖 AI Model Tab | Shows training accuracy/loss curves once trained |
| 🌓 Dark / Light Mode | Toggle theme from the top bar |

---

## 🗂️ Project Structure

```
PlantAI/
├── main.py                     # App entry point — run this
├── config.py                   # Central paths & constants
├── requirements.txt
├── README.md
│
├── data/
│   ├── plants_data.json        # Editable knowledge base (15 plants)
│   ├── history.json            # Auto-created: recognition history
│   └── favorites.json          # Auto-created: starred plants
│
├── model/
│   ├── model_builder.py        # CNN architecture (Keras Sequential)
│   ├── train.py                # Train from dataset/ folder
│   ├── predict.py              # Loads model, runs predictions
│   ├── plant_model.h5          # Created after training (not included)
│   ├── class_indices.json      # Created after training
│   └── training_history.json   # Created after training
│
├── utils/
│   ├── database.py             # JSON-backed history/favorites/search/stats
│   ├── image_utils.py          # OpenCV/Pillow preprocessing & thumbnails
│   └── report_utils.py         # .txt / .html report generation
│
├── gui/
│   ├── app.py                  # Main Tkinter window & all 6 tabs
│   ├── theme.py                # Light/Dark ttk themes
│   └── charts.py                # Dependency-free canvas bar/line charts
│
├── dataset/                    # YOU add training images here (empty folders provided)
│   ├── Tulsi/
│   ├── Neem/
│   ├── ... (15 folders total, one per plant)
│
├── reports/                    # Saved recognition reports land here
├── sample_images/              # Put test images here for quick manual testing
└── assets/                     # Reserved for icons/screenshots
```

---

## ⚙️ Installation

1. Make sure you have **Python 3.9–3.11** installed (TensorFlow does not
   yet support every newer Python version — check TensorFlow's release
   notes if unsure).
2. Tkinter usually ships with Python. On Linux, if it's missing:
   ```bash
   sudo apt-get install python3-tk
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the App

```bash
python main.py
```

The app opens immediately, even **before** you train a model:
- The **Plant Library**, **Favorites**, **History**, and **Statistics**
  tabs work right away, since they only depend on the local JSON data.
- The **Recognize** tab will tell you the model isn't trained yet if
  you try to predict before training one (see below).

---

## 🧠 Training the Recognition Model

The project ships **without** a pre-trained `.h5` file or any leaf
photos, since a real plant-image dataset is large and copyrighted
photos can't be bundled. To train your own model:

1. Collect photos for each plant (50–300+ images per class is a good
   starting point; more is always better). Good sources: your own
   photos, or a properly licensed/public-domain leaf dataset.
2. Drop them into the matching folder under `dataset/`:
   ```
   dataset/
     Tulsi/        <- put Tulsi leaf photos here
     Neem/         <- put Neem leaf photos here
     ... etc.
   ```
   (Empty folders for all 15 starter plants are already created for you.
   You can also add your own new plant folders — just also add a matching
   entry to `data/plants_data.json` so its details show up in the app.)
3. Run the trainer from the project root:
   ```bash
   python model/train.py
   ```
   This will:
   - Split your images 80/20 into train/validation automatically
   - Apply light data augmentation (rotation, zoom, flips) since
     medicinal-plant datasets are usually small
   - Train a 4-block CNN and save the best-performing weights to
     `model/plant_model.h5`
   - Save `model/class_indices.json` (maps model output -> plant name)
   - Save `model/training_history.json` (used by the **AI Model** tab
     to draw accuracy/loss curves)
4. Restart the app (or just re-open the Recognize tab) — the header
   will now show "✅ Model loaded" and predictions will work.

### Adding a brand-new plant (16th, 17th, ...)

1. Create `dataset/<NewPlantName>/` and add training photos.
2. Add a matching entry to `data/plants_data.json` with the same key
   (`scientific_name`, `family`, `category`, `description`,
   `parts_used`, `traditional_uses`, `precautions`).
3. Re-run `python model/train.py` to retrain including the new class.

---

## 📄 Reports

From the **Recognize** tab, click **Save Report** after a prediction
to write both:
- `reports/<Plant>_<timestamp>.txt` — plain text
- `reports/<Plant>_<timestamp>.html` — styled, includes the uploaded
  image embedded inline, viewable in any browser

Use **History → Export CSV** to get a spreadsheet of every recognition
ever made (great for quick analysis in Excel/Pandas).

---

## 🧩 Tech Stack

- **TensorFlow / Keras** — CNN model definition, training, inference
- **OpenCV** — image reading, color conversion, resizing/normalizing
- **NumPy** — array operations for the model pipeline
- **Pandas** — available for any tabular analysis you want to add on
  top of `history.json` / the exported CSV
- **Pillow** — GUI-friendly thumbnails and HTML report image embedding
- **Tkinter / ttk** — the entire desktop GUI, including a from-scratch
  dependency-free bar/line chart renderer (no matplotlib needed)

Everything is 100% local — no external API calls, no internet
required to run.

---

## 🛣️ Ideas for Further Extension

- Swap the from-scratch CNN for transfer learning (e.g. MobileNetV2)
  for higher accuracy with fewer training images.
- Add a confusion matrix view to the AI Model tab once you have a
  held-out test set (scikit-learn's `confusion_matrix` pairs nicely
  with `train.py`'s validation generator).
- Add a "Plant of the Day" widget on the Library tab.
- Package with PyInstaller for a one-click `.exe` / binary.
