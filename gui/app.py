"""
app.py
The main Tkinter application window and all its tabs:

  1. Recognize   - upload/preview an image, run prediction, view result, save report
  2. Library     - browse & search all known plants, filter by category, favorite them
  3. History     - past recognitions, export to CSV, reopen a saved report
  4. Favorites   - quick view of starred plants
  5. Statistics  - counts, most-recognized plants, category breakdown (bar chart)
  6. AI Model    - training accuracy/loss curves + basic model info

This module intentionally keeps all the "wiring" (button -> action) in
one place, while heavy lifting (prediction, DB, reports, image prep)
lives in utils/ and model/.
"""
import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from config import APP_TITLE, APP_MIN_SIZE, REPORTS_DIR, ASSETS_DIR
from utils.database import PlantDatabase
from utils.image_utils import make_preview_thumbnail
from utils.report_utils import save_report
from model.predict import PlantPredictor
from gui.theme import LIGHT, DARK, apply_theme
from gui.charts import draw_bar_chart, draw_line_chart


class PlantAIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(f"{APP_MIN_SIZE[0]}x{APP_MIN_SIZE[1]}")
        self.root.minsize(*APP_MIN_SIZE)

        self.db = PlantDatabase()
        self.predictor = PlantPredictor()

        self.dark_mode = False
        self.palette = LIGHT
        self.style = ttk.Style(self.root)

        self.current_image_path = None
        self.current_result = None  # last prediction dict
        self.preview_photo = None   # keep a reference so Tk doesn't GC it

        self._build_layout()
        apply_theme(self.root, self.style, self.palette)
        self._refresh_library()
        self._refresh_history()
        self._refresh_favorites()
        self._refresh_stats()
        self._refresh_model_tab()

    # ------------------------------------------------------------------
    # Layout scaffolding
    # ------------------------------------------------------------------
    def _build_layout(self):
        top_bar = ttk.Frame(self.root)
        top_bar.pack(fill="x", padx=16, pady=(14, 6))

        ttk.Label(top_bar, text="\U0001F33F PlantAI", style="Title.TLabel").pack(side="left")
        self.model_status_lbl = ttk.Label(
            top_bar,
            text=self._model_status_text(),
        )
        self.model_status_lbl.pack(side="left", padx=16)

        self.theme_btn = ttk.Button(top_bar, text="\U0001F319 Dark Mode",
                                     style="Secondary.TButton", command=self._toggle_theme)
        self.theme_btn.pack(side="right")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.tab_recognize = ttk.Frame(self.notebook)
        self.tab_library = ttk.Frame(self.notebook)
        self.tab_history = ttk.Frame(self.notebook)
        self.tab_favorites = ttk.Frame(self.notebook)
        self.tab_stats = ttk.Frame(self.notebook)
        self.tab_model = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_recognize, text="  \U0001F4F7 Recognize  ")
        self.notebook.add(self.tab_library, text="  \U0001F4DA Plant Library  ")
        self.notebook.add(self.tab_history, text="  \U0001F4DC History  ")
        self.notebook.add(self.tab_favorites, text="  \u2B50 Favorites  ")
        self.notebook.add(self.tab_stats, text="  \U0001F4C8 Statistics  ")
        self.notebook.add(self.tab_model, text="  \U0001F916 AI Model  ")

        self._build_recognize_tab()
        self._build_library_tab()
        self._build_history_tab()
        self._build_favorites_tab()
        self._build_stats_tab()
        self._build_model_tab()

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, _event):
        tab_text = self.notebook.tab(self.notebook.select(), "text")
        if "History" in tab_text:
            self._refresh_history()
        elif "Favorites" in tab_text:
            self._refresh_favorites()
        elif "Statistics" in tab_text:
            self._refresh_stats()
        elif "AI Model" in tab_text:
            self._refresh_model_tab()

    def _model_status_text(self):
        return ("\u2705 Model loaded" if self.predictor.available
                else "\u26A0 No trained model - run model/train.py")

    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.palette = DARK if self.dark_mode else LIGHT
        self.theme_btn.configure(text="\u2600 Light Mode" if self.dark_mode else "\U0001F319 Dark Mode")
        apply_theme(self.root, self.style, self.palette)
        # re-draw canvases so their manual colors match the new theme
        self._refresh_stats()
        self._refresh_model_tab()

    # ------------------------------------------------------------------
    # TAB 1: Recognize
    # ------------------------------------------------------------------
    def _build_recognize_tab(self):
        container = ttk.Frame(self.tab_recognize)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(container, style="Panel.TFrame")
        left.pack(side="left", fill="both", expand=False, padx=(0, 10), ipadx=10, ipady=10)

        self.image_canvas = tk.Label(left, text="No image selected\n\U0001F5BC",
                                      width=44, height=18, bg="#dbe6db", relief="flat")
        self.image_canvas.pack(padx=16, pady=16)

        btn_row = ttk.Frame(left, style="Panel.TFrame")
        btn_row.pack(pady=(0, 16))
        ttk.Button(btn_row, text="\U0001F4C1 Upload Image", command=self._on_upload).grid(row=0, column=0, padx=4)
        ttk.Button(btn_row, text="\U0001F50D Predict", command=self._on_predict).grid(row=0, column=1, padx=4)
        ttk.Button(btn_row, text="\u2716 Clear", style="Secondary.TButton",
                   command=self._on_clear).grid(row=0, column=2, padx=4)

        right = ttk.Frame(container, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True, ipadx=10, ipady=10)

        self.result_title = ttk.Label(right, text="Upload a leaf/plant photo to begin",
                                       style="Heading.TLabel")
        self.result_title.pack(anchor="w", padx=16, pady=(16, 4))

        self.confidence_lbl = ttk.Label(right, text="", style="Confidence.TLabel")
        self.confidence_lbl.pack(anchor="w", padx=16)

        self.details_text = tk.Text(right, height=16, wrap="word", relief="flat",
                                     font=("Segoe UI", 10), borderwidth=0)
        self.details_text.pack(fill="both", expand=True, padx=16, pady=10)
        self.details_text.configure(state="disabled")

        action_row = ttk.Frame(right, style="Panel.TFrame")
        action_row.pack(anchor="w", padx=16, pady=(0, 16))
        ttk.Button(action_row, text="\U0001F4BE Save Report", command=self._on_save_report).grid(row=0, column=0, padx=4)
        self.fav_btn = ttk.Button(action_row, text="\u2B50 Add to Favorites",
                                   style="Secondary.TButton", command=self._on_toggle_favorite)
        self.fav_btn.grid(row=0, column=1, padx=4)

    def _on_upload(self):
        path = filedialog.askopenfilename(
            title="Select a plant/leaf image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not path:
            return
        try:
            self.preview_photo = make_preview_thumbnail(path)
            self.image_canvas.configure(
                image=self.preview_photo, text="",
                width=self.preview_photo.width(),
                height=self.preview_photo.height(),
            )
            self.current_image_path = path
            self.current_result = None
            self._set_result_placeholder("Image loaded. Click 'Predict' to identify it.")
        except Exception as exc:
            messagebox.showerror("Image Error", f"Could not load image:\n{exc}")

    def _on_clear(self):
        self.current_image_path = None
        self.current_result = None
        self.preview_photo = None
        self.image_canvas.configure(
            image="", text="No image selected\n\U0001F5BC",
            width=44, height=18,
        )
        self._set_result_placeholder("Upload a leaf/plant photo to begin")
        self.confidence_lbl.configure(text="")

    def _set_result_placeholder(self, text):
        self.result_title.configure(text=text)
        self._set_details_text("")

    def _set_details_text(self, text):
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", text)
        self.details_text.configure(state="disabled")

    def _on_predict(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        try:
            result = self.predictor.predict(self.current_image_path)
        except RuntimeError as exc:
            messagebox.showwarning("Model Not Ready", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Prediction Error", f"Something went wrong:\n{exc}")
            return

        self.current_result = result
        plant_name = result["plant"]
        info = self.db.get_plant(plant_name)

        self.result_title.configure(text=f"Detected: {plant_name}"
                                     + (f"  ({info['scientific_name']})" if info else ""))
        self.confidence_lbl.configure(text=f"{result['confidence']:.1f}% confidence")
        self._set_details_text(self._format_plant_details(info, result))
        self._update_fav_button(plant_name)

        record = self.db.add_history(
            image_name=os.path.basename(self.current_image_path),
            image_path=self.current_image_path,
            predicted_plant=plant_name,
            confidence=result["confidence"],
        )
        self._last_record = record

    def _format_plant_details(self, info, result):
        lines = []
        if result.get("top3") and len(result["top3"]) > 1:
            alt = ", ".join(f"{n} ({c:.1f}%)" for n, c in result["top3"][1:])
            lines.append(f"Other possibilities: {alt}\n")

        if not info:
            lines.append("No reference details found in the local plant database for this class.")
            return "\n".join(lines)

        lines.append(f"Family: {info.get('family', '-')}    Category: {info.get('category', '-')}")
        lines.append(f"Parts commonly used: {', '.join(info.get('parts_used', []))}")
        lines.append("")
        lines.append("Description:")
        lines.append(info.get("description", "-"))
        lines.append("")
        lines.append("Traditional / commonly discussed uses:")
        for u in info.get("traditional_uses", []):
            lines.append(f"  \u2022 {u}")
        lines.append("")
        lines.append("\u26A0 Safety & Precautions:")
        for p in info.get("precautions", []):
            lines.append(f"  \u2022 {p}")
        lines.append("")
        lines.append("Note: AI identification does not confirm a plant is safe to "
                      "consume or suitable for treating any condition. Consult a "
                      "qualified professional before medicinal use.")
        return "\n".join(lines)

    def _update_fav_button(self, plant_name):
        is_fav = self.db.is_favorite(plant_name)
        self.fav_btn.configure(text="\u2B50 Remove Favorite" if is_fav else "\u2B50 Add to Favorites")

    def _on_toggle_favorite(self):
        if not self.current_result:
            messagebox.showinfo("No Result", "Predict a plant first.")
            return
        plant_name = self.current_result["plant"]
        added = self.db.toggle_favorite(plant_name)
        self._update_fav_button(plant_name)
        self._refresh_favorites()
        messagebox.showinfo("Favorites", f"{plant_name} {'added to' if added else 'removed from'} favorites.")

    def _on_save_report(self):
        if not self.current_result:
            messagebox.showinfo("No Result", "Predict a plant first.")
            return
        info = self.db.get_plant(self.current_result["plant"])
        record = getattr(self, "_last_record", None) or self.db.add_history(
            image_name=os.path.basename(self.current_image_path),
            image_path=self.current_image_path,
            predicted_plant=self.current_result["plant"],
            confidence=self.current_result["confidence"],
        )
        txt_path, html_path = save_report(record, info or {}, self.current_image_path)
        messagebox.showinfo("Report Saved",
                             f"Report saved to:\n{txt_path}\n{html_path}")

    # ------------------------------------------------------------------
    # TAB 2: Plant Library
    # ------------------------------------------------------------------
    def _build_library_tab(self):
        container = ttk.Frame(self.tab_library)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(container)
        left.pack(side="left", fill="y", padx=(0, 10))

        search_row = ttk.Frame(left)
        search_row.pack(fill="x", pady=(0, 8))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._refresh_library())
        ttk.Entry(search_row, textvariable=self.search_var, width=26).pack(side="left", fill="x", expand=True)

        cat_row = ttk.Frame(left)
        cat_row.pack(fill="x", pady=(0, 8))
        ttk.Label(cat_row, text="Category:").pack(side="left")
        self.category_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(cat_row, textvariable=self.category_var,
                                            values=self.db.all_categories(), state="readonly", width=16)
        self.category_combo.pack(side="left", padx=6)
        self.category_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_library())

        self.library_list = tk.Listbox(left, width=28, height=24, activestyle="none",
                                        highlightthickness=0, relief="flat")
        self.library_list.pack(fill="y", expand=True)
        self.library_list.bind("<<ListboxSelect>>", self._on_library_select)

        right = ttk.Frame(container, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True, ipadx=10, ipady=10)

        self.lib_title = ttk.Label(right, text="Select a plant from the list",
                                    style="Heading.TLabel")
        self.lib_title.pack(anchor="w", padx=16, pady=(16, 4))

        self.lib_fav_btn = ttk.Button(right, text="\u2B50 Add to Favorites",
                                       style="Secondary.TButton",
                                       command=self._on_library_toggle_favorite)
        self.lib_fav_btn.pack(anchor="w", padx=16, pady=(0, 8))

        self.lib_text = tk.Text(right, height=20, wrap="word", relief="flat",
                                 font=("Segoe UI", 10), borderwidth=0)
        self.lib_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.lib_text.configure(state="disabled")

        self._selected_library_plant = None

    def _refresh_library(self):
        query = self.search_var.get() if hasattr(self, "search_var") else ""
        category = self.category_var.get() if hasattr(self, "category_var") else "All"

        names = set(self.db.search(query)) & set(self.db.plants_in_category(category))
        names = sorted(names)

        self.library_list.delete(0, "end")
        for name in names:
            star = " \u2B50" if self.db.is_favorite(name) else ""
            self.library_list.insert("end", f"{name}{star}")
        self._current_library_names = names

    def _on_library_select(self, _event):
        selection = self.library_list.curselection()
        if not selection:
            return
        name = self._current_library_names[selection[0]]
        self._selected_library_plant = name
        info = self.db.get_plant(name)
        self.lib_title.configure(text=f"{name}  ({info.get('scientific_name', '-')})")
        self.lib_fav_btn.configure(
            text="\u2B50 Remove Favorite" if self.db.is_favorite(name) else "\u2B50 Add to Favorites")
        self._set_lib_text(self._format_plant_details(info, {"top3": []}))

    def _set_lib_text(self, text):
        self.lib_text.configure(state="normal")
        self.lib_text.delete("1.0", "end")
        self.lib_text.insert("1.0", text)
        self.lib_text.configure(state="disabled")

    def _on_library_toggle_favorite(self):
        if not self._selected_library_plant:
            return
        self.db.toggle_favorite(self._selected_library_plant)
        self._on_library_select(None) if False else None
        # refresh label + list stars
        name = self._selected_library_plant
        self.lib_fav_btn.configure(
            text="\u2B50 Remove Favorite" if self.db.is_favorite(name) else "\u2B50 Add to Favorites")
        self._refresh_library()
        self._refresh_favorites()

    # ------------------------------------------------------------------
    # TAB 3: History
    # ------------------------------------------------------------------
    def _build_history_tab(self):
        container = ttk.Frame(self.tab_history)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        toolbar = ttk.Frame(container)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="\U0001F5D1 Clear History", style="Secondary.TButton",
                   command=self._on_clear_history).pack(side="left", padx=4)
        ttk.Button(toolbar, text="\U0001F4E4 Export CSV", style="Secondary.TButton",
                   command=self._on_export_csv).pack(side="left", padx=4)
        ttk.Button(toolbar, text="\U0001F4C2 Open Reports Folder", style="Secondary.TButton",
                   command=self._on_open_reports).pack(side="left", padx=4)

        columns = ("timestamp", "image", "plant", "confidence")
        self.history_tree = ttk.Treeview(container, columns=columns, show="headings", height=20)
        for col, label, width in [
            ("timestamp", "Date / Time", 160),
            ("image", "Image", 220),
            ("plant", "Predicted Plant", 180),
            ("confidence", "Confidence (%)", 120),
        ]:
            self.history_tree.heading(col, text=label)
            self.history_tree.column(col, width=width, anchor="w")
        self.history_tree.pack(fill="both", expand=True)

    def _refresh_history(self):
        self.history_tree.delete(*self.history_tree.get_children())
        for r in self.db.get_history():
            self.history_tree.insert("", "end", values=(
                r["timestamp"], r["image_name"], r["predicted_plant"], r["confidence"]))

    def _on_clear_history(self):
        if messagebox.askyesno("Confirm", "Clear all recognition history? This cannot be undone."):
            self.db.clear_history()
            self._refresh_history()
            self._refresh_stats()

    def _on_export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")],
                                             initialfile="plantai_history.csv")
        if not path:
            return
        self.db.export_history_csv(path)
        messagebox.showinfo("Exported", f"History exported to:\n{path}")

    def _on_open_reports(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(REPORTS_DIR)  # noqa: S606 (Windows only)
            elif shutil.which("xdg-open"):
                os.system(f'xdg-open "{REPORTS_DIR}"')
            elif shutil.which("open"):
                os.system(f'open "{REPORTS_DIR}"')
            else:
                messagebox.showinfo("Reports Folder", REPORTS_DIR)
        except Exception:
            messagebox.showinfo("Reports Folder", REPORTS_DIR)

    # ------------------------------------------------------------------
    # TAB 4: Favorites
    # ------------------------------------------------------------------
    def _build_favorites_tab(self):
        container = ttk.Frame(self.tab_favorites)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        self.fav_list = tk.Listbox(container, width=30, height=24, activestyle="none",
                                    highlightthickness=0, relief="flat")
        self.fav_list.pack(side="left", fill="y", padx=(0, 10))
        self.fav_list.bind("<<ListboxSelect>>", self._on_fav_select)

        right = ttk.Frame(container, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True, ipadx=10, ipady=10)
        self.fav_title = ttk.Label(right, text="Select a favorite plant",
                                    style="Heading.TLabel")
        self.fav_title.pack(anchor="w", padx=16, pady=16)
        self.fav_text = tk.Text(right, height=20, wrap="word", relief="flat",
                                 font=("Segoe UI", 10), borderwidth=0)
        self.fav_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.fav_text.configure(state="disabled")

    def _refresh_favorites(self):
        self.fav_list.delete(0, "end")
        for name in self.db.get_favorites():
            self.fav_list.insert("end", name)

    def _on_fav_select(self, _event):
        selection = self.fav_list.curselection()
        if not selection:
            return
        name = self.fav_list.get(selection[0])
        info = self.db.get_plant(name)
        self.fav_title.configure(text=f"{name}  ({info.get('scientific_name', '-') if info else '-'})")
        self.fav_text.configure(state="normal")
        self.fav_text.delete("1.0", "end")
        self.fav_text.insert("1.0", self._format_plant_details(info, {"top3": []}))
        self.fav_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # TAB 5: Statistics
    # ------------------------------------------------------------------
    def _build_stats_tab(self):
        container = ttk.Frame(self.tab_stats)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        summary = ttk.Frame(container, style="Panel.TFrame")
        summary.pack(fill="x", pady=(0, 10), ipady=10)
        self.stat_labels = {}
        for key, label in [
            ("total_predictions", "Total Predictions"),
            ("unique_plants_identified", "Unique Plants Identified"),
            ("average_confidence", "Avg. Confidence (%)"),
            ("total_known_plants", "Plants in Library"),
            ("total_favorites", "Favorites"),
        ]:
            box = ttk.Frame(summary, style="Panel.TFrame")
            box.pack(side="left", expand=True, fill="x", padx=10)
            val = ttk.Label(box, text="0", style="Confidence.TLabel")
            val.pack()
            ttk.Label(box, text=label, style="Muted.TLabel").pack()
            self.stat_labels[key] = val

        chart_frame = ttk.Frame(container, style="Panel.TFrame")
        chart_frame.pack(fill="both", expand=True, ipady=10)
        ttk.Label(chart_frame, text="Most Recognized Plants", style="Heading.TLabel").pack(
            anchor="w", padx=16, pady=(10, 0))
        self.stats_canvas = tk.Canvas(chart_frame, bg=self.palette["panel"], highlightthickness=0, height=280)
        self.stats_canvas.pack(fill="both", expand=True, padx=16, pady=16)

    def _refresh_stats(self):
        stats = self.db.get_statistics()
        self.stat_labels["total_predictions"].configure(text=str(stats["total_predictions"]))
        self.stat_labels["unique_plants_identified"].configure(text=str(stats["unique_plants_identified"]))
        self.stat_labels["average_confidence"].configure(text=f"{stats['average_confidence']}")
        self.stat_labels["total_known_plants"].configure(text=str(stats["total_known_plants"]))
        self.stat_labels["total_favorites"].configure(text=str(stats["total_favorites"]))

        self.stats_canvas.configure(bg=self.palette["panel"])
        data = [(name, count) for name, count in stats["most_common"]]
        draw_bar_chart(self.stats_canvas, data, accent=self.palette["accent"],
                        muted=self.palette["muted"], text=self.palette["text"])

    # ------------------------------------------------------------------
    # TAB 6: AI Model info / performance
    # ------------------------------------------------------------------
    def _build_model_tab(self):
        container = ttk.Frame(self.tab_model)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        info_frame = ttk.Frame(container, style="Panel.TFrame")
        info_frame.pack(fill="x", pady=(0, 10), ipady=10)
        self.model_info_lbl = ttk.Label(info_frame, text="", style="Panel.TLabel",
                                         justify="left")
        self.model_info_lbl.pack(anchor="w", padx=16, pady=10)

        chart_frame = ttk.Frame(container, style="Panel.TFrame")
        chart_frame.pack(fill="both", expand=True, ipady=10)
        ttk.Label(chart_frame, text="Training Accuracy / Loss", style="Heading.TLabel").pack(
            anchor="w", padx=16, pady=(10, 0))
        self.model_canvas = tk.Canvas(chart_frame, bg=self.palette["panel"], highlightthickness=0, height=280)
        self.model_canvas.pack(fill="both", expand=True, padx=16, pady=16)

        ttk.Label(container,
                  text="To (re)train the model: place labeled images under dataset/<PlantName>/ "
                       "and run `python model/train.py` from the project root.",
                  style="Muted.TLabel", wraplength=900, justify="left").pack(anchor="w", padx=6, pady=(6, 0))

    def _refresh_model_tab(self):
        import json
        from config import TRAINING_HISTORY_PATH, CLASS_INDEX_PATH

        status = self._model_status_text()
        n_classes = "-"
        if os.path.exists(CLASS_INDEX_PATH):
            try:
                with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
                    n_classes = str(len(json.load(f)))
            except Exception:
                pass

        self.model_info_lbl.configure(
            text=f"Status: {status}\nClasses trained: {n_classes}\n"
                 f"Input size: 128x128 RGB   |   Architecture: 4-block CNN (Keras)"
        )

        self.model_canvas.configure(bg=self.palette["panel"])
        history = {}
        if os.path.exists(TRAINING_HISTORY_PATH):
            try:
                with open(TRAINING_HISTORY_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                history = {k: v for k, v in raw.items() if k in ("accuracy", "val_accuracy")}
            except Exception:
                history = {}
        draw_line_chart(self.model_canvas, history, muted=self.palette["muted"],
                         text=self.palette["text"], colors=(self.palette["accent"], "#e53935"))