"""
theme.py
Light and dark palettes + a helper to apply them to ttk widgets.
Kept as plain dictionaries so any widget can just do THEME["bg"] etc.
"""
from tkinter import ttk

LIGHT = {
    "bg": "#f4f8f4",
    "panel": "#ffffff",
    "text": "#1f2d1f",
    "muted": "#5a6b5a",
    "accent": "#2e7d32",
    "accent_dark": "#1b5e20",
    "border": "#dbe6db",
    "warn_bg": "#fff8e1",
    "warn_border": "#f9a825",
    "tree_bg": "#ffffff",
    "tree_alt": "#f0f6f0",
}

DARK = {
    "bg": "#141a14",
    "panel": "#1e261e",
    "text": "#e8f0e8",
    "muted": "#9db09d",
    "accent": "#66bb6a",
    "accent_dark": "#43a047",
    "border": "#2c372c",
    "warn_bg": "#332b12",
    "warn_border": "#f9a825",
    "tree_bg": "#1e261e",
    "tree_alt": "#242e24",
}


def apply_theme(root, style: ttk.Style, palette: dict):
    root.configure(bg=palette["bg"])

    style.theme_use("clam")

    style.configure("TFrame", background=palette["bg"])
    style.configure("Panel.TFrame", background=palette["panel"])

    style.configure("TLabel", background=palette["bg"], foreground=palette["text"])
    style.configure("Panel.TLabel", background=palette["panel"], foreground=palette["text"])
    style.configure("Muted.TLabel", background=palette["panel"], foreground=palette["muted"])
    style.configure("Title.TLabel", background=palette["bg"], foreground=palette["accent_dark"],
                     font=("Segoe UI", 18, "bold"))
    style.configure("Heading.TLabel", background=palette["panel"], foreground=palette["accent_dark"],
                     font=("Segoe UI", 13, "bold"))
    style.configure("Confidence.TLabel", background=palette["panel"], foreground=palette["accent"],
                     font=("Segoe UI", 22, "bold"))

    style.configure("TButton", background=palette["accent"], foreground="#ffffff",
                     font=("Segoe UI", 10, "bold"), padding=8, borderwidth=0)
    style.map("TButton", background=[("active", palette["accent_dark"])])

    style.configure("Secondary.TButton", background=palette["border"], foreground=palette["text"],
                     font=("Segoe UI", 10), padding=8, borderwidth=0)
    style.map("Secondary.TButton", background=[("active", palette["muted"])])

    style.configure("TNotebook", background=palette["bg"], borderwidth=0)
    style.configure("TNotebook.Tab", background=palette["panel"], foreground=palette["text"],
                     padding=(14, 8), font=("Segoe UI", 10, "bold"))
    style.map("TNotebook.Tab",
              background=[("selected", palette["accent"])],
              foreground=[("selected", "#ffffff")])

    style.configure("Treeview", background=palette["tree_bg"], fieldbackground=palette["tree_bg"],
                     foreground=palette["text"], rowheight=26, borderwidth=0)
    style.configure("Treeview.Heading", background=palette["accent"], foreground="#ffffff",
                     font=("Segoe UI", 10, "bold"))
    style.map("Treeview", background=[("selected", palette["accent"])],
              foreground=[("selected", "#ffffff")])

    style.configure("TEntry", fieldbackground=palette["panel"], foreground=palette["text"],
                     bordercolor=palette["border"])
    style.configure("TCombobox", fieldbackground=palette["panel"], foreground=palette["text"])
