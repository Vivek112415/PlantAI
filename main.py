"""
main.py
Entry point for the PlantAI desktop application.

Run:
    python main.py
"""
import tkinter as tk
from gui.app import PlantAIApp


def main():
    root = tk.Tk()
    app = PlantAIApp(root)  # noqa: F841 - kept alive by Tk mainloop
    root.mainloop()


if __name__ == "__main__":
    main()
