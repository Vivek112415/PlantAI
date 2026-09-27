"""
charts.py
Very small, dependency-free chart drawing helpers using plain Tkinter
Canvas. This keeps the project on exactly the requested stack
(no matplotlib) while still giving the Statistics / AI Performance
tabs a visual bar/line chart.
"""


def draw_bar_chart(canvas, data, accent="#2e7d32", muted="#5a6b5a", text="#1f2d1f"):
    """
    data: list of (label, value) tuples.
    Draws simple vertical bars scaled to the canvas size.
    """
    canvas.delete("all")
    canvas.update_idletasks()
    width = max(canvas.winfo_width(), 300)
    height = max(canvas.winfo_height(), 200)

    if not data:
        canvas.create_text(width / 2, height / 2, text="No data yet",
                            fill=muted, font=("Segoe UI", 11))
        return

    padding_left = 40
    padding_bottom = 50
    padding_top = 20
    chart_w = width - padding_left - 20
    chart_h = height - padding_bottom - padding_top

    max_val = max(v for _, v in data) or 1
    bar_count = len(data)
    bar_gap = 18
    bar_w = max((chart_w - bar_gap * (bar_count + 1)) / bar_count, 10)

    # Y axis line
    canvas.create_line(padding_left, padding_top, padding_left, padding_top + chart_h,
                        fill=muted)
    canvas.create_line(padding_left, padding_top + chart_h, width - 10, padding_top + chart_h,
                        fill=muted)

    x = padding_left + bar_gap
    for label, value in data:
        bar_h = (value / max_val) * chart_h
        y0 = padding_top + chart_h - bar_h
        y1 = padding_top + chart_h
        canvas.create_rectangle(x, y0, x + bar_w, y1, fill=accent, outline="")
        canvas.create_text(x + bar_w / 2, y0 - 10, text=str(round(value, 1)),
                            fill=text, font=("Segoe UI", 9, "bold"))
        canvas.create_text(x + bar_w / 2, padding_top + chart_h + 16,
                            text=label[:10], fill=text, font=("Segoe UI", 8), angle=0)
        x += bar_w + bar_gap


def draw_line_chart(canvas, series_dict, muted="#5a6b5a", text="#1f2d1f",
                     colors=("#2e7d32", "#e53935")):
    """
    series_dict: {"accuracy": [...], "val_accuracy": [...]} style dict of
    equal-length lists (e.g. training_history.json). Draws each as a line.
    """
    canvas.delete("all")
    canvas.update_idletasks()
    width = max(canvas.winfo_width(), 300)
    height = max(canvas.winfo_height(), 200)

    if not series_dict:
        canvas.create_text(width / 2, height / 2, text="No training history yet",
                            fill=muted, font=("Segoe UI", 11))
        return

    padding = 40
    chart_w = width - padding * 2
    chart_h = height - padding * 2

    all_vals = [v for series in series_dict.values() for v in series]
    if not all_vals:
        canvas.create_text(width / 2, height / 2, text="No data", fill=muted)
        return
    max_val = max(all_vals) or 1
    min_val = min(min(all_vals), 0)
    n_points = max(len(v) for v in series_dict.values())

    canvas.create_line(padding, padding, padding, padding + chart_h, fill=muted)
    canvas.create_line(padding, padding + chart_h, padding + chart_w, padding + chart_h, fill=muted)

    legend_y = 10
    for idx, (name, series) in enumerate(series_dict.items()):
        color = colors[idx % len(colors)]
        canvas.create_text(padding + idx * 140, legend_y, text=name, fill=color,
                            font=("Segoe UI", 9, "bold"), anchor="w")
        if len(series) < 2:
            continue
        points = []
        for i, v in enumerate(series):
            x = padding + (i / max(n_points - 1, 1)) * chart_w
            y = padding + chart_h - ((v - min_val) / (max_val - min_val + 1e-9)) * chart_h
            points.extend([x, y])
        canvas.create_line(*points, fill=color, width=2, smooth=True)
