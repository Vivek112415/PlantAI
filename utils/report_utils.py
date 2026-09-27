"""
report_utils.py
Builds a human-readable report (.txt and .html) for a single recognition
result, saved locally under reports/. No external services involved.
"""
import os
from datetime import datetime

from config import REPORTS_DIR
from utils.image_utils import image_to_base64

DISCLAIMER = (
    "Identification by AI does not confirm that a plant is safe to consume "
    "or suitable for treating any disease. Traditional/health information "
    "shown here is drawn from a locally stored reference dataset and is for "
    "general educational purposes only - it is not medical advice. Always "
    "consult a qualified professional before medicinal use of any plant."
)


def _safe_filename(name):
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()


def generate_text_report(record, plant_info):
    lines = []
    lines.append("=" * 60)
    lines.append("PlantAI - Recognition Report")
    lines.append("=" * 60)
    lines.append(f"Date/Time      : {record['timestamp']}")
    lines.append(f"Image File     : {record['image_name']}")
    lines.append(f"Predicted Plant: {record['predicted_plant']}")
    lines.append(f"Confidence     : {record['confidence']}%")
    lines.append("-" * 60)

    if plant_info:
        lines.append(f"Scientific Name : {plant_info.get('scientific_name', '-')}")
        lines.append(f"Family          : {plant_info.get('family', '-')}")
        lines.append(f"Category        : {plant_info.get('category', '-')}")
        lines.append(f"Parts Used      : {', '.join(plant_info.get('parts_used', []))}")
        lines.append("")
        lines.append("Description:")
        lines.append(f"  {plant_info.get('description', '-')}")
        lines.append("")
        lines.append("Traditional / Commonly Discussed Uses:")
        for use in plant_info.get("traditional_uses", []):
            lines.append(f"  - {use}")
        lines.append("")
        lines.append("Safety & Precautions:")
        for note in plant_info.get("precautions", []):
            lines.append(f"  - {note}")
    else:
        lines.append("No additional reference information found for this plant.")

    lines.append("-" * 60)
    lines.append("DISCLAIMER:")
    lines.append(DISCLAIMER)
    lines.append("=" * 60)
    return "\n".join(lines)


def generate_html_report(record, plant_info, image_path=None):
    img_tag = ""
    if image_path and os.path.exists(image_path):
        try:
            b64 = image_to_base64(image_path)
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" class="thumb"/>'
        except Exception:
            img_tag = ""

    uses_html = "".join(f"<li>{u}</li>" for u in plant_info.get("traditional_uses", [])) \
        if plant_info else "<li>No data available</li>"
    precautions_html = "".join(f"<li>{p}</li>" for p in plant_info.get("precautions", [])) \
        if plant_info else ""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>PlantAI Report - {record['predicted_plant']}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; background:#f4f8f4; color:#1f2d1f; padding:24px; }}
.card {{ background:#fff; border-radius:12px; padding:24px; max-width:640px; margin:auto;
         box-shadow:0 2px 10px rgba(0,0,0,0.08); }}
h1 {{ color:#2e7d32; margin-top:0; }}
.thumb {{ max-width:100%; border-radius:10px; margin-bottom:16px; }}
.badge {{ display:inline-block; background:#e8f5e9; color:#2e7d32; padding:4px 10px;
          border-radius:20px; font-size:13px; margin-right:6px; }}
.section-title {{ font-weight:600; margin-top:18px; color:#33691e; }}
.warn {{ background:#fff8e1; border-left:4px solid #f9a825; padding:12px 16px;
         margin-top:20px; font-size:13px; border-radius:6px; }}
ul {{ margin:6px 0 0 18px; padding:0; }}
</style></head>
<body>
  <div class="card">
    {img_tag}
    <h1>{record['predicted_plant']}</h1>
    <span class="badge">Confidence: {record['confidence']}%</span>
    <span class="badge">{record['timestamp']}</span>
    <div class="section-title">Scientific Name</div>
    <div>{plant_info.get('scientific_name', '-') if plant_info else '-'}</div>
    <div class="section-title">Family / Category</div>
    <div>{plant_info.get('family', '-') if plant_info else '-'} /
         {plant_info.get('category', '-') if plant_info else '-'}</div>
    <div class="section-title">Description</div>
    <div>{plant_info.get('description', '-') if plant_info else '-'}</div>
    <div class="section-title">Traditional / Commonly Discussed Uses</div>
    <ul>{uses_html}</ul>
    <div class="section-title">Safety &amp; Precautions</div>
    <ul>{precautions_html}</ul>
    <div class="warn"><b>Disclaimer:</b> {DISCLAIMER}</div>
  </div>
</body></html>"""


def save_report(record, plant_info, image_path=None):
    """Writes both .txt and .html reports; returns (txt_path, html_path)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _safe_filename(f"{record['predicted_plant']}_{stamp}")

    txt_path = os.path.join(REPORTS_DIR, base + ".txt")
    html_path = os.path.join(REPORTS_DIR, base + ".html")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(generate_text_report(record, plant_info))

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(generate_html_report(record, plant_info, image_path))

    return txt_path, html_path
