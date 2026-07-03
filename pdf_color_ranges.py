"""
pdf_color_ranges.py

Builds the CBC-style color-coded reference range table for the PDF report.
Pulls all boundaries from metric_ranges.py (single source of truth) — does
not redefine or duplicate any range values.
"""

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Table, TableStyle, Paragraph

import metric_ranges as mr


def _metric_value_map(metrics: dict) -> dict:
    """Maps metric_ranges keys to the numeric value the app already computed."""
    return {
        "front_knee_bracing": metrics.get("front_knee_bracing", {}).get("degrees"),
        "hip_shoulder_separation": metrics.get("hip_shoulder_separation", {}).get("degrees"),
        "trunk_lean": metrics.get("trunk_lean", {}).get("degrees"),
        "release_height": metrics.get("release_height", {}).get("ratio"),
        "head_stability": (
            metrics.get("head_stability", {}).get("value")
            or metrics.get("head_stability", {}).get("deviation_index")
        ),
    }


def _format_value(metric_key: str, value) -> str:
    if value is None:
        return "No Data"
    r = mr.RANGES[metric_key]
    if r.unit == "°":
        return f"{round(float(value), 1)}°"
    if r.unit == "%":
        return f"{round(float(value) * 100, 1)}%"
    return str(round(float(value), 4))


def build_color_coded_range_table(metrics: dict, bold_body: ParagraphStyle) -> Table:
    values = _metric_value_map(metrics)

    header = [
        Paragraph("<b>Metric</b>", bold_body),
        Paragraph("<b>Value</b>", bold_body),
        Paragraph("<b>Zone</b>", bold_body),
        Paragraph("<b>Optimal Range</b>", bold_body),
    ]

    rows = [header]
    row_tiers = ["header"]  # tracks which color to paint each row

    for key in mr.all_metric_keys():
        r = mr.RANGES[key]
        val = values.get(key)
        tier = mr.classify(key, val)
        zone_label = {"green": "Optimal", "amber": "Acceptable",
                      "red": "Critical", "unknown": "No Data"}[tier]
        rows.append([r.label, _format_value(key, val), zone_label, r.display_optimal])
        row_tiers.append(tier)

    table = Table(rows, colWidths=[150, 90, 100, 110])

    style_commands = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ]

    for i, tier in enumerate(row_tiers):
        if tier == "header":
            continue
        fill = colors.HexColor(mr.TIER_COLORS_PDF[tier])
        style_commands.append(('BACKGROUND', (0, i), (-1, i), fill))

    table.setStyle(TableStyle(style_commands))
    return table
