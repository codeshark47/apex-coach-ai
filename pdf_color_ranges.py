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
    return {key: mr.extract_metric_value(metrics, key) for key in mr.all_metric_keys()}


def _format_value(metric_key: str, value) -> str:
    if value is None:
        return "No Data"
    r = mr.RANGES[metric_key]
    if r.unit == "°":
        return f"{round(float(value), 1)}°"
    if r.unit == "%":
        return f"{round(float(value) * 100, 1)}%"
    return str(round(float(value), 4))


def build_color_coded_range_table(metrics: dict, bold_body: ParagraphStyle, bowler_type: str = None) -> Table:
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
        val = values.get(key)
        tier = mr.classify(key, val, bowler_type)
        zone_label = {"green": "Optimal", "amber": "Acceptable", "red": "Critical",
                      "unknown": "No Data", "descriptive": "Descriptive (no benchmark yet)"}[tier]
        if tier == "descriptive":
            r = mr.RANGES[key]
            rows.append([r.label, _format_value(key, val), zone_label, "N/A for this bowling style"])
        else:
            r = mr.SPIN_RANGE_OVERRIDES.get((key, bowler_type), mr.RANGES[key])
            optimal_text = r.display_optimal
            # See orchestrator.calculate_release_height_ratio_safe's and
            # kinematics.calculate_head_stability's "recalibration_pending"
            # comments (2026-08-05): both measurements were just corrected
            # to fix real camera-distance bugs, but their bands were tuned
            # against the OLD, uncorrected measurement and haven't been
            # re-validated against the new one yet — say so in the report
            # itself, not just on screen, so the PDF can't overstate
            # confidence either.
            if metrics.get(key, {}).get("recalibration_pending"):
                optimal_text += " (bands under re-validation — see report)"
            rows.append([r.label, _format_value(key, val), zone_label, optimal_text])
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
