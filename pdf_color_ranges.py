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

    # REAL BUG FOUND (2026-08-07, coach's actual PDF): the header row uses
    # Paragraph objects (which word-wrap inside their column), but every
    # DATA row below used plain strings, which reportlab's Table does NOT
    # wrap — a long string like "Descriptive (no benchmark yet)" or a
    # descriptive_note()/recalibration caveat just overflows straight into
    # the next column instead of wrapping to a second line, producing
    # visibly overlapping text in the rendered PDF (confirmed directly
    # from the coach's screenshot: "Descriptive (no benchmark yet)" and
    # "Flexed-Knee technique..." overlapping character-for-character).
    # bold_body.parent is the plain (non-bold) body style already defined
    # at every call site — reused here instead of adding a new required
    # parameter, so every existing caller keeps working unchanged.
    body_style = bold_body.parent or bold_body

    def cell(text) -> Paragraph:
        return Paragraph(str(text), body_style)

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
        r = mr.RANGES[key]
        if tier == "unknown":
            # FIX (2026-08-06, found on a real clip): "unknown" (a genuine
            # tracking failure this delivery) used to fall into the same
            # branch as a real validated band, showing e.g. "No Data" next
            # to "Optimal: 160-180deg" for front_knee_bracing — a band
            # that's DEAD for classification now regardless (always-
            # descriptive for pace — see metric_ranges._ALWAYS_DESCRIPTIVE_
            # METRICS). A missing value must never be shown next to ANY
            # band text, real or not.
            rows.append([cell(r.label), cell("No Data"), cell(zone_label), cell("—")])
        elif tier == "descriptive":
            # FIX (2026-08-06): was a generic "N/A for this bowling style"
            # for every descriptive-tier metric — confusing for
            # front_knee_bracing/hip_shoulder_separation, which are
            # descriptive for PACE too (real audit found no universal
            # band exists for either, for any bowler_type — see
            # metric_ranges._ALWAYS_DESCRIPTIVE_METRICS), so "this bowling
            # style" read oddly when the style literally was pace.
            # descriptive_note() gives the real, metric-specific, sourced
            # text instead (e.g. the actual Extended/Flexed-Knee
            # classification for front_knee_bracing).
            rows.append([cell(r.label), cell(_format_value(key, val)), cell(zone_label),
                         cell(mr.descriptive_note(key, val, bowler_type))])
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
            # See orchestrator.calculate_release_height_ratio_safe's
            # br_tracking_confidence docstring (2026-08-07) — release_height
            # and head_stability use different field names for the same
            # concern (release_frame_tracking_uncertain / release_window_
            # tracking_uncertain) since one covers a single frame, the
            # other a multi-frame window.
            metric_flags = metrics.get(key, {})
            if metric_flags.get("release_frame_tracking_uncertain") or metric_flags.get("release_window_tracking_uncertain"):
                optimal_text += " (tracking near release flagged unstable — verify before trusting)"
            rows.append([cell(r.label), cell(_format_value(key, val)), cell(zone_label), cell(optimal_text)])
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
