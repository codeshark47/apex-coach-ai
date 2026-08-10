"""
tests/test_pdf_color_ranges.py

Regression test for a real bug found on an actual clip (2026-08-06):
front_knee_bracing came back with no data (a genuine tracking failure this
delivery) but the PDF table still showed the OLD, now-dead pace band
("Optimal: 160-180deg") in the Optimal Range column, right next to
"No Data" in the Value column — a stale, unsourced band appearing to
apply to a metric that has neither a real measurement NOR (per
metric_ranges._ALWAYS_DESCRIPTIVE_METRICS) a validated band at all.
tier == "unknown" must be handled before any band text (real or
descriptive) is ever considered.
"""

from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

import pdf_color_ranges as pcr
import metric_ranges as mr


def _plain(cell) -> str:
    """Unwrap a table cell back to plain text. FIX (2026-08-07): cells
    are now Paragraph objects (see TestCellsWrapInsteadOfOverlapping) so
    they word-wrap inside their column instead of overflowing into the
    next one — tests need the underlying text, not the Paragraph object
    itself, which doesn't compare equal to a plain string."""
    return cell.text if isinstance(cell, Paragraph) else cell


def _table_rows(metrics: dict, bowler_type=None):
    style = ParagraphStyle("bold_body", parent=ParagraphStyle("body"))
    table = pcr.build_color_coded_range_table(metrics, style, bowler_type=bowler_type)
    return [[_plain(c) for c in row] for row in table._cellvalues[1:]]  # skip header row


class TestMissingValueNeverShowsABand:
    def test_front_knee_bracing_missing_shows_no_data_not_the_dead_pace_band(self):
        rows = _table_rows({})
        knee_row = next(r for r in rows if r[0] == "Lead Knee Bracing")
        assert knee_row[1] == "No Data"
        assert "160" not in knee_row[3]
        assert "180" not in knee_row[3]

    def test_missing_value_row_has_no_data_in_every_column_that_could_carry_a_band(self):
        rows = _table_rows({})
        for row in rows:
            assert row[1] == "No Data"
            # The Optimal Range column must never show a real or dead band
            # string when there's no measurement to judge in the first place.
            assert row[3] == "—"

    def test_descriptive_metric_with_a_real_value_still_shows_the_real_note(self):
        """The fix for the missing-value case must not have broken the
        genuinely-descriptive (real value, no band) case."""
        metrics = {"front_knee_bracing": {"degrees": 175.0}}
        rows = _table_rows(metrics)
        knee_row = next(r for r in rows if r[0] == "Lead Knee Bracing")
        assert knee_row[1] != "No Data"
        assert "Extended-Knee" in knee_row[3]

    def test_validated_metric_with_a_real_value_still_shows_the_real_band(self):
        metrics = {"release_height": {"ratio": 1.20}}
        rows = _table_rows(metrics)
        release_row = next(r for r in rows if r[0] == "Release Height")
        assert release_row[3] == "118%+"


class TestCellsWrapInsteadOfOverlapping:
    """Regression test for a real bug found in the coach's actual PDF
    (2026-08-07): the header row used Paragraph objects (which word-wrap
    inside their column), but every data row used plain strings, which
    reportlab's Table does NOT wrap — a long string like "Descriptive (no
    benchmark yet)" just overflowed straight into the next column instead
    of wrapping to a second line. Confirmed directly from the coach's
    screenshot: "Descriptive (no benchmark yet)" visibly overlapping
    character-for-character with "Flexed-Knee technique..." in the next
    column. Every cell must now be a Paragraph, not a plain string, so
    reportlab actually wraps long content instead of overflowing it."""

    def _raw_table(self, metrics: dict, bowler_type=None):
        style = ParagraphStyle("bold_body", parent=ParagraphStyle("body"))
        return pcr.build_color_coded_range_table(metrics, style, bowler_type=bowler_type)

    def test_every_data_cell_is_a_paragraph_not_a_plain_string(self):
        table = self._raw_table({"front_knee_bracing": {"degrees": 150.0}})
        for row in table._cellvalues[1:]:
            for cell in row:
                assert isinstance(cell, Paragraph), f"cell {cell!r} is not wrap-capable"

    def test_long_descriptive_text_is_preserved_intact_in_its_own_cell(self):
        """The real failure mode: long text must stay contained in its
        own cell's Paragraph (able to wrap onto multiple lines within the
        column), never silently truncated or merged with a neighbor."""
        table = self._raw_table({"front_knee_bracing": {"degrees": 150.0}})
        rows = [[_plain(c) for c in row] for row in table._cellvalues[1:]]
        knee_row = next(r for r in rows if r[0] == "Lead Knee Bracing")
        assert "Flexed-Knee technique" in knee_row[3]
        assert "Portus" in knee_row[3]
