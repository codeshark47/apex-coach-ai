"""
tests/test_main_smoothing.py

Regression test for a real bug found during a broader audit:
main.py's landmark-smoothing step (a rolling mean applied after gap-fill
interpolation) used min_periods=1, which silently resurrected frames an
earlier interpolate(..., limit_area="inside") step had correctly left as
genuine NaN — a real tracking gap longer than the fill limit, deliberately
left unpatched. See smooth_without_resurrecting_gaps's docstring for the
full real-footage detail.
"""

import numpy as np
import pandas as pd

import main


class TestSmoothWithoutResurrectingGaps:
    def test_genuine_long_gap_stays_nan_after_smoothing(self):
        """THE exact regression: a long real gap must not get partially
        filled in by the rolling mean just because a real value sits a
        couple of frames away on one side."""
        vals = list(range(13)) + [np.nan] * 21 + list(range(34, 40))
        df = pd.DataFrame({"x": vals})

        gap_fill_limit = 3
        interpolated = df.interpolate(
            method="linear", limit=gap_fill_limit, limit_direction="both", limit_area="inside"
        )
        result = main.smooth_without_resurrecting_gaps(interpolated)

        # Frames deep inside the gap (past the fill limit reaching in
        # from EITHER edge — 13,14,15 fillable from the left, 31,32,33
        # fillable from the right) must still be NaN — not fabricated
        # from 1-2 neighbors by the rolling mean alone.
        assert result["x"].iloc[16:31].isna().all()

    def test_real_data_still_gets_smoothed(self):
        """The whole point of this function is to still smooth genuine,
        present data — only the gap-resurrection behavior is the bug."""
        df = pd.DataFrame({"x": [10.0, 11.0, 12.0, 20.0, 12.0, 11.0, 10.0]})
        result = main.smooth_without_resurrecting_gaps(df)
        # A single spike surrounded by real data should be pulled toward
        # its neighbors by the rolling mean, not left untouched.
        assert result["x"].iloc[3] < 20.0
        assert not result["x"].isna().any()

    def test_short_gap_within_fill_limit_is_smoothed_normally(self):
        """A short gap that interpolate() already bridged (within its
        limit) is real, usable data by the time smoothing runs — must be
        smoothed like any other data, not treated as missing."""
        vals = [10.0, 11.0, 12.0, np.nan, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        df = pd.DataFrame({"x": vals})
        interpolated = df.interpolate(
            method="linear", limit=3, limit_direction="both", limit_area="inside"
        )
        result = main.smooth_without_resurrecting_gaps(interpolated)
        assert not result["x"].isna().any()
