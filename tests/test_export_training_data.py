"""
tests/test_export_training_data.py

flatten_session() is the part of tools/export_training_data.py that turns
a raw Supabase `sessions` row into the flat training-data schema — pure
dict-shuffling, no network needed. These tests pin down exactly how the
"was this auto-detected value corrected by the coach" signal is derived,
since that's the actual point of the export: an accidental sign flip or
key-name mismatch here would silently corrupt training labels rather than
raising an error anyone would notice.
"""

from tools.export_training_data import flatten_session


def _row(**metrics_overrides):
    metrics = {}
    metrics.update(metrics_overrides)
    return {
        "id": "session-1",
        "athlete_id": "athlete-1",
        "session_date": "2026-07-01T00:00:00+00:00",
        "camera_mode": "single",
        "fps": 240,
        "metrics": metrics,
    }


class TestFlattenSession:
    def test_corrected_br_frame_is_flagged(self):
        row = _row(_release_frame_confirmed={
            "auto_detected": 62, "auto_confidence": 0.8, "coach_confirmed": 72,
        })
        flat = flatten_session(row)
        assert flat["br_auto_detected"] == 62
        assert flat["br_coach_confirmed"] == 72
        assert flat["br_corrected"] is True

    def test_accepted_br_frame_is_not_flagged_as_corrected(self):
        """Coach clicked confirm without moving anything — auto and
        confirmed are identical, this must NOT count as a correction."""
        row = _row(_release_frame_confirmed={
            "auto_detected": 11, "auto_confidence": 0.9, "coach_confirmed": 11,
        })
        flat = flatten_session(row)
        assert flat["br_corrected"] is False

    def test_unconfirmed_event_reports_none_not_false(self):
        """No confirmation step ran for this session (e.g. dual-camera
        mode, or an older session) — must be None (unlabeled), never
        False, which would wrongly claim 'confirmed correct.'"""
        row = _row()
        flat = flatten_session(row)
        assert flat["br_auto_detected"] is None
        assert flat["br_corrected"] is None
        assert flat["ffc_corrected"] is None
        assert flat["bfc_corrected"] is None

    def test_ffc_and_bfc_pairs_have_no_confidence_field(self):
        row = _row(
            _ffc_frame_confirmed={"auto_detected": 30, "coach_confirmed": 33},
            _bfc_frame_confirmed={"auto_detected": 10, "coach_confirmed": 10},
        )
        flat = flatten_session(row)
        assert flat["ffc_corrected"] is True
        assert flat["bfc_corrected"] is False
        assert "ffc_auto_confidence" not in flat

    def test_wrist_point_correction_flag_and_coordinates(self):
        row = _row(_wrist_point_corrected=[512, 340])
        flat = flatten_session(row)
        assert flat["wrist_point_corrected"] is True
        assert flat["wrist_x"] == 512
        assert flat["wrist_y"] == 340

    def test_wrist_point_not_corrected_reports_false_not_none(self):
        """Auto-tracked wrist position was accepted as-is — a real,
        meaningful label (0% correction), not a missing value."""
        row = _row(_wrist_point_corrected=None)
        flat = flatten_session(row)
        assert flat["wrist_point_corrected"] is False
        assert flat["wrist_x"] is None
        assert flat["wrist_y"] is None

    def test_camera_angle_confirmed_passthrough(self):
        row = _row(_camera_angle_confirmed="front-on")
        flat = flatten_session(row)
        assert flat["camera_angle_confirmed"] == "front-on"

    def test_session_identity_fields_passthrough(self):
        flat = flatten_session(_row())
        assert flat["session_id"] == "session-1"
        assert flat["athlete_id"] == "athlete-1"
        assert flat["camera_mode"] == "single"
        assert flat["fps"] == 240
