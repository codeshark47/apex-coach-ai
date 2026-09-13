"""
tests/test_coaching_agent_timeout.py

Regression test for a real gap found in a robustness audit (2026-09-13):
neither Gemini call in coaching_agent.py set any request timeout. A
slow/hanging connection had no bound and would hang the "Generate AI
Coaching Report" action indefinitely instead of failing predictably into
the existing outer try/except.
"""

import os
from unittest.mock import MagicMock, patch

import coaching_agent as ca


def _minimal_bowling_payload():
    return {
        "time_indices": {"back_foot_contact_frame": 10, "front_foot_contact_frame": 20, "ball_release_frame": 30},
        "video_metadata": {"fps": 30, "source_file": "clip.mp4", "total_frames": 60},
        "biomechanical_metrics": {
            "front_knee_bracing": {"degrees": 170.0, "tier": "Extended-Knee Technique"},
            "trunk_lean": {"degrees": 20.0, "tier": "Effective Forward Drive"},
            "hip_shoulder_separation": {"degrees": 30.0, "tier": "Moderate Separation"},
            "release_height": {"ratio": 1.2, "classification": "Standard Mid-Arm Release"},
            "head_stability": {"value": 0.05, "classification": "Elite Fixed Gaze Focus"},
            "rear_knee_angle": {"degrees": 160.0, "tier": "Flexed Rear Knee"},
            "rear_hip_flexion": {"degrees": 15.0, "tier": "Within Typical Range"},
        },
    }


def test_bowling_gemini_call_sets_a_bounded_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    mock_response = MagicMock()
    mock_response.text = "SECTION 1 narrative\n---\nDRILL: x"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        ca.generate_biomechanical_coaching_report(_minimal_bowling_payload())

    _, kwargs = mock_client.models.generate_content.call_args
    config = kwargs["config"]
    assert config.http_options is not None
    assert config.http_options.timeout is not None
    assert config.http_options.timeout > 0
