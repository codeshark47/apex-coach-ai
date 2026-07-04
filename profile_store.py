"""
profile_store.py

Athlete profile + session history persistence, backed by Supabase (Postgres).

Design choices, deliberately:
  - No local fallback store. If Supabase isn't configured, functions raise
    a clear RuntimeError rather than silently degrading to fake/in-memory
    "history" that vanishes and misleads the coach.
  - Reads credentials from Streamlit secrets first (st.secrets), then env
    vars, matching how the rest of this app already reads GEMINI_API_KEY.
  - Every function does exactly one real DB operation. No caching layer
    that could serve stale history without the caller knowing.

Setup (one-time):
  1. Create a free Supabase project.
  2. Run supabase_schema.sql in its SQL editor.
  3. Add to .streamlit/secrets.toml (or Streamlit Cloud's secrets panel):
       SUPABASE_URL = "https://xxxx.supabase.co"
       SUPABASE_KEY = "your-anon-or-service-key"
"""

import os
import math
from typing import Optional

_client = None


def _sanitize_for_json(obj):
    """
    Recursively replaces NaN/Infinity with None. These are valid Python floats
    but NOT valid JSON — Supabase's client will reject them outright. This is
    not "faking" the data: None/null honestly represents "no valid value,"
    which is exactly what NaN meant here (a failed/undefined computation).
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _get_credentials():
    url = None
    key = None
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
    except Exception:
        pass
    url = url or os.environ.get("SUPABASE_URL")
    key = key or os.environ.get("SUPABASE_KEY")
    return url, key


def get_client():
    global _client
    if _client is not None:
        return _client

    url, key = _get_credentials()
    if not url or not key:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to .streamlit/secrets.toml (or env vars) before using athlete "
            "history. See supabase_schema.sql for the required tables."
        )

    from supabase import create_client
    _client = create_client(url, key)
    return _client


def get_or_create_athlete(name: str) -> str:
    """Returns the athlete's UUID, creating the row if it doesn't exist."""
    name = name.strip()
    if not name:
        raise ValueError("Athlete name cannot be empty.")

    client = get_client()
    existing = client.table("athletes").select("id").eq("name", name).execute()
    if existing.data:
        return existing.data[0]["id"]

    created = client.table("athletes").insert({"name": name}).execute()
    if not created.data:
        raise RuntimeError(f"Failed to create athlete profile for '{name}'.")
    return created.data[0]["id"]


def list_athletes() -> list:
    """Returns [{"id": ..., "name": ...}, ...] ordered alphabetically."""
    client = get_client()
    result = client.table("athletes").select("id, name").order("name").execute()
    return result.data or []


def save_session(athlete_id: str, video_filename: str, camera_mode: str,
                  fps: float, metrics: dict, phase_durations: Optional[dict],
                  release_arm_speed_kmh: Optional[float], speed_status: str) -> dict:
    """Persists one analysis run against an athlete's history."""
    client = get_client()
    row = {
        "athlete_id": athlete_id,
        "video_filename": video_filename,
        "camera_mode": camera_mode,
        "fps": fps,
        "metrics": _sanitize_for_json(metrics),
        "phase_durations": _sanitize_for_json(phase_durations),
        "release_arm_speed_kmh": release_arm_speed_kmh if release_arm_speed_kmh and math.isfinite(release_arm_speed_kmh) else None,
        "speed_status": speed_status,
    }
    result = client.table("sessions").insert(row).execute()
    if not result.data:
        raise RuntimeError("Failed to save session to athlete history.")
    return result.data[0]


def get_athlete_history(athlete_id: str, limit: int = 20) -> list:
    """Most recent sessions first."""
    client = get_client()
    result = (
        client.table("sessions")
        .select("*")
        .eq("athlete_id", athlete_id)
        .order("session_date", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []
