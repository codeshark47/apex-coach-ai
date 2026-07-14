"""
profile_store.py

Athlete profile + session history persistence, backed by Supabase (Postgres).

SECURITY: every function that touches athlete/session data now requires an
explicit coach_user_id and filters by it in the query itself. This is the
PRIMARY access control — not Row Level Security — because this module
intentionally uses SUPABASE_KEY (the secret/service key) to bypass RLS for
legitimate server-side operations. The service key bypasses RLS entirely,
always, regardless of any policies defined in Postgres. RLS policies added
alongside this are defense-in-depth only, for a future scenario where the
anon key might be used against these tables directly — they are not what
prevents one coach from reading another's data today.

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
  2. Run supabase_schema.sql, then add_coach_scoping.sql in its SQL editor.
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


def _require_coach_user_id(coach_user_id: str):
    if not coach_user_id or not isinstance(coach_user_id, str):
        raise ValueError(
            "coach_user_id is required and must be the signed-in user's ID — "
            "athlete data is scoped per-coach and cannot be accessed without it."
        )


def _assert_owns_athlete(client, athlete_id: str, coach_user_id: str):
    """Raises PermissionError if athlete_id doesn't belong to coach_user_id."""
    result = (
        client.table("athletes")
        .select("id")
        .eq("id", athlete_id)
        .eq("coach_user_id", coach_user_id)
        .execute()
    )
    if not result.data:
        raise PermissionError(
            "This athlete does not exist or does not belong to the signed-in coach."
        )


def get_or_create_athlete(name: str, coach_user_id: str) -> str:
    """
    Returns the athlete's UUID, creating the row if it doesn't exist,
    scoped to this coach. Two different coaches naming an athlete the same
    thing (e.g. two "John Smith"s) now correctly get separate athlete_ids
    instead of silently sharing one record.
    """
    name = name.strip()
    if not name:
        raise ValueError("Athlete name cannot be empty.")
    _require_coach_user_id(coach_user_id)

    client = get_client()
    existing = (
        client.table("athletes")
        .select("id")
        .eq("name", name)
        .eq("coach_user_id", coach_user_id)
        .execute()
    )
    if existing.data:
        return existing.data[0]["id"]

    created = client.table("athletes").insert(
        {"name": name, "coach_user_id": coach_user_id}
    ).execute()
    if not created.data:
        raise RuntimeError(f"Failed to create athlete profile for '{name}'.")
    return created.data[0]["id"]


def list_athletes(coach_user_id: str) -> list:
    """Returns [{"id": ..., "name": ...}, ...] for THIS coach only, alphabetical."""
    _require_coach_user_id(coach_user_id)
    client = get_client()
    result = (
        client.table("athletes")
        .select("id, name")
        .eq("coach_user_id", coach_user_id)
        .order("name")
        .execute()
    )
    return result.data or []


def save_session(athlete_id: str, coach_user_id: str, video_filename: str,
                  camera_mode: str, fps: float, metrics: dict,
                  phase_durations: Optional[dict],
                  release_arm_speed_kmh: Optional[float], speed_status: str) -> dict:
    """Persists one analysis run against an athlete's history, after
    verifying this athlete actually belongs to this coach."""
    _require_coach_user_id(coach_user_id)
    client = get_client()
    _assert_owns_athlete(client, athlete_id, coach_user_id)

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


def get_athlete_history(athlete_id: str, coach_user_id: str, limit: int = 20) -> list:
    """Most recent sessions first, after verifying this athlete belongs to
    this coach."""
    _require_coach_user_id(coach_user_id)
    client = get_client()
    _assert_owns_athlete(client, athlete_id, coach_user_id)

    result = (
        client.table("sessions")
        .select("*")
        .eq("athlete_id", athlete_id)
        .order("session_date", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []
