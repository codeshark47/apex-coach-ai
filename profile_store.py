"""
profile_store.py

Athlete profile + session history persistence, backed by Supabase (Postgres).

SECURITY: every function that touches athlete/session data requires an
explicit coach_user_id and filters by it in the query itself. This is the
PRIMARY access control — not Row Level Security — because this module
intentionally uses SUPABASE_KEY (the secret/service key) to bypass RLS for
legitimate server-side operations. The service key bypasses RLS entirely,
always, regardless of any policies defined in Postgres — RLS is not what
prevents one coach from reading another's data today, this scoping is.

RLS policies (add_rls_policies.sql) are real defense-in-depth on top of
that, for a scenario where the anon key is ever used against these tables
directly (it currently isn't — auth.py only uses it for sign-in/sign-up).
A previous version of this docstring claimed such policies already existed
"alongside this" — verified false during a full app audit (no RLS policy
existed anywhere in this repo's SQL). add_rls_policies.sql actually adds
them now; run it if you haven't.

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
  2. Run supabase_schema.sql, then add_coach_scoping.sql, then
     add_rls_policies.sql in its SQL editor (add_coach_scoping.sql was
     applied directly via Supabase's SQL editor in this project and isn't
     committed here — it adds the athletes.coach_user_id column these
     RLS policies and every query in this module depend on).
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


def get_athlete(athlete_id: str, coach_user_id: str) -> dict:
    """Full athlete row (name, team_id, photo_url, notes, ...), after
    verifying this athlete belongs to this coach."""
    _require_coach_user_id(coach_user_id)
    client = get_client()
    result = (
        client.table("athletes")
        .select("*")
        .eq("id", athlete_id)
        .eq("coach_user_id", coach_user_id)
        .execute()
    )
    if not result.data:
        raise PermissionError(
            "This athlete does not exist or does not belong to the signed-in coach."
        )
    return result.data[0]


def update_athlete_profile(athlete_id: str, coach_user_id: str,
                            photo_url: Optional[str] = None,
                            notes: Optional[str] = None) -> None:
    """Updates the optional profile fields (photo_url, notes) for an athlete.
    Only fields explicitly passed (not None) are changed — passing neither
    is a no-op, it does not clear existing values. Requires photo_url and
    notes columns from add_teams.sql to exist on the athletes table."""
    _require_coach_user_id(coach_user_id)
    client = get_client()
    _assert_owns_athlete(client, athlete_id, coach_user_id)

    updates = {}
    if photo_url is not None:
        updates["photo_url"] = photo_url
    if notes is not None:
        updates["notes"] = notes
    if not updates:
        return
    client.table("athletes").update(updates).eq("id", athlete_id).execute()


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


# ---------------------------------------------------------------
# TEAMS / ACADEMY ROSTER — same coach-scoping security model as athletes
# above: every function requires coach_user_id and filters by it directly,
# since the service key bypasses RLS. Requires add_teams.sql to have been
# run against the live database first (teams table + athletes.team_id
# column) — these functions will raise a clear Postgres error otherwise,
# not silently degrade.
# ---------------------------------------------------------------

def get_or_create_team(name: str, coach_user_id: str) -> str:
    """Returns the team's UUID, creating it if it doesn't exist for this coach."""
    name = name.strip()
    if not name:
        raise ValueError("Team name cannot be empty.")
    _require_coach_user_id(coach_user_id)

    client = get_client()
    existing = (
        client.table("teams")
        .select("id")
        .eq("name", name)
        .eq("coach_user_id", coach_user_id)
        .execute()
    )
    if existing.data:
        return existing.data[0]["id"]

    created = client.table("teams").insert(
        {"name": name, "coach_user_id": coach_user_id}
    ).execute()
    if not created.data:
        raise RuntimeError(f"Failed to create team '{name}'.")
    return created.data[0]["id"]


def list_teams(coach_user_id: str) -> list:
    """Returns [{"id": ..., "name": ...}, ...] for THIS coach only, alphabetical."""
    _require_coach_user_id(coach_user_id)
    client = get_client()
    result = (
        client.table("teams")
        .select("id, name")
        .eq("coach_user_id", coach_user_id)
        .order("name")
        .execute()
    )
    return result.data or []


def assign_athlete_to_team(athlete_id: str, team_id: Optional[str], coach_user_id: str) -> None:
    """
    Sets (or clears, if team_id is None) which team an athlete belongs to.
    Verifies the athlete belongs to this coach first; the team_id itself is
    a foreign key, so an invalid/other-coach's team_id fails at the
    database level rather than silently succeeding.
    """
    _require_coach_user_id(coach_user_id)
    client = get_client()
    _assert_owns_athlete(client, athlete_id, coach_user_id)

    if team_id is not None:
        team_check = (
            client.table("teams")
            .select("id")
            .eq("id", team_id)
            .eq("coach_user_id", coach_user_id)
            .execute()
        )
        if not team_check.data:
            raise PermissionError("This team does not exist or does not belong to the signed-in coach.")

    client.table("athletes").update({"team_id": team_id}).eq("id", athlete_id).execute()


def list_athletes_by_team(team_id: str, coach_user_id: str) -> list:
    """Returns [{"id": ..., "name": ...}, ...] for athletes on this team,
    after verifying the team belongs to this coach."""
    _require_coach_user_id(coach_user_id)
    client = get_client()
    team_check = (
        client.table("teams")
        .select("id")
        .eq("id", team_id)
        .eq("coach_user_id", coach_user_id)
        .execute()
    )
    if not team_check.data:
        raise PermissionError("This team does not exist or does not belong to the signed-in coach.")

    result = (
        client.table("athletes")
        .select("id, name")
        .eq("team_id", team_id)
        .eq("coach_user_id", coach_user_id)
        .order("name")
        .execute()
    )
    return result.data or []


def list_unassigned_athletes(coach_user_id: str) -> list:
    """Athletes belonging to this coach that aren't on any team yet."""
    _require_coach_user_id(coach_user_id)
    client = get_client()
    result = (
        client.table("athletes")
        .select("id, name")
        .eq("coach_user_id", coach_user_id)
        .is_("team_id", "null")
        .order("name")
        .execute()
    )
    return result.data or []
