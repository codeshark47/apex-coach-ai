"""
usage_limits.py
Tracks per-user free-analysis usage in Supabase, reusing the same
server-side client pattern as profile_store.py (SUPABASE_KEY, bypasses
RLS deliberately for this server-side counter — same justification as
profile_store: this is app-controlled bookkeeping, not user-owned data
that needs per-row access policies).
"""
import os
from profile_store import get_client

DEFAULT_FREE_LIMIT = int(os.environ.get("DEMO_FREE_LIMIT", "2"))


def get_usage(user_id: str) -> dict:
    """
    Returns {"used": int, "limit": int, "remaining": int} for this user.
    Creates a row with used=0 on first-ever call for a given user.
    """
    client = get_client()
    result = client.table("demo_usage").select("*").eq("user_id", user_id).execute()

    if result.data:
        row = result.data[0]
        used = row["used_count"]
        limit = row["free_limit"]
    else:
        used = 0
        limit = DEFAULT_FREE_LIMIT
        client.table("demo_usage").insert({
            "user_id": user_id,
            "used_count": used,
            "free_limit": limit,
        }).execute()

    return {"used": used, "limit": limit, "remaining": max(0, limit - used)}


def record_usage(user_id: str) -> dict:
    """
    Increments this user's used_count by 1 and returns the updated usage
    dict. Call this ONLY after an analysis has genuinely completed
    successfully — never on a failed run or a mere button click.
    """
    current = get_usage(user_id)  # ensures row exists
    client = get_client()
    new_used = current["used"] + 1
    client.table("demo_usage").update({"used_count": new_used}).eq("user_id", user_id).execute()
    return {"used": new_used, "limit": current["limit"], "remaining": max(0, current["limit"] - new_used)}


def is_admin(email: str) -> bool:
    """
    Checks if this email is in the ADMIN_EMAILS allowlist (comma-separated),
    read from Streamlit secrets or env vars — same pattern as every other
    credential in this app. Admins bypass the demo usage limit entirely and
    are never written to the demo_usage table.
    """
    if not email:
        return False
    raw = None
    try:
        import streamlit as st
        raw = st.secrets.get("ADMIN_EMAILS")
    except Exception:
        pass
    raw = raw or os.environ.get("ADMIN_EMAILS")
    if not raw:
        return False
    allowlist = {e.strip().lower() for e in raw.split(",") if e.strip()}
    return email.strip().lower() in allowlist
