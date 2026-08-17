"""
usage_limits.py
Tracks per-user free-analysis usage in Supabase, reusing the same
server-side client pattern as profile_store.py (SUPABASE_KEY, bypasses
RLS deliberately for this server-side counter — same justification as
profile_store: this is app-controlled bookkeeping, not user-owned data
that needs per-row access policies).

DAILY RESET: demo_usage.usage_date (add_daily_usage_reset.sql) tracks
which calendar day used_count applies to. Previously this was a pure
LIFETIME cap with no date concept at all — once a coach used up
free_limit, they were capped forever, not "for today." Whenever
get_usage/record_usage see usage_date isn't today, they reset
used_count to 0 and stamp usage_date to today FIRST, before reading or
incrementing it — so "10 analyses" means 10 fresh ones every day, not
10 ever.

MIGRATION SAFETY: every write that touches usage_date is wrapped so a
"column does not exist" error (the migration hasn't been run against
this project yet) falls back to the old lifetime-cap write instead of
crashing a real coach's session — reported via monitoring.capture so
it's visible that the migration is still pending, not silently masked
forever. Remove this fallback once add_daily_usage_reset.sql has been
run everywhere this app is deployed.
"""
import os
from datetime import date

import monitoring
from profile_store import get_client

DEFAULT_FREE_LIMIT = int(os.environ.get("DEMO_FREE_LIMIT", "2"))


def _write_usage(client, user_id: str, used_count: int, today: str, extra: dict = None):
    """INSERT or UPDATE (extra decides which) including usage_date, with a
    fallback to the same write minus usage_date if that column doesn't
    exist yet — see the module docstring's MIGRATION SAFETY note."""
    payload = {"used_count": used_count, "usage_date": today, **(extra or {})}
    try:
        if extra is not None:  # insert path — extra carries user_id/free_limit
            client.table("demo_usage").insert(payload).execute()
        else:
            client.table("demo_usage").update(payload).eq("user_id", user_id).execute()
    except Exception as e:
        monitoring.capture(e)
        payload.pop("usage_date")
        if extra is not None:
            client.table("demo_usage").insert(payload).execute()
        else:
            client.table("demo_usage").update(payload).eq("user_id", user_id).execute()


def get_usage(user_id: str) -> dict:
    """
    Returns {"used": int, "limit": int, "remaining": int} for this user,
    for TODAY specifically. Creates a row with used=0 on first-ever call
    for a given user, and resets used_count to 0 (persisted immediately,
    not just returned) the first time this is called on a new day.

    limit is the GREATER of the row's stored free_limit (a manual
    per-user override — e.g. the two demo-coach accounts set to 10, see
    tests) and payments.effective_limit() (2026-08-17, a real
    subscription tier). Two different override mechanisms, both real:
    a manual grant should never be clawed back by a coach's tier being
    "free," and a genuine paid subscription should never be capped by
    whatever free_limit happened to be written when the row was first
    created (that column predates paid tiers, written once at row-
    creation time — without this max(), a coach who upgrades after
    their row already exists would stay capped at day-one's limit).
    """
    import payments
    client = get_client()
    result = client.table("demo_usage").select("*").eq("user_id", user_id).execute()
    today = date.today().isoformat()
    tier_limit = payments.effective_limit(user_id)

    if result.data:
        row = result.data[0]
        limit = max(row["free_limit"], tier_limit)
        # "usage_date" absent (pre-migration row/column) reads as None,
        # which is never == today — but with no date column to tell "new
        # day" apart from "column just doesn't exist," treating every
        # call as same-day (skip the reset) is the correct pre-migration
        # fallback: it's exactly today's old lifetime-cap behavior.
        if "usage_date" in row and row["usage_date"] != today:
            used = 0
            _write_usage(client, user_id, 0, today)
        else:
            used = row["used_count"]
    else:
        used = 0
        limit = tier_limit
        _write_usage(client, user_id, used, today, extra={"user_id": user_id, "free_limit": limit})

    return {"used": used, "limit": limit, "remaining": max(0, limit - used)}


def record_usage(user_id: str) -> dict:
    """
    Increments this user's used_count by 1 (for TODAY — get_usage above
    already reset it if this is a new day) and returns the updated usage
    dict. Call this ONLY after an analysis has genuinely completed
    successfully — never on a failed run or a mere button click.
    """
    current = get_usage(user_id)  # ensures row exists and today's reset has happened
    client = get_client()
    new_used = current["used"] + 1
    _write_usage(client, user_id, new_used, date.today().isoformat())
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
