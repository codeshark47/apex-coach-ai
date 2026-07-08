"""
auth.py

Real user sign-up/sign-in using Supabase Auth — reuses the Supabase
project already set up for athlete history, rather than inventing a
separate password system or rolling custom password hashing (which would
be a real security risk to build by hand).

IMPORTANT KEY DISTINCTION:
  - profile_store.py uses SUPABASE_KEY (the "secret" key) to read/write
    athlete/session data, deliberately bypassing Row Level Security for
    server-side history operations.
  - Auth operations (sign up / sign in / sign out) should use the
    "anon"/"publishable" key instead — that's what it's designed for, and
    Supabase's Auth endpoints have their own protections independent of
    table-level RLS. Do NOT reuse the secret key for this.

Setup required (one-time):
  Add a SECOND secret to .streamlit/secrets.toml / Streamlit Cloud secrets:
    SUPABASE_ANON_KEY = "sb_publishable_xxxxxxxxxxxxxxxxxxxx"
  (the same "Publishable key" from Supabase's API settings page you
  already saw earlier — NOT the secret key, which stays as SUPABASE_KEY).

No new database schema needed — Supabase Auth manages its own internal
user table automatically.
"""

import os

_auth_client = None


def _get_anon_credentials():
    url = None
    anon_key = None
    secrets_error = None
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL")
        anon_key = st.secrets.get("SUPABASE_ANON_KEY")
    except Exception as e:
        secrets_error = str(e)
    url = url or os.environ.get("SUPABASE_URL")
    anon_key = anon_key or os.environ.get("SUPABASE_ANON_KEY")
    return url, anon_key, secrets_error


def get_auth_client():
    global _auth_client
    if _auth_client is not None:
        return _auth_client

    url, anon_key, secrets_error = _get_anon_credentials()
    if not url or not anon_key:
        if secrets_error:
            raise RuntimeError(f"Could not read Streamlit secrets: {secrets_error}")
        raise RuntimeError(
            "Sign-in is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY "
            "(the publishable key, NOT the secret key) to secrets."
        )

    from supabase import create_client
    _auth_client = create_client(url, anon_key)
    return _auth_client


def sign_up(email: str, password: str) -> dict:
    email = email.strip()
    if not email or "@" not in email:
        return {"status": "error", "message": "Enter a valid email address."}
    if not password or len(password) < 6:
        return {"status": "error", "message": "Password must be at least 6 characters."}

    try:
        client = get_auth_client()
        client.auth.sign_up({"email": email, "password": password})
        return {
            "status": "success",
            "message": "Account created. Check your email to confirm your address before signing in.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def sign_in(email: str, password: str) -> dict:
    email = email.strip()
    if not email or not password:
        return {"status": "error", "message": "Enter both email and password."}

    try:
        client = get_auth_client()
        result = client.auth.sign_in_with_password({"email": email, "password": password})
        if result and result.user:
            return {
                "status": "success",
                "user": {"id": result.user.id, "email": result.user.email},
            }
        return {"status": "error", "message": "Sign-in failed — no user returned."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def sign_out():
    try:
        client = get_auth_client()
        client.auth.sign_out()
    except Exception:
        pass
