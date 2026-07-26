"""
monitoring.py

Optional error tracking via Sentry — the "we'd only find out about a
production bug if a coach happens to email us" gap. If SENTRY_DSN isn't
configured (e.g. local dev), init_sentry() and capture() are both no-ops,
so nothing here can break a deployment that hasn't set it up.

Streamlit does NOT share module-level state across pages/ files at the
script level — each page runs as its own top-level script — so
init_sentry() must be called near the top of streamlit_app.py AND every
pages/*.py file. It's safe to call repeatedly: the module (and its
_initialized flag) is cached in sys.modules once per server process, so
this only actually calls sentry_sdk.init() once.
"""

import os

_initialized = False


def _get_dsn():
    dsn = None
    try:
        import streamlit as st
        dsn = st.secrets.get("SENTRY_DSN")
    except Exception:
        pass
    return dsn or os.environ.get("SENTRY_DSN")


def init_sentry():
    global _initialized
    if _initialized:
        return
    dsn = _get_dsn()
    if not dsn:
        return
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.0,
            # Athlete/coach data (names, video-derived metrics) must never
            # ride along on an error report.
            send_default_pii=False,
        )
        _initialized = True
    except Exception:
        pass


def capture(exc: Exception):
    """Report an exception that's already been caught and handled at the
    call site (the user still sees the existing fallback message) — this
    only adds visibility for us, it changes no existing behavior."""
    if not _initialized:
        return
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
