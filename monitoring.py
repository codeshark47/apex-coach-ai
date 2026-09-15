"""
monitoring.py

Optional error tracking via Sentry — the "we'd only find out about a
production bug if a coach happens to email us" gap. If SENTRY_DSN isn't
configured (e.g. local dev), the Sentry side of init_sentry()/capture()
is a no-op, so nothing here can break a deployment that hasn't set it up.

Streamlit does NOT share module-level state across pages/ files at the
script level — each page runs as its own top-level script — so
init_sentry() must be called near the top of streamlit_app.py AND every
pages/*.py file. It's safe to call repeatedly: the module (and its
_initialized flag) is cached in sys.modules once per server process, so
this only actually calls sentry_sdk.init() once.

LOCAL FALLBACK LOG (2026-09-15, real gap found while debugging a coach's
actual live session): capture() being a complete no-op without Sentry
meant every "skipped N frames"/"rejected candidate" diagnostic this
codebase's own raw-re-extraction gates raise via monitoring.capture()
was invisible for a real coach's real session — undebuggable after the
fact, only reproducible by guessing at their exact clicks. capture() now
always appends a line to a local log file (logs/monitoring.log,
git-ignored) regardless of whether Sentry is configured, so any local/
dev run — including a coach testing against a locally-hosted instance —
leaves a trail. This is purely additive to the existing Sentry path and
never raises (a logging failure must never break the caller's own
already-handled exception flow).
"""

import datetime
import os
import traceback

_initialized = False
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "monitoring.log")


def _log_locally(exc: Exception):
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- {datetime.datetime.now().isoformat()} ---\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


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
    _log_locally(exc)
    if not _initialized:
        return
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
