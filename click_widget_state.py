"""
click_widget_state.py

Extracted so the fix below is independently testable without needing a
full Streamlit runtime — streamlit_app.py executes UI code at import
time, so it can't be imported directly in a test.

THE BUG THIS FIXES: streamlit_image_coordinates (and similar click-image
components) replay their LAST reported click value on every rerun that
doesn't give them a new widget key — Streamlit reconnects to "the same
widget" rather than starting fresh. render_zoomable_click_image's key
already varies with the crop region (zoom/pan), but at the DEFAULT zoom
level (1x, no zoom — where calibration stays permanently, since it never
enables zoom at all) the crop region is always the full image regardless
of what point(s) the caller has tracked, so the key never changed when a
point was added, moved, or cleared. Reported directly: pressing "Reset"
didn't reset, and a misclicked point stayed forever no matter what —
because clearing the caller's own session state doesn't touch the
widget's frontend-side memory, and the stale click gets replayed right
back into the just-cleared state on the very next rerun.
"""


def next_click_generation(session_state, key_prefix: str, marker_point, extra_markers: list) -> int:
    """
    Returns a monotonically increasing generation number, bumped only
    when the caller's actual tracked point state (marker_point, or the
    set of extra_markers) differs from what it was last render. Fold
    this into a click-widget's key so a reset/add/move always forces a
    genuinely fresh widget instance.

    A raw snapshot of the state (e.g. "no points yet") is NOT safe to
    use as the key directly — that exact shape recurs every time a coach
    resets, which would recreate a key string used earlier in the
    session, and risks Streamlit reconnecting to THAT old widget's stale
    value instead of starting fresh. A counter that only ever increases
    cannot collide with a prior key no matter how many times the same
    state shape (empty, one point, etc.) recurs.

    session_state: anything dict-like (st.session_state in real use; a
    plain dict is a faithful stand-in for testing this in isolation).
    """
    token = f"{marker_point}|{[m['point'] for m in extra_markers]}"
    token_key = f"_{key_prefix}_last_marker_token"
    gen_key = f"_{key_prefix}_click_gen"
    if session_state.get(token_key) != token:
        session_state[gen_key] = session_state.get(gen_key, 0) + 1
        session_state[token_key] = token
    return session_state[gen_key]


def seed_confirmation_status(session_state, key_prefix: str, seed_point, seed_frame, extra_seeds: list):
    """
    Returns (is_ready: bool, pending_identity: str) for the seed-
    placement confirmation gate — is_ready=True means the coach already
    explicitly confirmed this EXACT (seed_point, seed_frame, extra_seeds)
    combination, so the caller can safely proceed to the expensive
    extraction stage.

    BUG FIX, found from a real production crash: this gate used to
    return "ready" immediately whenever extra_seeds was empty — i.e. the
    single-seed case (the common one, used by every analysis) had NO
    gate at all. Every click adjusting the primary seed point's
    placement (coaches naturally click a few times to get it precise)
    changed the pending identity and re-triggered a full video
    extraction. This was believed harmless when first found (just
    "annoying/slow"), but is now confirmed to be a real memory leak, not
    just wasted time: repeated PoseLandmarker creation within one
    process was measured directly to permanently leak ~30-40MB per
    extraction call, even after .close() and gc.collect() — a live
    session's server log showed 11 separate extraction passes within 90
    seconds of ordinary seed placement, and the app was OOM-killed by
    Streamlit Cloud shortly after. Every seed configuration — including
    a single point with no extra confirmations — now requires one
    explicit confirmation before extraction runs, closing off the most
    direct, reproducible way a coach could unknowingly trigger repeated
    leaking extraction passes.
    """
    pending_identity = f"{seed_point}_{seed_frame}_{extra_seeds}"
    locked_key = f"_{key_prefix}_seeds_locked_identity"
    is_ready = session_state.get(locked_key) == pending_identity
    return is_ready, pending_identity


def lock_seed_confirmation(session_state, key_prefix: str, pending_identity: str):
    """Records that the coach has confirmed this exact seed configuration
    is final — called from the "Continue" button's on-click handling."""
    session_state[f"_{key_prefix}_seeds_locked_identity"] = pending_identity
