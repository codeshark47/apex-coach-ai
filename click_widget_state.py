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
