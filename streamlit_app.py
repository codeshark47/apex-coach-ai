import os
import base64
from dotenv import load_dotenv
load_dotenv()

import monitoring
monitoring.init_sentry()

import streamlit as st
import pandas as pd

from orchestrator import run_complete_bowling_analysis
import orchestrator as o
from coaching_agent import generate_biomechanical_coaching_report

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime

# Phase 2 modules — single source of truth for ranges, real timing/speed, history
import metric_ranges as mr
import pdf_color_ranges as pcr
import speed_estimation as se
import calibration as cal
import profile_store as store
import data_quality as dq
import run_up_analysis as rua
import click_widget_state


def _render_frame_jump_box(slider_key: str, min_value: int, max_value: int):
    """
    A direct "type the frame number" box targeting one frame-scrubbing
    slider. Dragging a slider to the EXACT right frame is fiddly on a
    real video — a coach who already knows roughly which frame they want
    (e.g. from scrubbing the source video in another player) should be
    able to just type it. The box's own up/down stepper arrows already
    cover the "nudge by one frame" case (an earlier version of this had
    separate ◀/▶ buttons for that — removed as redundant, per direct
    coach feedback, once this box existed). Mutates the slider's own
    session_state key via a callback (the supported way to adjust a
    keyed widget from outside itself) — call this immediately before the
    st.slider(key=slider_key) call it targets, no other wiring needed,
    the slider just picks up the new value on rerun.

    The number box's own widget key is derived from the CURRENT slider
    value (not a fixed string) — this is deliberate: Streamlit only reads
    a keyed widget's value= argument on first creation, so a fixed key
    would go stale and show the wrong number after the slider itself is
    dragged directly. Deriving the key from the current value forces a
    fresh widget each time something else changes it, so the displayed
    number is always correct regardless of which control last touched it.
    """
    current_value = st.session_state.get(slider_key, min_value)
    jump_key = f"{slider_key}_jump_input_{current_value}"

    def _jump_to_typed():
        typed = st.session_state.get(jump_key)
        if typed is not None:
            st.session_state[slider_key] = max(min_value, min(max_value, int(typed)))

    st.number_input(
        "Jump to an exact frame number", min_value=min_value, max_value=max_value,
        value=current_value, key=jump_key, on_change=_jump_to_typed,
        help="Type an exact frame number (or use the arrows) and press Enter",
    )


def _framed_image_container():
    """
    A smaller, centered, bordered container for reference-frame images —
    was full-page-width with no visual frame, which read as a raw debug
    dump rather than a deliberate part of the UI. Purely a layout change:
    streamlit_image_coordinates still reports back its own actual
    rendered width/height regardless of container size, and every call
    site already scales click coordinates using that reported size (not
    a hardcoded one) — so shrinking the display here doesn't touch the
    click-to-pixel math at all.
    """
    _, mid, _ = st.columns([1, 3, 1])
    return mid.container(border=True)


def render_zoomable_click_image(pil_img, key_prefix: str, marker_point=None, marker_color: str = "lime",
                                 extra_markers: list = None, enable_zoom: bool = False):
    """
    Displays a reference-frame image, with an optional digital zoom, and
    returns a click position in ORIGINAL image pixel coordinates (or None
    if nothing was clicked this run).

    enable_zoom: OFF by default — direct coach feedback was that zoom
    should be reserved for the one click that genuinely needs
    pixel-level precision (the release-point/wrist correction), not
    every clickable image. Seed-point (roughly which person is the
    bowler) and calibration (usually a stump, a large target) don't need
    it and it was just adding a control to skip past.

    WHY A CROP-BASED "DIGITAL" ZOOM, NOT CSS/BROWSER PINCH-ZOOM: on a
    phone screen the reference frame is small enough that clicking the
    exact right pixel (a wrist, a stump edge) is genuinely hard. The
    click-to-original-pixel math already depends on knowing exactly what
    region of the source image is on screen and at what scale (see the
    scale_x/scale_y pattern used everywhere in this file) — a
    server-side crop keeps that fully under this app's control, where an
    uncontrolled browser pinch-zoom gesture would risk desyncing
    reported click coordinates from what's actually visible.

    marker_point: (x, y) in ORIGINAL image coordinates, if a PRIMARY
    marker should be drawn — also used to CENTER the zoom, since a coach
    zooming in almost always wants to fine-tune a point they already
    placed roughly, not an arbitrary corner of the frame. None if no
    marker exists yet (zoom centers on the last extra_marker, or the
    image middle if there's nothing to center on at all).

    extra_markers: optional list of {"point": (x,y), "color": str,
    "label": str|None} — for the calibration flow, which shows up to two
    numbered reference points at once, unlike every other call site here
    (seed point, wrist correction) which only ever has one marker.

    At the default "1x (no zoom)" level the crop offset is exactly (0, 0)
    (no .crop() call happens at all) — the ONE other transformation that
    can still apply regardless of zoom level is the display downscale
    below, for images larger than MAX_DISPLAY_DIM (verified necessary on
    real 1080x1920 phone footage — see that comment for why).
    """
    from PIL import Image, ImageDraw
    from streamlit_image_coordinates import streamlit_image_coordinates

    orig_w, orig_h = pil_img.size
    extra_markers = extra_markers or []

    zoom = 1.0
    if enable_zoom:
        zoom_options = {"1x (no zoom)": 1.0, "2x": 2.0, "3x": 3.0, "4x": 4.0}
        zoom_label = st.select_slider(
            "Zoom", options=list(zoom_options.keys()), value="1x (no zoom)",
            key=f"{key_prefix}_zoom_level",
            help="Zoom in for more precise clicking — especially useful on a phone screen.",
        )
        zoom = zoom_options[zoom_label]

    crop_w = max(1, int(orig_w / zoom))
    crop_h = max(1, int(orig_h / zoom))
    if marker_point is not None:
        center_x, center_y = marker_point
    elif extra_markers:
        center_x, center_y = extra_markers[-1]["point"]
    else:
        center_x, center_y = orig_w // 2, orig_h // 2
    crop_x0 = max(0, min(orig_w - crop_w, center_x - crop_w // 2))
    crop_y0 = max(0, min(orig_h - crop_h, center_y - crop_h // 2))

    display_img = pil_img.copy()
    if extra_markers:
        draw = ImageDraw.Draw(display_img)
        r = max(4, orig_w // 150)
        for m in extra_markers:
            px, py = m["point"]
            draw.ellipse((px - r, py - r, px + r, py + r), outline=m.get("color", "red"), width=3)
            if m.get("label"):
                draw.text((px + r + 4, py - r - 4), str(m["label"]), fill=m.get("color", "red"))
    if marker_point is not None:
        draw = ImageDraw.Draw(display_img)
        r = max(5, orig_w // 100)
        px, py = marker_point
        draw.ellipse((px - r, py - r, px + r, py + r), outline=marker_color, width=4)

    if zoom > 1.0:
        display_img = display_img.crop((crop_x0, crop_y0, crop_x0 + crop_w, crop_y0 + crop_h))
        st.caption(f"🔍 Zoomed {zoom_label} — click anywhere in this cropped view to place the marker there.")

    # crop_disp_w/h = the TRUE pixel size of the region being shown, before
    # any resize below — this is what the final click math scales back
    # from, so it must be captured here regardless of what happens next.
    crop_disp_w, crop_disp_h = display_img.size

    # DOWNSCALE FOR THE COMPONENT — BUG FIX found on real footage: a
    # phone-shot rear-view clip (1080x1920, verified directly — 5x the
    # pixels of a same-session side-on clip at 848x478) was slow enough
    # sending its full-resolution image to the streamlit_image_coordinates
    # component that it looked like the image "wasn't loading" at all,
    # every time, until zoomed in — which incidentally shrank the actual
    # pixel data enough to render quickly, creating the false impression
    # that zoom was required for the image to work. The component
    # transmits the PIL image's actual pixel data every render regardless
    # of its CSS-displayed size (use_column_width scales it visually in
    # the browser, but doesn't reduce what gets sent) — resizing down to a
    # fixed max dimension keeps rendering fast and CONSISTENT regardless
    # of the source video's native resolution. MAX_DISPLAY_DIM is an
    # engineering choice (comfortably above any phone screen's effective
    # width) balancing load speed against not needlessly blurring detail,
    # not a validated number.
    MAX_DISPLAY_DIM = 960
    downscale = min(1.0, MAX_DISPLAY_DIM / max(crop_disp_w, crop_disp_h))
    if downscale < 1.0:
        display_img = display_img.resize(
            (max(1, round(crop_disp_w * downscale)), max(1, round(crop_disp_h * downscale))),
            Image.LANCZOS,
        )

    # BUG FIX found on real footage (a click reported "in the sky", nowhere
    # near the bowler): this widget's key was FIXED regardless of zoom or
    # crop region, so switching zoom level didn't create a fresh widget —
    # streamlit_image_coordinates can replay a STALE click position (valid
    # for the previous, differently-cropped image) against the new one,
    # which lands the translated coordinate somewhere that has nothing to
    # do with where the coach actually clicked. Deriving the key from the
    # crop region itself (same "dynamic key" pattern as the frame-number
    # jump box above) forces a genuinely new widget instance whenever the
    # displayed image changes, so a click can never be misattributed to
    # the wrong crop.
    #
    # SECOND BUG FIX (reported directly: "reset" doesn't reset; a
    # misclicked point stays forever no matter what): at the DEFAULT zoom
    # level (1x, no zoom — every call site's starting state, and where
    # calibration stays permanently since it never enables zoom at all),
    # crop_w/crop_h always equal the full image size, so crop_x0/crop_y0
    # are ALWAYS (0, 0) regardless of marker_point or extra_markers — the
    # key above never changes when a point is added, moved, or cleared.
    # Streamlit then replays the component's last-known click value on
    # every rerun a "Reset" button causes, which immediately re-appends
    # the very point that was just supposedly cleared. See
    # click_widget_state.py's docstring for the full reasoning — pulled
    # out to its own module since streamlit_app.py can't be imported in
    # a test (it runs UI code at import time).
    click_gen = click_widget_state.next_click_generation(
        st.session_state, key_prefix, marker_point, extra_markers
    )
    click_widget_key = (
        f"{key_prefix}_zoomclick_widget_{crop_x0}_{crop_y0}_{crop_w}_{crop_h}_{click_gen}"
    )
    # BUG FIX (reported directly: the image "wouldn't load" until zoomed in
    # and back out): streamlit_image_coordinates defaults to PNG with ZERO
    # compression for any PIL image passed in. Verified directly — a
    # realistic 960x540 photographic frame (post the MAX_DISPLAY_DIM
    # downscale above) serializes to ~2MB of base64 text as PNG regardless
    # of compress_level (photographic noise doesn't deflate well), versus
    # ~375KB as JPEG at quality=80 — a >5x smaller payload stuffed into a
    # single WebSocket message on every render. That gap, not the pixel
    # count, is what stalled the first paint until zooming forced a
    # smaller (cropped) image through. No precision is lost by this — the
    # click-to-pixel math only depends on displayed dimensions, never on
    # image encoding quality.
    with _framed_image_container():
        click = streamlit_image_coordinates(
            display_img, key=click_widget_key,
            use_column_width="always",
            image_format="JPEG", jpeg_quality=80,
        )

    if click is None:
        return None

    rendered_w = click.get("width") or crop_disp_w
    rendered_h = click.get("height") or crop_disp_h
    scale_x = crop_disp_w / rendered_w
    scale_y = crop_disp_h / rendered_h
    click_x_in_crop = click["x"] * scale_x
    click_y_in_crop = click["y"] * scale_y
    return (round(crop_x0 + click_x_in_crop), round(crop_y0 + click_y_in_crop))


def render_stream_event_confirmation(stage12_result, ref_path: str, file_identity: str,
                                      key_prefix: str, bowling_arm: str, stream_label: str):
    """
    Mandatory BFC/FFC/BR + release-point confirmation for ONE video stream
    — built so Dual Camera mode gets the exact same reliability safeguards
    Single Camera already has, instead of trusting raw auto-detection with
    no human review at all. Every one of these steps exists because a
    real, confirmed failure mode was found on actual footage where
    auto-detection looked confident and was still wrong (see the comments
    in Single Camera's own confirmation flow, further down this file, for
    the specific clips that proved each one out).

    Deliberately a NEW, separate function rather than a refactor of Single
    Camera's existing (already working, already verified) inline flow —
    this is pure addition with its own session_state keys (prefixed by
    key_prefix, e.g. "side"/"rear"), so it carries zero risk of changing
    Single Camera's behavior.

    The release-point step runs for BOTH streams, not just whichever one
    was originally expected to feed Release Height — BUG FIX from real
    footage: Release Height used to be computed only from the side
    stream with no way to correct a failed reading, and when it failed,
    there was no fallback. dual_camera_orchestrator.py now falls back to
    the rear stream's own corrected release point when the side stream's
    is unavailable, so both need the same correction opportunity.

    Returns None if BFC/FFC/BR/release-point aren't all confirmed yet. Once confirmed,
    returns {"BFC": int, "FFC": int, "BR": int, "BFC_auto_detected": ...,
    "FFC_auto_detected": ..., "BR_auto_detected": ..., "BR_auto_confidence": ...,
    "wrist_override_x": float|None, "wrist_override_y": float|None}.
    """
    br_key, ffc_key, bfc_key = f"_{key_prefix}_br_confirmed_frame", f"_{key_prefix}_ffc_confirmed_frame", f"_{key_prefix}_bfc_confirmed_frame"
    br_id_key, ffc_id_key, bfc_id_key = f"_{key_prefix}_br_identity", f"_{key_prefix}_ffc_identity", f"_{key_prefix}_bfc_identity"

    if st.session_state.get(br_id_key) != file_identity:
        st.session_state[br_key] = None
        st.session_state[br_id_key] = file_identity
    if st.session_state.get(ffc_id_key) != file_identity:
        st.session_state[ffc_key] = None
        st.session_state[ffc_id_key] = file_identity
    if st.session_state.get(bfc_id_key) != file_identity:
        st.session_state[bfc_key] = None
        st.session_state[bfc_id_key] = file_identity

    br_auto, br_confidence, ffc_auto, bfc_auto = None, None, None, None
    if stage12_result is not None and stage12_result.get("status") == "success":
        br_auto = stage12_result["events"].get("BR")
        br_confidence = stage12_result["events"].get("BR_confidence")
        ffc_auto = stage12_result["events"].get("FFC")
        bfc_auto = stage12_result["events"].get("BFC")

    total_frames = cal.get_frame_count(ref_path)

    with st.expander(f"🎯 Confirm Ball Release Frame — {stream_label}",
                      expanded=st.session_state.get(br_key) is None):
        if br_auto is None:
            st.error("Couldn't detect a release frame at all — check tracking above.")
        else:
            conf_note = {"high": "high confidence", "low": "low confidence"}.get(br_confidence, "unknown confidence")
            st.info(
                f"Algorithm's best guess: **frame {br_auto}** ({conf_note}). Scrub to the exact "
                f"frame where the ball actually leaves the hand and confirm. Auto-detection can "
                f"be wrong even when it reports high confidence, so this step always runs."
            )
            slider_key = f"{key_prefix}_br_confirm_slider"
            _render_frame_jump_box(slider_key, 0, max(total_frames - 1, 0))
            br_slider_val = st.slider(
                "Scrub to the true ball-release frame",
                min_value=0, max_value=max(total_frames - 1, 0),
                value=min(max(br_auto, 0), max(total_frames - 1, 0)),
                key=slider_key
            )
            br_frame_img = cal.extract_reference_frame(ref_path, frame_index=br_slider_val)
            if br_frame_img is not None:
                with _framed_image_container():
                    st.image(br_frame_img, use_column_width=True,
                              caption=f"Frame {br_slider_val} — is the ball leaving the hand here?")
            if st.button("✅ Confirm this is the release frame", key=f"{key_prefix}_confirm_br_button"):
                st.session_state[br_key] = br_slider_val
                st.rerun()
            if st.session_state.get(br_key) is not None:
                st.success(f"Confirmed: release at frame {st.session_state[br_key]}.")

    confirmed_br_frame = st.session_state.get(br_key)
    wrist_override_x, wrist_override_y = None, None
    wrist_confirmed_key = f"_{key_prefix}_wrist_step_confirmed"
    if confirmed_br_frame is not None:
        wrist_point_key = f"_{key_prefix}_wrist_confirmed_point"
        wrist_id_key = f"_{key_prefix}_wrist_identity"
        wrist_identity = f"{file_identity}_{confirmed_br_frame}"
        if st.session_state.get(wrist_id_key) != wrist_identity:
            st.session_state[wrist_point_key] = None
            st.session_state[wrist_id_key] = wrist_identity
            st.session_state[wrist_confirmed_key] = False

        # MANDATORY, not optional (BUG FIX from direct coach feedback: this
        # used to be a skippable "(optional)" panel, but release_height is
        # exactly as prone to silent wrist mistracking as any other event —
        # same reasoning as why BFC/FFC/BR are never skippable either).
        with st.expander(f"🖐️ Confirm Release Point — {stream_label}",
                          expanded=not st.session_state.get(wrist_confirmed_key, False)):
            if br_confidence == "low":
                st.warning(
                    "Auto-detection reported **low confidence** on this release frame. This specific "
                    "signal correlates with the wrist/hand tracking silently undershooting the real "
                    "arm extension — the drawn Release Height line can look plausible while still "
                    "being wrong. Please check the marker below carefully before confirming."
                )
            st.caption(
                "The yellow marker is the auto-tracked ball/hand position at your confirmed "
                "release frame. If it doesn't sit on the real ball/hand, click the real position "
                "to correct it, then confirm. This feeds Release Height directly."
            )
            wrist_frame_img = cal.extract_reference_frame(ref_path, frame_index=confirmed_br_frame)
            if wrist_frame_img is not None:
                from PIL import Image

                pil_img = Image.fromarray(wrist_frame_img)
                orig_w, orig_h = pil_img.size

                auto_point = None
                stage12_df = stage12_result.get("df") if stage12_result else None
                bowl_side_for_wrist = "RIGHT" if bowling_arm == "right" else "LEFT"
                if stage12_df is not None:
                    wrist_rows = stage12_df[stage12_df["frame"] == confirmed_br_frame]
                    if not wrist_rows.empty:
                        wx = wrist_rows.iloc[0].get(f"{bowl_side_for_wrist}_WRIST_x")
                        wy = wrist_rows.iloc[0].get(f"{bowl_side_for_wrist}_WRIST_y")
                        if wx is not None and wy is not None and not pd.isna(wx) and not pd.isna(wy):
                            auto_point = (round(wx * orig_w), round(wy * orig_h))

                corrected_point = st.session_state.get(wrist_point_key)
                display_point = corrected_point or auto_point

                if corrected_point is None:
                    st.caption("🟡 Auto-tracked position shown. Zoom in and click the image below to correct it if wrong.")
                else:
                    st.caption("✅ Corrected — click again to move the marker, or reset below.")

                new_point = render_zoomable_click_image(
                    pil_img, key_prefix=f"{key_prefix}_wrist", marker_point=display_point,
                    marker_color=("lime" if corrected_point is not None else "yellow"),
                    enable_zoom=True,
                )
                if new_point is not None and st.session_state.get(wrist_point_key) != new_point:
                    st.session_state[wrist_point_key] = new_point
                    st.rerun()

                if corrected_point is not None and st.button("↺ Reset to auto-tracked position", key=f"{key_prefix}_reset_wrist_point"):
                    st.session_state[wrist_point_key] = None
                    st.rerun()

            if st.button("✅ Confirm this release point", key=f"{key_prefix}_confirm_wrist_button"):
                st.session_state[wrist_confirmed_key] = True
                st.rerun()
            if st.session_state.get(wrist_confirmed_key):
                st.success("Release point confirmed.")

        confirmed_point = st.session_state.get(wrist_point_key)
        if confirmed_point is not None:
            wrist_frame_dims = cal.extract_reference_frame(ref_path, frame_index=confirmed_br_frame)
            if wrist_frame_dims is not None:
                _wf_h, _wf_w = wrist_frame_dims.shape[:2]
                wrist_override_x = confirmed_point[0] / _wf_w
                wrist_override_y = confirmed_point[1] / _wf_h

    with st.expander(f"🦶 Confirm Front Foot Contact Frame — {stream_label}",
                      expanded=st.session_state.get(ffc_key) is None):
        if ffc_auto is None:
            st.error("Couldn't detect a front-foot-contact frame at all — check tracking above.")
        else:
            st.info(
                f"Algorithm's best guess: **frame {ffc_auto}**. Scrub to the exact frame where "
                f"the front (lead) foot first plants on the ground and confirm."
            )
            slider_key = f"{key_prefix}_ffc_confirm_slider"
            _render_frame_jump_box(slider_key, 0, max(total_frames - 1, 0))
            ffc_slider_val = st.slider(
                "Scrub to the true front-foot-contact frame",
                min_value=0, max_value=max(total_frames - 1, 0),
                value=min(max(ffc_auto, 0), max(total_frames - 1, 0)),
                key=slider_key
            )
            ffc_frame_img = cal.extract_reference_frame(ref_path, frame_index=ffc_slider_val)
            if ffc_frame_img is not None:
                with _framed_image_container():
                    st.image(ffc_frame_img, use_column_width=True,
                              caption=f"Frame {ffc_slider_val} — has the front foot just planted here?")
            if st.button("✅ Confirm this is the front-foot-contact frame", key=f"{key_prefix}_confirm_ffc_button"):
                st.session_state[ffc_key] = ffc_slider_val
                st.rerun()
            if st.session_state.get(ffc_key) is not None:
                st.success(f"Confirmed: front-foot contact at frame {st.session_state[ffc_key]}.")

    with st.expander(f"👟 Confirm Back Foot Contact Frame — {stream_label}",
                      expanded=st.session_state.get(bfc_key) is None):
        if bfc_auto is None:
            st.error("Couldn't detect a back-foot-contact frame at all — check tracking above.")
        else:
            st.info(
                f"Algorithm's best guess: **frame {bfc_auto}**. Scrub to the frame where the back "
                f"(rear) foot plants just before the final delivery stride and confirm."
            )
            slider_key = f"{key_prefix}_bfc_confirm_slider"
            _render_frame_jump_box(slider_key, 0, max(total_frames - 1, 0))
            bfc_slider_val = st.slider(
                "Scrub to the true back-foot-contact frame",
                min_value=0, max_value=max(total_frames - 1, 0),
                value=min(max(bfc_auto, 0), max(total_frames - 1, 0)),
                key=slider_key
            )
            bfc_frame_img = cal.extract_reference_frame(ref_path, frame_index=bfc_slider_val)
            if bfc_frame_img is not None:
                with _framed_image_container():
                    st.image(bfc_frame_img, use_column_width=True,
                              caption=f"Frame {bfc_slider_val} — has the back foot just planted here?")
            if st.button("✅ Confirm this is the back-foot-contact frame", key=f"{key_prefix}_confirm_bfc_button"):
                st.session_state[bfc_key] = bfc_slider_val
                st.rerun()
            if st.session_state.get(bfc_key) is not None:
                st.success(f"Confirmed: back-foot contact at frame {st.session_state[bfc_key]}.")

    br_confirmed = st.session_state.get(br_key)
    ffc_confirmed = st.session_state.get(ffc_key)
    bfc_confirmed = st.session_state.get(bfc_key)
    wrist_confirmed = st.session_state.get(wrist_confirmed_key, False)
    if br_confirmed is None or ffc_confirmed is None or bfc_confirmed is None or not wrist_confirmed:
        return None

    return {
        "BFC": bfc_confirmed, "FFC": ffc_confirmed, "BR": br_confirmed,
        "BFC_auto_detected": bfc_auto, "FFC_auto_detected": ffc_auto,
        "BR_auto_detected": br_auto, "BR_auto_confidence": br_confidence,
        "wrist_override_x": wrist_override_x, "wrist_override_y": wrist_override_y,
    }


def stream_confirmation_resolved(key_prefix: str) -> bool:
    """Whether all 4 mandatory steps are confirmed for this stream — used
    to gate the Execute button, same purpose as Single Camera's br/ffc/bfc
    _resolved checks. Release point is included since it's no longer an
    optional extra (see render_stream_event_confirmation)."""
    return (
        st.session_state.get(f"_{key_prefix}_br_confirmed_frame") is not None
        and st.session_state.get(f"_{key_prefix}_ffc_confirmed_frame") is not None
        and st.session_state.get(f"_{key_prefix}_bfc_confirmed_frame") is not None
        and st.session_state.get(f"_{key_prefix}_wrist_step_confirmed", False)
    )


# ====================================================================
# PAGE CONFIG & ELITE DARK UI  (unchanged from Phase 1)
# ====================================================================
st.set_page_config(page_title="Apex Coach AI", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span { color: #E2E8F0 !important; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #121824, #1A2333);
    border: 1px solid #00B4D8; border-radius: 10px; padding: 16px;
    box-shadow: 0 4px 15px rgba(0,180,216,0.1);
}
div[data-testid="stMetricValue"] { font-size: 28px !important; color: #00B4D8 !important; font-weight: 800 !important; }
div[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 12px !important; }
div[data-testid="stMetricDelta"] { color: #38BDF8 !important; font-size: 11px !important; }
h1, h2, h3, h4 { color: #00B4D8 !important; font-family: 'Helvetica Neue', sans-serif; }
section[data-testid="stSidebar"] { background-color: #0F1524 !important; border-right: 1px solid #00B4D8; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #00B4D8, #0077B6) !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-weight: 700 !important;
    padding: 14px 28px !important; transition: all 0.3s ease !important; width: 100% !important;
}
div[data-testid="stButton"] button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(0,180,216,0.4) !important; }
div[data-testid="stDownloadButton"] button {
    background: linear-gradient(90deg, #0077B6, #023E8A) !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
div[data-testid="stSuccess"] { background-color: #0D2818 !important; border-left: 4px solid #00C853 !important; }
div[data-testid="stError"] { background-color: #2D0A0A !important; border-left: 4px solid #FF3D3D !important; }
div[data-testid="stInfo"] { background-color: #0A1628 !important; border-left: 4px solid #00B4D8 !important; }
hr { border-color: #00B4D8 !important; opacity: 0.2 !important; }
div[data-testid="stFileUploader"] { background-color: #121824 !important; border: 1px dashed #00B4D8 !important; border-radius: 8px !important; padding: 8px !important; }
div[data-testid="stRadio"] label { color: #E2E8F0 !important; }
div[data-testid="stExpander"] { background-color: #121824 !important; border: 1px solid #1E3A5F !important; border-radius: 8px !important; }
div[data-testid="stSpinner"] { color: #00B4D8 !important; }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# LOGO  (unchanged)
# ====================================================================
script_directory = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_directory, "apex_logo.png.png")

log_col1, log_col2, log_col3 = st.columns([1.5, 1, 1.5])
with log_col2:
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""<div style="display:flex;justify-content:center;align-items:center;
            width:100%;height:90px;overflow:hidden;margin-bottom:-20px;">
            <img src="data:image/png;base64,{encoded}"
            style="max-width:160px;height:auto;transform:translateY(-15px);
            filter:drop-shadow(0px 2px 6px rgba(0,180,216,0.3));"></div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<h1 style='text-align:center;color:#00B4D8;'>⚡ APEX COACH AI</h1>", unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align:center;color:#94A3B8;font-weight:400;"
    "letter-spacing:2px;'>AUTONOMOUS BIOMECHANICAL PERFORMANCE HUB</h3>",
    unsafe_allow_html=True
)
st.divider()

# ====================================================================
# AUTHENTICATION GATE (real per-user sign-in via Supabase Auth)
# Replaces the earlier single shared beta password with real accounts.
# ====================================================================
import auth as auth_module

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if st.session_state.auth_user is None:
    st.markdown(
        "<h2 style='text-align:center;color:#00B4D8;'>🔐 Sign In</h2>",
        unsafe_allow_html=True
    )
    auth_tab_signin, auth_tab_signup = st.tabs(["Sign In", "Create Account"])

    with auth_tab_signin:
        with st.form("signin_form"):
            signin_email = st.text_input("Email", key="signin_email")
            signin_password = st.text_input("Password", type="password", key="signin_password")
            signin_submitted = st.form_submit_button("Sign In", use_container_width=True)
        if signin_submitted:
            try:
                result = auth_module.sign_in(signin_email, signin_password)
                if result["status"] == "success":
                    st.session_state.auth_user = result["user"]
                    st.rerun()
                else:
                    st.error(result["message"])
            except RuntimeError as e:
                st.error(str(e))

    with auth_tab_signup:
        with st.form("signup_form"):
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
            st.markdown(
                "<div style='font-size:0.85rem;color:#94A3B8;line-height:1.5;margin-top:6px;'>"
                "<b>Data use notice:</b> creating an account means you're a coach or academy "
                "staff member submitting athlete performance data. We collect the bowling-action "
                "video you upload, the biomechanical measurements and timing data extracted from "
                "it, any corrections you make to that data, and the athlete profile details "
                "(name, team, notes) you enter. This is used only to generate coaching reports "
                "and track an athlete's progress in your account — it is never sold or shared "
                "outside your account's coaching use. <b>If an athlete you submit data for is "
                "under 18, creating an account confirms you have the consent of that athlete's "
                "parent or guardian</b> to have this data collected and analyzed."
                "</div>",
                unsafe_allow_html=True,
            )
            signup_consent = st.checkbox(
                "I have read the data use notice above and confirm I have the authority "
                "and any required consent (including parent/guardian consent for athletes "
                "under 18) to submit this data.",
                key="signup_consent",
            )
            signup_submitted = st.form_submit_button("Create Account", use_container_width=True)
        if signup_submitted:
            try:
                result = auth_module.sign_up(signup_email, signup_password, consent_given=signup_consent)
                if result["status"] == "success":
                    st.success(result["message"])
                else:
                    st.error(result["message"])
            except RuntimeError as e:
                st.error(str(e))

    st.stop()  # nothing below this renders until signed in

else:
    top_l, top_r = st.columns([4, 1])
    with top_r:
        st.caption(f"Signed in: {st.session_state.auth_user['email']}")
        if st.button("Sign Out", use_container_width=True):
            auth_module.sign_out()
            st.session_state.auth_user = None
            st.rerun()
os.makedirs("input", exist_ok=True)


# ====================================================================
# PDF GENERATOR — now uses pdf_color_ranges + metric_ranges (single source)
# ====================================================================
def generate_pdf_report(metrics, frames, ai_insights, bowler_name="Elite Athlete",
                         camera_mode="Single Camera", phase_durations=None,
                         speed_result=None, quality=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'],
                                  fontSize=22, leading=26, textColor=colors.HexColor('#1A365D'))
    h2_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'],
                               fontSize=14, leading=18, textColor=colors.HexColor('#2B6CB0'),
                               spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'],
                                 fontSize=10, leading=14, textColor=colors.HexColor('#2D3748'))
    bold_body = ParagraphStyle('ReportBodyBold', parent=body_style, fontName='Helvetica-Bold')

    current_date = datetime.now().strftime("%Y-%m-%d")
    story.append(Paragraph("APEX COACH AI — BIOMECHANICAL REPORT", title_style))
    story.append(Paragraph(
        f"<b>Target Athlete:</b> {bowler_name} | "
        f"<b>Date:</b> {current_date} | "
        f"<b>Camera Mode:</b> {camera_mode} | "
        f"<b>Status:</b> AI-Assisted Biomechanical Estimate",
        body_style
    ))
    story.append(Spacer(1, 15))

    if quality and quality.get("confidence") == "low":
        warn_style = ParagraphStyle('Warn', parent=body_style,
                                     textColor=colors.HexColor('#C53030'),
                                     fontName='Helvetica-Bold')
        missing_labels = [mr.RANGES[k].label for k in quality["missing_metrics"]]
        story.append(Paragraph(
            f"⚠ LOW TRACKING CONFIDENCE: {quality['missing_count']} of 5 metrics "
            f"failed to compute ({', '.join(missing_labels)}). Remaining values in "
            f"this report came from the same degraded tracking and should not be "
            f"treated as reliable. Re-shoot this delivery before acting on these results.",
            warn_style
        ))
        story.append(Spacer(1, 12))

    # KINEMATIC MILESTONES + PHASE TIMING
    story.append(Paragraph("Kinematic Sequence Milestones", h2_style))
    time_rows = [
        [Paragraph("<b>Milestone Phase</b>", bold_body), Paragraph("<b>Frame</b>", bold_body),
         Paragraph("<b>Duration</b>", bold_body)],
        ["Back Foot Contact (BFC)", f"Frame {frames.get('back_foot_contact_frame', 'N/A')}", "—"],
        ["Front Foot Contact (FFC)", f"Frame {frames.get('front_foot_contact_frame', 'N/A')}",
         f"{phase_durations['bfc_to_ffc_seconds']}s from BFC" if phase_durations else "N/A"],
        ["Ball Release Point (BR)", f"Frame {frames.get('ball_release_frame', 'N/A')}",
         f"{phase_durations['ffc_to_br_seconds']}s from FFC" if phase_durations else "N/A"],
    ]
    t_time = Table(time_rows, colWidths=[190, 150, 160])
    t_time.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EDF2F7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_time)
    story.append(Spacer(1, 10))

    # SPEED (only if calibrated — never fabricated)
    story.append(Paragraph("Delivery Stride Tempo & Release Speed", h2_style))
    if phase_durations:
        story.append(Paragraph(
            f"Full delivery stride (BFC → BR): <b>{phase_durations['bfc_to_br_seconds']}s</b> "
            f"at {phase_durations['fps']} fps.",
            body_style
        ))
    if speed_result and speed_result.get("status") == "success":
        story.append(Paragraph(
            f"Estimated release arm speed: <b>{speed_result['kmh']} km/h</b> "
            f"({speed_result['bowling_arm'].replace('_', ' ').title()}). "
            f"<i>This tracks the bowling hand near release, not the ball itself — "
            f"treat as a session-over-session trend indicator, not a radar-equivalent reading.</i>",
            body_style
        ))
    elif speed_result and speed_result.get("status") == "not_calibrated":
        story.append(Paragraph(
            "Release speed not available — camera not calibrated for this session. "
            "Run calibration once per camera setup to enable this.",
            body_style
        ))
    else:
        story.append(Paragraph("Release speed not available for this session.", body_style))
    story.append(Spacer(1, 15))

    # COLOR-CODED CBC-STYLE REFERENCE RANGE TABLE
    story.append(Paragraph("Core Biomechanical Telemetry", h2_style))
    story.append(Paragraph(
        "<i>Reference ranges — Optimal (green) | Acceptable (amber) | Critical (red).</i>",
        body_style
    ))
    story.append(Spacer(1, 6))
    story.append(pcr.build_color_coded_range_table(metrics, bold_body))
    story.append(Spacer(1, 15))

    # AI NARRATIVE
    story.append(Paragraph("Autonomous AI Coach Assessment", h2_style))
    narrative = ai_insights.get("narrative_analysis", "No narrative generated.")
    narrative = narrative.replace(
        "SECTION 1 — BIOMECHANICAL NARRATIVE ASSESSMENT:", ""
    ).replace("SECTION 1 — BIOMECHANICAL NARRATIVE:", "").strip()
    story.append(Paragraph(narrative, body_style))
    story.append(Spacer(1, 15))

    # DRILLS
    story.append(Paragraph("Prescribed Training Drills", h2_style))
    drills = ai_insights.get("prescribed_drills", [])
    if drills:
        for drill in drills:
            story.append(Paragraph(f"• {drill}", body_style))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("All metrics within acceptable range. No critical interventions required.", body_style))

    story.append(Spacer(1, 25))
    story.append(Paragraph("—" * 60, body_style))
    story.append(Paragraph("<b>Shoaib Nazar</b>, Founder | Apex Coach AI", bold_body))
    story.append(Paragraph("Automated Video-Based Coaching Report", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ====================================================================
# SIDEBAR
# ====================================================================
st.sidebar.markdown("<h2 style='color:#00B4D8;text-align:center;'>⚡ Control Panel</h2>", unsafe_allow_html=True)

st.sidebar.header("📝 Player Profile")
player_name = st.sidebar.text_input(
    "Athlete Full Name", value="Elite Athlete",
    help="Appears in the official PDF report header and links this session to their history."
)

# ---------------------------------------------------------------
# ATHLETE HISTORY (Supabase) — real DB, explicit error if not configured
# ---------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("📊 Athlete History")
history_enabled = True
try:
    store.get_client()
except RuntimeError as e:
    history_enabled = False
    st.sidebar.warning(f"History disabled: {e}")

if history_enabled:
    with st.sidebar.expander("View past sessions", expanded=False):
        try:
            athletes = store.list_athletes(st.session_state.auth_user["id"])
        except Exception as e:
            monitoring.capture(e)
            athletes = []
            st.error(f"Could not load athletes: {e}")

        if athletes:
            names = [a["name"] for a in athletes]
            selected = st.selectbox("Select athlete", names, key="history_athlete_select")
            selected_id = next(a["id"] for a in athletes if a["name"] == selected)
            try:
                history = store.get_athlete_history(selected_id, st.session_state.auth_user["id"])
            except Exception as e:
                monitoring.capture(e)
                history = []
                st.error(f"Could not load history: {e}")

            if history:
                for s in history:
                    date_str = s.get("session_date", "")[:10]
                    speed = s.get("release_arm_speed_kmh")
                    speed_str = f"{speed} km/h" if speed else "n/a"
                    st.markdown(f"**{date_str}** — {s.get('camera_mode', '?')} — speed: {speed_str}")
            else:
                st.caption("No sessions recorded yet for this athlete.")
        else:
            st.caption("No athlete profiles yet. Run an analysis to create one.")

st.sidebar.divider()

# ---------------------------------------------------------------
# CAMERA CALIBRATION — real two-point calibration, no assumed constants
# ---------------------------------------------------------------
# Rendered in the MAIN content area (not the sidebar), same fix and same
# reason as the bowler-click picker: the sidebar is a narrow, fixed-width
# column on desktop browsers, which made it hard to click precisely on
# the top/bottom of a stump. The main area uses the full page width
# (layout="wide" is set above).
st.header("📏 Speed Calibration")
if "calibration" not in st.session_state:
    st.session_state.calibration = None

with st.expander("Calibrate camera for speed (once per setup)", expanded=False):
    st.caption(
        "This is a ONE-TIME setup per fixed camera position — not per delivery. "
        "Upload any clip from that camera spot (can be a dedicated short clip of "
        "just the stumps, doesn't need to be an actual delivery), scrub to a frame "
        "where your reference points are visible, then click both directly on the "
        "image below."
    )

    # PRESET REFERENCE DISTANCES — guided, foolproof calibration instead of a
    # generic "click two points" prompt. Modeled on a competitor's calibration
    # screen (two labeled stump-alignment guides), adapted for this app's
    # pre-recorded-video workflow rather than a live camera feed. The full-
    # pitch preset also matters for more than just guidance: calibrating
    # against the popping-crease-to-popping-crease distance (a known 20.12m)
    # uses a MUCH longer pixel baseline than a single stump's 0.2286m width,
    # which is inherently more precise (small pixel-click error matters far
    # less as a fraction of a longer real-world distance) — and it's the
    # same full-pitch calibration Phase 2 ball-tracking will eventually need
    # for pitch mapping, so setting it up now is not wasted effort.
    CALIBRATION_PRESETS = {
        "Stump width (0.2286m) — single stump set close-up": {
            "distance_m": 0.2286, "label": "stump width",
            "point1_prompt": "one edge of the stumps",
            "point2_prompt": "the other edge of the stumps",
        },
        "Popping crease to popping crease (20.12m) — full pitch in frame": {
            "distance_m": 20.12, "label": "popping crease to popping crease",
            "point1_prompt": "the STRIKER END stumps",
            "point2_prompt": "the NON-STRIKER END stumps",
        },
        "Custom distance": {
            "distance_m": None, "label": "custom",
            "point1_prompt": "the first reference point",
            "point2_prompt": "the second reference point",
        },
    }
    calib_preset_choice = st.selectbox(
        "What are you calibrating against?", list(CALIBRATION_PRESETS.keys()),
        key="calib_preset_choice"
    )
    calib_preset = CALIBRATION_PRESETS[calib_preset_choice]

    calib_video = st.file_uploader("Reference video/frame source (.mp4 or .mov)", type=["mp4", "mov", "m4v"], key="calib_video")

    if "calib_points" not in st.session_state:
        st.session_state.calib_points = []

    if calib_video is not None:
        temp_path = os.path.join("input", "calibration_ref.mp4")
        with open(temp_path, "wb") as f:
            f.write(calib_video.getbuffer())

        total_frames = cal.get_frame_count(temp_path)
        if total_frames > 1:
            _render_frame_jump_box("calib_frame_idx", 0, max(total_frames - 1, 0))
            frame_idx = st.slider(
                "Scrub to a frame where your reference points (e.g. stumps) are clearly visible",
                min_value=0, max_value=max(total_frames - 1, 0),
                value=min(total_frames - 1, total_frames // 2),
                key="calib_frame_idx"
            )
        else:
            frame_idx = 0

        frame = cal.extract_reference_frame(temp_path, frame_index=frame_idx)

        if st.session_state.get("_calib_last_frame_idx") != frame_idx:
            st.session_state.calib_points = []
            st.session_state["_calib_last_frame_idx"] = frame_idx

        # Widgets keyed "cal_dist"/"cal_label" keep whatever value the coach
        # last set, even across reruns — changing the preset selector alone
        # wouldn't update the pre-filled number/label unless that stored
        # state is cleared here, the same reset pattern used above for a
        # changed frame.
        if st.session_state.get("_calib_last_preset") != calib_preset_choice:
            st.session_state.pop("cal_dist", None)
            st.session_state.pop("cal_label", None)
            st.session_state["_calib_last_preset"] = calib_preset_choice

        if frame is not None:
            from PIL import Image

            pil_img = Image.fromarray(frame)

            if len(st.session_state.calib_points) < 2:
                which_point = (calib_preset["point1_prompt"] if len(st.session_state.calib_points) == 0
                               else calib_preset["point2_prompt"])
                st.caption(f"📍 Click **{which_point}** on the image below.")
            else:
                st.caption("✅ Both points selected — see below.")

            existing_markers = [
                {"point": pt, "color": "red", "label": str(i + 1)}
                for i, pt in enumerate(st.session_state.calib_points)
            ]
            new_point = render_zoomable_click_image(
                pil_img, key_prefix="calib", extra_markers=existing_markers,
            )

            if new_point is not None and len(st.session_state.calib_points) < 2:
                if not st.session_state.calib_points or st.session_state.calib_points[-1] != new_point:
                    st.session_state.calib_points.append(new_point)
                    st.rerun()

            if st.button("↺ Reset points", key="reset_calib_points"):
                st.session_state.calib_points = []
                st.rerun()

            if len(st.session_state.calib_points) == 2:
                default_dist = calib_preset["distance_m"] if calib_preset["distance_m"] is not None else 0.2286
                real_dist = st.number_input(
                    "Real-world distance between the two points you clicked (meters)",
                    min_value=0.0, value=default_dist, step=0.01, key="cal_dist",
                    help="Pre-filled from your selection above — adjust if you clicked a different reference."
                )
                ref_label = st.text_input("Reference label (e.g. 'stump width')",
                                           value=calib_preset["label"], key="cal_label")
                if st.button("Compute calibration"):
                    try:
                        calibration = cal.compute_scale(
                            st.session_state.calib_points[0],
                            st.session_state.calib_points[1],
                            real_dist, ref_label or "custom"
                        )
                        st.session_state.calibration = calibration
                        st.session_state.calib_points = []
                        st.success(f"Calibrated: {calibration.meters_per_pixel:.6f} m/px")
                        warning = cal.implausibility_warning(calibration, pil_img.width)
                        if warning:
                            st.warning(warning)
                    except ValueError as e:
                        st.error(str(e))
        else:
            st.error("Could not read a frame from that video.")

    if st.session_state.calibration:
        c = st.session_state.calibration
        st.info(f"Active calibration: {c.reference_label} ({c.reference_distance_m}m) — "
                f"{c.meters_per_pixel:.6f} m/px")
        if st.button("Clear calibration"):
            st.session_state.calibration = None

st.sidebar.header("🎥 Camera Mode")
camera_mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["Single Camera", "Dual Camera — Recommended"],
    help=(
        "Dual camera gives highest accuracy.\n"
        "Side-on: knee angle, trunk lean, release height.\n"
        "Rear-view: hip-shoulder separation, head stability."
    )
)

st.sidebar.divider()
# FIX: was defaulting to "Auto-detect (recommended)" — proven unreliable
# twice on real footage in this project (most recently: silently locked
# onto the WRONG arm for a left-arm bowler, producing a release-frame
# detection deep in follow-through with a knee angle that matched exactly
# what a coach saw on a garbage result). Auto-detect is kept as an
# option for convenience, but it no longer pre-selects itself or claims
# to be "recommended" — a coach must now actively choose Left or Right,
# which is the one setting proven to make this pipeline reliable.
bowling_arm_choice = st.sidebar.selectbox(
    "🎯 Bowling Arm (required)",
    ["-- Select bowling arm --", "Right-arm", "Left-arm", "Auto-detect (unreliable)"],
    help="Auto-detect has been wrong on real footage — always set this manually for a trustworthy result."
)
bowling_arm_override = {
    "-- Select bowling arm --": None,
    "Auto-detect (unreliable)": None,
    "Right-arm": "right",
    "Left-arm": "left",
}[bowling_arm_choice]
bowling_arm_selected = bowling_arm_choice in ("Right-arm", "Left-arm")
if bowling_arm_choice == "Auto-detect (unreliable)":
    st.sidebar.warning(
        "⚠️ Auto-detect has produced wrong-arm results on real footage. "
        "Set Right-arm or Left-arm explicitly for a result you can trust."
    )

# CAMERA ANGLE (optional manual override) — the geometry-based auto-detect
# (shoulder-width/height ratio) has been verified to misclassify some real
# footage: the same genuinely side-on setup produced ratios ranging from
# clearly-side-on to clearly-front-or-rear across different frames of the
# SAME clip, depending on the bowler's incidental running pose. That
# misclassification silently disables a release-detection check that only
# applies to side-on footage, producing a release frame deep in the wrong
# part of the delivery. If you know the filming angle, set it here rather
# than trusting auto-detect.
camera_angle_choice = st.sidebar.selectbox(
    "📐 Filming Angle",
    ["Auto-detect", "Side-on", "Rear-view / Front-on"],
    help="Auto-detect can misjudge this on some footage — set it manually if you know the angle."
)
camera_angle_override = {
    "Auto-detect": None,
    "Side-on": "side_on",
    "Rear-view / Front-on": "front_or_rear",
}[camera_angle_choice]


def _reject_if_empty_upload(uploaded_file):
    """
    Real failure mode found from a coach's screen recording + device
    testing across iPhone/Samsung vs. Pixel/Infinix: iCloud "Optimize
    Storage" and Samsung Cloud both keep only a low-res stub on-device
    for videos backed up to the cloud, silently fetching the real file
    only when something tries to read it. Pixel/Infinix don't do this
    by default — matching why only iPhone/Samsung ever showed the
    problem. A stub handed to the browser before it's finished
    resolving can surface here as a 0-byte (or near-zero) upload rather
    than the picker failing to attach anything at all. Catch it with a
    specific, actionable message instead of letting it fail silently
    downstream.
    """
    if uploaded_file is not None and uploaded_file.size < 1024:
        st.sidebar.error(
            "⚠️ This video came through as (almost) empty. If it's backed up to "
            "iCloud or Samsung Cloud, your phone may not have downloaded the full "
            "file yet — open it once in your Gallery/Photos app until it fully "
            "plays, then upload it again."
        )
        return True
    return False


st.sidebar.divider()
st.sidebar.header("📁 Upload Video")
uploaded_side = None
uploaded_rear = None
uploaded_single = None

if camera_mode == "Single Camera":
    uploaded_single = st.sidebar.file_uploader("Bowling Video (.mp4 or .mov)", type=["mp4", "mov", "m4v"])
    st.sidebar.caption(
        "📱 Video not attaching after a few tries? If it's backed up to iCloud/"
        "Samsung Cloud, open it fully in your Gallery app first, then retry here."
    )
    if _reject_if_empty_upload(uploaded_single):
        uploaded_single = None
else:
    st.sidebar.info("Upload both angles for maximum accuracy. Events are detected independently on each stream.")
    uploaded_side = st.sidebar.file_uploader("📹 Side-On Video (.mp4 or .mov)", type=["mp4", "mov", "m4v"], key="side")
    uploaded_rear = st.sidebar.file_uploader("📹 Rear-View Video (.mp4 or .mov)", type=["mp4", "mov", "m4v"], key="rear")
    st.sidebar.caption(
        "📱 Video not attaching after a few tries? If it's backed up to iCloud/"
        "Samsung Cloud, open it fully in your Gallery app first, then retry here."
    )
    if _reject_if_empty_upload(uploaded_side):
        uploaded_side = None
    if _reject_if_empty_upload(uploaded_rear):
        uploaded_rear = None


def render_bowler_seed_ui(uploaded_file, key_prefix: str, label: str, save_key: str = None):
    """
    One-time-per-video step: save the upload, show a scrubbable reference
    frame, and have the coach click directly on the bowler. Returns
    ((x_px, y_px), frame_index) once clicked, else (None, 0).

    This is the credible fix for the "skeleton locks onto the wrong
    person" risk (documented at length in main.py) — instead of another
    automatic guess, the coach tells the app who to track, once, and
    main.py's seeded tracker just follows whoever's torso stays closest
    to that identity frame to frame.

    save_key: the STREAM identifier ("single"/"side"/"rear") the on-disk
    reference copy is saved under — defaults to key_prefix, which is
    correct for the primary seed call. BUG FIX (reported directly as a
    real, noticeable loading delay between seed-confirmation steps):
    render_extra_seed_ui calls this once per extra slot with a DIFFERENT
    key_prefix each time (…_extra0, _extra1, _extra2), and every one of
    those was writing its OWN full copy of the SAME uploaded video to
    disk — up to 4 byte-for-byte-identical copies of a possibly
    multi-MB file, purely from opening a slot that hadn't even been
    clicked in yet. Passing the stream's real save_key here means every
    slot for the same video reads/writes exactly one shared file, while
    each slot still keeps its OWN independent click/point state via
    key_prefix — reset in slot 2 can never affect slot 3's marker.
    """
    if uploaded_file is None:
        return None, 0

    save_key = save_key or key_prefix
    file_identity = f"{uploaded_file.name}_{uploaded_file.size}"
    point_key = f"{key_prefix}_seed_point"
    frame_key = f"{key_prefix}_seed_frame"
    identity_key = f"{key_prefix}_seed_identity"
    shared_ref_identity_key = f"_{save_key}_shared_seed_ref_identity"
    # Deliberately the SAME name the old key_prefix-scoped version used
    # (e.g. "single_seed_ref_path") — several places further down the
    # file read this directly by that exact string (single_ref_path,
    # side_ref_path, rear_ref_path), all keyed only by the stream's
    # save_key, never by an extra-seed slot's key_prefix. Keeping the
    # identical name here means every one of those call sites keeps
    # working with zero changes, instead of silently reading a stale/
    # missing key the moment this got scoped to save_key.
    shared_ref_path_key = f"{save_key}_seed_ref_path"

    if st.session_state.get(shared_ref_identity_key) != file_identity:
        # New file uploaded (or first time this save_key has seen it) —
        # cache ONE temp copy on disk, shared by every seed slot for this
        # same stream, so we can pull reference frames from it.
        os.makedirs("input", exist_ok=True)
        ref_path = os.path.abspath(os.path.join("input", f"_seed_ref_{save_key}_{uploaded_file.name}"))
        try:
            o.save_uploaded_video_capped(uploaded_file, ref_path)
        except RuntimeError as e:
            st.error(f"⚠️ {e}")
            st.stop()
        st.session_state[shared_ref_identity_key] = file_identity
        st.session_state[shared_ref_path_key] = ref_path

    if st.session_state.get(identity_key) != file_identity:
        # Still per-slot: a genuinely new file must reset THIS slot's own
        # click state, independent of whether the shared copy above was
        # just (re)written or already existed from an earlier slot.
        st.session_state[identity_key] = file_identity
        st.session_state[point_key] = None
        st.session_state[frame_key] = 0

    ref_path = st.session_state[shared_ref_path_key]
    total_frames = cal.get_frame_count(ref_path)

    # Rendered in the MAIN content area (not the sidebar) — the sidebar is
    # a narrow, fixed-width column on desktop browsers, which was
    # squeezing a wide cricket video frame down to a size too small to
    # click accurately. The main area uses the full page width
    # (layout="wide" is set above), giving a much larger, clearer image.
    with st.expander(f"🎯 Confirm the bowler — {label}", expanded=st.session_state.get(point_key) is None):
        st.caption(
            "Scrub to any frame where the bowler is clearly visible, then click "
            "directly on him. This tells the app exactly who to track for the "
            "whole clip, instead of guessing — the single most reliable fix for "
            "the skeleton ever locking onto the wrong person."
        )
        if total_frames > 1:
            _render_frame_jump_box(f"{key_prefix}_seed_slider", 0, max(total_frames - 1, 0))
            frame_idx = st.slider(
                "Scrub to a frame with the bowler visible",
                min_value=0, max_value=max(total_frames - 1, 0),
                value=min(total_frames - 1, total_frames // 3),
                key=f"{key_prefix}_seed_slider"
            )
        else:
            frame_idx = 0

        frame = cal.extract_reference_frame(ref_path, frame_index=frame_idx)

        if st.session_state.get(f"_{key_prefix}_last_frame_idx") != frame_idx:
            st.session_state[point_key] = None
            st.session_state[f"_{key_prefix}_last_frame_idx"] = frame_idx

        if frame is not None:
            from PIL import Image

            pil_img = Image.fromarray(frame)
            point = st.session_state.get(point_key)

            if point is None:
                st.caption("📍 Click directly on the bowler below.")
            else:
                st.caption("✅ Bowler confirmed — click again to move the marker.")

            new_point = render_zoomable_click_image(
                pil_img, key_prefix=f"{key_prefix}_seed", marker_point=point,
            )

            if new_point is not None and st.session_state.get(point_key) != new_point:
                st.session_state[point_key] = new_point
                st.session_state[frame_key] = frame_idx
                st.rerun()

            if st.session_state.get(point_key) is not None:
                if st.button("↺ Reset marker", key=f"{key_prefix}_reset_seed"):
                    st.session_state[point_key] = None
                    st.rerun()
        else:
            st.error("Could not read a frame from that video.")

    return st.session_state.get(point_key), st.session_state.get(frame_key, 0)


def render_extra_seed_ui(uploaded_file, key_prefix: str, label: str):
    """
    Optional additional confirmation points, spread across the clip.
    Exists for exactly the failure mode found on real footage: a single
    seed's "walk" gives up permanently once it can't find a matching
    candidate for MAX_GAP_FRAMES in a row (see main._walk_from_seed) —
    on a fast run-up or a clip where the bowler gets small/distant/
    blurry for a stretch, this can leave large parts of the clip with
    NO tracked skeleton at all, confirmed directly on real footage (a
    126-frame clip where only 37 frames — one contiguous window around
    the delivery — had any tracked landmarks; the rest of the run-up
    and follow-through were blank). Re-confirming identity at MULTIPLE
    points lets the tracker split the clip into more, shorter zones,
    each only needing to survive the (much shorter) gap to its nearest
    seed instead of one seed carrying the whole clip.

    Supports up to 3 extra confirmations (4 seeds total including the
    primary) — deliberately still a fixed small number of manual clicks,
    not a new automatic re-acquisition heuristic. This codebase already
    went through 9 commits of automatic identity-tracking heuristics
    that each fixed one real clip and broke another before landing on
    manual seeding as the reliable approach (see main.py's top-of-file
    history) — the fix for insufficient COVERAGE is more manual anchor
    points, not smarter automatic guessing.

    PROGRESSIVELY REVEALED, not all 3 shown at once (BUG FIX from direct
    coach feedback: "why do we have 4 expanders just to identify the
    bowler" — showing empty "add a 3rd"/"add a 4th" slots before they
    were ever needed was clutter for the large majority of clips where
    one extra confirmation is plenty). Only the 2nd slot shows by
    default; the 3rd only appears once the 2nd is actually filled in,
    and the 4th only once the 3rd is filled — so a coach who never needs
    more than one extra confirmation only ever sees ONE extra expander
    (two total, including the primary seed), while a genuinely hard clip
    can still escalate all the way to 4 seeds exactly as before.

    Returns a list of (frame_index, point) tuples for every extra
    confirmation the coach has added, or None if none were added.
    """
    if uploaded_file is None:
        return None
    extra_seeds = []
    ordinals = ["2nd", "3rd", "4th"]
    for i, ordinal in enumerate(ordinals):
        with st.expander(f"➕ Tracking lost partway through — add a {ordinal} confirmation ({label})",
                          expanded=(i == 0)):
            st.caption(
                "Only needed if the skeleton is missing or drifts onto someone else "
                "for part of this clip (fast run-up, bowler far from camera, or "
                "another player nearby). Scrub to a frame in that gap where the "
                "bowler is clearly visible and click him — same as the first "
                "confirmation above, just at a different point in the clip."
            )
            point, frame_idx = render_bowler_seed_ui(uploaded_file, f"{key_prefix}_extra{i}",
                                                       f"{label} — {ordinal} confirmation",
                                                       save_key=key_prefix)
            if point is not None:
                extra_seeds.append((frame_idx, point))
        # Reveal the NEXT slot only once this one is actually filled in —
        # keeps the common case (0 or 1 extra confirmation needed) down
        # to at most one extra expander instead of always showing all 3.
        if point is None:
            break
    return extra_seeds if extra_seeds else None


def seeds_ready_for_extraction(key_prefix: str, seed_point, seed_frame, extra_seeds: list) -> bool:
    """
    Gates the expensive extraction/event-detection stage behind an
    explicit confirmation from the coach — see
    click_widget_state.seed_confirmation_status for the real production
    crash that made this apply to EVERY seed configuration, not just
    when extra confirmations are added (a prior version of this
    docstring described the single-seed case as "the common case, no
    reported problem" — that's no longer true now that repeated
    extraction is confirmed to leak real memory, not just cost time).

    Returns True when it's safe to proceed with (seed_point, seed_frame,
    extra_seeds) exactly as currently set, because the coach explicitly
    confirmed this exact configuration is final. Returns False (and
    renders a "Continue" button) while still waiting on that
    confirmation — the caller should skip extraction/downstream UI
    entirely in that case.
    """
    if seed_point is None:
        # Nothing placed yet for this stream — BUG FOUND directly from a
        # coach screenshot: the side/rear calls below run unconditionally
        # (see the comment above them), so without this check, the
        # "confirm tracking point(s)" prompt rendered for BOTH streams
        # even in Single Camera mode with no video uploaded at all, since
        # side_seed_point/rear_seed_point default to None there and never
        # get set. Nothing to confirm yet means nothing to render.
        return True

    is_ready, pending_identity = click_widget_state.seed_confirmation_status(
        st.session_state, key_prefix, seed_point, seed_frame, extra_seeds
    )
    if is_ready:
        return True

    st.info(
        "📍 Happy with the marker placement? Add another confirmation above if "
        "tracking still needs it, or confirm below to continue — this waits for "
        "you to finish instead of re-running tracking after every click."
    )
    if st.button("✅ Confirm tracking point(s) — continue", key=f"{key_prefix}_seeds_continue"):
        click_widget_state.lock_seed_confirmation(st.session_state, key_prefix, pending_identity)
        st.rerun()
    return False


single_seed_point, single_seed_frame = (None, 0)
side_seed_point, side_seed_frame = (None, 0)
rear_seed_point, rear_seed_frame = (None, 0)
single_extra_seeds = None
side_extra_seeds = None
rear_extra_seeds = None

if camera_mode == "Single Camera":
    single_seed_point, single_seed_frame = render_bowler_seed_ui(uploaded_single, "single", "Bowling video")
    if single_seed_point is not None:
        single_extra_seeds = render_extra_seed_ui(uploaded_single, "single", "Bowling video")
else:
    side_seed_point, side_seed_frame = render_bowler_seed_ui(uploaded_side, "side", "Side-on video")
    rear_seed_point, rear_seed_frame = render_bowler_seed_ui(uploaded_rear, "rear", "Rear-view video")
    if side_seed_point is not None:
        side_extra_seeds = render_extra_seed_ui(uploaded_side, "side", "Side-on video")
    if rear_seed_point is not None:
        rear_extra_seeds = render_extra_seed_ui(uploaded_rear, "rear", "Rear-view video")

# CAMERA ANGLE — confirmed UPFRONT, before Execute, not after the analysis
# has already run with a guess. Only for Single Camera: Dual Camera already
# knows its two streams' angles by construction (side-on + rear/front),
# no ambiguity to resolve.
confirmed_angle_functional = None   # "side_on" | "front_or_rear" — feeds the actual computation
confirmed_angle_label = None        # "side_on" | "front" | "rear" | "unknown" — for captions + logging

if (camera_mode == "Single Camera" and uploaded_single is not None and single_seed_point is not None
        and seeds_ready_for_extraction("single", single_seed_point, single_seed_frame, single_extra_seeds)):
    single_ref_path = st.session_state.get("single_seed_ref_path")
    # Includes bowling-arm, seed choices, AND the sidebar angle override —
    # any of these changing (coach corrects the arm, adds a second seed,
    # flips the angle dropdown) means the extraction/events below are
    # stale and must rerun, not just be reused because the file itself
    # is unchanged.
    single_file_identity = (
        f"{uploaded_single.name}_{uploaded_single.size}_{bowling_arm_override}"
        f"_{single_seed_point}_{single_seed_frame}_{single_extra_seeds}_{camera_angle_override}"
    )

    # Runs the REAL extraction+event-detection stage now (cached per
    # file — this is real work, not free) instead of a cheap isolated-
    # frame heuristic. Verified directly that a lightweight shortcut
    # here (single-frame detection, or sampling early run-up frames)
    # is unreliable on real footage: isolated frames often fail to
    # detect a small/distant bowler at all, and early run-up frames
    # can show a misleadingly rotated torso if the bowler curves into
    # his approach. The full extraction already needs to run before
    # Execute's metrics/rendering stage anyway — doing it here just
    # means the angle/release-frame questions get asked right after THIS
    # (cheaper) stage instead of after the full pipeline including video
    # rendering has already run once with a guess. Runs UNCONDITIONALLY
    # now (previously skipped entirely when the coach set a manual angle
    # override in the sidebar) — BUG FIX: the release-frame confirmation
    # below needs stage12_result/single_ref_path regardless of which
    # angle path is taken; skipping this for the manual-override branch
    # left those referenced before assignment, crashing with a NameError
    # the moment a coach used the sidebar override on real (Streamlit
    # Cloud) traffic.
    if st.session_state.get("_stage12_identity") != single_file_identity:
        with st.spinner("📐 Extracting tracking data and checking filming angle..."):
            st.session_state["_stage12_result"] = o.extract_and_detect_events(
                single_ref_path, output_dir="output",
                bowling_arm_override=bowling_arm_override,
                seed_point=single_seed_point, seed_frame_index=single_seed_frame,
                extra_seeds=single_extra_seeds,
                camera_angle_override=camera_angle_override,
            )
        st.session_state["_stage12_identity"] = single_file_identity
        st.session_state["_angle_confirmed_choice"] = None

    stage12_result = st.session_state.get("_stage12_result")

    if camera_angle_override is not None:
        # Coach already told us via the sidebar — trust that over any guess.
        # Only a real "Not sure" (no override given, and the upfront radio
        # left unconfirmed) should count as unresolved.
        confirmed_angle_functional = camera_angle_override
        confirmed_angle_label = "side_on" if camera_angle_override == "side_on" else "front_or_rear_manual"
    else:
        angle_estimate = stage12_result.get("angle_estimate") if stage12_result and stage12_result.get("status") == "success" else None

        choice_labels = ["Side-on", "Rear-view (behind bowler)", "Front-on (facing bowler)", "Not sure"]
        default_idx = 3
        if angle_estimate is not None and angle_estimate.angle == "side_on":
            default_idx = 0

        with st.expander("📐 Filming Angle Check",
                          expanded=st.session_state.get("_angle_confirmed_choice") is None):
            if stage12_result is None or stage12_result.get("status") != "success":
                st.error(stage12_result.get("message", "Tracking extraction failed.") if stage12_result else "Tracking extraction failed.")
            elif angle_estimate is None or angle_estimate.angle == "unavailable":
                st.info("Couldn't auto-detect the filming angle from this clip — please confirm below. "
                        "This changes what Trunk Lean and Head Stability actually measure.")
            elif angle_estimate.angle == "side_on":
                st.success(f"Detected: **side-on** (shoulder ratio {angle_estimate.ratio}) — "
                           f"the best-supported angle for all 5 metrics. Confirm or correct below.")
            else:
                st.warning(f"📐 {angle_estimate.confidence_note} (shoulder-width ratio: {angle_estimate.ratio}) "
                           f"— front and rear look nearly identical from pose data alone, so please "
                           f"confirm which one this actually is.")

            angle_choice = st.radio(
                "Confirm the filming angle for this video",
                choice_labels, index=default_idx,
                key="angle_confirm_radio_upfront", horizontal=True
            )
            st.session_state["_angle_confirmed_choice"] = angle_choice

        confirmed_angle_label = {
            "Side-on": "side_on", "Rear-view (behind bowler)": "rear",
            "Front-on (facing bowler)": "front", "Not sure": "unknown",
        }[st.session_state["_angle_confirmed_choice"]]
        confirmed_angle_functional = "side_on" if confirmed_angle_label == "side_on" else "front_or_rear"

    # BALL RELEASE FRAME — CONFIRMED BY THE COACH, ALWAYS, not just when
    # confidence looks low. Verified directly on real footage (a leaping
    # delivery filmed rear-view): the auto-detector reported "high"
    # confidence (1.0 plausible fraction) while landing 38 frames early,
    # on an ordinary running stride instead of the real release swing —
    # every frame it looked at really was genuinely tracked, just from
    # the wrong part of the delivery, which is exactly the kind of error
    # this confidence score cannot see. Since it's proven unreliable as a
    # signal for when to skip confirmation, this step is never skipped:
    # a human directly watching the footage cannot make that mistake.
    # This also fixes release height's dependence on separately-timed
    # front-foot-plant detection (see _find_grounded_reference_near) —
    # once release is a verified anchor, the "grounded reference" search
    # centers on it instead of an independently-guessed plant frame.
    if st.session_state.get("_br_confirmed_identity") != single_file_identity:
        st.session_state["_br_confirmed_frame"] = None
        st.session_state["_br_confirmed_identity"] = single_file_identity

    br_auto = None
    br_confidence = None
    if stage12_result is not None and stage12_result.get("status") == "success":
        br_auto = stage12_result["events"].get("BR")
        br_confidence = stage12_result["events"].get("BR_confidence")

    with st.expander("🎯 Confirm Ball Release Frame",
                      expanded=st.session_state.get("_br_confirmed_frame") is None):
        if br_auto is None:
            st.error("Couldn't detect a release frame at all — check tracking above.")
        else:
            conf_note = {"high": "high confidence", "low": "low confidence"}.get(br_confidence, "unknown confidence")
            st.info(
                f"Algorithm's best guess: **frame {br_auto}** ({conf_note}). Scrub to the exact "
                f"frame where the ball actually leaves the hand and confirm — this feeds Release "
                f"Height, and every metric that depends on it, directly. Auto-detection can be "
                f"wrong even when it reports high confidence, so this step always runs."
            )
            total_frames_single = cal.get_frame_count(single_ref_path)
            _render_frame_jump_box("br_confirm_slider", 0, max(total_frames_single - 1, 0))
            br_slider_val = st.slider(
                "Scrub to the true ball-release frame",
                min_value=0, max_value=max(total_frames_single - 1, 0),
                value=min(max(br_auto, 0), max(total_frames_single - 1, 0)),
                key="br_confirm_slider"
            )
            br_frame_img = cal.extract_reference_frame(single_ref_path, frame_index=br_slider_val)
            if br_frame_img is not None:
                with _framed_image_container():
                    st.image(br_frame_img, use_column_width=True,
                              caption=f"Frame {br_slider_val} — is the ball leaving the hand here?")
            if st.button("✅ Confirm this is the release frame", key="confirm_br_button"):
                st.session_state["_br_confirmed_frame"] = br_slider_val
                st.rerun()
            if st.session_state.get("_br_confirmed_frame") is not None:
                st.success(f"Confirmed: release at frame {st.session_state['_br_confirmed_frame']}.")

    # RELEASE POINT (wrist/ball position) — optional correction, separate
    # from WHICH FRAME is release. Verified directly on real footage: even
    # on the correct frame, MediaPipe can systematically under-track how
    # far the hand extends during a fast, blurred swing — not a one-frame
    # glitch (shoulder-to-wrist distance grew smoothly right through the
    # bad reading, no anomaly for a filter to catch), so no automatic check
    # can fix it. MANDATORY, not optional (BUG FIX from direct coach
    # feedback: this used to be a skippable "(optional)" panel — same
    # reasoning as why BFC/FFC/BR are never skippable either, now also
    # gated into single_ready below).
    confirmed_br_frame = st.session_state.get("_br_confirmed_frame")
    if st.session_state.get("_wrist_step_identity") != single_file_identity:
        st.session_state["_wrist_step_confirmed"] = False
        st.session_state["_wrist_step_identity"] = single_file_identity
    if confirmed_br_frame is not None:
        wrist_identity = f"{single_file_identity}_{confirmed_br_frame}"
        if st.session_state.get("_wrist_identity") != wrist_identity:
            st.session_state["_wrist_confirmed_point"] = None
            st.session_state["_wrist_identity"] = wrist_identity
            st.session_state["_wrist_step_confirmed"] = False

        with st.expander("🖐️ Confirm Release Point",
                          expanded=not st.session_state.get("_wrist_step_confirmed", False)):
            if br_confidence == "low":
                st.warning(
                    "Auto-detection reported **low confidence** on this release frame. Verified on real "
                    "footage: this specific signal correlates with the wrist/hand tracking silently "
                    "undershooting the real arm extension — the drawn Release Height line can look "
                    "plausible while still being wrong. Please check the marker below carefully before "
                    "confirming."
                )
            st.caption(
                "The yellow marker is the auto-tracked ball/hand position at your "
                "confirmed release frame. If it doesn't sit on the real ball/hand — "
                "common during fast, motion-blurred swings — click the real position "
                "to correct it, then confirm. This feeds Release Height directly."
            )
            wrist_frame_img = cal.extract_reference_frame(single_ref_path, frame_index=confirmed_br_frame)
            if wrist_frame_img is not None:
                from PIL import Image

                pil_img = Image.fromarray(wrist_frame_img)
                orig_w, orig_h = pil_img.size

                auto_point = None
                stage12_df = stage12_result.get("df") if stage12_result else None
                bowl_side_for_wrist = "RIGHT" if bowling_arm_override == "right" else "LEFT"
                if stage12_df is not None:
                    wrist_rows = stage12_df[stage12_df["frame"] == confirmed_br_frame]
                    if not wrist_rows.empty:
                        wx = wrist_rows.iloc[0].get(f"{bowl_side_for_wrist}_WRIST_x")
                        wy = wrist_rows.iloc[0].get(f"{bowl_side_for_wrist}_WRIST_y")
                        if wx is not None and wy is not None and not pd.isna(wx) and not pd.isna(wy):
                            auto_point = (round(wx * orig_w), round(wy * orig_h))

                corrected_point = st.session_state.get("_wrist_confirmed_point")
                display_point = corrected_point or auto_point

                if corrected_point is None:
                    st.caption("🟡 Auto-tracked position shown. Zoom in and click the image below to correct it if wrong.")
                else:
                    st.caption("✅ Corrected — click again to move the marker, or reset below.")

                new_point = render_zoomable_click_image(
                    pil_img, key_prefix="single_wrist", marker_point=display_point,
                    marker_color=("lime" if corrected_point is not None else "yellow"),
                    enable_zoom=True,
                )
                if new_point is not None and st.session_state.get("_wrist_confirmed_point") != new_point:
                    st.session_state["_wrist_confirmed_point"] = new_point
                    st.rerun()

                if corrected_point is not None and st.button("↺ Reset to auto-tracked position", key="reset_wrist_point"):
                    st.session_state["_wrist_confirmed_point"] = None
                    st.rerun()

            if st.button("✅ Confirm this release point", key="confirm_wrist_button"):
                st.session_state["_wrist_step_confirmed"] = True
                st.rerun()
            if st.session_state.get("_wrist_step_confirmed"):
                st.success("Release point confirmed.")

    # FRONT FOOT CONTACT FRAME — same reasoning, same mandatory pattern,
    # for a real, separate bug: Hip-Shoulder Separation and the FFC-frame
    # Knee Bracing value are measured AT front-foot-plant specifically
    # (a genuinely different, earlier moment than release, with its own
    # coaching meaning — how much rotation has been built up BEFORE
    # release, not at it) — so confirming release alone doesn't fix
    # them. Verified directly on real footage (this same leaping
    # rear-view delivery): auto-detection landed FFC at frame 87, an
    # ordinary mid-run-up running stride, nowhere near the crease —
    # the coach's own frame-by-frame review placed the real plant at
    # 147-148, just 2-3 frames before release. For a bowler with no
    # clean "foot stops moving" moment (see the FFC/BR docstrings
    # elsewhere), auto-detection can miss this by a huge margin, and a
    # human watching the footage cannot make that mistake.
    if st.session_state.get("_ffc_confirmed_identity") != single_file_identity:
        st.session_state["_ffc_confirmed_frame"] = None
        st.session_state["_ffc_confirmed_identity"] = single_file_identity

    ffc_auto = None
    if stage12_result is not None and stage12_result.get("status") == "success":
        ffc_auto = stage12_result["events"].get("FFC")

    with st.expander("🦶 Confirm Front Foot Contact Frame",
                      expanded=st.session_state.get("_ffc_confirmed_frame") is None):
        if ffc_auto is None:
            st.error("Couldn't detect a front-foot-contact frame at all — check tracking above.")
        else:
            st.info(
                f"Algorithm's best guess: **frame {ffc_auto}**. Scrub to the exact frame where "
                f"the front (lead) foot first plants on the ground and confirm — this feeds Hip-"
                f"Shoulder Separation and the pre-release Knee Bracing reading directly. This is "
                f"usually shortly BEFORE the release frame you just confirmed, not necessarily "
                f"far earlier in the run-up."
            )
            total_frames_ffc = cal.get_frame_count(single_ref_path)
            _render_frame_jump_box("ffc_confirm_slider", 0, max(total_frames_ffc - 1, 0))
            ffc_slider_val = st.slider(
                "Scrub to the true front-foot-contact frame",
                min_value=0, max_value=max(total_frames_ffc - 1, 0),
                value=min(max(ffc_auto, 0), max(total_frames_ffc - 1, 0)),
                key="ffc_confirm_slider"
            )
            ffc_frame_img = cal.extract_reference_frame(single_ref_path, frame_index=ffc_slider_val)
            if ffc_frame_img is not None:
                with _framed_image_container():
                    st.image(ffc_frame_img, use_column_width=True,
                              caption=f"Frame {ffc_slider_val} — has the front foot just planted here?")
            if st.button("✅ Confirm this is the front-foot-contact frame", key="confirm_ffc_button"):
                st.session_state["_ffc_confirmed_frame"] = ffc_slider_val
                st.rerun()
            if st.session_state.get("_ffc_confirmed_frame") is not None:
                st.success(f"Confirmed: front-foot contact at frame {st.session_state['_ffc_confirmed_frame']}.")

    # BACK FOOT CONTACT FRAME — completes the set. Only feeds Head
    # Stability's measurement window (BFC to BR) — a smaller blast radius
    # than FFC/BR, but the same failure mode was confirmed on this exact
    # clip: auto-detection landed BFC at frame 73, while he's still near
    # his mark starting the run-up, not anywhere close to the real back-
    # foot-contact of the delivery stride (which should land shortly
    # before the confirmed FFC, not 70+ frames earlier).
    if st.session_state.get("_bfc_confirmed_identity") != single_file_identity:
        st.session_state["_bfc_confirmed_frame"] = None
        st.session_state["_bfc_confirmed_identity"] = single_file_identity

    bfc_auto = None
    if stage12_result is not None and stage12_result.get("status") == "success":
        bfc_auto = stage12_result["events"].get("BFC")

    with st.expander("👟 Confirm Back Foot Contact Frame",
                      expanded=st.session_state.get("_bfc_confirmed_frame") is None):
        if bfc_auto is None:
            st.error("Couldn't detect a back-foot-contact frame at all — check tracking above.")
        else:
            st.info(
                f"Algorithm's best guess: **frame {bfc_auto}**. Scrub to the frame where the back "
                f"(rear) foot plants just before the final delivery stride — this feeds Head "
                f"Stability's measurement window. Usually just a few frames before the front-foot "
                f"contact you just confirmed, not far back in the run-up."
            )
            total_frames_bfc = cal.get_frame_count(single_ref_path)
            _render_frame_jump_box("bfc_confirm_slider", 0, max(total_frames_bfc - 1, 0))
            bfc_slider_val = st.slider(
                "Scrub to the true back-foot-contact frame",
                min_value=0, max_value=max(total_frames_bfc - 1, 0),
                value=min(max(bfc_auto, 0), max(total_frames_bfc - 1, 0)),
                key="bfc_confirm_slider"
            )
            bfc_frame_img = cal.extract_reference_frame(single_ref_path, frame_index=bfc_slider_val)
            if bfc_frame_img is not None:
                with _framed_image_container():
                    st.image(bfc_frame_img, use_column_width=True,
                              caption=f"Frame {bfc_slider_val} — has the back foot just planted here?")
            if st.button("✅ Confirm this is the back-foot-contact frame", key="confirm_bfc_button"):
                st.session_state["_bfc_confirmed_frame"] = bfc_slider_val
                st.rerun()
            if st.session_state.get("_bfc_confirmed_frame") is not None:
                st.success(f"Confirmed: back-foot contact at frame {st.session_state['_bfc_confirmed_frame']}.")

br_resolved = (camera_mode != "Single Camera") or (st.session_state.get("_br_confirmed_frame") is not None)
ffc_resolved = (camera_mode != "Single Camera") or (st.session_state.get("_ffc_confirmed_frame") is not None)
bfc_resolved = (camera_mode != "Single Camera") or (st.session_state.get("_bfc_confirmed_frame") is not None)
wrist_resolved = (camera_mode != "Single Camera") or st.session_state.get("_wrist_step_confirmed", False)

# DUAL CAMERA — EVENT CONFIRMATION, same mandatory pattern as Single Camera
# above (BUG FOUND during a full app audit: Dual Camera was going straight
# from upload to fully-automatic BFC/FFC/BR detection with NO human review
# at all, while being labeled "Recommended" — every auto-detection failure
# mode this session found and fixed for Single Camera was fully exposed,
# unmitigated, here). Each stream is its own independently-filmed video
# with its own timeline, so each gets its own stage1+2 extraction and its
# own confirmation flow — reusing render_stream_event_confirmation defined
# above rather than duplicating Single Camera's inline code a third time.
side_stage12_result = None
rear_stage12_result = None
side_confirmed_events = None
rear_confirmed_events = None

# Evaluated as two separate statements, NOT chained with `and` in the
# if-condition below — Python's and short-circuits, which would mean
# the rear stream's own "done adding seeds?" prompt never even renders
# whenever the side stream still has one pending. Each stream needs its
# continue-button shown independently of the other's state.
side_seeds_ready = seeds_ready_for_extraction("side", side_seed_point, side_seed_frame, side_extra_seeds)
rear_seeds_ready = seeds_ready_for_extraction("rear", rear_seed_point, rear_seed_frame, rear_extra_seeds)

if (camera_mode != "Single Camera" and uploaded_side is not None and uploaded_rear is not None
        and side_seed_point is not None and rear_seed_point is not None
        and side_seeds_ready and rear_seeds_ready):
    side_ref_path = st.session_state.get("side_seed_ref_path")
    rear_ref_path = st.session_state.get("rear_seed_ref_path")

    side_file_identity = (
        f"{uploaded_side.name}_{uploaded_side.size}_{bowling_arm_override}"
        f"_{side_seed_point}_{side_seed_frame}_{side_extra_seeds}"
    )
    rear_file_identity = (
        f"{uploaded_rear.name}_{uploaded_rear.size}_{bowling_arm_override}"
        f"_{rear_seed_point}_{rear_seed_frame}_{rear_extra_seeds}"
    )

    # Fixed camera_angle_override per stream — Dual Camera knows each
    # stream's angle by construction (the side upload IS side-on, the rear
    # upload IS front/rear), same reasoning as the comment above on why
    # angle_resolved skips Dual Camera entirely.
    if st.session_state.get("_side_stage12_identity") != side_file_identity:
        with st.spinner("📐 Extracting side-on tracking data..."):
            st.session_state["_side_stage12_result"] = o.extract_and_detect_events(
                side_ref_path, output_dir="output",
                bowling_arm_override=bowling_arm_override,
                seed_point=side_seed_point, seed_frame_index=side_seed_frame,
                extra_seeds=side_extra_seeds, camera_angle_override="side_on",
            )
        st.session_state["_side_stage12_identity"] = side_file_identity
    side_stage12_result = st.session_state.get("_side_stage12_result")

    if st.session_state.get("_rear_stage12_identity") != rear_file_identity:
        with st.spinner("📐 Extracting rear-view tracking data..."):
            st.session_state["_rear_stage12_result"] = o.extract_and_detect_events(
                rear_ref_path, output_dir="output",
                bowling_arm_override=bowling_arm_override,
                seed_point=rear_seed_point, seed_frame_index=rear_seed_frame,
                extra_seeds=rear_extra_seeds, camera_angle_override="front_or_rear",
            )
        st.session_state["_rear_stage12_identity"] = rear_file_identity
    rear_stage12_result = st.session_state.get("_rear_stage12_result")

    st.markdown("#### 📹 Side-On Stream — Confirm Delivery Events")
    if side_stage12_result is None or side_stage12_result.get("status") != "success":
        st.error(side_stage12_result.get("message", "Side-on tracking extraction failed.")
                  if side_stage12_result else "Side-on tracking extraction failed.")
    else:
        side_confirmed_events = render_stream_event_confirmation(
            side_stage12_result, side_ref_path, side_file_identity,
            key_prefix="side", bowling_arm=side_stage12_result.get("bowling_arm", "right"),
            stream_label="Side-On",
        )

    st.markdown("#### 📹 Rear-View Stream — Confirm Delivery Events")
    if rear_stage12_result is None or rear_stage12_result.get("status") != "success":
        st.error(rear_stage12_result.get("message", "Rear-view tracking extraction failed.")
                  if rear_stage12_result else "Rear-view tracking extraction failed.")
    else:
        rear_confirmed_events = render_stream_event_confirmation(
            rear_stage12_result, rear_ref_path, rear_file_identity,
            key_prefix="rear", bowling_arm=rear_stage12_result.get("bowling_arm", "right"),
            stream_label="Rear-View",
        )

# Angle must be genuinely resolved (not left on "Not sure") before running —
# matches the same hard-gate already applied to bowling arm above. Dual
# Camera doesn't need this: each stream's angle is known by construction.
angle_resolved = (camera_mode != "Single Camera") or (
    confirmed_angle_label is not None and confirmed_angle_label != "unknown"
)

single_ready = (camera_mode == "Single Camera" and uploaded_single is not None
                 and single_seed_point is not None and bowling_arm_selected and angle_resolved
                 and br_resolved and ffc_resolved and bfc_resolved and wrist_resolved)
dual_ready = (camera_mode == "Dual Camera — Recommended"
              and uploaded_side is not None and uploaded_rear is not None
              and side_seed_point is not None and rear_seed_point is not None
              and bowling_arm_selected
              and side_confirmed_events is not None and rear_confirmed_events is not None)

if camera_mode == "Single Camera" and uploaded_single is not None and single_seed_point is None:
    st.sidebar.warning("👆 Click the bowler in the frame above to enable analysis.")
elif camera_mode != "Single Camera" and uploaded_side is not None and uploaded_rear is not None and (side_seed_point is None or rear_seed_point is None):
    st.sidebar.warning("👆 Click the bowler in both frames above to enable analysis.")
elif not bowling_arm_selected and (uploaded_single is not None or uploaded_side is not None):
    st.sidebar.warning("👆 Select Right-arm or Left-arm above to enable analysis — auto-detect is not reliable enough to run on by default.")
elif (camera_mode == "Single Camera" and uploaded_single is not None and single_seed_point is not None
      and bowling_arm_selected and not angle_resolved):
    st.sidebar.warning("👆 Confirm the filming angle above (not \"Not sure\") to enable analysis — "
                        "this changes what several metrics actually measure.")
elif (camera_mode == "Single Camera" and uploaded_single is not None and single_seed_point is not None
      and bowling_arm_selected and angle_resolved and not br_resolved):
    st.sidebar.warning("👆 Confirm the ball release frame above to enable analysis — "
                        "this feeds Release Height and every metric that depends on it.")
elif (camera_mode == "Single Camera" and uploaded_single is not None and single_seed_point is not None
      and bowling_arm_selected and angle_resolved and br_resolved and not ffc_resolved):
    st.sidebar.warning("👆 Confirm the front-foot-contact frame above to enable analysis — "
                        "this feeds Hip-Shoulder Separation and pre-release Knee Bracing.")
elif (camera_mode == "Single Camera" and uploaded_single is not None and single_seed_point is not None
      and bowling_arm_selected and angle_resolved and br_resolved and ffc_resolved and not bfc_resolved):
    st.sidebar.warning("👆 Confirm the back-foot-contact frame above to enable analysis — "
                        "this feeds Head Stability's measurement window.")
elif (camera_mode == "Single Camera" and uploaded_single is not None and single_seed_point is not None
      and bowling_arm_selected and angle_resolved and br_resolved and ffc_resolved and bfc_resolved
      and not wrist_resolved):
    st.sidebar.warning("👆 Confirm the release point above to enable analysis — "
                        "this feeds Release Height directly.")
elif (camera_mode != "Single Camera" and uploaded_side is not None and uploaded_rear is not None
      and side_seed_point is not None and rear_seed_point is not None and bowling_arm_selected
      and not (side_confirmed_events is not None and rear_confirmed_events is not None)):
    st.sidebar.warning("👆 Confirm the ball-release/front-foot/back-foot frames for BOTH streams "
                        "above to enable analysis — auto-detection can be wrong even when it "
                        "looks confident.")

import usage_limits
_is_admin_user = usage_limits.is_admin(st.session_state.auth_user.get("email", ""))
if _is_admin_user:
    _usage = {"used": 0, "limit": float("inf"), "remaining": float("inf")}
    st.sidebar.caption("🛠️ Admin account — unlimited analyses")
else:
    _usage = usage_limits.get_usage(st.session_state.auth_user["id"])
    if _usage["remaining"] <= 0:
        st.sidebar.error(
            f"You've used all {_usage['limit']} free analyses on this account. "
            "Contact us to unlock unlimited access."
        )
    else:
        st.sidebar.caption(f"🎟️ {_usage['remaining']} of {_usage['limit']} free analyses remaining")

if (single_ready or dual_ready) and _usage["remaining"] > 0:
    if st.sidebar.button("🚀 Execute Biomechanical Analysis Run", use_container_width=True):
        os.makedirs("input", exist_ok=True)

        try:
            if camera_mode == "Single Camera":
                video_path = os.path.abspath(os.path.join("input", uploaded_single.name))
                o.save_uploaded_video_capped(uploaded_single, video_path)
                st.sidebar.success(f"Cached: {uploaded_single.name}")
            else:
                video_path = os.path.abspath(os.path.join("input", uploaded_side.name))
                rear_path = os.path.abspath(os.path.join("input", uploaded_rear.name))
                o.save_uploaded_video_capped(uploaded_side, video_path)
                o.save_uploaded_video_capped(uploaded_rear, rear_path)
                st.sidebar.success(f"Cached: {uploaded_side.name} + {uploaded_rear.name}")
        except RuntimeError as e:
            st.error(f"⚠️ {e}")
            st.stop()

        with st.spinner("Executing kinematic extraction and landmark mapping..."):
            if camera_mode == "Dual Camera — Recommended":
                from dual_camera_orchestrator import run_dual_camera_analysis
                result_payload = run_dual_camera_analysis(
                    video_path, rear_path, bowling_arm_override=bowling_arm_override,
                    side_seed_point=side_seed_point, side_seed_frame_index=side_seed_frame,
                    rear_seed_point=rear_seed_point, rear_seed_frame_index=rear_seed_frame,
                    side_extra_seeds=side_extra_seeds, rear_extra_seeds=rear_extra_seeds,
                    side_precomputed=side_stage12_result, rear_precomputed=rear_stage12_result,
                    side_event_overrides=side_confirmed_events, rear_event_overrides=rear_confirmed_events,
                )
                active_camera_mode = "Dual Camera"
            else:
                # Reuse the extraction/events already computed for the
                # angle-check step ONLY if the coach's confirmed angle
                # matches what that step assumed — if they corrected the
                # auto-guess (e.g. it said front/rear, they confirmed
                # side-on), the cached events were built on the wrong
                # elbow-plausibility gating and must be redone, not reused.
                _stage12 = st.session_state.get("_stage12_result")
                _reuse = (
                    _stage12 is not None and _stage12.get("status") == "success"
                    and _stage12.get("camera_angle") == confirmed_angle_functional
                )
                if not _reuse:
                    _stage12 = o.extract_and_detect_events(
                        video_path, output_dir="output", bowling_arm_override=bowling_arm_override,
                        seed_point=single_seed_point, seed_frame_index=single_seed_frame,
                        extra_seeds=single_extra_seeds, camera_angle_override=confirmed_angle_functional,
                    )

                # Coach-confirmed release frame OVERRIDES the auto-detected
                # one here, before metrics/rendering run — see "Confirm Ball
                # Release Frame" above for why this is never skipped. The
                # original auto guess is kept alongside (not discarded) so
                # it can be logged for Phase 2 training data: every session
                # becomes a real (auto_guess, coach_confirmed) label pair.
                _br_confirmed = st.session_state.get("_br_confirmed_frame")
                if _stage12 is not None and _stage12.get("status") == "success" and _br_confirmed is not None:
                    _stage12 = dict(_stage12)
                    _stage12["events"] = dict(_stage12["events"])
                    _stage12["events"]["BR_auto_detected"] = _stage12["events"].get("BR")
                    _stage12["events"]["BR_auto_confidence"] = _stage12["events"].get("BR_confidence")
                    _stage12["events"]["BR"] = _br_confirmed
                    _stage12["events"]["BR_confidence"] = "coach_confirmed"
                    _stage12["events"]["BR_plausible_fraction"] = 1.0

                # Same for front-foot-contact — see "Confirm Front Foot
                # Contact Frame" above. Hip-Shoulder Separation and the
                # FFC-frame Knee Bracing value read events["FFC"] directly,
                # so overriding it here is enough to fix both.
                _ffc_confirmed = st.session_state.get("_ffc_confirmed_frame")
                if _stage12 is not None and _stage12.get("status") == "success" and _ffc_confirmed is not None:
                    _stage12 = dict(_stage12)
                    _stage12["events"] = dict(_stage12["events"])
                    _stage12["events"]["FFC_auto_detected"] = _stage12["events"].get("FFC")
                    _stage12["events"]["FFC"] = _ffc_confirmed

                # Same for back-foot-contact — see "Confirm Back Foot
                # Contact Frame" above. Only Head Stability's window reads
                # events["BFC"].
                _bfc_confirmed = st.session_state.get("_bfc_confirmed_frame")
                if _stage12 is not None and _stage12.get("status") == "success" and _bfc_confirmed is not None:
                    _stage12 = dict(_stage12)
                    _stage12["events"] = dict(_stage12["events"])
                    _stage12["events"]["BFC_auto_detected"] = _stage12["events"].get("BFC")
                    _stage12["events"]["BFC"] = _bfc_confirmed

                # Coach-corrected release POINT (wrist/ball position) — see
                # "Correct Release Point" above. Stored in pixel coords from
                # the click widget; normalize against this exact frame's real
                # dimensions before handing off, since orchestrator.py's
                # calculations all work in 0-1 normalized landmark space.
                _wrist_confirmed = st.session_state.get("_wrist_confirmed_point")
                _wrist_br_frame = st.session_state.get("_br_confirmed_frame")
                if (_stage12 is not None and _stage12.get("status") == "success"
                        and _wrist_confirmed is not None and _wrist_br_frame is not None):
                    _wrist_frame_for_norm = cal.extract_reference_frame(video_path, frame_index=_wrist_br_frame)
                    if _wrist_frame_for_norm is not None:
                        _wf_h, _wf_w = _wrist_frame_for_norm.shape[:2]
                        _stage12 = dict(_stage12)
                        _stage12["events"] = dict(_stage12["events"])
                        _stage12["events"]["wrist_override_x"] = _wrist_confirmed[0] / _wf_w
                        _stage12["events"]["wrist_override_y"] = _wrist_confirmed[1] / _wf_h

                result_payload = run_complete_bowling_analysis(
                    video_path, bowling_arm_override=bowling_arm_override,
                    seed_point=single_seed_point, seed_frame_index=single_seed_frame,
                    extra_seeds=single_extra_seeds,
                    camera_angle_override=confirmed_angle_functional,
                    precomputed=_stage12,
                )
                active_camera_mode = "Single Camera"
                st.session_state.pending_angle_label = confirmed_angle_label

        # Persist across reruns: Streamlit reruns the ENTIRE script on every
        # widget interaction (including the angle-confirmation radio button
        # further below). Without this, clicking that radio button would
        # make this "if button:" block report False again, wiping out the
        # whole results section and forcing a full re-analysis — which is
        # exactly the bug where confirming the camera angle appeared to
        # "reset and ask again."
        st.session_state.pending_result_payload = result_payload
        st.session_state.pending_video_path = video_path
        st.session_state.pending_active_camera_mode = active_camera_mode
        st.session_state.pending_player_name = player_name
        st.session_state.ai_insights_cache = None       # force regeneration for this NEW result
        st.session_state.history_saved_for_run = False  # allow exactly one history save for this NEW result
        st.session_state.usage_recorded_for_run = False  # allow exactly one usage-count increment for this NEW result

# Render results from session_state (not directly gated on this rerun's
# button click) so any later widget interaction on this page — like the
# angle-confirmation radio — doesn't discard everything computed above.
if st.session_state.get("pending_result_payload") is not None:
    result_payload = st.session_state.pending_result_payload
    video_path = st.session_state.pending_video_path
    active_camera_mode = st.session_state.pending_active_camera_mode
    player_name = st.session_state.pending_player_name

    if True:  # preserves original indentation/structure below unchanged

        # ================================================================
        # RESULTS DISPLAY
        # ================================================================
        if result_payload.get("status") == "success":
            st.success("✅ Kinematic Pipeline Finished Successfully!")
            if not st.session_state.get("usage_recorded_for_run", False):
                if _is_admin_user:
                    st.session_state.usage_recorded_for_run = True
                else:
                    try:
                        usage_limits.record_usage(st.session_state.auth_user["id"])
                        st.session_state.usage_recorded_for_run = True
                    except Exception as e:
                        monitoring.capture(e)
                        st.warning(f"Could not update usage count: {e}")

            metrics = result_payload["biomechanical_metrics"]
            frames = result_payload["time_indices"]
            fps = result_payload["video_metadata"]["fps"]
            total_frames = result_payload["video_metadata"]["total_frames"]
            import cv2 as _cv2_diag
            st.caption(
                f"🔍 Decoded as {fps:.2f} FPS, {total_frames} total frames "
                f"— source: {result_payload['video_metadata']['source_file']} "
                f"— OpenCV {_cv2_diag.__version__}"
            )

            events = {
                "BFC": frames["back_foot_contact_frame"],
                "FFC": frames["front_foot_contact_frame"],
                "BR": frames["ball_release_frame"],
            }

            # --- TRACKING QUALITY GUARD ---
            quality = dq.assess_quality(metrics)
            if quality["confidence"] == "low":
                missing_labels = [mr.RANGES[k].label for k in quality["missing_metrics"]]
                st.error(
                    f"⚠️ Low tracking confidence on this clip — "
                    f"{quality['missing_count']} of 5 metrics failed to compute "
                    f"({', '.join(missing_labels)}). This usually means motion blur, "
                    f"occlusion, or the bowler leaving frame during this delivery. "
                    f"Any remaining numeric values below came from the same degraded "
                    f"tracking and should not be trusted — **we recommend re-shooting "
                    f"this delivery** rather than acting on these results."
                )

            # --- RELEASE-FRAME CONFIDENCE GUARD ---
            # Separate from the guard above: this clip can have all 5 metrics
            # compute "successfully" and still have an unreliable release
            # frame, because the arm moves fastest (and blurs most) at the
            # exact instant it's trying to pinpoint. Verified directly: the
            # SAME footage decoded by two different video libraries can each
            # land on a different release frame, several frames apart — not
            # a bug in the detection logic, a property of the source footage
            # (motion blur right at release). No amount of algorithm tuning
            # closes that gap when the underlying pixels are ambiguous, so
            # the honest move is to disclose it rather than present a
            # specific frame number with false confidence.
            if frames.get("ball_release_confidence") == "low":
                st.warning(
                    "🎯 Release-frame timing has low confidence on this delivery "
                    f"(only {frames.get('ball_release_plausible_fraction', 0)*100:.0f}% of "
                    "the search window had clean tracking around release) — usually "
                    "motion blur at the exact instant the arm is fastest. Ball Release "
                    "Point may be off by a few frames, which would also shift Release "
                    "Height and Release Arm Speed. For a delivery you need precise "
                    "numbers on, re-shoot at a higher shutter speed or higher frame "
                    "rate if your camera supports it."
                )

            # --- PHASE TIMING (always available) ---
            phase_durations = None
            try:
                phase_durations = se.compute_phase_durations(events, fps)
            except ValueError as e:
                st.warning(f"Phase timing unavailable: {e}")

            # --- SPEED (only if calibrated) ---
            speed_result = None
            height_absolute_result = None
            landmarks_csv = o.landmarks_csv_path(active_camera_mode)
            if os.path.exists(landmarks_csv):
                landmarks_df = pd.read_csv(landmarks_csv)
                cap_w, cap_h = 1920, 1080  # overwritten below if we can read real dims
                try:
                    import cv2
                    _cap = cv2.VideoCapture(video_path)
                    cap_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cap_w
                    cap_h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cap_h
                    _cap.release()
                except Exception:
                    pass

                mpp = st.session_state.calibration.meters_per_pixel if st.session_state.calibration else None
                speed_result = se.compute_release_arm_speed(
                    landmarks_df, events, fps, cap_w, cap_h, meters_per_pixel=mpp,
                    video_path=video_path,
                    bowling_arm_override=metrics.get("bowling_arm_detected")
                )
                height_absolute_result = se.compute_release_height_absolute(
                    metrics.get("release_height", {}).get("debug_raw"), cap_h, meters_per_pixel=mpp
                )

            # --- RUN-UP ANALYSIS (stride detection + rhythm + strike pattern) ---
            run_up_result = None
            strike_summary = None
            if os.path.exists(landmarks_csv):
                run_up_result = rua.detect_run_up_strides(
                    landmarks_df, bfc_frame_idx=events["BFC"], fps=fps,
                    frame_width=cap_w, frame_height=cap_h
                )
                if run_up_result["status"] == "success":
                    annotated_contacts = rua.classify_strike_patterns(
                        landmarks_df, run_up_result["contacts"], frame_width=cap_w, frame_height=cap_h
                    )
                    strike_summary = rua.summarize_strike_patterns(annotated_contacts)

            # CAMERA ANGLE: already confirmed UPFRONT (before Execute) for Single
            # Camera mode — see the "Filming Angle Check" step. No longer asked
            # again here after the analysis has already run with a guess. Dual
            # Camera has no ambiguity to resolve (its two streams' angles are
            # known by construction), so it keeps the side-on-interpretation
            # default it always had.
            resolved_angle = st.session_state.get("pending_angle_label") or "side_on"

            # TIMELINE
            st.header("⏱️ Kinematic Sequence Timeline")
            t1, t2, t3 = st.columns(3)
            t1.metric("Back Foot Contact (BFC)", f"Frame {frames['back_foot_contact_frame']}")
            t2.metric("Front Foot Contact (FFC)", f"Frame {frames['front_foot_contact_frame']}",
                       f"+{phase_durations['bfc_to_ffc_seconds']}s" if phase_durations else None)
            t3.metric("Ball Release Point (BR)", f"Frame {frames['ball_release_frame']}",
                       f"+{phase_durations['ffc_to_br_seconds']}s" if phase_durations else None)

            if speed_result:
                if speed_result["status"] == "success":
                    st.metric("🏏 Estimated Release Arm Speed",
                              f"{speed_result['kmh']} km/h",
                              help="Tracks the bowling hand near release — a strong correlate "
                                   "of ball speed, not a direct radar reading.")
                elif speed_result["status"] == "not_calibrated":
                    st.info("📏 " + speed_result["message"])
                else:
                    st.warning(f"Speed estimate unavailable: {speed_result['message']}")

            st.divider()
            # Expanded by default — was collapsed, which combined with the
            # peak-detection bug (see run_up_analysis.py) to make this
            # section effectively invisible even after the underlying data
            # started working. A coach shouldn't have to know to click
            # into a collapsed section to find out this feature exists.
            with st.expander("🏃 Run-Up Analysis", expanded=True):
                if run_up_result is None:
                    st.info("Run-up data unavailable — landmark file not found.")
                elif run_up_result["status"] != "success":
                    st.info(run_up_result.get("message", "Run-up analysis unavailable for this clip."))
                else:
                    ru1, ru2, ru3 = st.columns(3)
                    ru1.metric("Run-Up Duration", f"{run_up_result['run_up_duration_seconds']}s")
                    ru2.metric("Detected Foot Contacts", run_up_result["stride_count"])
                    cv = run_up_result["rhythm_consistency_cv"]
                    ru3.metric("Rhythm Consistency (CV)", f"{cv}" if cv is not None else "N/A",
                               help="Coefficient of variation of time between foot contacts. "
                                    "Lower = more consistent pacing. There's no universal "
                                    "'good' cutoff — compare this bowler's own value across "
                                    "sessions over time rather than against a fixed target.")
                    if strike_summary:
                        st.markdown("**Foot Strike Pattern (run-up)**")
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Heel-Strike", strike_summary["heel"])
                        sc2.metric("Midfoot", strike_summary["midfoot"])
                        sc3.metric("Forefoot", strike_summary["forefoot"])
                        total_known = strike_summary["heel"] + strike_summary["midfoot"] + strike_summary["forefoot"]
                        if total_known > 0 and strike_summary["heel"] / total_known > 0.6:
                            st.caption(
                                "ℹ️ This run-up shows a heel-strike-dominant pattern. Heel-striking "
                                "during a sprint approach is generally considered less efficient than "
                                "midfoot/forefoot contact — worth discussing with the bowler, though "
                                "this is a general running-mechanics observation, not a cricket-specific "
                                "validated threshold."
                            )

            st.divider()

            col_graph, col_insights = st.columns([1, 1.2])

            with col_graph:
                st.header("🎞️ Visual Verification")
                clean_slug = player_name.replace(" ", "_")
                video_output = result_payload.get("annotated_video_output")
                rear_video_output = result_payload.get("rear_annotated_video_output")

                if rear_video_output and os.path.exists(rear_video_output):
                    tab_side, tab_rear = st.tabs(["📹 Side-On", "📹 Rear-View"])
                    with tab_side:
                        if video_output and os.path.exists(video_output):
                            st.video(video_output)
                            st.download_button(
                                label="📥 Download Side-On Video",
                                data=open(video_output, "rb").read(),
                                file_name=f"Annotated_SideOn_{clean_slug}.mp4",
                                mime="video/mp4",
                                use_container_width=True,
                                key="dl_side_video"
                            )
                        else:
                            st.info("Side-on annotated video rendering in progress...")
                    with tab_rear:
                        st.video(rear_video_output)
                        st.download_button(
                            label="📥 Download Rear-View Video",
                            data=open(rear_video_output, "rb").read(),
                            file_name=f"Annotated_RearView_{clean_slug}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            key="dl_rear_video"
                        )
                elif video_output and os.path.exists(video_output):
                    st.video(video_output)
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=open(video_output, "rb").read(),
                        file_name=f"Annotated_{clean_slug}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        key="dl_single_video"
                    )
                else:
                    st.info("Annotated video rendering in progress...")

                st.divider()
                st.header("📈 Biomechanical Measurements")

                knee_deg = mr.extract_metric_value(metrics, "front_knee_bracing")
                hip_deg = mr.extract_metric_value(metrics, "hip_shoulder_separation")
                trunk_deg = mr.extract_metric_value(metrics, "trunk_lean")
                rel_ratio = mr.extract_metric_value(metrics, "release_height")
                head_val = mr.extract_metric_value(metrics, "head_stability")

                def ui_deg(val):
                    return f"{round(float(val), 1)}°" if val is not None else "N/A"

                def ui_pct(val):
                    return f"{round(float(val) * 100, 1)}%" if val is not None else "N/A"

                def ui_val(val):
                    return str(round(float(val), 4)) if val is not None else "N/A"

                detected_arm = metrics.get("bowling_arm_detected")
                if detected_arm:
                    arm_source = "manually selected" if bowling_arm_override else "auto-detected"
                    st.caption(f"🎯 Bowling arm ({arm_source}): **{detected_arm.title()}-arm**")

                m1, m2 = st.columns(2)
                m1.metric("Lead Knee Bracing Angle", ui_deg(knee_deg),
                           metrics.get('front_knee_bracing', {}).get('tier', 'N/A'))
                yield_delta = metrics.get('front_knee_bracing', {}).get('yield_delta_degrees')
                yield_status = metrics.get('front_knee_bracing', {}).get('yield_status')
                if yield_delta is not None:
                    deg_at_release = metrics.get('front_knee_bracing', {}).get('degrees_at_release')
                    if yield_status == "yielding":
                        m1.caption(f"⚠️ Yields to {round(deg_at_release, 1)}° at release ({yield_delta:+.1f}°) — soft knee")
                    elif yield_status == "braced":
                        m1.caption(f"✅ Holds/extends to {round(deg_at_release, 1)}° at release ({yield_delta:+.1f}°) — braced")
                    else:
                        m1.caption(f"ℹ️ {round(deg_at_release, 1)}° at release ({yield_delta:+.1f}°)")
                m2.metric("Hip-Shoulder Rotation Twist", ui_deg(hip_deg),
                           metrics.get('hip_shoulder_separation', {}).get('tier', 'N/A'))

                st.write("")
                m3, m4, m5 = st.columns(3)
                m3.metric("Trunk Lean Deflection", ui_deg(trunk_deg),
                           metrics.get('trunk_lean', {}).get('tier', 'N/A'))
                m4.metric("Release Height Ratio", ui_pct(rel_ratio),
                           (metrics.get('release_height', {}).get('classification')
                            or metrics.get('release_height', {}).get('tier', 'N/A')))
                if height_absolute_result and height_absolute_result.get("status") == "success":
                    m4.caption(f"📏 {height_absolute_result['cm']} cm above ground (stump-calibrated)")
                elif height_absolute_result and height_absolute_result.get("status") == "not_calibrated":
                    m4.caption("📏 Calibrate camera (sidebar) for an absolute height in cm")
                m5.metric("Head Stability Variance", ui_val(head_val),
                           (metrics.get('head_stability', {}).get('classification')
                            or metrics.get('head_stability', {}).get('tier', 'N/A')))
                rel_debug = metrics.get('release_height', {}).get('debug_raw')
                if rel_debug:
                    with st.expander("🔧 Debug Info — Release Height Ratio"):
                        st.json(rel_debug)



                if resolved_angle in ("rear", "front", "unknown"):
                    if resolved_angle == "rear":
                        st.caption(
                            "📐 Filmed rear-view: Trunk Lean and Head Stability above reflect "
                            "**lateral sway**, not forward lean/gaze drift — the formula measures "
                            "left-right frame movement, which means something different from this "
                            "angle than from side-on. Knee Bracing may also be foreshortened. "
                            "Hip-Shoulder Separation is likely the most reliable metric from this angle."
                        )
                    elif resolved_angle == "front":
                        st.caption(
                            "📐 Filmed front-on: this angle isn't validated for any of these 5 "
                            "metrics — the bowling arm crossing in front of the torso can also "
                            "confuse pose tracking during the swing. Treat all values here with "
                            "reduced confidence."
                        )
                    else:
                        st.caption(
                            "📐 Filming angle unconfirmed — Trunk Lean, Head Stability, and Knee "
                            "Bracing assume a side-on view. If this wasn't filmed side-on, treat "
                            "those three with reduced confidence."
                        )

                # CBC REFERENCE RANGES — now sourced from metric_ranges.py, colored dots reflect real classification
                st.divider()
                with st.expander("🩺 Reference Ranges", expanded=False):
                    metric_value_lookup = {
                        "front_knee_bracing": knee_deg,
                        "hip_shoulder_separation": hip_deg,
                        "trunk_lean": trunk_deg,
                        "release_height": rel_ratio,
                        "head_stability": head_val,
                    }
                    dot = {"green": "🟢", "amber": "🟡", "red": "🔴", "unknown": "⚪"}
                    for key in mr.all_metric_keys():
                        r = mr.RANGES[key]
                        tier = mr.classify(key, metric_value_lookup.get(key))
                        st.markdown(
                            f"**{r.label}** {dot[tier]} — "
                            f"🟢 Optimal: `{r.display_optimal}`"
                        ) 
                    

            with col_insights:
                st.header("🧠 Autonomous AI Coach Assessment")
                if st.session_state.get("ai_insights_cache") is not None:
                    # Already generated for this result on an earlier rerun
                    # (e.g. before the angle-confirmation radio was clicked) —
                    # reuse it instead of calling Gemini again for no reason.
                    ai_insights = st.session_state.ai_insights_cache
                elif quality["confidence"] == "low":
                    st.warning(
                        "AI coaching analysis withheld for this delivery — tracking "
                        "confidence was too low to generate reliable coaching advice. "
                        "Re-shoot this delivery and re-run analysis."
                    )
                    ai_insights = {
                        "narrative_analysis": (
                            "Not generated: this delivery had insufficient tracking "
                            "quality (see warning above). Re-shoot and re-analyze "
                            "before drawing coaching conclusions."
                        ),
                        "prescribed_drills": [],
                    }
                    st.session_state.ai_insights_cache = ai_insights
                else:
                    with st.spinner("Generating expert coaching analysis..."):
                        ai_insights = generate_biomechanical_coaching_report(result_payload)
                    st.session_state.ai_insights_cache = ai_insights

                clean_slug = player_name.replace(" ", "_")
                pdf_data = generate_pdf_report(
                    metrics, frames, ai_insights,
                    bowler_name=player_name,
                    camera_mode=active_camera_mode,
                    phase_durations=phase_durations,
                    speed_result=speed_result,
                    quality=quality,
                )
                st.download_button(
                    label="📄 Download Official PDF Report",
                    data=pdf_data,
                    file_name=f"Biomechanical_Report_{clean_slug}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

                st.write("")
                st.markdown("### 📝 Technical Narrative")
                narrative = ai_insights.get("narrative_analysis", "")
                narrative = narrative.replace(
                    "SECTION 1 — BIOMECHANICAL NARRATIVE ASSESSMENT:", ""
                ).replace("SECTION 1 — BIOMECHANICAL NARRATIVE:", "").strip()
                st.write(narrative)

                st.markdown("### 🎯 Prescribed Training Drills")
                drills = ai_insights.get("prescribed_drills", [])
                if drills:
                    for i, drill in enumerate(drills, 1):
                        st.markdown(
                            f"<div style='background:#121824;border-left:3px solid "
                            f"#00B4D8;padding:12px;border-radius:6px;margin-bottom:"
                            f"8px;'><b style='color:#00B4D8;'>Drill {i}</b><br>"
                            f"<span style='color:#E2E8F0;'>{drill}</span></div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("All metrics within acceptable range. No critical interventions required.")

            # --- SAVE TO ATHLETE HISTORY (Supabase) ---
            # Guarded to fire exactly ONCE per analysis result — without this,
            # every later rerun (e.g. clicking the angle-confirmation radio)
            # would re-save a duplicate row to Supabase.
            if history_enabled and not st.session_state.get("history_saved_for_run", False):
                try:
                    athlete_id = store.get_or_create_athlete(player_name, st.session_state.auth_user["id"])
                    metrics_with_quality = {
                        **metrics,
                        "_data_quality": quality,
                        "_run_up": {"analysis": run_up_result, "strike_summary": strike_summary},
                        # Coach-confirmed camera angle, not a guess — real labeled
                        # data for a future trained angle classifier (Phase 2),
                        # collected for free as a side effect of normal use.
                        "_camera_angle_confirmed": resolved_angle,
                        # Same idea for release-frame timing: real (auto-guess,
                        # coach-confirmed) label pairs, logged whenever the
                        # single-camera confirmation step ran. Only present for
                        # Single Camera — see run_complete_bowling_analysis.
                        "_release_frame_confirmed": {
                            "auto_detected": frames.get("ball_release_frame_auto_detected"),
                            "auto_confidence": frames.get("ball_release_auto_confidence"),
                            "coach_confirmed": frames.get("ball_release_frame"),
                        } if frames.get("ball_release_frame_auto_detected") is not None else None,
                        # Same idea for front-foot-contact timing.
                        "_ffc_frame_confirmed": {
                            "auto_detected": frames.get("front_foot_contact_frame_auto_detected"),
                            "coach_confirmed": frames.get("front_foot_contact_frame"),
                        } if frames.get("front_foot_contact_frame_auto_detected") is not None else None,
                        # Same idea for back-foot-contact timing.
                        "_bfc_frame_confirmed": {
                            "auto_detected": frames.get("back_foot_contact_frame_auto_detected"),
                            "coach_confirmed": frames.get("back_foot_contact_frame"),
                        } if frames.get("back_foot_contact_frame_auto_detected") is not None else None,
                        # Whether the coach had to manually correct the
                        # auto-tracked wrist/ball position — real signal for
                        # how often MediaPipe's release-point tracking needs
                        # a human fix, same data-collection reasoning as the
                        # frame confirmations above.
                        "_wrist_point_corrected": st.session_state.get("_wrist_confirmed_point"),
                    }
                    store.save_session(
                        athlete_id=athlete_id,
                        coach_user_id=st.session_state.auth_user["id"],
                        video_filename=os.path.basename(video_path),
                        camera_mode=active_camera_mode,
                        fps=fps,
                        metrics=metrics_with_quality,
                        phase_durations=phase_durations,
                        release_arm_speed_kmh=(speed_result.get("kmh") if speed_result and speed_result.get("status") == "success" else None),
                        speed_status=(speed_result.get("status") if speed_result else "unavailable"),
                    )
                    st.session_state.history_saved_for_run = True
                    st.toast(f"Session saved to {player_name}'s history.")
                except Exception as e:
                    monitoring.capture(e)
                    st.warning(f"Could not save this session to history: {e}")

        else:
            st.error(
                f"Pipeline interrupted at stage "
                f"[{result_payload.get('stage', 'unknown')}]: "
                f"{result_payload.get('message', 'Unknown error')}"
            )
if st.session_state.get("pending_result_payload") is None:
    if camera_mode == "Single Camera":
        st.info("Upload a bowling video in the sidebar to begin analysis.")
    else:
        st.info("Upload both Side-On and Rear-View videos in the sidebar to begin dual-camera analysis.")

# ====================================================================
# CAMERA INSTRUCTIONS
# ====================================================================
st.sidebar.divider()
with st.sidebar.expander("📐 Camera Positioning Guide", expanded=False):
    st.markdown("""
**For maximum tracking accuracy:**
- **Right-arm bowlers:** Camera on the **left** side of the pitch
- **Left-arm bowlers:** Camera on the **right** side of the pitch
- **Alignment:** Parallel to the popping crease line
- **Distance:** 10–12 feet from the bowler
- **Frame rate:** 30 or 60 FPS only
- **Dual camera:** Both phones start recording before the bowler begins run-up
""")
