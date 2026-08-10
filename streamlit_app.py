import os
import base64
import html
import math
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
import batting_kinematics as bk
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


# ====================================================================
# BATTING ANALYSIS — VISUAL THEME  ("floodlit night match")
# ====================================================================
# Purely presentational: a dark, willow-gold-accented skin for the
# Batting Analysis output (approved by the coach from a standalone HTML
# mockup). Every selector is prefixed with .apex-batting-theme so this
# can never bleed onto the Bowling Analysis page — this <style> block is
# only ever injected from inside render_batting_analysis_ui, which is a
# dead-end branch (it ends in st.stop()), and every themed HTML block
# below is additionally wrapped in its own .apex-batting-theme div as a
# second, belt-and-braces layer of scoping. No data/orchestration logic
# lives here — only markup for values already computed elsewhere.
_BATTING_THEME_CSS = """
<style>
.apex-batting-theme {
  --bg: #0d1310; --surface: #172019; --surface-2: #1d2721;
  --border: #2a352d; --border-soft: #212b24;
  --ink: #ece8db; --ink-muted: #93a79b; --ink-faint: #647266;
  --accent: #d3a54d;
  --good: #34c77b; --good-soft: rgba(52,199,123,0.14);
  --warn: #e8a33d; --warn-soft: rgba(232,163,61,0.14);
  --crit: #e5484d; --crit-soft: rgba(229,72,77,0.16);
  --font-display: "Barlow Condensed","Bahnschrift","Oswald","Arial Narrow",sans-serif;
  --font-body: Charter,"Iowan Old Style","Georgia Pro",Georgia,serif;
  --font-mono: "Cascadia Mono","SFMono-Regular",Consolas,"Liberation Mono",monospace;
  color: var(--ink);
}
.apex-batting-theme, .apex-batting-theme * { box-sizing: border-box; }

.apex-batting-theme .topbar {
  display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding:14px 18px; margin-bottom:14px;
  background: radial-gradient(ellipse 700px 300px at 30% -40%, rgba(211,165,77,0.10), transparent 60%), var(--bg);
  border:1px solid var(--border); border-radius:12px;
}
.apex-batting-theme .brand { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.apex-batting-theme .brand .logo-mark { height:26px; width:auto; display:block; filter:drop-shadow(0 1px 5px rgba(211,165,77,0.18)); }
.apex-batting-theme .brand .mark { font-family:var(--font-display); font-weight:700; font-size:19px; letter-spacing:0.02em; color:var(--ink); line-height:1.2; }
.apex-batting-theme .brand .mark em { color:var(--accent); font-style:normal; }
.apex-batting-theme .brand .tagline {
  font-family:var(--font-body); font-size:11.5px; font-style:italic; color:var(--ink-muted); line-height:1.3;
}
.apex-batting-theme .brand .module {
  font-family:var(--font-display); font-size:11px; letter-spacing:0.16em; text-transform:uppercase;
  color:var(--ink-muted); padding:3px 9px; border:1px solid var(--border); border-radius:3px; margin-left:6px;
}
.apex-batting-theme .identity { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.apex-batting-theme .intro-note {
  font-size:13px; color:var(--ink-muted); line-height:1.55; padding:2px 4px 10px;
}
.apex-batting-theme .intro-note b { color:var(--ink); font-weight:600; }
.apex-batting-theme .pill {
  font-family:var(--font-mono); font-size:12px; padding:4px 10px; border-radius:999px;
  border:1px solid var(--border); background:var(--surface); color:var(--ink-muted); white-space:nowrap;
}
.apex-batting-theme .pill b { color:var(--ink); font-weight:600; }
.apex-batting-theme .pill.pill-warn { border-color:rgba(232,163,61,0.4); background:var(--warn-soft); color:var(--warn); }

.apex-batting-theme .flag {
  display:flex; gap:14px; align-items:flex-start; background:var(--crit-soft);
  border:1px solid rgba(229,72,77,0.4); border-radius:10px; padding:14px 18px; margin:14px 0;
}
.apex-batting-theme .flag.is-clear { background:var(--good-soft); border-color:rgba(52,199,123,0.35); }
.apex-batting-theme .flag.is-neutral { background:var(--surface); border-color:var(--border); }
.apex-batting-theme .flag .stripe { width:4px; align-self:stretch; border-radius:3px; background:var(--crit); flex:none; }
.apex-batting-theme .flag.is-clear .stripe { background:var(--good); }
.apex-batting-theme .flag.is-neutral .stripe { background:var(--ink-faint); }
.apex-batting-theme .flag-body { flex:1; min-width:0; }
.apex-batting-theme .flag-title {
  font-family:var(--font-display); font-weight:700; letter-spacing:0.03em; font-size:15px;
  color:var(--crit); text-transform:uppercase; margin-bottom:3px;
}
.apex-batting-theme .flag.is-clear .flag-title { color:var(--good); }
.apex-batting-theme .flag.is-neutral .flag-title { color:var(--ink-muted); }
.apex-batting-theme .flag-desc { font-size:14px; color:var(--ink); max-width:70ch; }
.apex-batting-theme .flag-meters { display:flex; gap:22px; margin-top:10px; flex-wrap:wrap; }
.apex-batting-theme .flag-meter { font-family:var(--font-mono); font-size:12px; color:var(--ink-muted); }
.apex-batting-theme .flag-meter b { color:var(--ink); font-size:13px; }
.apex-batting-theme .flag-bar { width:110px; height:5px; border-radius:3px; background:var(--border); margin-top:4px; overflow:hidden; }
.apex-batting-theme .flag-bar > span { display:block; height:100%; background:var(--crit); }
.apex-batting-theme .flag.is-clear .flag-bar > span { background:var(--good); }

.apex-batting-theme .section-label {
  font-family:var(--font-display); font-size:13px; letter-spacing:0.12em; text-transform:uppercase;
  color:var(--ink-faint); margin:18px 2px 10px;
}

/* RESTRUCTURE (2026-08-04): grid narrowed from minmax(230px,...) — the
   metrics grid now sits in a Streamlit column beside the video (real
   st.columns, not a CSS trick) instead of spanning the full page width,
   so it needs to comfortably fit 2 cards per row in a narrower space. */
.apex-batting-theme .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(155px, 1fr)); gap:9px; margin-bottom:8px; }
.apex-batting-theme .card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px 16px; position:relative; }
.apex-batting-theme .card-head { display:flex; align-items:flex-start; justify-content:space-between; gap:8px; margin-bottom:10px; }
.apex-batting-theme .card-title { font-family:var(--font-display); font-size:13px; letter-spacing:0.04em; color:var(--ink-muted); text-transform:uppercase; line-height:1.25; }
.apex-batting-theme .chip { font-family:var(--font-mono); font-size:10.5px; letter-spacing:0.04em; padding:2px 7px; border-radius:4px; font-weight:700; white-space:nowrap; flex:none; }
.apex-batting-theme .chip.good { background:var(--good-soft); color:var(--good); }
.apex-batting-theme .chip.warn { background:var(--warn-soft); color:var(--warn); }
.apex-batting-theme .chip.crit { background:var(--crit-soft); color:var(--crit); }
.apex-batting-theme .chip.unknown { background:var(--border-soft); color:var(--ink-faint); }
.apex-batting-theme .value-row { display:flex; align-items:baseline; gap:5px; margin-bottom:2px; }
.apex-batting-theme .value { font-family:var(--font-mono); font-variant-numeric:tabular-nums; font-size:28px; font-weight:600; letter-spacing:-0.01em; }
.apex-batting-theme .value.good { color:var(--good); }
.apex-batting-theme .value.warn { color:var(--warn); }
.apex-batting-theme .value.crit { color:var(--crit); }
.apex-batting-theme .value.unknown { color:var(--ink-faint); }
.apex-batting-theme .unit { font-family:var(--font-mono); font-size:13px; color:var(--ink-faint); }
.apex-batting-theme .card-note { font-size:12.5px; color:var(--ink-muted); margin-bottom:6px; min-height:1.2em; }
.apex-batting-theme .card-range { font-size:11px; color:var(--ink-faint); font-family:var(--font-mono); }
.apex-batting-theme .view-tag {
  position:absolute; top:14px; right:16px; font-family:var(--font-mono); font-size:9px; letter-spacing:0.05em;
  color:var(--warn); border:1px solid rgba(232,163,61,0.4); border-radius:3px; padding:1px 5px;
}

/* Per-metric visualizations (2026-08-03) — ported from the approved
   mockup, computed server-side as static SVG/HTML per card (the mockup
   drew these with client-side JS; Streamlit's markdown sandbox doesn't
   reliably execute injected <script> tags, and these values are already
   final by the time the card renders, so there's nothing to compute live
   here anyway). */
.apex-batting-theme .gauge-wrap { display:flex; justify-content:center; margin-top:8px; }
.apex-batting-theme svg.gauge { width:100%; height:auto; max-width:170px; }
.apex-batting-theme .gauge .band { fill:none; stroke-width:8; }
.apex-batting-theme .gauge .needle { stroke:var(--ink); stroke-width:2.5; stroke-linecap:round; }
.apex-batting-theme .gauge .hub { fill:var(--ink); }
.apex-batting-theme .gauge .tick-label { font-family:var(--font-mono); font-size:8px; fill:var(--ink-faint); }

.apex-batting-theme .foot-gauge { margin-top:8px; }
.apex-batting-theme .foot-track {
  position:relative; height:10px; border-radius:6px; margin:10px 2px 4px;
  background:linear-gradient(90deg, rgba(79,138,201,0.28), rgba(79,138,201,0.06) 46%, var(--border) 50%, rgba(201,122,79,0.06) 54%, rgba(201,122,79,0.28));
  border:1px solid var(--border);
}
.apex-batting-theme .foot-target {
  position:absolute; top:-3px; bottom:-3px; background:rgba(211,165,77,0.32);
  border-left:1px solid var(--accent); border-right:1px solid var(--accent); border-radius:3px;
}
.apex-batting-theme .foot-marker { position:absolute; top:-6px; width:2px; height:22px; background:var(--ink); }
.apex-batting-theme .foot-marker::after {
  content:""; position:absolute; top:-5px; left:50%; transform:translateX(-50%);
  border:5px solid transparent; border-top-color:var(--ink);
}
.apex-batting-theme .foot-labels { display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:9.5px; color:var(--ink-faint); padding:0 2px; }

.apex-batting-theme .bar-gauge { margin-top:10px; }
.apex-batting-theme .bar-track { position:relative; height:10px; border-radius:6px; background:var(--border); overflow:hidden; }
.apex-batting-theme .bar-fill { position:absolute; inset:0; border-radius:6px; }
.apex-batting-theme .bar-ticks { position:relative; height:14px; margin-top:2px; }
.apex-batting-theme .bar-ticks span { position:absolute; top:0; font-family:var(--font-mono); font-size:8.5px; color:var(--ink-faint); transform:translateX(-50%); }

.apex-batting-theme .report-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:18px 20px; margin-top:8px; }
.apex-batting-theme .report-head { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
.apex-batting-theme .report-head .dot { width:7px; height:7px; border-radius:50%; background:var(--accent); }
.apex-batting-theme .report-title { font-family:var(--font-display); font-size:15px; letter-spacing:0.03em; text-transform:uppercase; color:var(--ink); }
.apex-batting-theme .narrative { font-family:var(--font-body); font-size:15px; color:var(--ink); max-width:75ch; line-height:1.6; margin:0; }
.apex-batting-theme .narrative b { color:var(--accent); font-weight:700; }
.apex-batting-theme .drills { list-style:none; margin:14px 0 0; padding:0; display:flex; flex-direction:column; gap:10px; }
.apex-batting-theme .drills li { padding:10px 0 0; border-top:1px solid var(--border-soft); font-size:14px; font-family:var(--font-body); color:var(--ink); }
.apex-batting-theme .drills .dname { font-family:var(--font-display); letter-spacing:0.02em; color:var(--accent); font-weight:700; font-size:13.5px; text-transform:uppercase; display:block; margin-bottom:2px; }

/* NATIVE STREAMLIT WIDGETS (2026-08-03, coach feedback: expanders/video
   still looked like the old plain UI) — this whole <style> block is only
   ever injected while the Batting Analysis branch is actually running
   (see render_batting_analysis_ui), so scoping these to the raw
   data-testid selectors (rather than nesting under .apex-batting-theme,
   which native Streamlit widgets aren't rendered inside of) still can't
   leak onto the Bowling Analysis page — that's a completely separate
   script run where this CSS is never injected at all. !important matches
   the specificity of the app's own pre-existing global expander/video
   rules further down this file (the "PAGE CONFIG & ELITE DARK UI" block)
   — source order (this one renders later) settles the tie in its favor. */
/* NOTE: Streamlit 1.60.0's expander header label has no stable, dedicated
   testid of its own to target directly (verified by reading the actual
   JS bundle — it's an internal, minified styled-component, not something
   safe to hard-code a class name for). Setting color on the outer
   container and relying on CSS inheritance is the robust way to reach it
   without depending on an implementation detail that could rename itself
   on a Streamlit update. */
div[data-testid="stExpander"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; margin-bottom: 10px !important; color: var(--ink) !important;
}
div[data-testid="stExpander"] svg { fill: var(--ink-muted) !important; }

/* BUG FIX (2026-08-03, real coach test): "huge expanders" — the frame-
   preview images inside the seed-click/stance/backlift/contact
   confirmation expanders are portrait phone-video frames rendered at
   use_column_width=True, i.e. stretched to the FULL width of the main
   content column. For a portrait image that means the height scales up
   proportionally to something far taller than the column is wide — the
   exact same "unconstrained portrait media" bug as the video fix above,
   just for st.image instead of st.video. Same fix: cap the max-width so
   the image (and the expander containing it) settles to a sane size
   regardless of the source photo's native resolution.
   [data-testid="stImage"] is confirmed correct from the JS bundle (the
   image *container*, not the raw <img> — the actual <img> tag is nested
   inside it, hence the descendant selector rather than assuming stImage
   itself is the <img>, unlike the stVideo mistake above). */
div[data-testid="stImage"] { display: flex !important; justify-content: center !important; }
div[data-testid="stImage"] img {
  max-width: 420px !important; width: 100% !important; height: auto !important;
  max-height: 70vh !important; object-fit: contain !important;
  border-radius: 8px !important; border: 1px solid var(--border) !important;
}

/* BUG FIX (2026-08-03, real coach test, second attempt): the first fix
   used "div[data-testid='stVideo']" and did nothing at all — verified by
   reading Streamlit 1.60.0's own JS bundle directly: data-testid="stVideo"
   is set on the <video> element ITSELF, not a wrapping div (there is no
   such div). !important is needed because Streamlit's React component
   sets inline width/height styles on the element directly, which beat a
   plain CSS rule of equal-or-lower specificity. Without a size cap, a
   tall portrait phone clip stretched to "width:100%" of the full main
   content column scales its height proportionally, easily overflowing
   well past the viewport — confirmed directly in the coach's screenshot
   (one video frame taller than the whole page). */
video[data-testid="stVideo"] {
  max-width: 380px !important; width: 100% !important; height: auto !important;
  max-height: 70vh !important; object-fit: contain !important; display: block !important;
  margin: 0 auto !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; background: var(--bg) !important;
}
</style>
"""


def _batting_html(raw: str) -> None:
    """
    BUG FIX (2026-08-03, found on a real coach test): st.markdown(...,
    unsafe_allow_html=True) still runs its input through Streamlit's
    Markdown/CommonMark renderer before allowing raw HTML through it —
    and CommonMark treats a line indented 4+ spaces as a literal indented
    code block, rendering it as visible plain text instead of parsing the
    HTML inside it. The themed HTML blocks below were written nicely
    indented for source readability (nested <div>s), which tripped
    exactly this: the coach saw raw HTML/text where the styled cards
    should have been. Stripping each line's leading whitespace here means
    the source can stay readable while the string handed to st.markdown
    is always a single unindented block, immune to this.
    """
    st.markdown(_dedent_html(raw), unsafe_allow_html=True)


def _dedent_html(raw: str) -> str:
    """Strips every line's leading whitespace — see _batting_html's
    docstring above. Used both there AND by any helper (like
    _batting_metric_card_html below) that builds a multi-line indented
    HTML fragment which will eventually be joined into a larger string
    and handed to st.markdown — the CommonMark code-block trap applies
    just the same whether the indented string reaches st.markdown
    directly or via a join() of several such fragments. FOUND ON A REAL
    COACH TEST (2026-08-03): the first fix only covered direct
    st.markdown(f\"\"\"...\"\"\") call sites and missed this function
    returning its own indented fragment, which still broke the metric
    cards grid the exact same way."""
    return "\n".join(line.strip() for line in raw.strip("\n").split("\n"))


# Full named-shot vocabulary for the "Shot Played" dropdown (2026-08-03,
# coach-provided real cricket shot taxonomy — vertical-bat drives/defense
# plus horizontal-bat/unorthodox shots) mapped to
# batting_kinematics.SHOT_TARGET_CENTERS_DEGREES / NOT_APPLICABLE_SHOTS
# keys. Ordered so the shots that DO get a real front-foot-alignment
# target come first, then the ones that don't (grouped to match the
# coach's own categorization: defense, cuts, pull/hook, sweeps,
# unorthodox) — see NOT_APPLICABLE_SHOTS in batting_kinematics.py for why
# that second group isn't scored on this particular metric.
_SHOT_PLAYED_OPTIONS = {
    "Not sure / skip this check": None,
    "Straight Drive": "straight_drive",
    "On Drive": "on_drive",
    "Off Drive": "off_drive",
    "Cover Drive": "cover_drive",
    "Flick / Leg Glance": "flick_leg_glance",
    "Forward Defense": "forward_defense",
    "Backward Defense": "backward_defense",
    "Square Cut": "square_cut",
    "Late Cut": "late_cut",
    "Pull Shot": "pull_shot",
    "Hook Shot": "hook_shot",
    "Sweep": "standard_sweep",
    "Reverse Sweep": "reverse_sweep",
    "Slog Sweep": "slog_sweep",
    "Scoop / Ramp Shot": "scoop_ramp",
    "Switch Hit": "switch_hit",
}


# Maps each metric_ranges.py batting_* key to the sub-dict key
# batting_orchestrator.run_batting_analysis actually stores it under in
# biomechanical_metrics — same mapping mr.extract_batting_metric_value
# already uses internally, duplicated here only to also reach each
# metric's "tier" descriptor text (extract_batting_metric_value only
# returns the numeric value, not the tier), which the metric cards below
# display as their one-line descriptor.
_BATTING_SUBKEY_BY_MKEY = {
    "batting_head_movement": "head_movement",
    "batting_front_foot_alignment": "front_foot_alignment",
    "batting_weight_transfer": "weight_transfer",
    "batting_downswing_plane": "downswing_plane",
    "batting_top_elbow_angle": "top_elbow_angle",
    "batting_front_knee_flexion": "front_knee_flexion",
    "batting_xfactor_separation": "xfactor_separation",
}
_ZONE_CSS_CLASS = {"green": "good", "amber": "warn", "red": "crit", "unknown": "unknown"}
_ZONE_CSS_VAR = {"good": "--good", "warn": "--warn", "crit": "--crit", "unknown": "--ink-faint"}

# Display axis extents for the semicircle gauges below — a PRESENTATION
# choice (how much of the dial to draw), not a duplicated threshold: the
# actual green/amber/red boundaries always come from mr.RANGES directly
# (see _bands_for_metric), never repeated here.
_GAUGE_DISPLAY_RANGE = {
    "batting_downswing_plane": (0.0, 55.0),
    "batting_top_elbow_angle": (60.0, 180.0),
    "batting_front_knee_flexion": (60.0, 185.0),
    "batting_xfactor_separation": (0.0, 70.0),
}
_BAR_DISPLAY_MAX = {
    "batting_head_movement": 0.06,
    "batting_weight_transfer": 100.0,
}


def _svg_polar(cx: float, cy: float, r: float, angle_deg: float):
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def _svg_arc_path(cx: float, cy: float, r: float, a0: float, a1: float) -> str:
    x0, y0 = _svg_polar(cx, cy, r, a0)
    x1, y1 = _svg_polar(cx, cy, r, a1)
    large = 1 if abs(a1 - a0) > 180 else 0
    return f"M {x0:.2f} {y0:.2f} A {r} {r} 0 {large} 1 {x1:.2f} {y1:.2f}"


def _bands_for_metric(mkey: str, display_min: float, display_max: float):
    """Derives the gauge's colored zone segments straight from mr.RANGES
    (never a second, hand-copied set of thresholds) — only supports
    kind="band", the only kind used by the metrics routed to a semicircle
    gauge below."""
    r = mr.RANGES[mkey]
    if r.kind != "band":
        return []
    g_lo, g_hi = r.green
    a_lo, a_hi = r.amber
    ah_lo, ah_hi = r.amber_high
    bands = []
    if display_min < a_lo:
        bands.append((display_min, a_lo, "--crit"))
    bands.append((a_lo, g_lo, "--warn"))
    bands.append((g_lo, g_hi, "--good"))
    bands.append((g_hi, ah_hi, "--warn"))
    if display_max > ah_hi:
        bands.append((ah_hi, display_max, "--crit"))
    return bands


def _svg_band_gauge(value: float, display_min: float, display_max: float, bands) -> str:
    """Static semicircle band gauge (needle + colored zone arcs), computed
    server-side — same math as the approved mockup's client-side JS
    version, ported to Python since these values are already final by the
    time the card renders (nothing here needs to be interactive)."""
    cx, cy, r = 60.0, 58.0, 46.0
    span = display_max - display_min
    parts = ['<svg class="gauge" viewBox="0 0 120 68">']
    for lo, hi, css_var in bands:
        a0 = 180 + ((lo - display_min) / span) * 180
        a1 = 180 + ((hi - display_min) / span) * 180
        parts.append(f'<path class="band" style="stroke:var({css_var})" d="{_svg_arc_path(cx, cy, r, a0, a1)}"></path>')
    clamped = max(display_min, min(display_max, value))
    a = 180 + ((clamped - display_min) / span) * 180
    tip_x, tip_y = _svg_polar(cx, cy, r - 12, a)
    parts.append(f'<line class="needle" x1="{cx}" y1="{cy}" x2="{tip_x:.2f}" y2="{tip_y:.2f}"></line>')
    parts.append(f'<circle class="hub" cx="{cx}" cy="{cy}" r="3.5"></circle>')
    for t in (display_min, (display_min + display_max) / 2, display_max):
        ta = 180 + ((t - display_min) / span) * 180
        lx, ly = _svg_polar(cx, cy, r + 9, ta)
        parts.append(f'<text class="tick-label" x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle">{round(t)}</text>')
    parts.append("</svg>")
    return f'<div class="gauge-wrap">{"".join(parts)}</div>'


def _linear_bar_gauge_html(value: float, display_max: float, ticks, css_zone: str) -> str:
    """Horizontal fill bar for the two metrics that read as a single
    linear number (higher_better/lower_better), not a band — weight
    transfer and head movement. Fill color follows the metric's own
    already-computed zone; ticks mark the real green/amber boundaries
    from mr.RANGES (passed in by the caller, not re-derived here)."""
    pct = max(0.0, min(1.0, value / display_max)) * 100 if display_max else 0.0
    tick_html = "".join(
        f'<span style="left:{(t / display_max) * 100:.0f}%">{t:g}</span>' for t in ticks
    )
    css_var = _ZONE_CSS_VAR.get(css_zone, "--ink-faint")
    return (
        '<div class="bar-gauge">'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:var({css_var})"></div></div>'
        f'<div class="bar-ticks">{tick_html}</div>'
        "</div>"
    )


def _foot_strip_gauge_html(signed_degrees, target_shot) -> str:
    """Off/leg strip gauge for front_foot_alignment: -45 (leg) to +45
    (off), a highlighted wedge for the shot's real target window (only
    when the shot actually has one — see batting_kinematics.
    SHOT_TARGET_CENTERS_DEGREES/NOT_APPLICABLE_SHOTS), and a marker at
    the real measured foot direction."""
    gmin, gmax = -45.0, 45.0

    def pct(a):
        return max(0.0, min(1.0, (a - gmin) / (gmax - gmin))) * 100

    target_html = ""
    if target_shot in bk.SHOT_TARGET_CENTERS_DEGREES:
        center = bk.SHOT_TARGET_CENTERS_DEGREES[target_shot]
        lo = max(gmin, center - bk.FRONT_FOOT_DEVIATION_TOLERANCE)
        hi = min(gmax, center + bk.FRONT_FOOT_DEVIATION_TOLERANCE)
        target_html = f'<div class="foot-target" style="left:{pct(lo):.1f}%;width:{max(0.0, pct(hi) - pct(lo)):.1f}%"></div>'

    marker_html = ""
    if signed_degrees is not None:
        clamped = max(gmin, min(gmax, signed_degrees))
        marker_html = f'<div class="foot-marker" style="left:{pct(clamped):.1f}%"></div>'

    return (
        '<div class="foot-gauge"><div class="foot-track">'
        f"{target_html}{marker_html}"
        '</div><div class="foot-labels"><span>LEG</span><span>STRAIGHT</span><span>OFF</span></div></div>'
    )


def _batting_metric_visual_html(mkey: str, value, css_zone: str, metrics: dict) -> str:
    """Dispatches to the right visualization for this metric key. Returns
    "" (no visual) when the underlying value is None — a missing/
    inapplicable reading has nothing real to plot, and a gauge sitting at
    a fixed "0" position would misleadingly look like a genuine reading
    of zero."""
    if mkey == "batting_front_foot_alignment":
        foot_data = metrics.get("front_foot_alignment", {})
        return _foot_strip_gauge_html(foot_data.get("signed_degrees"), foot_data.get("target_shot"))
    if value is None:
        return ""
    if mkey in _GAUGE_DISPLAY_RANGE:
        display_min, display_max = _GAUGE_DISPLAY_RANGE[mkey]
        return _svg_band_gauge(float(value), display_min, display_max, _bands_for_metric(mkey, display_min, display_max))
    if mkey in _BAR_DISPLAY_MAX:
        r = mr.RANGES[mkey]
        ticks = sorted(set(r.amber) | set(r.green)) if r.kind != "band" else []
        # Only the boundary that actually separates zones is worth marking
        # (e.g. weight_transfer: 20 and 40); drop any tick beyond the
        # gauge's own display ceiling.
        ticks = [t for t in ticks if 0 < t < _BAR_DISPLAY_MAX[mkey]]
        return _linear_bar_gauge_html(float(value), _BAR_DISPLAY_MAX[mkey], ticks, css_zone)
    return ""


def _batting_metric_card_html(mkey: str, metrics: dict, view_caveats: list) -> str:
    """
    Renders ONE metric as a themed card. Reads the zone straight from
    mr.classify()/mr.RANGES (never invents its own thresholds) — this
    function only decides how that already-computed classification looks
    on screen, not what it is.
    """
    value = mr.extract_batting_metric_value(metrics, mkey)
    zone = mr.classify(mkey, value)
    r = mr.RANGES[mkey]
    css_zone = _ZONE_CSS_CLASS.get(zone, "unknown")
    chip_label = zone.upper()
    subkey = _BATTING_SUBKEY_BY_MKEY.get(mkey, "")
    tier_text = metrics.get(subkey, {}).get("tier", "") if subkey else ""

    if value is not None:
        formatted = mr.format_value(mkey, value)
        if r.unit and formatted.endswith(r.unit):
            numeric_part, unit_part = formatted[: -len(r.unit)], r.unit
        else:
            numeric_part, unit_part = formatted, ""
    else:
        numeric_part, unit_part = "N/A", ""

    view_tag_html = '<span class="view-tag">view-sensitive</span>' if mkey in (view_caveats or []) else ""
    visual_html = _batting_metric_visual_html(mkey, value, css_zone, metrics)

    return _dedent_html(f"""<div class="card">
      {view_tag_html}
      <div class="card-head">
        <div class="card-title">{html.escape(r.label)}</div>
        <div class="chip {css_zone}">{html.escape(chip_label)}</div>
      </div>
      <div class="value-row">
        <span class="value {css_zone}">{html.escape(numeric_part)}</span>
        <span class="unit">{html.escape(unit_part)}</span>
      </div>
      <div class="card-note">{html.escape(tier_text)}</div>
      <div class="card-range">Target: {html.escape(r.display_optimal)}</div>
      {visual_html}
    </div>""")


def _batting_narrative_to_html(text: str) -> str:
    """
    Pure display conversion for the AI coach narrative paragraph: HTML-
    escapes the raw LLM text (never trusted as HTML) then re-applies the
    same **bold** emphasis markdown already rendered via st.write() in
    the previous plain version of this UI — same words, same emphasis,
    only the surrounding typography changes.
    """
    escaped = html.escape(text or "")
    parts = escaped.split("**")
    rebuilt = "".join(f"<b>{p}</b>" if i % 2 == 1 else p for i, p in enumerate(parts))
    return rebuilt.replace("\n", "<br>")


def render_batting_event_confirmation(stage12_result, ref_path: str, file_identity: str):
    """
    Batting equivalent of render_stream_event_confirmation — mandatory
    STANCE/BACKLIFT/CONTACT confirmation, same reasoning: batting_events.py
    is a first-pass heuristic (documented in its own module docstring as
    NOT yet tuned against real batting footage the way bowling's detector
    was), so nothing here gets trusted without a human looking at it.
    FOLLOW_THROUGH is not confirmable — it's a video-trim reference only,
    not used by any metric calculation (see batting_orchestrator.py).

    Returns None until all three are confirmed; once confirmed, returns
    {"STANCE": int, "BACKLIFT": int, "CONTACT": int,
    "STANCE_auto_detected": ..., "BACKLIFT_auto_detected": ...,
    "CONTACT_auto_detected": ..., "CONTACT_auto_confidence": ...}.
    """
    stance_key, backlift_key, contact_key = (
        "_batting_stance_confirmed_frame", "_batting_backlift_confirmed_frame", "_batting_contact_confirmed_frame",
    )
    stance_id_key, backlift_id_key, contact_id_key = (
        "_batting_stance_identity", "_batting_backlift_identity", "_batting_contact_identity",
    )

    if st.session_state.get(stance_id_key) != file_identity:
        st.session_state[stance_key] = None
        st.session_state[stance_id_key] = file_identity
    if st.session_state.get(backlift_id_key) != file_identity:
        st.session_state[backlift_key] = None
        st.session_state[backlift_id_key] = file_identity
    if st.session_state.get(contact_id_key) != file_identity:
        st.session_state[contact_key] = None
        st.session_state[contact_id_key] = file_identity

    stance_auto = backlift_auto = contact_auto = None
    contact_confidence = None
    if stage12_result is not None and stage12_result.get("status") == "success":
        stance_auto = stage12_result["events"].get("STANCE")
        backlift_auto = stage12_result["events"].get("BACKLIFT")
        contact_auto = stage12_result["events"].get("CONTACT")
        contact_confidence = stage12_result["events"].get("CONTACT_confidence")

    total_frames = cal.get_frame_count(ref_path)

    def _confirm_step(label, emoji, auto_frame, session_key, slider_suffix, question, confidence_note=None):
        with st.expander(f"{emoji} Confirm {label}", expanded=st.session_state.get(session_key) is None):
            if auto_frame is None:
                st.error("Couldn't auto-detect this frame at all — scrub manually below.")
                auto_frame = 0
            else:
                note = f" ({confidence_note} confidence)" if confidence_note else ""
                st.info(f"Algorithm's best guess: **frame {auto_frame}**{note}. This is a first-pass "
                        f"heuristic, not yet tuned against real batting footage — always check it.")
            slider_key = f"batting_{slider_suffix}_confirm_slider"
            _render_frame_jump_box(slider_key, 0, max(total_frames - 1, 0))
            slider_val = st.slider(
                f"Scrub to the true {label.lower()} frame",
                min_value=0, max_value=max(total_frames - 1, 0),
                value=min(max(auto_frame, 0), max(total_frames - 1, 0)),
                key=slider_key,
            )
            frame_img = cal.extract_reference_frame(ref_path, frame_index=slider_val)
            if frame_img is not None:
                with _framed_image_container():
                    st.image(frame_img, use_column_width=True, caption=f"Frame {slider_val} — {question}")
            if st.button(f"✅ Confirm this is {label.lower()}", key=f"batting_confirm_{slider_suffix}_button"):
                st.session_state[session_key] = slider_val
                st.rerun()
            if st.session_state.get(session_key) is not None:
                st.success(f"Confirmed: {label.lower()} at frame {st.session_state[session_key]}.")

    _confirm_step("Stance", "🧍", stance_auto, stance_key, "stance",
                   "is the batter set in their stance, before any movement, here?")
    _confirm_step("Backlift (top of swing)", "🏏", backlift_auto, backlift_key, "backlift",
                   "is the bat at the top of the backlift here?")
    _confirm_step("Point of Contact", "🎯", contact_auto, contact_key, "contact",
                   "is this the moment the bat meets the ball?", confidence_note=contact_confidence)

    stance_confirmed = st.session_state.get(stance_key)
    backlift_confirmed = st.session_state.get(backlift_key)
    contact_confirmed = st.session_state.get(contact_key)
    if stance_confirmed is None or backlift_confirmed is None or contact_confirmed is None:
        return None

    return {
        "STANCE": stance_confirmed,
        "BACKLIFT": backlift_confirmed,
        "CONTACT": contact_confirmed,
        "STANCE_auto_detected": stance_auto,
        "BACKLIFT_auto_detected": backlift_auto,
        "CONTACT_auto_detected": contact_auto,
        "CONTACT_auto_confidence": contact_confidence,
    }


def render_batting_analysis_ui(player_name: str, history_enabled: bool):
    """
    Self-contained Batting Analysis flow — deliberately isolated from
    every bowling code path below in this file (per the explicit decision
    to add batting as a separate module, not touch the working, already-
    verified bowling pipeline). Single video upload; the camera can be
    EITHER side-on or front-on/rear-on (auto-detected, same as bowling) —
    a coach filming in the nets often can't get a side-on shot at all
    because the net physically obstructs it, so both must work.

    Called from the top-level "Analysis Type" branch, followed by
    st.stop() — nothing below that call site in the file executes when
    Batting Analysis is selected.
    """
    import batting_orchestrator as bto
    import usage_limits
    from coaching_agent import generate_batting_coaching_report

    st.markdown(_BATTING_THEME_CSS, unsafe_allow_html=True)

    # THEMED INTRO HEADER — the coach's own feedback (2026-08-03): the report
    # section got the "floodlit night match" redesign, but the page still
    # opened with a bare st.title()/st.caption() indistinguishable from the
    # old plain UI, so it "looked the same" until Execute was clicked.
    #
    # BUG FIX (2026-08-03, "double header" spotted on a real coach test):
    # this used to re-render the logo + "APEX COACH AI" wordmark + tagline
    # a SECOND time — but that exact branding (logo + "AUTONOMOUS
    # BIOMECHANICAL PERFORMANCE HUB") already renders once, globally, at
    # the very top of the whole app (see the "LOGO (unchanged)" section
    # further down this file, which runs before the sidebar/auth gate for
    # every analysis type). Repeating it here stacked two near-identical
    # headers on the same page. Kept only the "Batting Analysis" section
    # marker here — that's the one piece of information this page adds
    # that the global header doesn't already show.
    _batting_html("""
<div class="apex-batting-theme">
  <div class="topbar">
    <div class="brand">
      <span class="module">🏏 Batting Analysis</span>
    </div>
  </div>
  <div class="intro-note">
    Stance, head position, weight transfer, downswing plane, and top-elbow control — from a
    single phone, filmed either side-on or front-on/rear-on. <b>front_foot_alignment,
    downswing_plane, and top_elbow_angle</b> are derived from body-pose landmarks only —
    downswing_plane specifically uses the midpoint of both wrists as a proxy for the bat
    handle's path, not the bat's actual face angle. Treat borderline readings on these three
    with appropriate caution, same as any first-version metric in this app.
  </div>
</div>
""")

    uploaded_batting_video = st.sidebar.file_uploader(
        "Batting Video (.mp4 or .mov)", type=["mp4", "mov", "m4v"], key="batting_video_upload",
    )
    if uploaded_batting_video is None:
        st.warning("👆 Upload a batting video in the sidebar to begin.")
        return

    st.sidebar.divider()
    batting_hand_choice = st.sidebar.selectbox(
        "🏏 Batting Hand", ["Auto-detect", "Right-handed", "Left-handed"],
        help="Auto-detected from which wrist sits higher at stance. Override here if you know it.",
        key="batting_hand_choice",
    )
    batting_hand_override = {"Auto-detect": None, "Right-handed": "left", "Left-handed": "right"}[batting_hand_choice]

    # CAMERA ANGLE — auto-detected the same way bowling already does
    # (camera_angle_detection.py's shoulder-width/height ratio), because
    # nets sessions often can't get a side-on shot (the net obstructs it).
    # front_foot_alignment and the falling-over check are unaffected by
    # this either way (see batting_kinematics._derive_batting_axes); only
    # weight_transfer and downswing_plane get a reduced-confidence caveat
    # under front-on/rear-on filming.
    camera_angle_choice = st.sidebar.selectbox(
        "📐 Filming Angle", ["Auto-detect", "Side-on", "Front-on / Rear-on"],
        help="Auto-detect can misjudge this — set it manually if you know how this clip was filmed.",
        key="batting_camera_angle_choice",
    )
    camera_angle_override = {
        "Auto-detect": None, "Side-on": "side_on", "Front-on / Rear-on": "front_or_rear",
    }[camera_angle_choice]

    file_identity = f"{uploaded_batting_video.name}_{uploaded_batting_video.size}"
    if st.session_state.get("_batting_ref_identity") != file_identity:
        os.makedirs("input", exist_ok=True)
        ref_path = os.path.abspath(os.path.join("input", f"_batting_ref_{uploaded_batting_video.name}"))
        try:
            o.save_uploaded_video_capped(uploaded_batting_video, ref_path)
        except RuntimeError as e:
            st.error(f"⚠️ {e}")
            st.stop()
        st.session_state["_batting_ref_identity"] = file_identity
        st.session_state["_batting_ref_path"] = ref_path
        st.session_state["_batting_seed_point"] = None
        st.session_state["_batting_stage12"] = None

    ref_path = st.session_state["_batting_ref_path"]
    total_frames = cal.get_frame_count(ref_path)

    with st.expander("🎯 Confirm the batter", expanded=st.session_state.get("_batting_seed_point") is None):
        st.caption(
            "Scrub to any frame where the batter is clearly visible, then click directly on them. "
            "This tells the app exactly who to track — the same reliable identity-tracking fix "
            "bowling analysis already relies on, instead of guessing."
        )
        seed_slider_key = "batting_seed_frame_slider"
        _render_frame_jump_box(seed_slider_key, 0, max(total_frames - 1, 0))
        seed_frame_idx = st.slider(
            "Scrub to a frame where the batter is clearly visible",
            min_value=0, max_value=max(total_frames - 1, 0),
            value=0, key=seed_slider_key,
        )
        # BUG FOUND (2026-08-03, comparing against render_bowler_seed_ui's
        # proven version): that function clears its stored point whenever
        # the scrub slider moves to a different frame — this one didn't,
        # so a point clicked on frame 40 could still be reused as the
        # "existing point" while previewing frame 200, silently mismatched
        # against a frame it was never actually clicked on. Not the cause
        # of the zero-detection framing issue found the same day (that's
        # camera positioning, confirmed by direct visual inspection of two
        # different clips), but a real correctness gap regardless.
        if st.session_state.get("_batting_seed_last_frame_idx") != seed_frame_idx:
            st.session_state["_batting_seed_point"] = None
            st.session_state["_batting_seed_last_frame_idx"] = seed_frame_idx

        seed_frame_img = cal.extract_reference_frame(ref_path, frame_index=seed_frame_idx)
        if seed_frame_img is not None:
            from PIL import Image
            pil_seed_img = Image.fromarray(seed_frame_img)
            existing_point = st.session_state.get("_batting_seed_point")
            new_point = render_zoomable_click_image(
                pil_seed_img, key_prefix="batting_seed", marker_point=existing_point, marker_color="lime",
            )
            if new_point is not None and new_point != existing_point:
                st.session_state["_batting_seed_point"] = new_point
                st.session_state["_batting_seed_frame_idx"] = seed_frame_idx
                st.session_state["_batting_stage12"] = None  # a new seed invalidates any cached stage1+2
                st.rerun()

    seed_point = st.session_state.get("_batting_seed_point")
    if seed_point is None:
        st.warning("👆 Click the batter above to continue.")
        return

    # Re-run stage 1+2 if the coach changes the handedness/camera-angle
    # override after it already ran once — same reasoning as the seed-
    # point invalidation above.
    _batting_overrides_key = (batting_hand_override, camera_angle_override)
    if st.session_state.get("_batting_overrides_key") != _batting_overrides_key:
        st.session_state["_batting_stage12"] = None
        st.session_state["_batting_overrides_key"] = _batting_overrides_key

    if st.session_state.get("_batting_stage12") is None:
        with st.spinner("Tracking the batter and detecting phase events..."):
            stage12 = bto.extract_and_detect_batting_events(
                ref_path, output_dir="output",
                seed_point=seed_point,
                seed_frame_index=st.session_state.get("_batting_seed_frame_idx", 0),
                batting_hand_override=batting_hand_override,
                camera_angle_override=camera_angle_override,
            )
        st.session_state["_batting_stage12"] = stage12

    stage12 = st.session_state["_batting_stage12"]
    if stage12.get("status") != "success":
        st.error(f"⚠️ Tracking failed: {stage12.get('message', 'unknown error')}")
        return

    if stage12.get("angle_estimate") is not None:
        _ae = stage12["angle_estimate"]
        _angle_label = {"side_on": "Side-on", "front_or_rear": "Front-on / Rear-on",
                         "uncertain": "Uncertain"}.get(_ae.angle, _ae.angle)
        _angle_source = "manually set" if camera_angle_override else "auto-detected"
        st.markdown(
            f'<div class="apex-batting-theme"><span class="pill">📐 Filming angle '
            f'({html.escape(_angle_source)}): <b>{html.escape(str(_angle_label))}</b> '
            f'— {html.escape(_ae.confidence_note)}</span></div>',
            unsafe_allow_html=True,
        )

    # DIAGNOSTIC (2026-08-03, added after two real tests both came back
    # with zero usable metrics): report tracking coverage immediately
    # after stage 1+2, instead of only surfacing this after the coach has
    # already confirmed all three events and run the full analysis. Both
    # real failures were traced to camera framing (the batter too small/
    # distant in frame for MediaPipe to place joints at all, confirmed by
    # directly inspecting the actual video frames) — this makes that
    # visible in seconds instead of requiring a database/CSV inspection.
    _stage12_df = stage12["df"]
    _tracked_frac = _stage12_df["NOSE_x"].notna().mean() if len(_stage12_df) else 0.0
    if _tracked_frac < 0.10:
        st.error(
            f"⚠️ Only {_tracked_frac:.0%} of frames had a detectable pose for the person you clicked. "
            "This almost always means the batter is too small/distant in this camera framing — "
            "film with the camera closer to the batter (not from the bowler's end down the full "
            "pitch length), the same way a bowling clip is filmed close to the bowler. Metrics below "
            "are very unlikely to compute from this clip."
        )
    elif _tracked_frac < 0.50:
        st.warning(
            f"⚠️ Only {_tracked_frac:.0%} of frames had a detectable pose — tracking is partial. "
            "Results may be incomplete; a camera position closer to the batter will help."
        )

    confirmed_events = render_batting_event_confirmation(stage12, ref_path, file_identity)
    if confirmed_events is None:
        st.warning("👆 Confirm Stance, Backlift, and Point of Contact above to enable analysis.")
        return

    # STRUCTURAL SPLIT (2026-08-04, coach feedback: "the fundamental grid
    # remains exactly the same" across several rounds of CSS-only
    # redesign) — this is a real change in composition, not another
    # repaint: Configure/Run and Report are now genuinely separate
    # Streamlit tabs (a different interaction model, not just a
    # different-looking single scroll), and the report's video sits
    # beside its metrics in real st.columns rather than stacked full-
    # width sections. Both use Streamlit's own native layout primitives
    # instead of CSS-injection tricks, deliberately — repeated selector
    # mismatches earlier this project (stVideo, stImage) were the direct
    # cost of fighting the framework's DOM instead of using it.
    tab_setup, tab_report = st.tabs(["⚙️ Configure & Run", "📊 Report"])

    with tab_setup:
        st.caption(
            "There's no ball-tracking signal yet to know the shot/line automatically — tell the "
            "app what happened on this delivery so it can judge front-foot direction against the "
            "actual shot, and check for the head/foot 'falling over' fault against the actual "
            "line bowled."
        )
        col_shot, col_line = st.columns(2)
        with col_shot:
            shot_choice = st.selectbox(
                "🏏 Shot Played",
                list(_SHOT_PLAYED_OPTIONS.keys()),
                key="batting_shot_played_choice",
            )
            if _SHOT_PLAYED_OPTIONS[shot_choice] in bk.NOT_APPLICABLE_SHOTS:
                st.caption(
                    "ℹ️ Front-foot alignment isn't scored for this shot — \"foot points toward "
                    "the shot\" is a front-foot-drive/defense concept, not a back-foot or "
                    "horizontal-bat one. The foot's actual direction is still shown, just "
                    "without a pass/fail target."
                )
        with col_line:
            line_choice = st.selectbox(
                "🎯 Line Bowled (for falling-over check)",
                ["Not sure / skip this check", "Off Stump", "Middle Stump", "Leg Stump"],
                key="batting_ball_line_choice",
            )
        shot_played = _SHOT_PLAYED_OPTIONS[shot_choice]
        ball_line = {
            "Not sure / skip this check": None, "Off Stump": "off", "Middle Stump": "middle", "Leg Stump": "leg",
        }[line_choice]

        _is_admin_user = usage_limits.is_admin(st.session_state.auth_user.get("email", ""))
        _usage = {"remaining": 1, "used": 0, "limit": 1}
        if not _is_admin_user:
            _usage = usage_limits.get_usage(st.session_state.auth_user["id"])
            if _usage["remaining"] <= 0:
                st.error(
                    f"🚫 Free analysis limit reached ({_usage['used']}/{_usage['limit']} today). "
                    "Contact us to unlock unlimited access."
                )
            else:
                st.caption(f"🎟️ {_usage['remaining']} of {_usage['limit']} free analyses remaining")

        # BUG FIX: this used to be a bare `return` when usage was exhausted
        # — harmless in the old single-scroll layout (nothing further to
        # show anyway), but here it would also skip rendering the Report
        # tab entirely, hiding an already-completed earlier result. Gating
        # just the button avoids that.
        if _usage["remaining"] > 0:
            if st.button("🚀 Execute Batting Analysis", use_container_width=True):
                with st.spinner("Calculating batting technique metrics..."):
                    stage12_with_overrides = dict(stage12)
                    stage12_with_overrides["events"] = {**stage12["events"], **{
                        "STANCE": confirmed_events["STANCE"],
                        "BACKLIFT": confirmed_events["BACKLIFT"],
                        "CONTACT": confirmed_events["CONTACT"],
                    }}
                    result_payload = bto.run_batting_analysis(
                        ref_path, output_dir="output", precomputed=stage12_with_overrides,
                        shot_played=shot_played, ball_line=ball_line,
                    )

                if result_payload.get("status") != "success":
                    st.error(f"⚠️ Analysis failed: {result_payload.get('message', 'unknown error')}")
                else:
                    # Fold the auto-vs-confirmed pairs into the result so they
                    # get saved to history below — same (auto_detected,
                    # coach_confirmed) training-signal pattern already used
                    # for bowling's release point/foot contacts.
                    result_payload["time_indices"]["stance_frame_auto_detected"] = confirmed_events["STANCE_auto_detected"]
                    result_payload["time_indices"]["backlift_frame_auto_detected"] = confirmed_events["BACKLIFT_auto_detected"]
                    result_payload["time_indices"]["contact_frame_auto_detected"] = confirmed_events["CONTACT_auto_detected"]
                    result_payload["time_indices"]["contact_auto_confidence"] = confirmed_events["CONTACT_auto_confidence"]

                    st.session_state["_batting_result_payload"] = result_payload
                    if not _is_admin_user:
                        usage_limits.record_usage(st.session_state.auth_user["id"])
                    st.success("✅ Analysis complete — open the **📊 Report** tab above to view it.")

    with tab_report:
        result_payload = st.session_state.get("_batting_result_payload")
        if result_payload is None:
            st.info(
                "👈 Pick the shot/line and click **Execute Batting Analysis** in the "
                "**Configure & Run** tab to see the report here."
            )
            return

        metrics = result_payload["biomechanical_metrics"]
        hand_val = metrics.get("batting_hand_detected", "Unknown")
        _angle_label = {"side_on": "Side-on", "front_or_rear": "Front-on / Rear-on",
                         "uncertain": "Uncertain", "unavailable": "Unavailable"}.get(
            result_payload.get("camera_angle"), "Unknown")
        view_caveats = result_payload.get("view_confidence_caveats") or []
        caveat_pill_html = ""
        if view_caveats:
            _caveat_labels = ", ".join(mr.RANGES[k].label for k in view_caveats)
            caveat_pill_html = f'<span class="pill pill-warn">⚠ Extra caution: {html.escape(_caveat_labels)}</span>'

        # BUG FIX (2026-08-03, "double header"): dropped the repeated logo +
        # "APEX COACH AI" wordmark here too — same reasoning as the intro
        # header fix above, that branding already renders once, globally, at
        # the top of the whole app. This bar's actual job is the per-session
        # info (leading side, filming angle, caveats), which it keeps.
        _batting_html(f"""
    <div class="apex-batting-theme">
      <div class="topbar">
        <div class="brand">
          <span class="module">🏏 Batting Analysis — Report</span>
        </div>
        <div class="identity">
          <span class="pill">Leading side: <b>{html.escape(str(hand_val))}</b></span>
          <span class="pill">📐 Filming angle: <b>{html.escape(_angle_label)}</b></span>
          {caveat_pill_html}
        </div>
      </div>
    </div>
    """)

        falling_over = result_payload.get("falling_over_risk", {})
        _fo_status = falling_over.get("status")
        if _fo_status == "success" and falling_over.get("flagged"):
            _head_pct = falling_over.get("head_shift_pct")
            _foot_pct = falling_over.get("foot_cross_pct")
            _bar_scale = 40.0  # visual scale only, chosen so the 15% flag threshold reads clearly on the bar — the numbers shown as text are always the real, unscaled percentages
            _head_bar = min(abs(_head_pct or 0) / _bar_scale * 100, 100)
            _foot_bar = min(abs(_foot_pct or 0) / _bar_scale * 100, 100)
            _batting_html(f"""
    <div class="apex-batting-theme">
      <div class="flag">
        <div class="stripe"></div>
        <div class="flag-body">
          <div class="flag-title">🚩 Red Flag — Falling Over</div>
          <div class="flag-desc">{html.escape(falling_over.get('reason') or '')}</div>
          <div class="flag-meters">
            <div class="flag-meter">Head drift toward danger side<br><b>{html.escape(str(_head_pct))}%</b> of stance width
              <div class="flag-bar"><span style="width:{_head_bar:.0f}%"></span></div>
            </div>
            <div class="flag-meter">Front-foot cross toward danger side<br><b>{html.escape(str(_foot_pct))}%</b> of stance width
              <div class="flag-bar"><span style="width:{_foot_bar:.0f}%"></span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """)
        elif _fo_status == "success":
            _line = html.escape(str(result_payload.get("ball_line") or ""))
            _batting_html(f"""
    <div class="apex-batting-theme">
      <div class="flag is-clear">
        <div class="stripe"></div>
        <div class="flag-body">
          <div class="flag-title">Falling-Over Check — Clear</div>
          <div class="flag-desc">Head/front-foot drift did not both point toward the danger side of this
            {_line}-stump delivery. No compound fault detected on this ball.</div>
        </div>
      </div>
    </div>
    """)
        elif _fo_status == "not_applicable":
            st.markdown(
                '<div class="apex-batting-theme"><div class="flag is-neutral"><div class="stripe"></div>'
                '<div class="flag-body"><div class="flag-title">Falling-Over Check — Not Evaluated</div>'
                '<div class="flag-desc">This check only runs for Off Stump or Leg Stump lines, where a '
                '"wrong side" drift is well-defined.</div></div></div></div>',
                unsafe_allow_html=True,
            )
        elif _fo_status == "error":
            st.markdown(
                '<div class="apex-batting-theme"><div class="flag is-neutral"><div class="stripe"></div>'
                '<div class="flag-body"><div class="flag-title">Falling-Over Check — Unavailable</div>'
                '<div class="flag-desc">Could not be evaluated for this clip (insufficient tracking data at '
                'the stance/contact frames).</div></div></div></div>',
                unsafe_allow_html=True,
            )

        # RESTRUCTURE (2026-08-04): video and its metrics now sit side by
        # side in real Streamlit columns, instead of the video appearing
        # in its own full-width block below a separate full-width metrics
        # grid. Same content, genuinely different composition.
        col_video, col_metrics = st.columns([1, 1.5], gap="medium")
        with col_video:
            if result_payload.get("annotated_video_output"):
                st.video(result_payload["annotated_video_output"])
            else:
                st.caption("Annotated video unavailable for this session.")
        with col_metrics:
            st.markdown(
                '<div class="apex-batting-theme"><div class="section-label">'
                'Batting Technique Telemetry — Stance → Backlift → Contact</div></div>',
                unsafe_allow_html=True,
            )
            _cards_html = "".join(
                _batting_metric_card_html(mkey, metrics, view_caveats) for mkey in mr.all_batting_metric_keys()
            )
            st.markdown(f'<div class="apex-batting-theme"><div class="grid">{_cards_html}</div></div>',
                        unsafe_allow_html=True)

        with st.spinner("Generating expert batting coaching report..."):
            insights = generate_batting_coaching_report(result_payload)

        narrative_html = _batting_narrative_to_html(insights.get("narrative_analysis", ""))
        drills = insights.get("prescribed_drills") or []
        drills_html = ""
        if drills:
            _drill_items = []
            for drill in drills:
                dname, _sep, ddesc = drill.partition(":")
                if ddesc.strip():
                    _drill_items.append(
                        f'<li><span class="dname">{html.escape(dname.strip())}</span>'
                        f'{html.escape(ddesc.strip())}</li>'
                    )
                else:
                    _drill_items.append(f'<li>{html.escape(drill.strip())}</li>')
            drills_html = f'<ul class="drills">{"".join(_drill_items)}</ul>'

        _batting_html(f"""
    <div class="apex-batting-theme">
      <div class="report-card">
        <div class="report-head"><span class="dot"></span><span class="report-title">AI Batting Coach — Assessment</span></div>
        <p class="narrative">{narrative_html}</p>
        {drills_html}
      </div>
    </div>
    """)

        clean_slug = player_name.replace(" ", "_")
        pdf_data = generate_batting_pdf_report(
            metrics, result_payload["time_indices"], insights, batter_name=player_name,
            falling_over_risk=falling_over,
        )
        st.download_button(
            label="📄 Download Official PDF Report",
            data=pdf_data,
            file_name=f"Batting_Report_{clean_slug}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        if history_enabled and not st.session_state.get("batting_history_saved_for_run", False):
            try:
                athlete_id = store.get_or_create_athlete(player_name, st.session_state.auth_user["id"])
                store.save_session(
                    athlete_id=athlete_id,
                    coach_user_id=st.session_state.auth_user["id"],
                    video_filename=os.path.basename(ref_path),
                    camera_mode="Batting - Single Camera",
                    fps=result_payload["video_metadata"]["fps"],
                    metrics=metrics,
                    phase_durations=None,
                    release_arm_speed_kmh=None,
                    speed_status="unavailable",
                )
                st.session_state["batting_history_saved_for_run"] = True
            except Exception as e:
                monitoring.capture(e)
                st.warning(f"Could not save this session to athlete history: {e}")


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
                         speed_result=None, quality=None, bowler_type=None):
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
    story.append(pcr.build_color_coded_range_table(metrics, bold_body, bowler_type))
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


def generate_batting_pdf_report(metrics, frames, ai_insights, batter_name="Elite Athlete",
                                 falling_over_risk=None):
    """
    Batting equivalent of generate_pdf_report — same reportlab primitives,
    same visual language, simplified for batting's metrics (no speed/
    phase-duration section, which doesn't apply to batting analysis).
    Deliberately builds its own color-coded table with mr.classify/
    mr.format_value directly rather than reusing pdf_color_ranges.py,
    which iterates the bowling-only metric set.

    falling_over_risk: optional result dict from
    batting_kinematics.detect_falling_over_risk (via
    result_payload["falling_over_risk"]) — when flagged, called out as
    its own red-flag paragraph, same as the Streamlit UI's alert box.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()

    # Print-safe echo of the on-screen "floodlit night match" palette —
    # the raw --ink/--accent hex values from the Streamlit theme are too
    # light/washed-out as TEXT on white paper, so these are darkened,
    # readable equivalents of the same two hues (dark pitch-green ink,
    # dark willow-gold) rather than a copy-paste of the screen colors.
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'],
                                  fontSize=22, leading=26, textColor=colors.HexColor('#182019'))
    h2_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'],
                               fontSize=14, leading=18, textColor=colors.HexColor('#8A6A24'),
                               spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'],
                                 fontSize=10, leading=14, textColor=colors.HexColor('#2D3748'))
    bold_body = ParagraphStyle('ReportBodyBold', parent=body_style, fontName='Helvetica-Bold')

    current_date = datetime.now().strftime("%Y-%m-%d")
    story.append(Paragraph("APEX COACH AI — BATTING TECHNIQUE REPORT", title_style))
    story.append(Paragraph(
        f"<b>Target Athlete:</b> {batter_name} | "
        f"<b>Date:</b> {current_date} | "
        f"<b>Batting Hand:</b> {metrics.get('batting_hand_detected', 'Unknown')} | "
        f"<b>Status:</b> AI-Assisted Biomechanical Estimate",
        body_style
    ))
    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "front_foot_alignment, downswing_plane, and top_elbow_angle are derived from body-pose "
        "landmarks only (no bat-tracking sensor) — downswing_plane uses the midpoint of both "
        "wrists as a proxy for the bat handle's path, not the bat's actual face angle.",
        ParagraphStyle('Caveat', parent=body_style, fontSize=8, textColor=colors.HexColor('#718096'))
    ))
    story.append(Spacer(1, 10))

    if falling_over_risk and falling_over_risk.get("status") == "success" and falling_over_risk.get("flagged"):
        story.append(Paragraph(
            f"🚩 RED FLAG — FALLING OVER: {falling_over_risk.get('reason', '')}",
            ParagraphStyle('RedFlag', parent=bold_body, fontSize=10, textColor=colors.HexColor('#C53030'))
        ))
        story.append(Spacer(1, 10))

    story.append(Paragraph("Phase Milestones", h2_style))
    time_rows = [
        [Paragraph("<b>Phase</b>", bold_body), Paragraph("<b>Frame</b>", bold_body)],
        ["Stance", f"Frame {frames.get('stance_frame', 'N/A')}"],
        ["Backlift (top of swing)", f"Frame {frames.get('backlift_frame', 'N/A')}"],
        ["Point of Contact", f"Frame {frames.get('contact_frame', 'N/A')}"],
    ]
    t_time = Table(time_rows, colWidths=[250, 250])
    t_time.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3ECD9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_time)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Batting Technique Telemetry", h2_style))
    story.append(Paragraph(
        "<i>Reference ranges — Optimal (green) | Acceptable (amber) | Critical (red).</i>",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Reuses metric_ranges.py's own PDF tint table (mr.TIER_COLORS_PDF)
    # instead of a second, separately-hardcoded copy of the same four
    # colors — this table already matches the on-screen good/warn/crit
    # zone colors' intent, just as soft print-safe fills.
    zone_bg = {k: colors.HexColor(v) for k, v in mr.TIER_COLORS_PDF.items()}
    metric_rows = [[Paragraph("<b>Metric</b>", bold_body), Paragraph("<b>Value</b>", bold_body),
                    Paragraph("<b>Zone</b>", bold_body), Paragraph("<b>Optimal</b>", bold_body)]]
    row_colors = [colors.HexColor('#F3ECD9')]
    for mkey in mr.all_batting_metric_keys():
        value = mr.extract_batting_metric_value(metrics, mkey)
        zone = mr.classify(mkey, value)
        r = mr.RANGES[mkey]
        metric_rows.append([
            r.label,
            mr.format_value(mkey, value) if value is not None else "N/A",
            zone.upper(),
            r.display_optimal,
        ])
        row_colors.append(zone_bg.get(zone, zone_bg["unknown"]))

    t_metrics = Table(metric_rows, colWidths=[190, 100, 90, 120])
    style_cmds = [
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]
    for i, bg in enumerate(row_colors):
        style_cmds.append(('BACKGROUND', (0, i), (-1, i), bg))
    t_metrics.setStyle(TableStyle(style_cmds))
    story.append(t_metrics)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Autonomous AI Batting Coach Assessment", h2_style))
    narrative = ai_insights.get("narrative_analysis", "No narrative generated.")
    narrative = narrative.replace(
        "SECTION 1 — BATTING TECHNIQUE NARRATIVE ASSESSMENT:", ""
    ).strip()
    story.append(Paragraph(narrative, body_style))
    story.append(Spacer(1, 15))

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
# ANALYSIS TYPE — batting is a deliberately separate, isolated module
# (see render_batting_analysis_ui's docstring). Placed here, before
# Speed Calibration, because batting's 5 metrics need no real-world
# scale at all — nothing below this branch runs for Batting Analysis.
# ---------------------------------------------------------------
st.sidebar.header("🏏 Analysis Type")
analysis_type = st.sidebar.radio(
    "What are we analyzing?", ["Bowling Analysis", "Batting Analysis"], key="analysis_type_choice",
)
if analysis_type == "Batting Analysis":
    render_batting_analysis_ui(player_name, history_enabled)
    st.stop()  # nothing below this renders for Batting Analysis — bowling code path is untouched
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

    # LOAD A SAVED CALIBRATION — the "once per setup" promise above only
    # held within a single browser session before this: st.session_state
    # is wiped the moment the tab closes or the server sleeps, so a coach
    # was redoing the same upload-and-click flow far more often than the
    # design intended. See add_camera_calibrations.sql for the real fix —
    # this lets a coach reload a named setup instantly instead.
    if not st.session_state.calibration:
        _saved_calibrations = store.list_calibrations(st.session_state.auth_user["id"])
        if _saved_calibrations:
            _saved_labels = [c["setup_label"] for c in _saved_calibrations]
            _load_choice = st.selectbox(
                "Or load a saved camera setup", _saved_labels, key="load_calib_choice",
            )
            if st.button("📥 Load this calibration"):
                _match = next(c for c in _saved_calibrations if c["setup_label"] == _load_choice)
                st.session_state.calibration = cal.Calibration.from_dict(_match["calibration"])
                st.session_state.calibration_frame_width = _match["frame_width_px"]
                st.success(f"Loaded calibration: {_load_choice}")
                st.rerun()
            st.divider()

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
        # ADDED (2026-08-05, coach request): stump width needs at least two
        # stumps clearly spaced apart in frame — unusable when only one
        # stump is visible (partial framing, occlusion, a rear-view camera
        # positioned close behind one stump). Stump HEIGHT (28 inches =
        # 0.7112m, the same official standard for every stump on any
        # ground) is a real, known vertical distance available from just
        # ONE stump — click its top and where it meets the ground.
        "Stump height (0.7112m) — works from just one visible stump": {
            "distance_m": 0.7112, "label": "stump height",
            "point1_prompt": "the top of one stump",
            "point2_prompt": "the base of that SAME stump, where it meets the ground",
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
                        st.session_state.calibration_frame_width = pil_img.width
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

        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            setup_name = st.text_input(
                "Save this camera setup as", key="save_calib_label",
                placeholder="e.g. Home nets — evening spot",
            )
        with save_col2:
            st.write("")
            if st.button("💾 Save for future sessions", use_container_width=True, disabled=not setup_name.strip()):
                store.save_calibration(
                    st.session_state.auth_user["id"], setup_name,
                    c.to_dict(), st.session_state.get("calibration_frame_width", 0),
                )
                st.success(f"Saved — reload it anytime as \"{setup_name.strip()}\".")

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

# BOWLER TYPE (2026-08-04) — real audit found spin bowling needs its own
# reference ranges, not fast bowling's: e.g. a wrist-spinner's natural
# front-knee angle can sit well below what pace's "collapsing knee"
# threshold would flag, because dropping the center of mass is part of
# normal wrist-spin technique, not a fault. There is no auto-detection
# for this (unlike bowling arm) — the coach knows what they filmed.
# Only front-knee-bracing has a real, correctly-mapped source for
# wrist-spin so far (see metric_ranges.SPIN_RANGE_OVERRIDES); every other
# metric — and everything for finger-spin — shows as a plain measured
# number with no green/amber/red verdict rather than silently borrowing
# pace's band (a real spinner's normal technique could otherwise get
# falsely flagged against fast-bowling-calibrated numbers).
bowler_type_choice = st.sidebar.selectbox(
    "🌀 Bowler Type",
    ["Pace (fast bowling)", "Finger-Spin (off-spin / left-arm orthodox)",
     "Wrist-Spin (leg-spin / left-arm wrist-spin)"],
    help="Changes which reference ranges your metrics are judged against. "
         "Where no validated spin-specific range exists yet, that metric shows "
         "as a plain measurement instead of a pass/fail verdict — see the "
         "Reference Ranges section in your report for exactly which ones."
)
bowler_type = {
    "Pace (fast bowling)": None,
    "Finger-Spin (off-spin / left-arm orthodox)": "finger_spin",
    "Wrist-Spin (leg-spin / left-arm wrist-spin)": "wrist_spin",
}[bowler_type_choice]

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
            "the skeleton ever locking onto the wrong person. ⚠️ If someone else "
            "(a batter, umpire, teammate) is visible before the bowler comes into "
            "frame, make sure you're clicking on the actual bowler, not them — "
            "this click is treated as ground truth, so it's the one step nothing "
            "else in the app can catch or correct if it's wrong."
        )
        if total_frames > 1:
            _render_frame_jump_box(f"{key_prefix}_seed_slider", 0, max(total_frames - 1, 0))
            frame_idx = st.slider(
                "Scrub to a frame with the bowler visible",
                min_value=0, max_value=max(total_frames - 1, 0),
                # FIX (2026-08-08, real bug the coach caught): was
                # total_frames // 3 — for a typical bowling recording
                # (run-up, then delivery, then follow-through, with
                # recording often starting before the bowler even enters
                # frame — see orchestrator._compute_segment_sum_body_
                # height's scale-consistency-guard comment for the real
                # clip this was found on), 1/3 of the way through a clip
                # is frequently still early run-up or even before the
                # bowler has entered frame at all — confirmed directly:
                # for the coach's actual 173-frame clip (bowler entering
                # at frame ~75), this default landed on frame 57, squarely
                # in the region where a DIFFERENT person (the batsman) was
                # the only one visible. A coach who clicks without
                # scrubbing first could seed the wrong person entirely —
                # the single most severe version of that failure mode,
                # since a coach-given seed is trusted as absolute ground
                # truth. 3/4 of the way through lands much more reliably
                # at or after delivery, well past any pre-entry footage.
                value=min(total_frames - 1, total_frames * 3 // 4),
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
                    bowler_type=bowler_type,
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
                    bowler_type=bowler_type,
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
                # release_height/head_stability's "tracking uncertain" flag
                # (see orchestrator.calculate_release_height_ratio_safe's
                # br_tracking_confidence docstring) is driven by detect_
                # delivery_events' br_confidence — a coarse, whole-search-
                # window aggregate. speed_result's OWN instability check
                # (_corroborated_peak_speed_px_s, a much stricter frame-by-
                # frame R^2 fit-quality test right around release) is a
                # SEPARATE signal that can fire even when br_confidence
                # still reads "high" — confirmed live: a real session showed
                # "Tracking around release was too unstable..." for the
                # speed estimate while release_height's warning never
                # appeared. Both checks look at the same underlying release-
                # frame landmark quality from different angles — combine
                # them here (the one place both results already exist)
                # rather than trusting either signal alone.
                #
                # REAL BUG FOUND (2026-08-07, coach pushed back hard and was
                # right): this unconditionally set release_height's flag
                # even when the coach had manually clicked the exact release
                # point (_wrist_confirmed_point) — the ONE piece of real
                # human ground truth this whole app is built to prioritize
                # (calculate_release_height_ratio_safe already correctly
                # skips its OWN br_tracking_confidence check for a coach-
                # confirmed point; this cross-check was silently overriding
                # that correct behavior from outside). speed_result's
                # instability check has no way to know about or use the
                # coach's click at all — it measures raw frame-to-frame
                # velocity, which a single confirmed point can't fix — so it
                # can fail for reasons a confirmed release point doesn't
                # address, but that's not evidence the confirmed point
                # itself is wrong. head_stability has no equivalent
                # confirmable point (it's a whole-window variance, not one
                # click), so it keeps applying the cross-check unconditionally.
                _coach_confirmed_release_point = (
                    st.session_state.get("_wrist_confirmed_point") is not None
                    and st.session_state.get("_br_confirmed_frame") is not None
                )
                if speed_result.get("reason") == "tracking_unstable":
                    if not _coach_confirmed_release_point:
                        metrics.setdefault("release_height", {})["release_frame_tracking_uncertain"] = True
                    metrics.setdefault("head_stability", {})["release_window_tracking_uncertain"] = True
                height_absolute_result = se.compute_release_height_absolute(
                    metrics.get("release_height", {}).get("debug_raw"), cap_h, meters_per_pixel=mpp
                )
                # Automatic bowler-height estimate — no manual entry, ever
                # (a coach explicitly said no coach will realistically
                # provide this): converted through the same stump
                # calibration. Also a real cross-check on the release-
                # height cm figure above — a release point taller than
                # ~1.3x the bowler's own estimated height is implausible
                # regardless of what the ratio-based check already allows.
                #
                # FIX (roadmap item #1, 2026-08-06): this used to reuse
                # release_height's own debug_raw["body_height"] — the WIDE
                # early-run-up segment-sum baseline. That's the right
                # baseline for the release-height RATIO (scale-invariant),
                # but wrong here: this feature divides by meters_per_pixel,
                # a scale only valid at the stump-calibration plane's
                # depth, and early run-up frames are physically much
                # farther from the stumps than release is. That mismatch
                # is exactly what produced a real, confirmed 444cm
                # implausible reading. Use the separate BFC±15 narrow-
                # window baseline instead — see orchestrator.py's
                # segment_sum_body_height_for_cm and
                # _compute_segment_sum_body_height's search_start_frame
                # docstring for the full reasoning.
                _segment_sum_for_height = metrics.get("release_height", {}).get("segment_sum_body_height_for_cm")
                estimated_height_result = se.compute_estimated_standing_height(
                    _segment_sum_for_height, cap_h, meters_per_pixel=mpp
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

                # kinematics.py's own "tier" strings (e.g. "Collapsing Knee
                # Joint") are pace-calibrated absolute-angle labels,
                # computed independently of metric_ranges.classify() —
                # showing one directly for a bowler_type with no validated
                # range for that metric (see SPIN_RANGE_OVERRIDES) would
                # silently reintroduce the exact misleading pace-vs-spin
                # verdict classify()'s "descriptive" tier exists to avoid.
                _display_bowler_type = result_payload.get("bowler_type")
                def _tier_caption(metric_key, raw_tier, value=None):
                    # FIX (2026-08-06): used to only check bowler_type, so
                    # front_knee_bracing/hip_shoulder_separation (real audit
                    # found NO universal pass/fail band exists for either,
                    # for ANY bowler_type — see metric_ranges._ALWAYS_
                    # DESCRIPTIVE_METRICS) still showed kinematics.py's raw
                    # "Elite Rigid Extension"/"Collapsing Knee Joint" pace
                    # verdict text for pace bowlers. mr.has_validated_range
                    # is the single source of truth every render site now
                    # shares (also used by video_overlay.py's hero chart and
                    # the Bowler Profile history chart).
                    #
                    # REGRESSION FOUND (2026-08-06, real clip test): this
                    # override fired even when value was None (a genuine
                    # tracking failure this delivery) — confirmed on a real
                    # session where Lead Knee Bracing showed "N/A" right
                    # next to the confident-sounding "Front knee action at
                    # release is a real technique CLASSIFICATION..." text,
                    # hiding the real diagnostic (raw_tier would have said
                    # "Data Deficit"/"Tracking Drop"). A missing value must
                    # show the real error state, not a classification note
                    # that implies something was actually measured.
                    if value is None or (isinstance(value, float) and value != value):  # None or NaN
                        return raw_tier
                    if not mr.has_validated_range(metric_key, _display_bowler_type):
                        return mr.descriptive_note(metric_key, value, _display_bowler_type)
                    return raw_tier

                detected_arm = metrics.get("bowling_arm_detected")
                if detected_arm:
                    arm_source = "manually selected" if bowling_arm_override else "auto-detected"
                    st.caption(f"🎯 Bowling arm ({arm_source}): **{detected_arm.title()}-arm**")
                if _display_bowler_type in ("finger_spin", "wrist_spin"):
                    _bt_label = {"finger_spin": "Finger-Spin", "wrist_spin": "Wrist-Spin"}[_display_bowler_type]
                    st.caption(f"🌀 Bowler type: **{_bt_label}**")
                # FIX (2026-08-06): used to only show this explanation for
                # spin bowler_types, but Lead Knee Bracing and Hip-Shoulder
                # Separation are ALWAYS descriptive now (real audit found
                # no universal pass/fail band for either, for any bowler
                # type — see metric_ranges._ALWAYS_DESCRIPTIVE_METRICS), so
                # every pace bowler now sees 🔵 dots too and needs the same
                # explanation.
                st.caption("Metrics without a 🔵 dot below have a validated benchmark; "
                           "🔵 means descriptive-only (see Reference Ranges).")

                m1, m2 = st.columns(2)
                m1.metric("Lead Knee Bracing Angle", ui_deg(knee_deg),
                           _tier_caption("front_knee_bracing", metrics.get('front_knee_bracing', {}).get('tier', 'N/A'), knee_deg))
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
                           _tier_caption("hip_shoulder_separation", metrics.get('hip_shoulder_separation', {}).get('tier', 'N/A'), hip_deg))

                st.write("")
                m3, m4, m5 = st.columns(3)
                m3.metric("Trunk Lean Deflection", ui_deg(trunk_deg),
                           _tier_caption("trunk_lean", metrics.get('trunk_lean', {}).get('tier', 'N/A'), trunk_deg))
                m4.metric("Release Height Ratio", ui_pct(rel_ratio),
                           _tier_caption("release_height", metrics.get('release_height', {}).get('classification')
                                         or metrics.get('release_height', {}).get('tier', 'N/A'), rel_ratio))
                if height_absolute_result and height_absolute_result.get("status") == "success":
                    m4.caption(f"📏 {height_absolute_result['cm']} cm above ground (stump-calibrated)")
                elif height_absolute_result and height_absolute_result.get("status") == "not_calibrated":
                    m4.caption("📏 Calibrate camera (sidebar) for an absolute height in cm")
                if metrics.get('release_height', {}).get('recalibration_pending'):
                    m4.caption(
                        "🔵 Using a newly-corrected body-height measurement (2026-08-05 fix) — "
                        "more trustworthy than before, but the Optimal/Acceptable bands above "
                        "haven't been re-tuned for it yet. Treat this reading as directional, "
                        "not a final verdict."
                    )
                if metrics.get('release_height', {}).get('release_frame_tracking_uncertain'):
                    m4.warning(
                        "⚠️ Tracking around the release frame was flagged unstable (heavy motion "
                        "blur is the common cause) — this same instability affects the wrist "
                        "landmark this ratio is measured from. Treat this reading with real "
                        "caution, or confirm the release point manually to override it."
                    )
                if estimated_height_result.get("status") == "success":
                    m4.caption(f"🧍 Estimated bowler height: ~{estimated_height_result['cm']:.0f} cm (auto, from stump calibration)")
                    # Real cross-check: a release point taller than ~1.3x
                    # the bowler's own estimated height is implausible
                    # regardless of what the ratio-based check upstream
                    # already allowed (see calculate_release_height_ratio_
                    # safe's 1.30 ceiling — same real-world reasoning,
                    # applied here in absolute cm instead of body-ratio terms).
                    if height_absolute_result and height_absolute_result.get("status") == "success":
                        _plausible_ceiling_cm = estimated_height_result["cm"] * 1.3
                        if height_absolute_result["cm"] > _plausible_ceiling_cm:
                            m4.warning(
                                f"⚠️ Release height ({height_absolute_result['cm']:.0f}cm) exceeds "
                                f"1.3x this bowler's estimated height ({estimated_height_result['cm']:.0f}cm) — "
                                f"likely a tracking or calibration issue, not a real reading."
                            )
                elif estimated_height_result.get("status") == "error":
                    m4.caption(f"⚠️ {estimated_height_result['message']}")
                m5.metric("Head Stability Variance", ui_val(head_val),
                           _tier_caption("head_stability", metrics.get('head_stability', {}).get('classification')
                                         or metrics.get('head_stability', {}).get('tier', 'N/A'), head_val))
                if metrics.get('head_stability', {}).get('recalibration_pending'):
                    m5.caption(
                        "🔵 Using a newly-corrected measurement (2026-08-05 fix, normalized for "
                        "camera distance) — more trustworthy than before, but the Optimal/Acceptable "
                        "bands haven't been re-tuned for it yet. Treat this reading as directional, "
                        "not a final verdict."
                    )
                if metrics.get('head_stability', {}).get('release_window_tracking_uncertain'):
                    m5.warning(
                        "⚠️ Tracking near the end of this window (right around release) was "
                        "flagged unstable (heavy motion blur is the common cause) — that can "
                        "inflate this variance reading. Treat it with real caution."
                    )
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
                    dot = {"green": "🟢", "amber": "🟡", "red": "🔴", "unknown": "⚪", "descriptive": "🔵"}
                    for key in mr.all_metric_keys():
                        tier = mr.classify(key, metric_value_lookup.get(key), _display_bowler_type)
                        r = mr.RANGES[key]
                        # FIX (2026-08-06): a genuinely missing value (tier
                        # "unknown") used to fall into the same branch as a
                        # real validated band, showing "Optimal: 160-180°"
                        # for front_knee_bracing even though that band is
                        # now DEAD for classification (always-descriptive
                        # for pace — see metric_ranges._ALWAYS_DESCRIPTIVE_
                        # METRICS). Confirmed on a real clip: Lead Knee
                        # Bracing showed "N/A" (no data this delivery) right
                        # next to "Optimal: 160-180deg" as if that stale
                        # band still applied. tier=="unknown" must be
                        # checked FIRST and independently of whether a
                        # validated band exists at all.
                        if tier == "unknown":
                            st.markdown(f"**{r.label}** {dot[tier]} — No data available this session")
                        elif tier == "descriptive":
                            st.markdown(
                                f"**{r.label}** {dot[tier]} — "
                                f"{mr.descriptive_note(key, metric_value_lookup.get(key), _display_bowler_type)}"
                            )
                        else:
                            band_r = mr.SPIN_RANGE_OVERRIDES.get((key, _display_bowler_type), r)
                            st.markdown(
                                f"**{r.label}** {dot[tier]} — "
                                f"🟢 Optimal: `{band_r.display_optimal}`"
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
                    bowler_type=result_payload.get("bowler_type"),
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
                        # Saved inside metrics (a flexible JSON column) rather
                        # than as a new "sessions" table column — same reason
                        # bowling_arm_detected already lives nested here, not
                        # as its own column: no schema migration needed, and
                        # the Bowler Profile history page can read it straight
                        # back out to re-classify past sessions with the
                        # correct spin/pace bands instead of assuming pace.
                        "bowler_type": result_payload.get("bowler_type"),
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

*Camera setup*
- **Right-arm bowlers:** Camera on the **left** side of the pitch. **Left-arm bowlers:** camera on the **right**
- **Alignment:** Parallel to the popping crease line, roughly 10–12 feet back
- **Mount:** Tripod or another stable stand — handheld shake degrades tracking on every metric

*Framing*
- Keep the bowler's **full body** (head to feet) in shot from run-up through follow-through — several measurements are built from early run-up frames, not just the release moment
- Don't zoom out just to fit a longer run-up — if the bowler becomes a tiny speck early on, tracking can fail to detect them at all. A cropped start to the run-up is fine; staying a clearly-sized figure throughout matters more
- Keep at least **one full stump** clearly visible and unobstructed — used for automatic speed/height calibration

*Recording*
- **Frame rate:** 30 or 60 FPS only
- **Lighting:** Even lighting works best — avoid filming straight into the sun or floodlights
- **Dual camera:** Both phones start recording before the bowler begins run-up
""")
