import os
import base64
from dotenv import load_dotenv
load_dotenv()

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
import camera_angle_detection as cad

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
            signup_submitted = st.form_submit_button("Create Account", use_container_width=True)
        if signup_submitted:
            try:
                result = auth_module.sign_up(signup_email, signup_password)
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
        f"<b>Status:</b> Certified Telemetry Data",
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
    story.append(Paragraph("Automated Digital Lab Performance Report", body_style))

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
            athletes = []
            st.error(f"Could not load athletes: {e}")

        if athletes:
            names = [a["name"] for a in athletes]
            selected = st.selectbox("Select athlete", names, key="history_athlete_select")
            selected_id = next(a["id"] for a in athletes if a["name"] == selected)
            try:
                history = store.get_athlete_history(selected_id, st.session_state.auth_user["id"])
            except Exception as e:
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
            from PIL import Image, ImageDraw
            from streamlit_image_coordinates import streamlit_image_coordinates

            pil_img = Image.fromarray(frame)
            orig_w, orig_h = pil_img.size

            display_img = pil_img.copy()
            draw = ImageDraw.Draw(display_img)
            for i, (px, py) in enumerate(st.session_state.calib_points):
                r = max(4, orig_w // 150)
                draw.ellipse((px - r, py - r, px + r, py + r), outline="red", width=3)
                draw.text((px + r + 4, py - r - 4), str(i + 1), fill="red")

            if len(st.session_state.calib_points) < 2:
                which_point = (calib_preset["point1_prompt"] if len(st.session_state.calib_points) == 0
                               else calib_preset["point2_prompt"])
                st.caption(f"📍 Click **{which_point}** on the image below.")
            else:
                st.caption("✅ Both points selected — see below.")

            click = streamlit_image_coordinates(
                display_img, key="calib_click_widget",
                use_column_width="always"
            )

            if click is not None and len(st.session_state.calib_points) < 2:
                # The component may report coords relative to its rendered size,
                # which can differ from the source image's native resolution —
                # rescale back to the original frame's pixel space to stay accurate.
                rendered_w = click.get("width") or orig_w
                rendered_h = click.get("height") or orig_h
                scale_x = orig_w / rendered_w
                scale_y = orig_h / rendered_h
                new_point = (round(click["x"] * scale_x), round(click["y"] * scale_y))

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
st.sidebar.divider()
st.sidebar.header("📁 Upload Video")
uploaded_side = None
uploaded_rear = None
uploaded_single = None

if camera_mode == "Single Camera":
    uploaded_single = st.sidebar.file_uploader("Bowling Video (.mp4 or .mov)", type=["mp4", "mov", "m4v"])
else:
    st.sidebar.info("Upload both angles for maximum accuracy. Events are detected independently on each stream.")
    uploaded_side = st.sidebar.file_uploader("📹 Side-On Video (.mp4 or .mov)", type=["mp4", "mov", "m4v"], key="side")
    uploaded_rear = st.sidebar.file_uploader("📹 Rear-View Video (.mp4 or .mov)", type=["mp4", "mov", "m4v"], key="rear")


def render_bowler_seed_ui(uploaded_file, key_prefix: str, label: str):
    """
    One-time-per-video step: save the upload, show a scrubbable reference
    frame, and have the coach click directly on the bowler. Returns
    ((x_px, y_px), frame_index) once clicked, else (None, 0).

    This is the credible fix for the "skeleton locks onto the wrong
    person" risk (documented at length in main.py) — instead of another
    automatic guess, the coach tells the app who to track, once, and
    main.py's seeded tracker just follows whoever's torso stays closest
    to that identity frame to frame.
    """
    if uploaded_file is None:
        return None, 0

    file_identity = f"{uploaded_file.name}_{uploaded_file.size}"
    point_key = f"{key_prefix}_seed_point"
    frame_key = f"{key_prefix}_seed_frame"
    identity_key = f"{key_prefix}_seed_identity"
    ref_path_key = f"{key_prefix}_seed_ref_path"

    if st.session_state.get(identity_key) != file_identity:
        # New file uploaded (or first time) — reset any prior click and
        # cache a temp copy on disk so we can pull reference frames from it.
        os.makedirs("input", exist_ok=True)
        ref_path = os.path.abspath(os.path.join("input", f"_seed_ref_{key_prefix}_{uploaded_file.name}"))
        with open(ref_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state[identity_key] = file_identity
        st.session_state[ref_path_key] = ref_path
        st.session_state[point_key] = None
        st.session_state[frame_key] = 0

    ref_path = st.session_state[ref_path_key]
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
            from PIL import Image, ImageDraw
            from streamlit_image_coordinates import streamlit_image_coordinates

            pil_img = Image.fromarray(frame)
            orig_w, orig_h = pil_img.size

            display_img = pil_img.copy()
            point = st.session_state.get(point_key)
            if point is not None:
                draw = ImageDraw.Draw(display_img)
                r = max(5, orig_w // 100)
                px, py = point
                draw.ellipse((px - r, py - r, px + r, py + r), outline="lime", width=4)

            if point is None:
                st.caption("📍 Click directly on the bowler below.")
            else:
                st.caption("✅ Bowler confirmed — click again to move the marker.")

            click = streamlit_image_coordinates(
                display_img, key=f"{key_prefix}_seed_click_widget",
                use_column_width="always"
            )

            if click is not None:
                rendered_w = click.get("width") or orig_w
                rendered_h = click.get("height") or orig_h
                scale_x = orig_w / rendered_w
                scale_y = orig_h / rendered_h
                new_point = (round(click["x"] * scale_x), round(click["y"] * scale_y))

                if st.session_state.get(point_key) != new_point:
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
    Optional second (or more) confirmation point, later in the same
    clip. Exists for exactly the failure mode found on real footage:
    tracking can lose the bowler for several seconds (occlusion,
    motion blur, or he's simply too small/distant for MediaPipe to
    detect at all during part of the clip) and the single seed has
    nothing to re-anchor to for the rest of the video. Re-confirming
    identity here lets the tracker split the clip into zones, each
    only needing to survive the gap to its NEAREST seed instead of one
    seed carrying the whole thing — see main._walk_from_seed.

    Returns a list of (frame_index, point) tuples for every extra
    confirmation the coach has added, or None if none were added.
    """
    if uploaded_file is None:
        return None
    with st.expander(f"➕ Tracking lost partway through — add a second confirmation ({label})", expanded=False):
        st.caption(
            "Only needed if the skeleton drifts onto someone else partway through "
            "this clip (e.g. a coach or another player standing nearby). Scrub to "
            "a later frame where the bowler is clearly visible again and click him "
            "— same as above, just a second time."
        )
        point, frame_idx = render_bowler_seed_ui(uploaded_file, f"{key_prefix}_extra", f"{label} — 2nd confirmation")
        if point is not None:
            return [(frame_idx, point)]
    return None


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

if camera_mode == "Single Camera" and uploaded_single is not None and single_seed_point is not None:
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
            br_slider_val = st.slider(
                "Scrub to the true ball-release frame",
                min_value=0, max_value=max(total_frames_single - 1, 0),
                value=min(max(br_auto, 0), max(total_frames_single - 1, 0)),
                key="br_confirm_slider"
            )
            br_frame_img = cal.extract_reference_frame(single_ref_path, frame_index=br_slider_val)
            if br_frame_img is not None:
                st.image(br_frame_img, use_column_width=True,
                          caption=f"Frame {br_slider_val} — is the ball leaving the hand here?")
            if st.button("✅ Confirm this is the release frame", key="confirm_br_button"):
                st.session_state["_br_confirmed_frame"] = br_slider_val
                st.rerun()
            if st.session_state.get("_br_confirmed_frame") is not None:
                st.success(f"Confirmed: release at frame {st.session_state['_br_confirmed_frame']}.")

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
            ffc_slider_val = st.slider(
                "Scrub to the true front-foot-contact frame",
                min_value=0, max_value=max(total_frames_ffc - 1, 0),
                value=min(max(ffc_auto, 0), max(total_frames_ffc - 1, 0)),
                key="ffc_confirm_slider"
            )
            ffc_frame_img = cal.extract_reference_frame(single_ref_path, frame_index=ffc_slider_val)
            if ffc_frame_img is not None:
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
            bfc_slider_val = st.slider(
                "Scrub to the true back-foot-contact frame",
                min_value=0, max_value=max(total_frames_bfc - 1, 0),
                value=min(max(bfc_auto, 0), max(total_frames_bfc - 1, 0)),
                key="bfc_confirm_slider"
            )
            bfc_frame_img = cal.extract_reference_frame(single_ref_path, frame_index=bfc_slider_val)
            if bfc_frame_img is not None:
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

# Angle must be genuinely resolved (not left on "Not sure") before running —
# matches the same hard-gate already applied to bowling arm above. Dual
# Camera doesn't need this: each stream's angle is known by construction.
angle_resolved = (camera_mode != "Single Camera") or (
    confirmed_angle_label is not None and confirmed_angle_label != "unknown"
)

single_ready = (camera_mode == "Single Camera" and uploaded_single is not None
                 and single_seed_point is not None and bowling_arm_selected and angle_resolved
                 and br_resolved and ffc_resolved and bfc_resolved)
dual_ready = (camera_mode == "Dual Camera — Recommended"
              and uploaded_side is not None and uploaded_rear is not None
              and side_seed_point is not None and rear_seed_point is not None
              and bowling_arm_selected)

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

        if camera_mode == "Single Camera":
            video_path = os.path.abspath(os.path.join("input", uploaded_single.name))
            with open(video_path, "wb") as f:
                f.write(uploaded_single.getbuffer())
            st.sidebar.success(f"Cached: {uploaded_single.name}")
        else:
            video_path = os.path.abspath(os.path.join("input", uploaded_side.name))
            rear_path = os.path.abspath(os.path.join("input", uploaded_rear.name))
            with open(video_path, "wb") as f:
                f.write(uploaded_side.getbuffer())
            with open(rear_path, "wb") as f:
                f.write(uploaded_rear.getbuffer())
            st.sidebar.success(f"Cached: {uploaded_side.name} + {uploaded_rear.name}")

        with st.spinner("Executing kinematic extraction and landmark mapping..."):
            if camera_mode == "Dual Camera — Recommended":
                from dual_camera_orchestrator import run_dual_camera_analysis
                result_payload = run_dual_camera_analysis(
                    video_path, rear_path, bowling_arm_override=bowling_arm_override,
                    side_seed_point=side_seed_point, side_seed_frame_index=side_seed_frame,
                    rear_seed_point=rear_seed_point, rear_seed_frame_index=rear_seed_frame,
                    side_extra_seeds=side_extra_seeds, rear_extra_seeds=rear_extra_seeds,
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
            landmarks_csv = os.path.join("output", "landmarks.csv")
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
            with st.expander("🏃 Run-Up Analysis", expanded=False):
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
