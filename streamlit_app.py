import os
import base64
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from orchestrator import run_complete_bowling_analysis
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
st.sidebar.header("📏 Speed Calibration")
if "calibration" not in st.session_state:
    st.session_state.calibration = None

with st.sidebar.expander("Calibrate camera for speed (once per setup)", expanded=False):
    st.caption(
        "This is a ONE-TIME setup per fixed camera position — not per delivery. "
        "Upload any clip from that camera spot (can be a dedicated short clip of "
        "just the stumps, doesn't need to be an actual delivery), scrub to a frame "
        "where two points of known distance are visible (e.g. the two stumps, "
        "0.2286m apart), then click both points directly on the image below."
    )
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
                st.caption(f"📍 Click point {len(st.session_state.calib_points) + 1} of 2 on the image below.")
            else:
                st.caption("✅ Both points selected — see below.")

            click = streamlit_image_coordinates(display_img, key="calib_click_widget")

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
                real_dist = st.number_input(
                    "Real-world distance between the two points you clicked (meters)",
                    min_value=0.0, value=0.2286, step=0.01, key="cal_dist",
                    help="Stump width = 0.2286m is the easiest reference if stumps are visible."
                )
                ref_label = st.text_input("Reference label (e.g. 'stump width')",
                                           value="stump width", key="cal_label")
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

st.sidebar.divider()

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

single_ready = camera_mode == "Single Camera" and uploaded_single is not None
dual_ready = (camera_mode == "Dual Camera — Recommended"
              and uploaded_side is not None and uploaded_rear is not None)

import usage_limits
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
                result_payload = run_dual_camera_analysis(video_path, rear_path)
                active_camera_mode = "Dual Camera"
            else:
                result_payload = run_complete_bowling_analysis(video_path)
                active_camera_mode = "Single Camera"

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
        st.session_state.pop("angle_confirm_radio", None)  # don't inherit the previous video's angle answer

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
                try:
                    usage_limits.record_usage(st.session_state.auth_user["id"])
                    st.session_state.usage_recorded_for_run = True
                except Exception as e:
                    st.warning(f"Could not update usage count: {e}")

            metrics = result_payload["biomechanical_metrics"]
            frames = result_payload["time_indices"]
            fps = result_payload["video_metadata"]["fps"]

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

            # --- PHASE TIMING (always available) ---
            phase_durations = None
            try:
                phase_durations = se.compute_phase_durations(events, fps)
            except ValueError as e:
                st.warning(f"Phase timing unavailable: {e}")

            # --- SPEED (only if calibrated) ---
            speed_result = None
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
                    landmarks_df, events, fps, cap_w, cap_h, meters_per_pixel=mpp
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

            # --- CAMERA ANGLE DETECTION (auto side-on vs not, ask only when ambiguous) ---
            angle_estimate = None
            if os.path.exists(landmarks_csv):
                angle_estimate = cad.estimate_camera_angle(
                    landmarks_df, reference_frame_idx=events["BFC"],
                    frame_width=cap_w, frame_height=cap_h
                )

            resolved_angle = "side_on"  # default assumption if detection unavailable
            if angle_estimate is not None:
                if angle_estimate.angle == "side_on":
                    resolved_angle = "side_on"
                elif angle_estimate.angle in ("front_or_rear", "uncertain"):
                    st.warning(
                        f"📐 Camera angle check: {angle_estimate.confidence_note} "
                        f"(shoulder-width ratio: {angle_estimate.ratio})"
                    )
                    angle_choice = st.radio(
                        "Which angle was this filmed from? This changes what Trunk Lean "
                        "and Head Stability actually measure.",
                        ["Side-on", "Rear-view (behind bowler)", "Front-on (facing bowler)", "Not sure"],
                        key="angle_confirm_radio", horizontal=True
                    )
                    resolved_angle = {
                        "Side-on": "side_on",
                        "Rear-view (behind bowler)": "rear",
                        "Front-on (facing bowler)": "front",
                        "Not sure": "unknown",
                    }[angle_choice]
                else:
                    resolved_angle = "unknown"

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

                m1, m2 = st.columns(2)
                m1.metric("Lead Knee Bracing Angle", ui_deg(knee_deg),
                           metrics.get('front_knee_bracing', {}).get('tier', 'N/A'))
                m2.metric("Hip-Shoulder Rotation Twist", ui_deg(hip_deg),
                           metrics.get('hip_shoulder_separation', {}).get('tier', 'N/A'))

                st.write("")
                m3, m4, m5 = st.columns(3)
                m3.metric("Trunk Lean Deflection", ui_deg(trunk_deg),
                           metrics.get('trunk_lean', {}).get('tier', 'N/A'))
                m4.metric("Release Height Ratio", ui_pct(rel_ratio),
                           (metrics.get('release_height', {}).get('classification')
                            or metrics.get('release_height', {}).get('tier', 'N/A')))
                m5.metric("Head Stability Variance", ui_val(head_val),
                           (metrics.get('head_stability', {}).get('classification')
                            or metrics.get('head_stability', {}).get('tier', 'N/A')))

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
