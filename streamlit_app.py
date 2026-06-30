import os
import base64
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from orchestrator import run_complete_bowling_analysis
from coaching_agent import generate_biomechanical_coaching_report

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime

# ====================================================================
# PAGE CONFIG & ELITE DARK UI
# ====================================================================
st.set_page_config(page_title="Apex Coach AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0B0E14; }

    /* All text readable on dark */
    .stApp, .stApp p, .stApp li, .stApp span {
        color: #E2E8F0 !important;
    }

    /* Metric tiles */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #121824, #1A2333);
        border: 1px solid #00B4D8;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,180,216,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: #00B4D8 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 12px !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #38BDF8 !important;
        font-size: 11px !important;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #00B4D8 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F1524 !important;
        border-right: 1px solid #00B4D8;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }

    /* Buttons */
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #00B4D8, #0077B6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 14px 28px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,180,216,0.4) !important;
    }

    /* Download buttons */
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, #0077B6, #023E8A) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Success / error / info boxes */
    div[data-testid="stSuccess"] {
        background-color: #0D2818 !important;
        border-left: 4px solid #00C853 !important;
    }
    div[data-testid="stError"] {
        background-color: #2D0A0A !important;
        border-left: 4px solid #FF3D3D !important;
    }
    div[data-testid="stInfo"] {
        background-color: #0A1628 !important;
        border-left: 4px solid #00B4D8 !important;
    }

    /* Dividers */
    hr { border-color: #00B4D8 !important; opacity: 0.2 !important; }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        background-color: #121824 !important;
        border: 1px dashed #00B4D8 !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }

    /* Radio buttons */
    div[data-testid="stRadio"] label {
        color: #E2E8F0 !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background-color: #121824 !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 8px !important;
    }

    /* Spinner */
    div[data-testid="stSpinner"] {
        color: #00B4D8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# LOGO
# ====================================================================
script_directory = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_directory, "apex_logo.png.png")

log_col1, log_col2, log_col3 = st.columns([1.5, 1, 1.5])
with log_col2:
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <div style="display:flex;justify-content:center;align-items:center;
                        width:100%;height:90px;overflow:hidden;margin-bottom:-20px;">
                <img src="data:image/png;base64,{encoded}"
                     style="max-width:160px;height:auto;transform:translateY(-15px);
                            filter:drop-shadow(0px 2px 6px rgba(0,180,216,0.3));">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h1 style='text-align:center;color:#00B4D8;'>⚡ APEX COACH AI</h1>",
            unsafe_allow_html=True
        )

st.markdown(
    "<h3 style='text-align:center;color:#94A3B8;font-weight:400;"
    "letter-spacing:2px;'>AUTONOMOUS BIOMECHANICAL PERFORMANCE HUB</h3>",
    unsafe_allow_html=True
)
st.divider()

os.makedirs("input", exist_ok=True)


# ====================================================================
# PDF GENERATOR
# ====================================================================
def generate_pdf_report(metrics, frames, ai_insights, bowler_name="Elite Athlete",
                        camera_mode="Single Camera"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontSize=22, leading=26,
        textColor=colors.HexColor('#1A365D')
    )
    h2_style = ParagraphStyle(
        'SectionHeader', parent=styles['Heading2'],
        fontSize=14, leading=18,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=12, spaceAfter=6
    )
    body_style = ParagraphStyle(
        'ReportBody', parent=styles['Normal'],
        fontSize=10, leading=14,
        textColor=colors.HexColor('#2D3748')
    )
    bold_body = ParagraphStyle(
        'ReportBodyBold', parent=body_style,
        fontName='Helvetica-Bold'
    )

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

    # KINEMATIC MILESTONES
    story.append(Paragraph("■■ Kinematic Sequence Milestones", h2_style))
    time_data = [
        [Paragraph("<b>Milestone Phase</b>", bold_body),
         Paragraph("<b>Target Frame Index</b>", bold_body)],
        ["Back Foot Contact (BFC)",
         f"Frame {frames.get('back_foot_contact_frame', 'N/A')}"],
        ["Front Foot Contact (FFC)",
         f"Frame {frames.get('front_foot_contact_frame', 'N/A')}"],
        ["Ball Release Point (BR)",
         f"Frame {frames.get('ball_release_frame', 'N/A')}"]
    ]
    t_time = Table(time_data, colWidths=[250, 250])
    t_time.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#EDF2F7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_time)
    story.append(Spacer(1, 15))

    # SAFE EXTRACTION
    knee_deg  = metrics.get('front_knee_bracing', {}).get('degrees')
    knee_tier = metrics.get('front_knee_bracing', {}).get('tier', 'N/A')
    hip_deg   = metrics.get('hip_shoulder_separation', {}).get('degrees')
    hip_tier  = metrics.get('hip_shoulder_separation', {}).get('tier', 'N/A')
    trunk_deg = metrics.get('trunk_lean', {}).get('degrees')
    trunk_tier= metrics.get('trunk_lean', {}).get('tier', 'N/A')
    rel_ratio = metrics.get('release_height', {}).get('ratio')
    rel_tier  = (metrics.get('release_height', {}).get('classification') or
                 metrics.get('release_height', {}).get('tier', 'N/A'))
    head_val  = (metrics.get('head_stability', {}).get('value') or
                 metrics.get('head_stability', {}).get('deviation_index'))
    head_tier = (metrics.get('head_stability', {}).get('classification') or
                 metrics.get('head_stability', {}).get('tier', 'N/A'))

    def safe_deg(val):
        return f"{round(float(val), 1)}°" if val is not None else "No Data"
    def safe_pct(val):
        return f"{round(float(val) * 100, 1)}%" if val is not None else "No Data"
    def safe_val(val):
        return str(round(float(val), 4)) if val is not None else "No Data"

    # REFERENCE RANGES NOTE
    story.append(Paragraph("■ Core Biomechanical Telemetry", h2_style))
    story.append(Paragraph(
        "<i>Reference ranges — Optimal (green) | Acceptable (amber) | "
        "Critical (red). CBC-style classification.</i>",
        body_style
    ))
    story.append(Spacer(1, 6))

    metrics_data = [
        [Paragraph("<b>Metric</b>", bold_body),
         Paragraph("<b>Value</b>", bold_body),
         Paragraph("<b>Tier</b>", bold_body),
         Paragraph("<b>Optimal Range</b>", bold_body)],
        ["Lead Knee Bracing",       safe_deg(knee_deg),  knee_tier,  "160–180°"],
        ["Hip-Shoulder Separation", safe_deg(hip_deg),   hip_tier,   "25–50°"],
        ["Trunk Lean",              safe_deg(trunk_deg), trunk_tier, "0–20°"],
        ["Release Height",          safe_pct(rel_ratio), rel_tier,   "85–105%"],
        ["Head Stability",          safe_val(head_val),  head_tier,  "0.00–0.02"],
    ]
    t_metrics = Table(metrics_data, colWidths=[150, 90, 150, 110])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 15))

    # AI NARRATIVE
    story.append(Paragraph("■ Autonomous AI Coach Assessment", h2_style))
    narrative = ai_insights.get("narrative_analysis", "No narrative generated.")
    narrative = narrative.replace(
        "SECTION 1 — BIOMECHANICAL NARRATIVE ASSESSMENT:", ""
    ).replace("SECTION 1 — BIOMECHANICAL NARRATIVE:", "").strip()
    story.append(Paragraph(narrative, body_style))
    story.append(Spacer(1, 15))

    # DRILLS
    story.append(Paragraph("■ Prescribed Training Drills", h2_style))
    drills = ai_insights.get("prescribed_drills", [])
    if drills:
        for drill in drills:
            story.append(Paragraph(f"• {drill}", body_style))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph(
            "All metrics within acceptable range. No critical interventions required.",
            body_style
        ))

    story.append(Spacer(1, 25))
    story.append(Paragraph("—" * 60, body_style))
    story.append(Paragraph(
        "<b>Shoaib Nazar</b>, Founder | Apex Coach AI", bold_body
    ))
    story.append(Paragraph("Automated Digital Lab Performance Report", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ====================================================================
# SIDEBAR
# ====================================================================
st.sidebar.markdown(
    "<h2 style='color:#00B4D8;text-align:center;'>⚡ Control Panel</h2>",
    unsafe_allow_html=True
)

st.sidebar.header("📝 Player Profile")
player_name = st.sidebar.text_input(
    "Athlete Full Name", value="Elite Athlete",
    help="Appears in the official PDF report header."
)
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

uploaded_side   = None
uploaded_rear   = None
uploaded_single = None

if camera_mode == "Single Camera":
    uploaded_single = st.sidebar.file_uploader(
        "Bowling Video (.mp4)", type=["mp4"]
    )
else:
    st.sidebar.info(
        "Upload both angles for maximum accuracy. "
        "Events are detected independently on each stream."
    )
    uploaded_side = st.sidebar.file_uploader(
        "📹 Side-On Video (.mp4)", type=["mp4"], key="side"
    )
    uploaded_rear = st.sidebar.file_uploader(
        "📹 Rear-View Video (.mp4)", type=["mp4"], key="rear"
    )

# Check readiness
single_ready = camera_mode == "Single Camera" and uploaded_single is not None
dual_ready   = (camera_mode == "Dual Camera — Recommended"
                and uploaded_side is not None
                and uploaded_rear is not None)

if single_ready or dual_ready:
    if st.sidebar.button("🚀 Execute Biomechanical Analysis Run",
                         use_container_width=True):

        os.makedirs("input", exist_ok=True)

        # SAVE UPLOADED FILES — abspath for cross-platform compatibility
        if camera_mode == "Single Camera":
            video_path = os.path.abspath(
                os.path.join("input", uploaded_single.name)
            )
            with open(video_path, "wb") as f:
                f.write(uploaded_single.getbuffer())
            st.sidebar.success(f"Cached: {uploaded_single.name}")

        else:
            video_path = os.path.abspath(
                os.path.join("input", uploaded_side.name)
            )
            rear_path = os.path.abspath(
                os.path.join("input", uploaded_rear.name)
            )
            with open(video_path, "wb") as f:
                f.write(uploaded_side.getbuffer())
            with open(rear_path, "wb") as f:
                f.write(uploaded_rear.getbuffer())
            st.sidebar.success(
                f"Cached: {uploaded_side.name} + {uploaded_rear.name}"
            )

        # RUN ANALYSIS
        with st.spinner("Executing kinematic extraction and landmark mapping..."):
            if camera_mode == "Dual Camera — Recommended":
                from dual_camera_orchestrator import run_dual_camera_analysis
                result_payload = run_dual_camera_analysis(video_path, rear_path)
                active_camera_mode = "Dual Camera"
            else:
                result_payload = run_complete_bowling_analysis(video_path)
                active_camera_mode = "Single Camera"

        # ================================================================
        # RESULTS DISPLAY
        # ================================================================
        if result_payload.get("status") == "success":
            st.success("✅ Kinematic Pipeline Finished Successfully!")

            metrics = result_payload["biomechanical_metrics"]
            frames  = result_payload["time_indices"]

            # TIMELINE
            st.header("⏱️ Kinematic Sequence Timeline")
            t1, t2, t3 = st.columns(3)
            t1.metric("Back Foot Contact (BFC)",
                      f"Frame {frames['back_foot_contact_frame']}")
            t2.metric("Front Foot Contact (FFC)",
                      f"Frame {frames['front_foot_contact_frame']}")
            t3.metric("Ball Release Point (BR)",
                      f"Frame {frames['ball_release_frame']}")
            st.divider()

            col_graph, col_insights = st.columns([1, 1.2])

            with col_graph:
                st.header("🎞️ Visual Verification")
                video_output = result_payload.get("annotated_video_output")
                if video_output and os.path.exists(video_output):
                    st.video(video_output)
                    clean_slug = player_name.replace(" ", "_")
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=open(video_output, "rb").read(),
                        file_name=f"Annotated_{clean_slug}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                else:
                    st.info("Annotated video rendering in progress...")

                st.divider()
                st.header("📈 Biomechanical Measurements")

                # SAFE UI EXTRACTION
                knee_deg  = metrics.get('front_knee_bracing', {}).get('degrees')
                hip_deg   = metrics.get('hip_shoulder_separation', {}).get('degrees')
                trunk_deg = metrics.get('trunk_lean', {}).get('degrees')
                rel_ratio = metrics.get('release_height', {}).get('ratio')
                head_val  = (
                    metrics.get('head_stability', {}).get('value') or
                    metrics.get('head_stability', {}).get('deviation_index')
                )

                def ui_deg(val):
                    return f"{round(float(val), 1)}°" if val is not None else "N/A"
                def ui_pct(val):
                    return (f"{round(float(val) * 100, 1)}%"
                            if val is not None else "N/A")
                def ui_val(val):
                    return str(round(float(val), 4)) if val is not None else "N/A"

                m1, m2 = st.columns(2)
                m1.metric(
                    "Lead Knee Bracing Angle", ui_deg(knee_deg),
                    metrics.get('front_knee_bracing', {}).get('tier', 'N/A')
                )
                m2.metric(
                    "Hip-Shoulder Rotation Twist", ui_deg(hip_deg),
                    metrics.get('hip_shoulder_separation', {}).get('tier', 'N/A')
                )
                st.write("")
                m3, m4, m5 = st.columns(3)
                m3.metric(
                    "Trunk Lean Deflection", ui_deg(trunk_deg),
                    metrics.get('trunk_lean', {}).get('tier', 'N/A')
                )
                m4.metric(
                    "Release Height Ratio", ui_pct(rel_ratio),
                    (metrics.get('release_height', {}).get('classification') or
                     metrics.get('release_height', {}).get('tier', 'N/A'))
                )
                m5.metric(
                    "Head Stability Variance", ui_val(head_val),
                    (metrics.get('head_stability', {}).get('classification') or
                     metrics.get('head_stability', {}).get('tier', 'N/A'))
                )

                # CBC REFERENCE RANGES
                st.divider()
                st.markdown("#### 🩺 Reference Ranges")
                ranges = {
                    "Knee Bracing":     ("160–180°", "145–160°", "< 145°"),
                    "Hip-Shoulder Sep": ("25–50°",   "15–25°",   "< 15°"),
                    "Trunk Lean":       ("0–20°",    "20–35°",   "> 35°"),
                    "Release Height":   ("85–105%",  "75–85%",   "< 75%"),
                    "Head Stability":   ("0–0.02",   "0.02–0.05","< 0.05"),
                }
                for metric_name, (optimal, acceptable, critical) in ranges.items():
                    st.markdown(
                        f"**{metric_name}** — "
                        f"🟢 Optimal: `{optimal}` | "
                        f"🟡 Acceptable: `{acceptable}` | "
                        f"🔴 Critical: `{critical}`"
                    )

            with col_insights:
                st.header("🧠 Autonomous AI Coach Assessment")

                with st.spinner("Generating expert coaching analysis..."):
                    ai_insights = generate_biomechanical_coaching_report(
                        result_payload
                    )

                clean_slug = player_name.replace(" ", "_")
                pdf_data = generate_pdf_report(
                    metrics, frames, ai_insights,
                    bowler_name=player_name,
                    camera_mode=active_camera_mode
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
                ).replace(
                    "SECTION 1 — BIOMECHANICAL NARRATIVE:", ""
                ).strip()
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
                    st.info(
                        "All metrics within acceptable range. "
                        "No critical interventions required."
                    )

        else:
            st.error(
                f"Pipeline interrupted at stage "
                f"[{result_payload.get('stage', 'unknown')}]: "
                f"{result_payload.get('message', 'Unknown error')}"
            )

elif camera_mode == "Single Camera":
    st.info("Upload a bowling video in the sidebar to begin analysis.")
else:
    st.info(
        "Upload both Side-On and Rear-View videos in the sidebar "
        "to begin dual-camera analysis."
    )

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
# --- RESILIENT PRODUCTION VIDEO STREAMING ---
import os
import streamlit as st

final_output_video = "output/annotated_delivery.mp4"

if os.path.exists(final_output_video):
    with open(final_output_video, 'rb') as video_file:
        video_bytes = video_file.read()
    
    st.video(video_bytes)
    st.write("---")
    st.subheader("📥 Export Performance Assets")
    st.download_button(
        label="🎬 Download Annotated Video",
        data=video_bytes,
        file_name="apex_coach_delivery_analysis.mp4",
        mime="video/mp4"
    )
