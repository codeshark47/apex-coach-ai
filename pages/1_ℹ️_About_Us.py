import streamlit as st

st.set_page_config(page_title="About Apex Coach AI", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span { color: #E2E8F0 !important; }
h1, h2, h3 { color: #00B4D8 !important; }
.story-card {
    background: linear-gradient(145deg, #121824, #1A2333);
    border: 1px solid #1E3A5F; border-radius: 12px; padding: 28px 32px; margin-bottom: 18px;
}
.story-card p, .story-card li { line-height: 1.7; font-size: 1.02rem; }
.step-num {
    display: inline-block; width: 26px; height: 26px; border-radius: 50%;
    background: #0F2A44; color: #4FD1E8 !important; text-align: center;
    line-height: 26px; font-size: 0.85rem; margin-right: 10px; border: 1px solid #1E3A5F;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>⚡ What is Apex Coach AI?</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class="story-card">

Apex Coach AI is a biomechanical analysis tool for cricket fast bowling. It turns a
bowling video into measurable, trackable data — so coaching feedback can be grounded
in real numbers, not just the naked eye.

</div>
""", unsafe_allow_html=True)

st.markdown("### How It Actually Works")

steps = [
    ("Pose extraction", "Every frame of the uploaded video is processed with MediaPipe's "
     "pose detection model, tracking 33 points on the bowler's body — joints, shoulders, "
     "hips, ankles, heels, and more."),
    ("Delivery event detection", "The system automatically identifies three key moments in "
     "the delivery: Back Foot Contact, Front Foot Contact, and Ball Release."),
    ("Five core biomechanical metrics", "Calculated directly from that landmark data: Lead "
     "Knee Bracing Angle, Hip-Shoulder Separation, Trunk Lean Deflection, Release Height "
     "Ratio, and Head Stability Variance."),
    ("Reference range classification", "Each metric is placed into an Optimal / Acceptable / "
     "Critical zone, shown with color-coded indicators throughout the report and PDF."),
    ("AI coaching narrative", "An AI-generated technical assessment and three prescribed "
     "training drills, grounded in the same classification zones shown in the report — not "
     "a separate, disconnected judgment."),
    ("Run-up analysis", "Stride count, pacing consistency, and foot-strike pattern (heel vs. "
     "midfoot vs. forefoot) detected from the approach, not just the delivery stride itself."),
    ("Speed estimation", "Once a camera is calibrated against a known real-world distance "
     "(like stump width), the system estimates release-arm speed. Without calibration, it "
     "will say so rather than guess a number."),
    ("Athlete history", "Sessions are saved per athlete, so a coach can track a bowler's "
     "technique across multiple recordings over time."),
]

for i, (title, desc) in enumerate(steps, start=1):
    st.markdown(f"""
    <div class="story-card">
    <span class="step-num">{i}</span><b>{title}</b><br>
    <span style="margin-left:36px; display:inline-block; margin-top:6px;">{desc}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="story-card">

### What It's For

Apex Coach AI is built for coaches, academies, and bowlers who want a repeatable,
data-backed way to look at technique — as a complement to expert coaching judgment,
not a replacement for it.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="story-card">

### Honesty, By Design

This tool is built to say "I don't know" rather than guess. If tracking quality is
too poor to trust a reading, the report says so. If a camera isn't calibrated, no
speed number is shown. Every classification a coach sees anywhere in this app comes
from the same single, transparent set of reference ranges — the report, the PDF, and
the AI narrative are never allowed to disagree with each other.

</div>
""", unsafe_allow_html=True)

st.divider()
st.caption("Apex Coach AI — Autonomous Biomechanical Performance Hub")
