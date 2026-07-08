import streamlit as st

st.set_page_config(page_title="About Apex Coach AI", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span { color: #E2E8F0 !important; }
h1, h2, h3 { color: #00B4D8 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>⚡ What is Apex Coach AI?</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("""
Apex Coach AI is a biomechanical analysis tool for cricket fast bowling. It turns a
bowling video into measurable, trackable data — so coaching feedback can be grounded
in real numbers, not just the naked eye.

### How it actually works

1. **Pose extraction** — every frame of the uploaded video is processed with
   MediaPipe's pose detection model, tracking 33 points on the bowler's body
   (joints, shoulders, hips, ankles, heels, and more).
2. **Delivery event detection** — the system automatically identifies three key
   moments in the delivery: Back Foot Contact, Front Foot Contact, and Ball Release.
3. **Five core biomechanical metrics**, calculated directly from that landmark data:
   - Lead Knee Bracing Angle
   - Hip-Shoulder Separation
   - Trunk Lean Deflection
   - Release Height Ratio
   - Head Stability Variance
4. **Reference range classification** — each metric is placed into an Optimal /
   Acceptable / Critical zone, shown with color-coded indicators throughout the
   report and PDF.
5. **AI coaching narrative** — an AI-generated technical assessment and three
   prescribed training drills, grounded in the same classification zones shown
   in the report (not a separate, disconnected judgment).
6. **Run-up analysis** — stride count, pacing consistency, and foot-strike pattern
   (heel vs. midfoot vs. forefoot) detected from the approach, not just the
   delivery stride itself.
7. **Speed estimation** — once a camera is calibrated against a known real-world
   distance (like stump width), the system estimates release-arm speed. Without
   calibration, it will say so rather than guess a number.
8. **Athlete history** — sessions are saved per athlete, so a coach can track a
   bowler's technique across multiple recordings over time.

### What it's for

Apex Coach AI is built for coaches, academies, and bowlers who want a repeatable,
data-backed way to look at technique — as a complement to expert coaching judgment,
not a replacement for it.

### Honesty, by design

This tool is built to say "I don't know" rather than guess. If tracking quality is
too poor to trust a reading, the report says so. If a camera isn't calibrated, no
speed number is shown. Every classification a coach sees anywhere in this app comes
from the same single, transparent set of reference ranges — the report, the PDF, and
the AI narrative are never allowed to disagree with each other.
""")

st.divider()
st.caption("Apex Coach AI — Autonomous Biomechanical Performance Hub")
