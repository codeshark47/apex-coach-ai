"""
pages/5_⚙️_How_It_Works.py

Explains the actual pipeline honestly — no overclaiming. The reference
range numbers below are pulled live from metric_ranges.py (describe_range)
rather than retyped here, so this page can never drift out of sync with
what the PDF report and dashboard actually use to classify a delivery.
"""

import streamlit as st

import metric_ranges as mr

st.set_page_config(page_title="How It Works - Apex Coach AI", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span { color: #E2E8F0 !important; }
h1, h2, h3 { color: #00B4D8 !important; }
section[data-testid="stSidebar"] { background-color: #0F1524 !important; border-right: 1px solid #00B4D8; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
.hiw-card {
    background: linear-gradient(145deg, #121824, #1A2333);
    border: 1px solid #1E3A5F; border-radius: 12px; padding: 26px 30px; margin-bottom: 18px;
}
.hiw-card p, .hiw-card li { line-height: 1.7; font-size: 1.0rem; }
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 30px; height: 30px; border-radius: 50%; background: #0F2A44;
    border: 1px solid #1E3A5F; color: #4FD1E8 !important; font-weight: 700; margin-right: 10px;
}
.range-row { display: flex; justify-content: space-between; border-bottom: 1px solid #1E3A5F; padding: 8px 0; }
.honesty-tag {
    display: inline-block; background: #2A1414; color: #FF9E9E !important;
    border: 1px solid #5C2626; border-radius: 999px; padding: 4px 14px; font-size: 0.82rem; margin: 4px 8px 4px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>⚙️ How Apex Coach AI Works</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class="hiw-card">

### The pipeline, step by step

<p><span class="step-num">1</span> <b>You upload a delivery video.</b> Side-on gives the most complete metric set;
front-on and rear-view are supported too, but some metrics (like hip-shoulder separation) are less reliable
from those angles and the app tells you so rather than silently guessing.</p>

<p><span class="step-num">2</span> <b>Pose detection.</b> Google's MediaPipe Pose Landmarker locks onto the bowler
frame by frame, tracking 33 body landmarks (ankles, knees, hips, shoulders, elbows, wrists, head) — automatically,
for both left-arm and right-arm bowlers, and for standard and leaping deliveries.</p>

<p><span class="step-num">3</span> <b>Delivery event detection.</b> The app estimates three key moments from ankle
and wrist motion: Back Foot Contact (BFC), Front Foot Contact (FFC), and Ball Release (BR). These are the hardest
frames to get right automatically — a leaping action, a partially-obscured foot, or unusual footwork can fool any
automatic detector.</p>

<p><span class="step-num">4</span> <b>You confirm the key frames.</b> Because event detection can be wrong on
real footage, BFC/FFC/BR are never silently trusted — you review and confirm (or correct) the frame the app picked
before any metric is calculated from it. This is a deliberate design choice: a wrong metric computed from a
confidently-wrong frame is worse than asking you to spend five seconds confirming it.</p>

<p><span class="step-num">5</span> <b>Metrics are calculated from the confirmed frames.</b> Five core
biomechanical metrics are computed (see the reference ranges below), each checked against a real anatomical range —
never a fabricated "everything looks perfect" number.</p>

<p><span class="step-num">6</span> <b>Data quality is reported alongside the numbers.</b> If tracking confidence
was low for a metric (poor lighting, occlusion, an unusual camera angle), the app says so explicitly instead of
publishing a number it can't stand behind.</p>

<p><span class="step-num">7</span> <b>Report + history.</b> You get an annotated video, a PDF report, and — once
an athlete has more than one session — trend charts on their Bowler Profile page so progress is visible over time,
not just a single snapshot.</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hiw-card">

### What this app will NOT do

<span class="honesty-tag">Never fabricate a metric it couldn't measure</span>
<span class="honesty-tag">Never claim 100% accuracy</span>
<span class="honesty-tag">Never silently trust an uncertain event frame</span>
<span class="honesty-tag">Never hide low tracking confidence</span>

<p>Automatic pose tracking on real-world footage is never perfect — camera shake, motion blur, unusual lighting,
and partial occlusion all degrade tracking quality. Where the app can't measure something reliably, it reports
"not available" or asks you to confirm/correct rather than presenting a guess as fact.</p>

</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hiw-card"><h3>Reference ranges used to classify every metric</h3></div>', unsafe_allow_html=True)
for key in mr.all_metric_keys():
    st.markdown(f"""
    <div class="hiw-card" style="padding:16px 24px;">
    {mr.describe_range(key)}
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Apex Coach AI — Autonomous Biomechanical Performance Hub")
