"""
pages/6_🛠️_How_To_Set_Up.py

Practical, step-by-step setup guide for a coach who has never used the
app before: filming guidance, account creation, calibration, and running
a first analysis. Purely informational — no dynamic data needed.
"""

import streamlit as st

st.set_page_config(page_title="How To Set Up - Apex Coach AI", page_icon="🛠️", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span { color: #E2E8F0 !important; }
h1, h2, h3 { color: #00B4D8 !important; }
section[data-testid="stSidebar"] { background-color: #0F1524 !important; border-right: 1px solid #00B4D8; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
.setup-card {
    background: linear-gradient(145deg, #121824, #1A2333);
    border: 1px solid #1E3A5F; border-radius: 12px; padding: 26px 30px; margin-bottom: 18px;
}
.setup-card p, .setup-card li { line-height: 1.7; font-size: 1.0rem; }
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 30px; height: 30px; border-radius: 50%; background: #0F2A44;
    border: 1px solid #1E3A5F; color: #4FD1E8 !important; font-weight: 700; margin-right: 10px;
}
.tip-tag {
    display: inline-block; background: #0F2A44; color: #4FD1E8 !important;
    border: 1px solid #1E3A5F; border-radius: 999px; padding: 4px 14px; font-size: 0.82rem; margin: 4px 8px 4px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🛠️ How To Set Up</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class="setup-card">

### 1. Filming your delivery

<span class="tip-tag">Side-on preferred</span>
<span class="tip-tag">Stable camera</span>
<span class="tip-tag">Full body in frame</span>
<span class="tip-tag">Good lighting</span>

<p><span class="step-num">1</span> <b>Camera angle.</b> Side-on (perpendicular to the bowler's run-up) gives the
most complete and reliable metric set. Front-on and rear-view are supported for the metrics that remain valid
from those angles, but hip-shoulder separation in particular is less reliable off-side-on — the app will flag this.</p>

<p><span class="step-num">2</span> <b>Camera position.</b> Mount on a tripod or stable surface at roughly hip-to-
chest height, far enough back that the bowler's full body — head to feet — stays in frame through the entire
run-up and delivery, including a leaping action.</p>

<p><span class="step-num">3</span> <b>Lighting and background.</b> Even, natural daylight works best. Avoid strong
backlighting (bowler in silhouette) and busy, high-motion backgrounds (moving players, traffic) which can confuse
pose tracking.</p>

<p><span class="step-num">4</span> <b>Frame rate.</b> Standard smartphone video (30fps or higher) is enough. Slow-
motion (60fps+) gives finer detail on fast actions and is a good option if your device supports it.</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="setup-card">

### 2. Create your coach account

<p><span class="step-num">1</span> On the main page, open the <b>Sign In</b> panel and switch to the
<b>Create Account</b> tab.</p>
<p><span class="step-num">2</span> Enter your email and a password (minimum 6 characters). Your account keeps
your athletes, sessions, and teams private — no other coach can see your data.</p>
<p><span class="step-num">3</span> Once created, sign in from the same panel. You'll stay signed in for the
session; use <b>Sign Out</b> in the top-right when you're done.</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="setup-card">

### 3. (Optional) Calibrate for release speed

<p><span class="step-num">1</span> Speed calibration is a <b>one-time setup per fixed camera position</b> — not
per delivery. Skip this if you only want the biomechanical metrics.</p>
<p><span class="step-num">2</span> Open <b>Speed Calibration</b> on the main page, upload any clip from that exact
camera spot (doesn't need to be a real delivery — even a clip of just the stumps works), and use one of the
guided presets (stump width, full popping crease, or a custom known distance) to click two reference points.</p>
<p><span class="step-num">3</span> Once calibrated, every analysis run from that camera position can estimate
release arm speed automatically.</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="setup-card">

### 4. Run your first analysis

<p><span class="step-num">1</span> Enter the bowler's name in <b>Player Profile</b> — this links the session to
their history automatically (a new profile is created the first time you use a name).</p>
<p><span class="step-num">2</span> Upload the delivery video and select the camera angle (or let the app
auto-detect it).</p>
<p><span class="step-num">3</span> When prompted, confirm — or correct — the Back Foot Contact, Front Foot Contact,
and Ball Release frames the app detected. This step matters: metrics are calculated from whichever frame is
confirmed.</p>
<p><span class="step-num">4</span> Review the annotated video, dashboard, and PDF report. Check the <b>Bowler
Profile</b> page afterward to see this session added to that athlete's trend history.</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="setup-card">

### 5. (Optional) Organize athletes into teams

<p>If you coach more than one bowler, open <b>Academy Profile</b> to create a team and assign athletes to it —
useful for academies, school teams, or training groups you want to track separately.</p>

</div>
""", unsafe_allow_html=True)

st.divider()
st.caption("Apex Coach AI — Autonomous Biomechanical Performance Hub")
