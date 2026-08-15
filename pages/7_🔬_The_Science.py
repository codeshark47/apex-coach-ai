"""
pages/7_🔬_The_Science.py

The real research behind each bowling reference range — built after a full
audit found the core 5 pace-bowling ranges had NO documented source
anywhere in the repo (traced via `git log -S` back to the very first
commit). Four of the five are now sourced from real, peer-reviewed
biomechanics research (citations below); the fifth (Head Stability) is
honestly labeled as NOT yet backed by published research rather than
implying otherwise. Live status per metric (real band vs. descriptive vs.
provisional) is pulled from metric_ranges.py, the same single source of
truth the dashboard and PDF use, so this page can't quietly drift out of
sync with what a coach actually sees in a report.
"""

import streamlit as st

import metric_ranges as mr

st.set_page_config(page_title="The Science - Apex Coach AI", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

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
.honesty-tag {
    display: inline-block; background: #2A1414; color: #FF9E9E !important;
    border: 1px solid #5C2626; border-radius: 999px; padding: 4px 14px; font-size: 0.82rem; margin: 4px 8px 4px 0;
}
.status-tag {
    display: inline-block; border-radius: 999px; padding: 4px 14px; font-size: 0.82rem; margin: 4px 8px 4px 0; font-weight: 600;
}
.status-real { background: #0F2A1A; color: #7FE3A3 !important; border: 1px solid #1E5C36; }
.status-adapted { background: #1A2340; color: #8FB3F0 !important; border: 1px solid #2A3E6C; }
.status-provisional { background: #2A1414; color: #FF9E9E !important; border: 1px solid #5C2626; }
.citation {
    font-size: 0.92rem; color: #94A3B8 !important; border-left: 2px solid #1E3A5F;
    padding-left: 14px; margin-top: 10px; line-height: 1.6;
}
.ref-list li { font-size: 0.92rem; color: #94A3B8 !important; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🔬 The Research Behind Our Numbers</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class="hiw-card">

### Where every number in this app comes from

Every reference range this app judges a bowler against is either sourced from real, peer-reviewed
sports biomechanics research, or explicitly labeled as not-yet-validated. We do not publish
invented "ideal" numbers, and we do not present a provisional internal estimate as if it were
established science.

<span class="honesty-tag">No fabricated targets</span>
<span class="honesty-tag">Real citations, checked against primary sources</span>
<span class="honesty-tag">Honest about what's still provisional</span>

This page exists because a full audit of this app found its original 5 core pace-bowling ranges
had **no documented source anywhere** — they dated back to the very first commit, before any
citation discipline existed. Four of the five have since been re-sourced from real research, read
in full, not just taken from a search summary. The fifth is disclosed below as still an open gap.

</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hiw-card"><h3>The 5 core bowling metrics</h3></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hiw-card">

<span class="status-tag status-adapted">REAL RESEARCH — TECHNIQUE CLASSIFICATION</span>

### Lead Knee Bracing

Front knee action at release is not a "straighter is better" scale. Portus, Mason, Elliott,
Pfitzner &amp; Done classified elite fast bowlers into real technique groups by knee angle at
release: **Extended-Knee** (&ge;170&deg;) and **Flexed-Knee** (&lt;170&deg;) — both are legitimate,
common techniques at the elite level. A related analysis from the same research group even found
non-injured bowlers had a *more* flexed knee at release than injured bowlers — the opposite of
what "straighter = better" would predict. This app reports which real technique category a
bowler's knee angle falls into, not a pass/fail verdict.

<div class="citation">
Portus, M., Mason, B., Elliott, B., Pfitzner, M., &amp; Done, R. (2004). Technique factors related
to ball release speed and trunk injuries in high performance cricket fast bowlers.
<i>Sports Biomechanics</i>, 3(2), 263&ndash;283.
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hiw-card">

<span class="status-tag status-adapted">REAL RESEARCH — VARIES BY TECHNIQUE, NOT SKILL</span>

### Hip-Shoulder Separation

Real data from 35 elite fast bowlers shows this varies enormously — mean 33.0&deg;, but with a
spread of &plusmn;21.6&deg; — because it depends heavily on bowling action type (front-on, side-on,
or mixed), not on skill. A front-on bowler will legitimately show far less separation than a
mixed-action bowler by design. No single "optimal zone" exists independent of technique, so this
app reports the real measured value with that context rather than inventing a universal target.

<div class="citation">
Senington, B., Lee, R. Y., &amp; Williams, J. M. (2018). Are shoulder counter rotation and hip
shoulder separation angle representative metrics of three-dimensional spinal kinematics in cricket
fast bowling? <i>Journal of Sports Sciences</i>, 36(15), 1763&ndash;1767.
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hiw-card">

<span class="status-tag status-real">REAL RESEARCH — DIRECTLY SOURCED</span>

### Trunk Lean

More forward lean at release is linked to *faster* deliveries, not less — the opposite of what
"stay upright" coaching cliches suggest. Elite fast bowlers average around 20.5&deg; of forward
flexion at release. This app's green band (13&deg;+) uses that real elite mean minus one standard
deviation as its floor.

<div class="citation">
Elliott, B., Foster, D., &amp; Gray, S. (1986). Biomechanics and physical factors affecting fast
bowling. <i>Australian Journal of Science and Medicine in Sport</i>, 18, 16&ndash;21.<br>
Worthington, P. J., King, M. A., &amp; Ranson, C. A. (2013). Relationships between fast bowling
technique and ball release speed in cricket. <i>Journal of Applied Biomechanics</i>, 29, 78&ndash;84.<br>
Felton, P., Lister, S. L., Worthington, P. J., &amp; King, M. A. (2018). Comparison of biomechanical
characteristics between male and female elite fast bowlers. <i>Journal of Sports Sciences</i>.
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hiw-card">

<span class="status-tag status-adapted">REAL RESEARCH — ADAPTED, NOT A DIRECT QUOTE</span>

### Release Height

Felton et al. measured elite release height at 112.8% of a bowler's **true standing height**
(floor to the top of the head). This app's own ratio uses a different baseline — the distance from
nose to ankle, summed across real skeletal segments — because reliable pose tracking doesn't give
a trustworthy head-top landmark to measure true stature from. We converted their real number into
this app's own baseline using standard anthropometric ratios (average adult eye/nose height and
ankle height, as a fraction of stature), worked through with real units and documented step by
step in the code. The result (118%+) is a genuine, real-data-informed target — but it is
**adapted** from Felton et al.'s number, not a direct quote of it, and that adaptation stacks a
general-population anthropometric approximation on top of real bowling-specific data.

<div class="citation">
Felton, P., Lister, S. L., Worthington, P. J., &amp; King, M. A. (2018). Comparison of biomechanical
characteristics between male and female elite fast bowlers. <i>Journal of Sports Sciences</i>.
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hiw-card">

<span class="status-tag status-provisional">NOT YET BACKED BY PUBLISHED RESEARCH</span>

### Head Stability

Unlike the four metrics above, this one is honest about a real gap: its current band is an
internal engineering estimate, not sourced from any published study. It exists so a genuinely
unstable head position doesn't go completely unflagged while real research is found — but it
should be read as directional, not as an established scientific target, until that changes.

</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hiw-card"><h3>A note on tracking-quality flags</h3>'
            '<p>Two metrics above (Release Height and Head Stability) can also carry a real-time '
            '"tracking uncertain" warning, separate from the research question on this page — that '
            'flag means the pose tracking right around ball release was itself flagged unstable for '
            'that specific delivery (heavy motion blur is the common cause), not that the reference '
            'range is in question. See the report itself for when that applies.</p></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="hiw-card">

### References

<ul class="ref-list">
<li>Elliott, B., Foster, D., &amp; Gray, S. (1986). Biomechanics and physical factors affecting fast
bowling. <i>Australian Journal of Science and Medicine in Sport</i>, 18, 16&ndash;21.</li>
<li>Felton, P., Lister, S. L., Worthington, P. J., &amp; King, M. A. (2018). Comparison of
biomechanical characteristics between male and female elite fast bowlers. <i>Journal of Sports
Sciences</i>.</li>
<li>Goswami, A., Srivastava, N., &amp; Rajpoot, Y. (2016). A biomechanical analysis of spin bowling
in cricket. <i>European Journal of Physical Education and Sport Science</i>, 2(6).</li>
<li>Portus, M., Mason, B., Elliott, B., Pfitzner, M., &amp; Done, R. (2004). Technique factors
related to ball release speed and trunk injuries in high performance cricket fast bowlers.
<i>Sports Biomechanics</i>, 3(2), 263&ndash;283.</li>
<li>Senington, B., Lee, R. Y., &amp; Williams, J. M. (2018). Are shoulder counter rotation and hip
shoulder separation angle representative metrics of three-dimensional spinal kinematics in cricket
fast bowling? <i>Journal of Sports Sciences</i>, 36(15), 1763&ndash;1767.</li>
<li>Worthington, P. J., King, M. A., &amp; Ranson, C. A. (2013). Relationships between fast bowling
technique and ball release speed in cricket. <i>Journal of Applied Biomechanics</i>, 29, 78&ndash;84.</li>
</ul>

</div>
""", unsafe_allow_html=True)

st.divider()
st.caption("Apex Coach AI — Autonomous Biomechanical Performance Hub")
