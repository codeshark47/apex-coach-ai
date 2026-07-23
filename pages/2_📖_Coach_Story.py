import streamlit as st

st.set_page_config(page_title="Coach Story - Apex Coach AI", page_icon="📖", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span { color: #E2E8F0 !important; }
h1, h2, h3 { color: #00B4D8 !important; }
.story-card {
    background: linear-gradient(145deg, #121824, #1A2333);
    border: 1px solid #1E3A5F; border-radius: 12px; padding: 28px 32px; margin-bottom: 18px;
}
.story-card p { line-height: 1.7; font-size: 1.02rem; }
.story-tag {
    display: inline-block; background: #0F2A44; color: #4FD1E8 !important;
    border: 1px solid #1E3A5F; border-radius: 999px; padding: 4px 14px;
    font-size: 0.82rem; margin: 4px 8px 4px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>📖 The Story Behind Apex Coach AI</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class="story-card">

### Shoaib Nazar — Founder

<span class="story-tag">ICC Level 2 Certified Coach</span>
<span class="story-tag">Founder, Strikers Den Sports Academy</span>
<span class="story-tag">18 years in international business</span>

Shoaib's path to Apex Coach AI runs through two very different worlds: competitive
cricket and corporate operations — and it's the collision of the two that shaped
how this app was built.

**The cricketer.** Growing up playing on Karachi's competitive club circuit, Shoaib
captained his zone at Under-15, Under-17, and Under-19 level as an opening batsman.
Years spent facing fast bowling built a close, practical feel for the mechanics of
the crease — front-leg brace, hip-shoulder timing, the split-second decisions a
batsman makes under pace. Cricket didn't stay a full-time path, but it never left.

**The operator.** Over an 18-year career spanning Pakistan, the UAE, and Singapore,
Shoaib worked across telesales, real estate consulting, and technology, eventually
founding a digital agency (AOS Formula), co-building the event platform Oyee.pk,
and serving as Assistant Vice President at Riztech Pvt Ltd — running brand,
engineering, and digital operations end to end. That stretch built a habit of
demanding measurable, verifiable results instead of taking things on faith.

**The convergence.** Returning to cricket as an ICC Level 2 Certified Coach, Shoaib
founded Strikers Den Sports Academy and went on to lead sport across six private
school campuses, authoring two coaching frameworks along the way — *Built to Win*,
a Cambridge PE curriculum for Grades 3–10, and *Built to Perform*, an elite cricket
coaching manual. Coaching at that scale exposed a real gap: technique assessment in
cricket still runs almost entirely on the coach's eye — valuable, but subjective,
and hard to track consistently over months of training. Apex Coach AI exists to put
real, repeatable numbers behind that judgment, not replace it.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="story-card">

### Why This Matters

Coaching by eye doesn't scale, and it doesn't leave a trail — two coaches can watch
the same delivery and disagree, and neither can easily show a bowler exactly how
this month compares to last month. Apex Coach AI was built to close that gap:
measure what can genuinely be measured, say "not confident" when tracking quality
doesn't support a claim, and never dress up a guess as a fact. The coaching
judgment still comes from the coach — this just gives it something solid to stand on.

</div>
""", unsafe_allow_html=True)

st.divider()
st.caption("Apex Coach AI — Autonomous Biomechanical Performance Hub")
