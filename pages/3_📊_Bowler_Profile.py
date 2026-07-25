"""
pages/3_📊_Bowler_Profile.py

Rich per-athlete profile: photo, team, coaching notes, and real session
history with metric trend charts pulled straight from Supabase — no
fabricated data, no placeholder numbers. If an athlete has zero sessions,
this says so plainly instead of drawing an empty/fake chart.

Streamlit does NOT share the main script's auth gate across pages/ files —
each page runs as its own script — so this page re-checks st.session_state
.auth_user itself before rendering anything athlete-specific.
"""

import html

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

import profile_store as store
import metric_ranges as mr

st.set_page_config(page_title="Bowler Profile - Apex Coach AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span, .stApp label { color: #E2E8F0 !important; }
h1, h2, h3 { color: #00B4D8 !important; }
section[data-testid="stSidebar"] { background-color: #0F1524 !important; border-right: 1px solid #00B4D8; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
.profile-card {
    background: linear-gradient(145deg, #121824, #1A2333);
    border: 1px solid #1E3A5F; border-radius: 12px; padding: 24px 28px; margin-bottom: 18px;
}
.avatar-circle {
    width: 96px; height: 96px; border-radius: 50%; object-fit: cover;
    border: 2px solid #00B4D8;
}
.avatar-placeholder {
    width: 96px; height: 96px; border-radius: 50%; background: #0F2A44;
    border: 2px solid #1E3A5F; display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
}
.team-pill {
    display: inline-block; background: #0F2A44; color: #4FD1E8 !important;
    border: 1px solid #1E3A5F; border-radius: 999px; padding: 3px 12px;
    font-size: 0.8rem; margin-top: 6px;
}
.stat-tile {
    background: #121824; border: 1px solid #1E3A5F; border-radius: 10px;
    padding: 14px 16px; text-align: center;
}
.stat-value { font-size: 1.6rem; font-weight: 700; color: #E2E8F0; }
.stat-label { font-size: 0.78rem; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 0.04em; }
.tier-pill { display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# AUTH GATE — re-checked here since pages/ scripts run independently
# ====================================================================
if not st.session_state.get("auth_user"):
    st.error("🔒 Please sign in from the main Apex Coach AI page first.")
    st.stop()

coach_user_id = st.session_state.auth_user["id"]

try:
    store.get_client()
except RuntimeError as e:
    st.error(f"Supabase is not configured: {e}")
    st.stop()

st.markdown("<h1 style='text-align:center;'>📊 Bowler Profile</h1>", unsafe_allow_html=True)
st.divider()

# ====================================================================
# ATHLETE SELECTOR
# ====================================================================
try:
    athletes = store.list_athletes(coach_user_id)
except Exception as e:
    st.error(f"Could not load athletes: {e}")
    st.stop()

if not athletes:
    st.info("No athlete profiles yet. Run an analysis from the main page first — a profile is created automatically the first time you name a bowler.")
    st.stop()

names = [a["name"] for a in athletes]
selected_name = st.selectbox("Select bowler", names, key="bowler_profile_select")
selected_id = next(a["id"] for a in athletes if a["name"] == selected_name)

try:
    athlete = store.get_athlete(selected_id, coach_user_id)
except Exception as e:
    st.error(f"Could not load athlete profile: {e}")
    st.stop()

try:
    teams = store.list_teams(coach_user_id)
except Exception:
    teams = []  # add_teams.sql not applied yet — profile still works, just no team feature
team_names_by_id = {t["id"]: t["name"] for t in teams}

# ====================================================================
# PROFILE HEADER CARD
# ====================================================================
photo_url = athlete.get("photo_url")
team_id = athlete.get("team_id")
notes = athlete.get("notes") or ""

header_l, header_r = st.columns([1, 5])
with header_l:
    if photo_url:
        # SECURITY FIX: photo_url is coach-typed free text (the "Edit
        # profile" form), embedded directly into an unsafe_allow_html
        # block — unescaped, a value like `"><script>...` would break out
        # of the src="" attribute and execute as live HTML/JS in whoever
        # views this profile. html.escape() neutralizes quotes/angle
        # brackets so it can only ever render as an (possibly broken)
        # image, never as markup.
        st.markdown(f'<img src="{html.escape(photo_url)}" class="avatar-circle">', unsafe_allow_html=True)
    else:
        st.markdown('<div class="avatar-placeholder">🏏</div>', unsafe_allow_html=True)
with header_r:
    st.markdown(f"### {athlete['name']}")
    if team_id and team_id in team_names_by_id:
        # Same fix — team name is coach-typed free text (the "Create team" form).
        st.markdown(f'<span class="team-pill">🏫 {html.escape(team_names_by_id[team_id])}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="team-pill">Unassigned</span>', unsafe_allow_html=True)
    if notes:
        st.caption(notes)

with st.expander("✏️ Edit profile", expanded=False):
    with st.form("edit_profile_form"):
        new_photo_url = st.text_input("Photo URL", value=photo_url or "",
                                       help="Paste a direct image link (e.g. from Google Drive share, imgur, etc.)")
        new_notes = st.text_area("Coaching notes", value=notes, height=90)

        if teams:
            team_options = ["Unassigned"] + [t["name"] for t in teams]
            current_team_name = team_names_by_id.get(team_id, "Unassigned")
            chosen_team_name = st.selectbox("Team", team_options,
                                             index=team_options.index(current_team_name) if current_team_name in team_options else 0)
        else:
            chosen_team_name = None
            st.caption("Team assignment requires add_teams.sql to be run in Supabase first — see the Academy/Team Profile page.")

        if st.form_submit_button("Save changes", use_container_width=True):
            try:
                store.update_athlete_profile(selected_id, coach_user_id,
                                              photo_url=new_photo_url, notes=new_notes)
                if teams:
                    new_team_id = None
                    if chosen_team_name != "Unassigned":
                        new_team_id = next(t["id"] for t in teams if t["name"] == chosen_team_name)
                    store.assign_athlete_to_team(selected_id, new_team_id, coach_user_id)
                st.success("Profile updated.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not save changes: {e}")

st.divider()

# ====================================================================
# SESSION HISTORY
# ====================================================================
try:
    history = store.get_athlete_history(selected_id, coach_user_id, limit=50)
except Exception as e:
    st.error(f"Could not load session history: {e}")
    st.stop()

if not history:
    st.info(f"No sessions recorded yet for {athlete['name']}. Run an analysis on the main page to start building this history.")
    st.stop()

METRIC_KEYS = ["front_knee_bracing", "hip_shoulder_separation", "trunk_lean", "release_height", "head_stability"]

TIER_LABELS = {"green": "Optimal", "amber": "Acceptable", "red": "Concern", "unknown": "No data"}


def _parse_date(raw):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


# most recent first from get_athlete_history — keep for the hero card,
# reverse for chronological charting
latest = history[0]
latest_metrics = latest.get("metrics", {}) or {}

st.subheader(f"Latest session — {(_parse_date(latest.get('session_date')) or datetime.now()).strftime('%b %d, %Y')}")

stat_cols = st.columns(len(METRIC_KEYS))
for col, key in zip(stat_cols, METRIC_KEYS):
    value = mr.extract_metric_value(latest_metrics, key)
    tier = mr.classify(key, value)
    tier_color = mr.TIER_COLORS[tier]
    display_value = mr.format_value(key, value) if value is not None else "—"
    with col:
        st.markdown(f"""
        <div class="stat-tile">
            <div class="stat-label">{mr.RANGES[key].label}</div>
            <div class="stat-value">{display_value}</div>
            <div class="tier-pill" style="background:{tier_color}22; color:{tier_color};">{TIER_LABELS[tier]}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ====================================================================
# TREND CHARTS — one metric per chart (small multiples, one axis each),
# shaded reference zones from metric_ranges.py, markers colored by tier.
# ====================================================================
st.subheader("Metric trends over time")
st.caption("Shaded bands show the same optimal / acceptable / concern zones used in the PDF report and dashboard. Line color is neutral; each point is colored by that session's classification.")

chronological = list(reversed(history))
dates = [_parse_date(s.get("session_date")) for s in chronological]

def _zone_shapes(metric_key: str, y_axis_max: float, y_axis_min: float = 0.0):
    r = mr.RANGES[metric_key]
    shapes = []

    def band(lo, hi, tier):
        shapes.append(dict(
            type="rect", xref="paper", x0=0, x1=1, yref="y", y0=lo, y1=hi,
            fillcolor=mr.TIER_COLORS[tier], opacity=0.10, line=dict(width=0), layer="below",
        ))

    g_lo, g_hi = r.green
    a_lo, a_hi = r.amber
    if r.kind == "higher_better":
        band(y_axis_min, a_lo, "red")
        band(a_lo, g_lo, "amber")
        band(g_lo, y_axis_max, "green")
    elif r.kind == "lower_better":
        band(y_axis_min, g_hi, "green")
        band(g_hi, a_hi, "amber")
        band(a_hi, y_axis_max, "red")
    elif r.kind == "band":
        band(y_axis_min, a_lo, "red")
        band(a_lo, g_lo, "amber")
        band(g_lo, g_hi, "green")
        ah_lo, ah_hi = r.amber_high
        band(g_hi, ah_hi, "amber")
        band(ah_hi, y_axis_max, "red")
    return shapes


chart_cols = st.columns(2)
for i, key in enumerate(METRIC_KEYS):
    r = mr.RANGES[key]
    raw_values = [mr.extract_metric_value(s.get("metrics", {}) or {}, key) for s in chronological]

    plot_x, plot_y, plot_colors, plot_text = [], [], [], []
    for d, v in zip(dates, raw_values):
        if d is None or v is None:
            continue
        display_v = v * 100 if r.unit == "%" else v
        tier = mr.classify(key, v)
        plot_x.append(d)
        plot_y.append(display_v)
        plot_colors.append(mr.TIER_COLORS[tier])
        plot_text.append(f"{mr.format_value(key, v)} — {TIER_LABELS[tier]}")

    target_col = chart_cols[i % 2]
    with target_col:
        if len(plot_x) < 2:
            st.markdown(f"**{r.label}**")
            st.caption("Need at least 2 sessions with a valid reading to plot a trend." if plot_x else "No valid readings recorded yet for this metric.")
            continue

        y_scale = 100 if r.unit == "%" else 1
        g_lo, g_hi = r.green[0] * y_scale, r.green[1] * y_scale
        data_max = max(plot_y)
        data_min = min(plot_y)
        ceiling_ref = r.amber_high[1] * y_scale if r.kind == "band" else g_hi
        y_axis_max = max(data_max, ceiling_ref) * 1.15
        y_axis_min = min(data_min, r.amber[0] * y_scale) * 0.85 if r.kind != "lower_better" else 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_x, y=plot_y, mode="lines+markers",
            line=dict(color="#00B4D8", width=2),
            marker=dict(size=10, color=plot_colors, line=dict(width=1, color="#0B0E14")),
            text=plot_text,
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{text}<extra></extra>",
            showlegend=False,
        ))
        fig.add_annotation(
            x=plot_x[-1], y=plot_y[-1], text=f"  {plot_text[-1].split(' — ')[0]}",
            showarrow=False, xanchor="left", font=dict(color="#E2E8F0", size=12),
        )
        fig.update_layout(
            title=dict(text=f"{r.label} ({r.unit})" if r.unit else r.label, font=dict(color="#E2E8F0", size=15)),
            shapes=_zone_shapes(key, y_axis_max, y_axis_min),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"),
            xaxis=dict(gridcolor="#1E3A5F", showgrid=False, zeroline=False),
            yaxis=dict(gridcolor="#1E3A5F", range=[y_axis_min, y_axis_max], zeroline=False),
            margin=dict(l=10, r=60, t=40, b=10), height=260,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ====================================================================
# SESSION TABLE — accessible alternative to the charts above
# ====================================================================
st.subheader("Session history")
table_rows = []
for s in history:
    d = _parse_date(s.get("session_date"))
    row = {
        "Date": d.strftime("%b %d, %Y") if d else "—",
        "Camera": s.get("camera_mode", "—"),
        "Speed (km/h)": s.get("release_arm_speed_kmh") or "—",
    }
    m = s.get("metrics", {}) or {}
    for key in METRIC_KEYS:
        value = mr.extract_metric_value(m, key)
        tier = mr.classify(key, value)
        icon = {"green": "🟢", "amber": "🟡", "red": "🔴", "unknown": "⚪"}[tier]
        row[mr.RANGES[key].label] = f"{icon} {mr.format_value(key, value)}" if value is not None else "⚪ —"
    table_rows.append(row)

st.dataframe(table_rows, use_container_width=True, hide_index=True)
