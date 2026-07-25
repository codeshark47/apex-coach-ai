"""
pages/4_🏫_Academy_Profile.py

Real multi-bowler roster management: create teams, assign athletes to
them, see who's on which team and who isn't assigned yet. Backed by the
teams table + athletes.team_id column (add_teams.sql) — if that migration
hasn't been run yet in Supabase, this page says so plainly instead of
crashing or faking a roster.
"""

import html

import streamlit as st

import profile_store as store

st.set_page_config(page_title="Academy Profile - Apex Coach AI", page_icon="🏫", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span, .stApp label { color: #E2E8F0 !important; }
h1, h2, h3 { color: #00B4D8 !important; }
section[data-testid="stSidebar"] { background-color: #0F1524 !important; border-right: 1px solid #00B4D8; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
.roster-card {
    background: linear-gradient(145deg, #121824, #1A2333);
    border: 1px solid #1E3A5F; border-radius: 12px; padding: 20px 24px; margin-bottom: 14px;
}
.athlete-row {
    display: flex; align-items: center; justify-content: space-between;
    background: #121824; border: 1px solid #1E3A5F; border-radius: 8px;
    padding: 10px 16px; margin-bottom: 8px;
}
.team-count-pill {
    display: inline-block; background: #0F2A44; color: #4FD1E8 !important;
    border: 1px solid #1E3A5F; border-radius: 999px; padding: 3px 12px; font-size: 0.8rem;
}
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

st.markdown("<h1 style='text-align:center;'>🏫 Academy / Team Profile</h1>", unsafe_allow_html=True)
st.divider()

# add_teams.sql detection: list_teams() hits a real "teams" table, so a
# missing-table Postgres error here means the migration hasn't run yet.
try:
    teams = store.list_teams(coach_user_id)
    teams_available = True
except Exception as e:
    teams = []
    teams_available = False
    st.warning(
        "The teams/roster feature needs a one-time database migration that hasn't "
        f"been applied yet (`add_teams.sql`). Run it in your Supabase project's SQL "
        f"Editor, then reload this page.\n\nDetails: {e}"
    )
    st.stop()

# ====================================================================
# CREATE A NEW TEAM
# ====================================================================
with st.expander("➕ Create a new team / academy group", expanded=(len(teams) == 0)):
    with st.form("create_team_form", clear_on_submit=True):
        new_team_name = st.text_input("Team / squad name", placeholder="e.g. U-17 Fast Bowlers")
        if st.form_submit_button("Create team", use_container_width=True):
            if new_team_name.strip():
                try:
                    store.get_or_create_team(new_team_name, coach_user_id)
                    st.success(f"Team '{new_team_name.strip()}' created.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not create team: {e}")
            else:
                st.error("Team name cannot be empty.")

if not teams:
    st.info("No teams yet — create one above to start building a roster.")
    st.stop()

st.divider()

# ====================================================================
# TEAM SELECTOR + ROSTER
# ====================================================================
team_names = [t["name"] for t in teams]
selected_team_name = st.selectbox("Select team", team_names, key="academy_team_select")
selected_team_id = next(t["id"] for t in teams if t["name"] == selected_team_name)

try:
    roster = store.list_athletes_by_team(selected_team_id, coach_user_id)
except Exception as e:
    st.error(f"Could not load roster: {e}")
    st.stop()

try:
    unassigned = store.list_unassigned_athletes(coach_user_id)
except Exception as e:
    unassigned = []
    st.error(f"Could not load unassigned athletes: {e}")

roster_col, add_col = st.columns([3, 2])

with roster_col:
    st.markdown(f"### Roster — {selected_team_name}")
    st.markdown(f'<span class="team-count-pill">{len(roster)} athlete(s)</span>', unsafe_allow_html=True)
    st.write("")

    if not roster:
        st.caption("No athletes on this team yet. Add one from the unassigned list on the right.")
    else:
        for athlete in roster:
            row_l, row_m, row_r = st.columns([3, 2, 2])
            with row_l:
                st.markdown(f"**{athlete['name']}**")
            with row_m:
                if st.button("View profile", key=f"view_{athlete['id']}", use_container_width=True):
                    st.session_state["bowler_profile_select"] = athlete["name"]
                    st.switch_page("pages/3_📊_Bowler_Profile.py")
            with row_r:
                if st.button("Remove from team", key=f"remove_{athlete['id']}", use_container_width=True):
                    try:
                        store.assign_athlete_to_team(athlete["id"], None, coach_user_id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not remove athlete: {e}")

with add_col:
    st.markdown("### Add to this team")
    if not unassigned:
        st.caption("No unassigned athletes — everyone with a profile is already on a team.")
    else:
        unassigned_names = [a["name"] for a in unassigned]
        pick_name = st.selectbox("Unassigned athletes", unassigned_names, key="academy_pick_unassigned")
        if st.button("Add to team", use_container_width=True):
            pick_id = next(a["id"] for a in unassigned if a["name"] == pick_name)
            try:
                store.assign_athlete_to_team(pick_id, selected_team_id, coach_user_id)
                st.success(f"Added {pick_name} to {selected_team_name}.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not assign athlete: {e}")

st.divider()

# ====================================================================
# ALL TEAMS OVERVIEW
# ====================================================================
st.subheader("All teams at a glance")
overview_cols = st.columns(min(len(teams), 4) or 1)
for i, t in enumerate(teams):
    try:
        count = len(store.list_athletes_by_team(t["id"], coach_user_id))
    except Exception:
        count = "—"
    with overview_cols[i % len(overview_cols)]:
        # SECURITY FIX: team name is coach-typed free text embedded into an
        # unsafe_allow_html block — escape it so it can only ever render as
        # text, never break out into markup (see the same fix in
        # Bowler Profile for the full reasoning).
        st.markdown(f"""
        <div class="roster-card" style="text-align:center;">
            <div style="font-size:1.1rem;font-weight:600;">{html.escape(t['name'])}</div>
            <div class="team-count-pill" style="margin-top:8px;">{count} athlete(s)</div>
        </div>
        """, unsafe_allow_html=True)

if unassigned:
    st.caption(f"{len(unassigned)} athlete(s) not yet assigned to any team.")
