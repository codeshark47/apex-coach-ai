"""
ball_tracking package — Phase 2, isolated from the working biomechanics
pipeline by design.

No file in here is imported by main.py, orchestrator.py,
dual_camera_orchestrator.py, or streamlit_app.py. It has its own data
path (see detect_ball_classical.py) and, once schema is applied, its own
Supabase table — never the existing athletes/sessions tables. If a
change here ever seems to require editing one of those existing files,
that's a signal to stop and reconsider the approach rather than cross
that boundary.

Status as of Phase 2 Day 1: feasibility testing only. Nothing in this
package has been validated against real footage yet — see
detect_ball_classical.py's module docstring for exactly what's proven
and what isn't.
"""
