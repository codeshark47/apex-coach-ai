"""
pages/9_🛡️_Admin_Billing.py

Manual payment-approval queue (2026-08-17) — see payments.py's module
docstring. Admin checks the real JazzCash/Easypaisa/bank transaction
history for each pending submission's reference number, then approves
or rejects it here; approval activates the coach's subscription.

Gated on usage_limits.is_admin(), same allowlist mechanism already used
elsewhere in this app (e.g. the ball-tracking beta's earlier admin-only
period) — re-checked here since pages/ scripts run independently of the
main script's own auth gate.
"""

import streamlit as st

import monitoring
monitoring.init_sentry()

import payments
import usage_limits

st.set_page_config(page_title="Admin Billing - Apex Coach AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span, .stApp label { color: #E2E8F0 !important; }
h1, h2, h3 { color: #00B4D8 !important; }
section[data-testid="stSidebar"] { background-color: #0F1524 !important; border-right: 1px solid #00B4D8; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
.submission-card {
    background: linear-gradient(145deg, #121824, #1A2333);
    border: 1px solid #1E3A5F; border-radius: 12px; padding: 20px 24px; margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.get("auth_user"):
    st.error("🔒 Please sign in from the main Apex Coach AI page first.")
    st.stop()

admin_email = st.session_state.auth_user.get("email", "")
if not usage_limits.is_admin(admin_email):
    st.error("🔒 This page is admin-only.")
    st.stop()

st.markdown("<h1 style='text-align:center;'>🛡️ Pending Payment Approvals</h1>", unsafe_allow_html=True)
st.divider()

pending = payments.list_pending_payments()
if not pending:
    st.success("No pending payments — all caught up.")
else:
    st.caption(
        "Check each transaction reference against your real JazzCash/Easypaisa/bank "
        "activity before approving — approving activates the coach's plan immediately."
    )
    for sub in pending:
        with st.container():
            st.markdown(f"""
            <div class="submission-card">
            <b>{sub['user_email']}</b><br>
            Plan: <b>{sub['tier_requested']}</b> ({sub['billing_period']}) — PKR {sub['amount_pkr']:,}<br>
            Method: {sub['payment_method']}<br>
            Reference: <code>{sub['transaction_reference']}</code><br>
            Submitted: {sub['submitted_at'][:16].replace('T', ' ')}
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("✅ Approve", key=f"approve_{sub['id']}"):
                    try:
                        payments.approve_payment(sub["id"], admin_email)
                        st.success(f"Approved — {sub['user_email']} is now on {sub['tier_requested']}.")
                        st.rerun()
                    except Exception as e:
                        monitoring.capture(e)
                        st.error(f"Failed to approve: {e}")
            with col2:
                if st.button("❌ Reject", key=f"reject_{sub['id']}"):
                    try:
                        payments.reject_payment(sub["id"], admin_email)
                        st.warning(f"Rejected {sub['user_email']}'s submission.")
                        st.rerun()
                    except Exception as e:
                        monitoring.capture(e)
                        st.error(f"Failed to reject: {e}")
