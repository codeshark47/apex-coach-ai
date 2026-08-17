"""
pages/8_💳_Upgrade.py

Manual-verification subscription flow (2026-08-17) — see payments.py's
module docstring for why this is manual review rather than an automated
checkout (no merchant/API payment account exists yet, personal
JazzCash/Easypaisa/bank accounts only).

Streamlit does NOT share the main script's auth gate across pages/
files — each page runs as its own script — so this page re-checks
st.session_state.auth_user itself before rendering anything, same
pattern as every other page in this app.
"""

import streamlit as st

import monitoring
monitoring.init_sentry()

import payments

st.set_page_config(page_title="Upgrade - Apex Coach AI", page_icon="💳", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
.stApp, .stApp p, .stApp li, .stApp span, .stApp label { color: #E2E8F0 !important; }
h1, h2, h3 { color: #00B4D8 !important; }
section[data-testid="stSidebar"] { background-color: #0F1524 !important; border-right: 1px solid #00B4D8; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
.tier-card {
    background: linear-gradient(145deg, #121824, #1A2333);
    border: 1px solid #1E3A5F; border-radius: 12px; padding: 24px 28px; margin-bottom: 18px;
}
.tier-card.current { border: 1px solid #00B4D8; }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# AUTH GATE — re-checked here since pages/ scripts run independently
# ====================================================================
if not st.session_state.get("auth_user"):
    st.error("🔒 Please sign in from the main Apex Coach AI page first.")
    st.stop()

user_id = st.session_state.auth_user["id"]
user_email = st.session_state.auth_user.get("email", "")

st.markdown("<h1 style='text-align:center;'>💳 Upgrade Your Plan</h1>", unsafe_allow_html=True)
st.divider()

sub = payments.get_subscription(user_id)
current_tier = sub["tier"]

st.markdown(f"""
<div class="tier-card current">
<b>Current plan:</b> {current_tier.upper()}
{"— expires " + sub["expires_at"][:10] if sub.get("expires_at") else ""}
</div>
""", unsafe_allow_html=True)

st.info(
    "💡 **How this works right now**: we don't yet have an automated payment gateway "
    "connected, so upgrades go through a quick manual check — send payment, tell us the "
    "transaction reference, and we confirm it against our own account activity, usually "
    "within a day. Your plan activates as soon as we approve it."
)

st.markdown("### Plans")
cols = st.columns(2)
tier_labels = {"pro": "Pro", "academy": "Academy"}
for col, tier in zip(cols, ["pro", "academy"]):
    with col:
        pricing = payments.TIER_PRICING_PKR[tier]
        limit = payments.TIER_LIMITS[tier]
        st.markdown(f"""
        <div class="tier-card">
        <h3>{tier_labels[tier]}</h3>
        <p>{limit} analyses/day</p>
        <p><b>PKR {pricing['monthly']:,}</b>/month or <b>PKR {pricing['annual']:,}</b>/year</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.markdown("### Submit a payment")

tier_choice = st.selectbox("Plan", ["pro", "academy"], format_func=lambda t: tier_labels[t], key="_pay_tier")
period_choice = st.radio("Billing period", ["monthly", "annual"], horizontal=True, key="_pay_period")
amount = payments.TIER_PRICING_PKR[tier_choice][period_choice]
st.write(f"**Amount to send: PKR {amount:,}**")

method_choice = st.radio(
    "Payment method", list(payments.PAYMENT_INSTRUCTIONS.keys()),
    format_func=lambda m: payments.PAYMENT_INSTRUCTIONS[m]["label"], horizontal=True, key="_pay_method",
)
instr = payments.PAYMENT_INSTRUCTIONS[method_choice]
if method_choice == "bank_transfer":
    st.markdown(
        f"Send to: **{instr['bank_name']}**, account **{instr['account_number']}**, "
        f"name **{instr['account_name']}**"
    )
else:
    st.markdown(f"Send to: **{instr['number']}** ({instr['account_name']})")

reference = st.text_input(
    "Transaction ID / reference number",
    help="The reference number from your JazzCash/Easypaisa/bank confirmation, so we can match your payment.",
    key="_pay_reference",
)

if st.button("Submit for review", key="_pay_submit"):
    if not reference.strip():
        st.error("Enter the transaction reference from your payment confirmation first.")
    else:
        try:
            payments.submit_payment(
                user_id, user_email, tier_choice, period_choice, method_choice, reference,
            )
            st.success(
                "✅ Submitted — we'll confirm your payment and activate your plan, usually "
                "within a day."
            )
        except Exception as e:
            monitoring.capture(e)
            st.error(f"Something went wrong submitting this: {e}")
