import streamlit as st
from supabase import create_client, Client

def init_supabase() -> Client:
    """Initializes the connection client by reading credentials safely from secrets."""
    try:
        # Check raw TOML structure parsing integrity
        _ = st.secrets.keys()
    except Exception as toml_err:
        st.error("🚨 **CRITICAL: Your .streamlit/secrets.toml file has a broken syntax typo!**")
        st.code(str(toml_err), language="python")
        st.stop()

    url = st.secrets.get("SUPABASE_URL")
    anon_key = st.secrets.get("SUPABASE_ANON_KEY")

    if not url or not anon_key:
        missing = []
        if not url: missing.append("SUPABASE_URL")
        if not anon_key: missing.append("SUPABASE_ANON_KEY")
        st.error(f"🚨 **MISSING KEYS:** Missing fields: {', '.join(missing)}")
        st.stop()

    try:
        return create_client(url, anon_key)
    except Exception as connection_err:
        st.error("🚨 **SUPABASE ARCHITECTURE CONNECTION FAILURE:** Connection rejected.")
        st.code(str(connection_err), language="python")
        st.stop()

def sign_in_user(email, password):
    """Attempts user login and catches explicit auth errors without masking them."""
    supabase = init_supabase()
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            st.session_state["user"] = response.user
            st.session_state["session"] = response.session
            st.success("🎉 Login successful! Redirecting...")
            st.rerun()
    except Exception as auth_error:
        error_msg = str(auth_error).lower()
        
        st.error("❌ **Authentication Failed: Invalid Login Credentials**")
        
        # Diagnostics Checklist Box
        with st.expander("🛠️ Debugger Checklist — Why am I seeing this?"):
            st.markdown("""
            1. **Email Confirmation Pending:** Did you click the confirmation link sent to your email inbox? Supabase blocks sign-ins until verified.
            2. **Key Assignment Mixup:** double-check your `.streamlit/secrets.toml`. 
               * `SUPABASE_ANON_KEY` **must** be your **ANON / PUBLIC** key.
               * `SUPABASE_KEY` **must** be your **SERVICE ROLE / SECRET** key.
               If these are flipped, password matching fails.
            """)
            st.warning(f"Raw system return string: {str(auth_error)}")

def sign_up_user(email, password):
    """Handles new user creation inside the database."""
    supabase = init_supabase()
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            st.info("📩 **Registration initialized successfully!**")
            st.success("Please check your email inbox and click the Supabase verification link before logging in.")
    except Exception as sign_up_err:
        st.error("❌ Registration Pipeline Blocked")
        st.code(str(sign_up_err), language="python")