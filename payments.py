"""
payments.py

Manual-verification subscription flow (2026-08-17) — built because the
business has personal JazzCash/Easypaisa/bank accounts only, no
merchant/API payment account yet (Stripe doesn't support Pakistan
payouts; the business's actual customers are Pakistan-based so a local
gateway like Safepay is the better fit anyway once ready). A coach
picks a tier, is shown where to send payment, and submits their own
transaction reference; an admin checks the real JazzCash/Easypaisa/
bank transaction history and approves or rejects the claim by hand.

This is a legitimate, common bootstrap pattern for a business without
merchant API access yet — NOT a permanent design. Swap
PAYMENT_INSTRUCTIONS/TIER_PRICING for a real gateway integration later
without touching usage_limits.py's tier-check logic, which reads from
`subscriptions` either way, not from how that table gets written.

REAL PRICING (2026-08-17, from the founder's own pricing deck) — four
tiers, each priced in BOTH PKR and USD, monthly or annual (annual =
10x monthly, i.e. 2 months free, matching the deck's own framing).
Limits are PER MONTH, not per day — a materially different tracking
model from the free tier's existing daily demo_usage table, which is
why paid usage is tracked here via period_start/used_this_period on
`subscriptions` (a rolling 30-day window, reset on boundary-cross, same
pattern as usage_limits.py's own daily reset) rather than reusing
demo_usage. An annual subscriber gets a FRESH monthly allowance every
30 days throughout their year, not one lump sum for all 12 months.

PAYMENT_INSTRUCTIONS below is still a TODO placeholder — the founder's
real JazzCash/Easypaisa/bank details haven't been provided yet. Only
PKR self-serve submission is wired up (JazzCash/Easypaisa/bank transfer
are all Pakistan-domestic); USD/international coaches are directed to
contact support until a real cross-border payment method exists.
"""
import os
from datetime import datetime, timedelta, timezone

import monitoring
from profile_store import get_client

TIER_ORDER = ["starter", "professional", "elite", "institutional"]

# Analyses per month. "institutional" is contractually unlimited — a
# large practical ceiling instead of None so every caller (this module,
# usage_limits.py, both UI pages) can do plain arithmetic on it without
# a None-special-case landmine; the UI displays it as "Unlimited" via a
# threshold check, not by reading a sentinel.
TIER_MONTHLY_LIMITS = {
    "free": int(os.environ.get("DEMO_FREE_LIMIT", "2")),  # still a DAILY limit — see usage_limits.py
    "starter": 10,
    "professional": 25,
    "elite": 50,
    "institutional": 100_000,
}
UNLIMITED_DISPLAY_THRESHOLD = 100_000

TIER_PRICING = {
    "starter":       {"pkr": {"monthly": 5_500,  "annual": 55_000},  "usd": {"monthly": 19,  "annual": 190}},
    "professional":  {"pkr": {"monthly": 13_000, "annual": 130_000}, "usd": {"monthly": 45,  "annual": 450}},
    "elite":         {"pkr": {"monthly": 23_000, "annual": 230_000}, "usd": {"monthly": 79,  "annual": 790}},
    "institutional": {"pkr": {"monthly": 40_000, "annual": 400_000}, "usd": {"monthly": 139, "annual": 1_390}},
}

# TODO: replace with the founder's real JazzCash number, Easypaisa number,
# and bank account details before this is shown to any real coach. USD/
# international payment has no self-serve path yet at all (see module
# docstring) — deliberately not listed here.
PAYMENT_INSTRUCTIONS = {
    "jazzcash": {"label": "JazzCash", "number": "TODO-ADD-JAZZCASH-NUMBER", "account_name": "TODO-ADD-NAME"},
    "easypaisa": {"label": "Easypaisa", "number": "TODO-ADD-EASYPAISA-NUMBER", "account_name": "TODO-ADD-NAME"},
    "bank_transfer": {"label": "Bank Transfer", "bank_name": "TODO-ADD-BANK", "account_number": "TODO-ADD-ACCOUNT-NUMBER", "account_name": "TODO-ADD-NAME"},
}

BILLING_PERIOD_DAYS = {"monthly": 30, "annual": 365}
USAGE_PERIOD_DAYS = 30  # every paid tier's analysis allowance refreshes on this cadence, independent of billing_period


def _table_missing(e: Exception) -> bool:
    # Same "migration hasn't been run against this project yet" tolerance
    # as usage_limits.py's MIGRATION SAFETY note — don't crash a coach's
    # session just because add_subscriptions.sql hasn't been run in this
    # particular deployment yet.
    return "does not exist" in str(e) or "PGRST" in str(e)


def _free_fallback() -> dict:
    return {"tier": "free", "status": "active", "expires_at": None}


def get_subscription(user_id: str) -> dict:
    """
    Returns {"tier": str, "status": str, "expires_at": str or None}.
    Defaults to free/active if no row exists yet, or if the migration
    hasn't been run in this deployment — never crashes a coach's session
    over a missing row or a missing table.

    Auto-expires a paid tier back to "free" the moment expires_at has
    passed (persisted immediately, not just returned) — so a lapsed
    subscription doesn't keep granting paid limits forever just because
    nothing else re-checked it.
    """
    try:
        client = get_client()
        result = client.table("subscriptions").select("*").eq("user_id", user_id).execute()
    except Exception as e:
        if not _table_missing(e):
            monitoring.capture(e)
        return _free_fallback()

    if not result.data:
        return _free_fallback()

    row = result.data[0]
    tier, status, expires_at = row["tier"], row["status"], row.get("expires_at")

    if tier != "free" and status == "active" and expires_at:
        try:
            expiry_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry_dt < datetime.now(expiry_dt.tzinfo):
                status = "expired"
                client.table("subscriptions").update(
                    {"status": "expired", "updated_at": datetime.now(timezone.utc).isoformat()}
                ).eq("user_id", user_id).execute()
        except (ValueError, TypeError) as e:
            monitoring.capture(e)

    return {"tier": tier if status == "active" else "free", "status": status, "expires_at": expires_at}


def get_monthly_usage(user_id: str) -> dict:
    """
    {"used", "limit", "remaining"} for an ACTIVE PAID subscriber, tracked
    on a rolling 30-day window (period_start -> period_start+30d) that's
    independent of billing_period — an annual subscriber still gets a
    FRESH allowance every 30 days, not one total for the whole year.
    Resets used_this_period to 0 the moment "now" has crossed the
    window boundary, persisted immediately — same reset-on-boundary-
    change pattern usage_limits.py already uses for the free tier's
    daily reset, just on a 30-day cadence instead of midnight.

    Free-tier users should never reach this — usage_limits.get_usage()
    only delegates here once get_subscription() confirms a paid tier.
    If called anyway (defensive), falls back to the free daily limit's
    number so callers never see a broken/zero limit.
    """
    try:
        client = get_client()
        result = client.table("subscriptions").select("*").eq("user_id", user_id).execute()
    except Exception as e:
        if not _table_missing(e):
            monitoring.capture(e)
        return {"used": 0, "limit": TIER_MONTHLY_LIMITS["free"], "remaining": TIER_MONTHLY_LIMITS["free"]}

    if not result.data:
        return {"used": 0, "limit": TIER_MONTHLY_LIMITS["free"], "remaining": TIER_MONTHLY_LIMITS["free"]}

    row = result.data[0]
    tier = row["tier"]
    limit = TIER_MONTHLY_LIMITS.get(tier, TIER_MONTHLY_LIMITS["free"])
    now = datetime.now(timezone.utc)
    period_start_raw = row.get("period_start")
    period_start = (
        datetime.fromisoformat(period_start_raw.replace("Z", "+00:00")) if period_start_raw else None
    )
    used = row.get("used_this_period") or 0

    if period_start is None or now >= period_start + timedelta(days=USAGE_PERIOD_DAYS):
        used = 0
        client.table("subscriptions").update(
            {"used_this_period": 0, "period_start": now.isoformat()}
        ).eq("user_id", user_id).execute()

    return {"used": used, "limit": limit, "remaining": max(0, limit - used)}


def record_monthly_usage(user_id: str) -> dict:
    """Increments this paid subscriber's used_this_period by 1 (for the
    CURRENT 30-day window — get_monthly_usage above already reset it if
    a new window just started). Call this ONLY after an analysis has
    genuinely completed successfully, same rule as usage_limits.record_usage."""
    current = get_monthly_usage(user_id)  # ensures the window reset has happened
    client = get_client()
    new_used = current["used"] + 1
    client.table("subscriptions").update({"used_this_period": new_used}).eq("user_id", user_id).execute()
    return {"used": new_used, "limit": current["limit"], "remaining": max(0, current["limit"] - new_used)}


def submit_payment(user_id: str, user_email: str, tier: str, billing_period: str,
                    currency: str, payment_method: str, transaction_reference: str) -> dict:
    """Records a coach's payment claim as PENDING — grants nothing yet.
    An admin must approve it (see approve_payment) before the tier
    actually changes. Raises ValueError on bad input rather than
    silently recording a nonsense claim."""
    if tier not in TIER_PRICING:
        raise ValueError(f"Unknown tier: {tier}")
    if billing_period not in BILLING_PERIOD_DAYS:
        raise ValueError(f"Unknown billing period: {billing_period}")
    if currency not in ("pkr", "usd"):
        raise ValueError(f"Unknown currency: {currency}")
    if not transaction_reference or not transaction_reference.strip():
        raise ValueError("A transaction reference is required.")

    amount = TIER_PRICING[tier][currency][billing_period]
    client = get_client()
    payload = {
        "user_id": user_id,
        "user_email": user_email,
        "tier_requested": tier,
        "billing_period": billing_period,
        "currency": currency,
        "amount": amount,
        "payment_method": payment_method,
        "transaction_reference": transaction_reference.strip(),
        # Set explicitly rather than left to the table's default — the
        # row this function hands back should be self-describing
        # without needing a real DB round-trip to know its own state.
        "status": "pending",
    }
    result = client.table("payment_submissions").insert(payload).execute()
    return result.data[0]


def list_pending_payments() -> list:
    client = get_client()
    result = (
        client.table("payment_submissions")
        .select("*")
        .eq("status", "pending")
        .order("submitted_at", desc=False)
        .execute()
    )
    return result.data


def approve_payment(submission_id: str, admin_email: str, notes: str = None) -> dict:
    """Marks the submission approved and activates/extends the coach's
    subscription for the paid billing period, starting from NOW (not
    from submission time) — a coach approved a few days after submitting
    still gets the full period they paid for. Also (re)starts the
    monthly usage-allowance window fresh from approval time, so a coach
    approved mid-month doesn't inherit a stale/partial window."""
    client = get_client()
    sub_result = client.table("payment_submissions").select("*").eq("id", submission_id).execute()
    if not sub_result.data:
        raise ValueError(f"No payment submission with id {submission_id}")
    submission = sub_result.data[0]

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=BILLING_PERIOD_DAYS[submission["billing_period"]])

    client.table("payment_submissions").update({
        "status": "approved", "reviewed_by": admin_email,
        "reviewed_at": now.isoformat(), "admin_notes": notes,
    }).eq("id", submission_id).execute()

    existing = client.table("subscriptions").select("*").eq("user_id", submission["user_id"]).execute()
    sub_payload = {
        "user_id": submission["user_id"], "tier": submission["tier_requested"],
        "status": "active", "currency": submission["currency"],
        "expires_at": expires_at.isoformat(), "updated_at": now.isoformat(),
        "period_start": now.isoformat(), "used_this_period": 0,
    }
    if existing.data:
        client.table("subscriptions").update(sub_payload).eq("user_id", submission["user_id"]).execute()
    else:
        client.table("subscriptions").insert(sub_payload).execute()

    return {"submission_id": submission_id, "tier": submission["tier_requested"], "expires_at": expires_at.isoformat()}


def reject_payment(submission_id: str, admin_email: str, notes: str = None) -> None:
    client = get_client()
    client.table("payment_submissions").update({
        "status": "rejected", "reviewed_by": admin_email,
        "reviewed_at": datetime.now(timezone.utc).isoformat(), "admin_notes": notes,
    }).eq("id", submission_id).execute()
