"""
payments.py

Manual-verification subscription flow (2026-08-17) — built because the
business has personal JazzCash/Easypaisa/bank accounts only, no
merchant/API payment account yet (Stripe doesn't support Pakistan
payouts; the business's actual customers are Pakistan-based so a local
gateway like Safepay is the better fit anyway once ready — see the
project's ball-tracking-strategy-adjacent memory for that research).
A coach picks a tier, is shown where to send payment, and submits their
own transaction reference; an admin checks the real JazzCash/Easypaisa/
bank transaction history and approves or rejects the claim by hand.

This is a legitimate, common bootstrap pattern for a business without
merchant API access yet — NOT a permanent design. Swap
PAYMENT_INSTRUCTIONS/TIER_PRICING_PKR for a real gateway integration
later without touching usage_limits.py's tier-check logic, which reads
from `subscriptions` either way, not from how that table gets written.

PLACEHOLDER VALUES BELOW (marked TODO) are NOT real business numbers —
pricing and payment destination are business decisions that must be
filled in with the founder's actual figures before this is shown to any
real coach. Shipping the TODO values as-is would show fabricated prices
and a payment number nobody can actually receive money at.
"""
import os
from datetime import datetime, timedelta, timezone

import monitoring
from profile_store import get_client

# TODO: confirm real tier limits with the founder before launch.
TIER_LIMITS = {
    "free": int(os.environ.get("DEMO_FREE_LIMIT", "2")),
    "pro": 15,
    "academy": 10_000,  # effectively unlimited for a single academy account
}

# TODO: replace with real PKR prices once decided — these are placeholders only.
TIER_PRICING_PKR = {
    "pro": {"monthly": 2000, "annual": 20000},
    "academy": {"monthly": 8000, "annual": 80000},
}

# TODO: replace with the founder's real JazzCash number, Easypaisa number,
# and bank account details before this is shown to any real coach.
PAYMENT_INSTRUCTIONS = {
    "jazzcash": {"label": "JazzCash", "number": "TODO-ADD-JAZZCASH-NUMBER", "account_name": "TODO-ADD-NAME"},
    "easypaisa": {"label": "Easypaisa", "number": "TODO-ADD-EASYPAISA-NUMBER", "account_name": "TODO-ADD-NAME"},
    "bank_transfer": {"label": "Bank Transfer", "bank_name": "TODO-ADD-BANK", "account_number": "TODO-ADD-ACCOUNT-NUMBER", "account_name": "TODO-ADD-NAME"},
}

BILLING_PERIOD_DAYS = {"monthly": 30, "annual": 365}


def _table_missing(e: Exception) -> bool:
    # Same "migration hasn't been run against this project yet" tolerance
    # as usage_limits.py's MIGRATION SAFETY note — don't crash a coach's
    # session just because add_subscriptions.sql hasn't been run in this
    # particular deployment yet.
    return "does not exist" in str(e) or "PGRST" in str(e)


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
        if _table_missing(e):
            return {"tier": "free", "status": "active", "expires_at": None}
        monitoring.capture(e)
        return {"tier": "free", "status": "active", "expires_at": None}

    if not result.data:
        return {"tier": "free", "status": "active", "expires_at": None}

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


def effective_limit(user_id: str) -> int:
    """The daily analysis limit this user should get RIGHT NOW, accounting
    for tier + expiry — usage_limits.get_usage() calls this instead of
    a single fixed constant."""
    sub = get_subscription(user_id)
    return TIER_LIMITS.get(sub["tier"], TIER_LIMITS["free"])


def submit_payment(user_id: str, user_email: str, tier: str, billing_period: str,
                    payment_method: str, transaction_reference: str) -> dict:
    """Records a coach's payment claim as PENDING — grants nothing yet.
    An admin must approve it (see approve_payment) before the tier
    actually changes. Raises ValueError on bad tier/period rather than
    silently recording a nonsense claim."""
    if tier not in TIER_PRICING_PKR:
        raise ValueError(f"Unknown tier: {tier}")
    if billing_period not in BILLING_PERIOD_DAYS:
        raise ValueError(f"Unknown billing period: {billing_period}")
    if not transaction_reference or not transaction_reference.strip():
        raise ValueError("A transaction reference is required.")

    amount = TIER_PRICING_PKR[tier][billing_period]
    client = get_client()
    payload = {
        "user_id": user_id,
        "user_email": user_email,
        "tier_requested": tier,
        "billing_period": billing_period,
        "amount_pkr": amount,
        "payment_method": payment_method,
        "transaction_reference": transaction_reference.strip(),
        # Set explicitly rather than left to the table's default —
        # the row this function hands back should be self-describing
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
    still gets the full period they paid for, not a period that started
    counting down before anyone confirmed the payment."""
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
        "status": "active", "expires_at": expires_at.isoformat(), "updated_at": now.isoformat(),
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
