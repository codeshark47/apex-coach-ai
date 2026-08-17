"""
tests/test_payments.py

Regression tests for payments.py's manual-verification subscription
flow (2026-08-17) — a fake Supabase client so no real network/Supabase
project is needed to run them, same pattern as test_usage_limits.py.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import payments as pm


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    """Minimal stand-in for supabase-py's table/query builder — enough
    chained-method surface (select/eq/order/update/insert/execute) for
    payments.py's exact call patterns. store is keyed by row "id" for
    payment_submissions, by "user_id" for subscriptions — both tables
    share one store dict in these tests, keyed however the test set it
    up, since _FakeTable itself is table-name-agnostic (same trick
    test_usage_limits.py's fake uses)."""

    def __init__(self, store):
        self.store = store
        self._filters = {}
        self._pending_update = None

    def select(self, *_args):
        return self

    def eq(self, col, value):
        self._filters[col] = value
        if self._pending_update is not None:
            for row in self.store.values():
                if row.get(col) == value:
                    row.update(self._pending_update)
            self._pending_update = None
        return self

    def order(self, *_args, **_kwargs):
        return self

    def update(self, values):
        self._pending_update = values
        return self

    def insert(self, values):
        key = values.get("id") or values.get("user_id")
        self.store[key] = values
        return self

    def execute(self):
        rows = [
            row for row in self.store.values()
            if all(row.get(k) == v for k, v in self._filters.items())
        ]
        return _FakeResult(rows)


class _FakeClient:
    def __init__(self, subs_store=None, payments_store=None):
        self.subs_store = subs_store if subs_store is not None else {}
        self.payments_store = payments_store if payments_store is not None else {}

    def table(self, name):
        if name == "subscriptions":
            return _FakeTable(self.subs_store)
        if name == "payment_submissions":
            return _FakeTable(self.payments_store)
        raise AssertionError(f"unexpected table: {name}")


@contextmanager
def _patched(subs_store=None, payments_store=None):
    client = _FakeClient(subs_store, payments_store)
    with patch("payments.get_client", return_value=client):
        yield client


class TestGetSubscription:
    def test_no_row_defaults_to_free(self):
        with _patched():
            result = pm.get_subscription("user-1")
        assert result == {"tier": "free", "status": "active", "expires_at": None}

    def test_active_paid_tier_is_returned(self):
        future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        subs = {"user-1": {"user_id": "user-1", "tier": "pro", "status": "active", "expires_at": future}}
        with _patched(subs_store=subs):
            result = pm.get_subscription("user-1")
        assert result["tier"] == "pro"
        assert result["status"] == "active"

    def test_expired_subscription_reports_and_persists_as_free(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        subs = {"user-1": {"user_id": "user-1", "tier": "pro", "status": "active", "expires_at": past}}
        with _patched(subs_store=subs) as client:
            result = pm.get_subscription("user-1")
        # Reported back as free (expired paid tier grants no paid limit)...
        assert result["tier"] == "free"
        assert result["status"] == "expired"
        # ...and persisted, not just returned — the next call shouldn't
        # need to re-derive this from a stale expires_at every time.
        assert client.subs_store["user-1"]["status"] == "expired"

    def test_table_missing_falls_back_to_free_without_crashing(self):
        # Simulates add_subscriptions.sql not having been run yet in
        # this deployment — same MIGRATION SAFETY tolerance as
        # usage_limits.py's own fallback.
        class _BrokenClient:
            def table(self, _name):
                raise Exception('relation "subscriptions" does not exist')

        with patch("payments.get_client", return_value=_BrokenClient()):
            result = pm.get_subscription("user-1")
        assert result == {"tier": "free", "status": "active", "expires_at": None}


class TestEffectiveLimit:
    def test_free_user_gets_free_tier_limit(self):
        with _patched():
            assert pm.effective_limit("user-1") == pm.TIER_LIMITS["free"]

    def test_active_pro_user_gets_pro_limit(self):
        future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        subs = {"user-1": {"user_id": "user-1", "tier": "pro", "status": "active", "expires_at": future}}
        with _patched(subs_store=subs):
            assert pm.effective_limit("user-1") == pm.TIER_LIMITS["pro"]


class TestSubmitPayment:
    def test_rejects_unknown_tier(self):
        with _patched(), pytest.raises(ValueError):
            pm.submit_payment("user-1", "a@b.com", "ultra", "monthly", "jazzcash", "TXN123")

    def test_rejects_blank_reference(self):
        with _patched(), pytest.raises(ValueError):
            pm.submit_payment("user-1", "a@b.com", "pro", "monthly", "jazzcash", "   ")

    def test_records_pending_submission_with_correct_amount(self):
        with _patched() as client:
            row = pm.submit_payment("user-1", "a@b.com", "pro", "monthly", "jazzcash", "TXN123")
        assert row["status"] == "pending"
        assert row["amount_pkr"] == pm.TIER_PRICING_PKR["pro"]["monthly"]
        assert row["transaction_reference"] == "TXN123"
        assert len(client.payments_store) == 1


class TestApprovePayment:
    def test_approve_activates_subscription_with_correct_expiry(self):
        payments_store = {
            "sub-1": {
                "id": "sub-1", "user_id": "user-1", "user_email": "a@b.com",
                "tier_requested": "pro", "billing_period": "monthly",
                "amount_pkr": pm.TIER_PRICING_PKR["pro"]["monthly"],
                "payment_method": "jazzcash", "transaction_reference": "TXN123",
                "status": "pending",
            }
        }
        with _patched(payments_store=payments_store) as client:
            result = pm.approve_payment("sub-1", "admin@apexcoach.ai")

        assert result["tier"] == "pro"
        assert client.payments_store["sub-1"]["status"] == "approved"
        assert client.payments_store["sub-1"]["reviewed_by"] == "admin@apexcoach.ai"

        new_sub = client.subs_store["user-1"]
        assert new_sub["tier"] == "pro"
        assert new_sub["status"] == "active"
        expires = datetime.fromisoformat(new_sub["expires_at"])
        # ~30 days out (monthly), starting from approval time, not
        # submission time — allow a wide tolerance, this just guards
        # against a gross unit error (days vs seconds etc.), not timing.
        assert timedelta(days=28) < (expires - datetime.now(timezone.utc)) < timedelta(days=32)

    def test_approve_unknown_submission_raises(self):
        with _patched(), pytest.raises(ValueError):
            pm.approve_payment("does-not-exist", "admin@apexcoach.ai")


class TestRejectPayment:
    def test_reject_marks_status_without_touching_subscription(self):
        payments_store = {
            "sub-1": {
                "id": "sub-1", "user_id": "user-1", "user_email": "a@b.com",
                "tier_requested": "pro", "billing_period": "monthly",
                "status": "pending",
            }
        }
        with _patched(payments_store=payments_store) as client:
            pm.reject_payment("sub-1", "admin@apexcoach.ai", notes="reference not found")
        assert client.payments_store["sub-1"]["status"] == "rejected"
        assert client.payments_store["sub-1"]["admin_notes"] == "reference not found"
        assert "user-1" not in client.subs_store
