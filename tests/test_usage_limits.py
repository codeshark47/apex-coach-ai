"""
tests/test_usage_limits.py

Regression tests for the daily-reset behavior added to usage_limits.py.
Before this, demo_usage was a pure LIFETIME cap with no date concept —
once used_count hit free_limit, that user was capped forever. These
tests pin down that a stale usage_date (yesterday or earlier) always
gets reset to 0 before being read OR incremented, using a fake Supabase
client so no real network/Supabase project is needed to run them.
"""

from contextlib import contextmanager
from datetime import date, timedelta
from unittest.mock import patch

import usage_limits as ul

TODAY = date.today().isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    """Minimal stand-in for supabase-py's table/query builder — just
    enough chained-method surface (select/eq/update/insert/execute) for
    usage_limits.py's exact call patterns."""

    def __init__(self, store, user_id=None):
        self.store = store
        self._user_id = user_id
        self._pending_update = None

    def select(self, *_args):
        return self

    def eq(self, _col, user_id):
        self._user_id = user_id
        if self._pending_update is not None:
            self.store[user_id] = {**self.store.get(user_id, {}), **self._pending_update}
            self._pending_update = None
        return self

    def update(self, values):
        self._pending_update = values
        return self

    def insert(self, values):
        self.store[values["user_id"]] = values
        return self

    def execute(self):
        if self._user_id is not None:
            row = self.store.get(self._user_id)
            return _FakeResult([row] if row else [])
        return _FakeResult([])


class _FakeClient:
    def __init__(self, store):
        self.store = store

    def table(self, _name):
        return _FakeTable(self.store)


@contextmanager
def _patched_client(store):
    """Also patches payments.get_client (2026-08-17) — get_usage() now
    calls payments.effective_limit(), a separate name binding of the
    same underlying profile_store.get_client, which the old single-patch
    version left unmocked. Without this, these tests silently made a
    real network call to the live Supabase project (harmless — it just
    hit a graceful fallback for a table that doesn't exist yet — but not
    a real unit test anymore). An empty store (no subscription rows) is
    fine: payments.get_subscription's own "no rows" path already returns
    the free tier by default."""
    with patch("usage_limits.get_client", return_value=_FakeClient(store)), \
         patch("payments.get_client", return_value=_FakeClient({})):
        yield


class TestDailyReset:
    def test_first_call_creates_row_with_todays_date(self):
        store = {}
        with _patched_client(store):
            result = ul.get_usage("user-1")
        assert result == {"used": 0, "limit": ul.DEFAULT_FREE_LIMIT, "remaining": ul.DEFAULT_FREE_LIMIT}
        assert store["user-1"]["usage_date"] == TODAY

    def test_same_day_repeated_calls_do_not_reset(self):
        store = {"user-1": {"user_id": "user-1", "used_count": 1, "free_limit": 10, "usage_date": TODAY}}
        with _patched_client(store):
            result = ul.get_usage("user-1")
        assert result == {"used": 1, "limit": 10, "remaining": 9}
        assert store["user-1"]["used_count"] == 1

    def test_stale_date_resets_used_count_to_zero(self):
        """The exact bug this migration fixes: a row from yesterday at
        its limit must NOT stay capped today."""
        store = {"user-1": {"user_id": "user-1", "used_count": 10, "free_limit": 10, "usage_date": YESTERDAY}}
        with _patched_client(store):
            result = ul.get_usage("user-1")
        assert result == {"used": 0, "limit": 10, "remaining": 10}
        assert store["user-1"]["used_count"] == 0
        assert store["user-1"]["usage_date"] == TODAY

    def test_record_usage_on_stale_row_resets_then_increments_to_one(self):
        """Not "yesterday's 10 + 1" — a new day's first recorded use must
        land on exactly 1, not 11."""
        store = {"user-1": {"user_id": "user-1", "used_count": 10, "free_limit": 10, "usage_date": YESTERDAY}}
        with _patched_client(store):
            result = ul.record_usage("user-1")
        assert result == {"used": 1, "limit": 10, "remaining": 9}
        assert store["user-1"]["used_count"] == 1
        assert store["user-1"]["usage_date"] == TODAY

    def test_record_usage_same_day_accumulates_normally(self):
        store = {"user-1": {"user_id": "user-1", "used_count": 3, "free_limit": 10, "usage_date": TODAY}}
        with _patched_client(store):
            result = ul.record_usage("user-1")
        assert result == {"used": 4, "limit": 10, "remaining": 6}

    def test_per_user_free_limit_override_is_respected(self):
        """The two demo-coach accounts get free_limit=10 while everyone
        else keeps DEFAULT_FREE_LIMIT — this must be read from the row,
        never overwritten by the default on a same-day or reset path."""
        store = {"coach-a": {"user_id": "coach-a", "used_count": 10, "free_limit": 10, "usage_date": YESTERDAY}}
        with _patched_client(store):
            result = ul.get_usage("coach-a")
        assert result["limit"] == 10

    def test_pre_migration_row_without_usage_date_does_not_crash_or_reset(self):
        """MIGRATION SAFETY: add_daily_usage_reset.sql hasn't necessarily
        been run everywhere yet — a row shaped like the OLD schema (no
        usage_date key at all, not even null) must be read exactly like
        before: no crash, no incorrect reset, existing used_count as-is."""
        store = {"coach-a": {"user_id": "coach-a", "used_count": 1, "free_limit": 2}}
        with _patched_client(store):
            result = ul.get_usage("coach-a")
        assert result == {"used": 1, "limit": 2, "remaining": 1}

    def test_new_user_insert_falls_back_when_usage_date_column_is_missing(self):
        """The realistic case this fallback exists for: a BRAND NEW user
        signs up before add_daily_usage_reset.sql has been run anywhere.
        There's no existing row to read a usage_date from (that's covered
        by test_pre_migration_row_without_usage_date_does_not_crash_or_reset
        above) — this is the INSERT path, which unconditionally tries to
        include usage_date. Must retry the insert without it and still
        create a usable row, not crash the coach's first-ever session."""
        store = {}

        class _NoUsageDateColumnTable(_FakeTable):
            def insert(self, values):
                if "usage_date" in values:
                    raise Exception('column "usage_date" of relation "demo_usage" does not exist')
                return super().insert(values)

        class _NoUsageDateColumnClient(_FakeClient):
            def table(self, _name):
                return _NoUsageDateColumnTable(self.store)

        with patch("usage_limits.get_client", return_value=_NoUsageDateColumnClient(store)), \
             patch("payments.get_client", return_value=_FakeClient({})):
            result = ul.get_usage("new-coach")
        assert result == {"used": 0, "limit": ul.DEFAULT_FREE_LIMIT, "remaining": ul.DEFAULT_FREE_LIMIT}
        assert store["new-coach"]["used_count"] == 0
        assert "usage_date" not in store["new-coach"]
