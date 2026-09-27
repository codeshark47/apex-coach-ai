"""
Regression test for a real crash (2026-09-27): a transient network
failure (httpx.ConnectError: [Errno 11001] getaddrinfo failed, hit
mid-upload of 47 new clips) took down the whole label_tool.py process
because every Supabase call in that file had zero error handling.

Verifies the fix's actual safety property: _run_query catches a
network-layer failure and returns None instead of raising, WITHOUT
also swallowing a genuine Postgres-level error response (which must
keep raising, since callers like _load_pitch_calibration deliberately
distinguish "not calibrated yet" from "the migration wasn't run").
"""
import httpx
import pytest

import ball_tracking.label_tool as lt


class _FakeQuery:
    def __init__(self, exc):
        self._exc = exc

    def execute(self):
        raise self._exc


def test_network_failure_is_caught_and_returns_none():
    query = _FakeQuery(httpx.ConnectError("[Errno 11001] getaddrinfo failed"))
    result = lt._run_query(query, action="doing a thing")
    assert result is None


def test_read_timeout_is_also_caught():
    query = _FakeQuery(httpx.ReadTimeout("timed out"))
    result = lt._run_query(query, action="doing a thing")
    assert result is None


def test_postgres_level_error_still_raises():
    class FakeAPIError(Exception):
        pass

    query = _FakeQuery(FakeAPIError("column pitch_calibration does not exist"))
    with pytest.raises(FakeAPIError):
        lt._run_query(query, action="doing a thing")


def test_successful_query_passes_through_unchanged():
    class _Result:
        data = [{"ok": True}]

    class _OkQuery:
        def execute(self):
            return _Result()

    result = lt._run_query(_OkQuery(), action="doing a thing")
    assert result.data == [{"ok": True}]
