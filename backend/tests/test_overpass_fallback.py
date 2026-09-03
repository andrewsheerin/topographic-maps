"""Regression tests for Overpass endpoint fallback (F-24). The failure that
motivated this: a decommissioned endpoint (dead DNS) and a hanging endpoint
were each retried 3x at full timeout, and the error message only showed the
last endpoint's failure. No network calls — requests.post is monkeypatched."""

import pytest
import requests

import constants
from core import roads


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_connection_error_fails_fast_to_next_endpoint(monkeypatch):
    calls = []

    def fake_post(endpoint, **kwargs):
        calls.append(endpoint)
        if endpoint == constants.OVERPASS_ENDPOINTS[0]:
            raise requests.exceptions.ConnectionError("getaddrinfo failed")
        return _FakeResponse({"elements": []})

    monkeypatch.setattr(roads.requests, "post", fake_post)
    monkeypatch.setattr(roads.time, "sleep", lambda s: pytest.fail("slept"))

    assert roads._post_overpass("q") == {"elements": []}
    # Dead endpoint tried exactly once — no retries, no sleeps.
    assert calls == [constants.OVERPASS_ENDPOINTS[0], constants.OVERPASS_ENDPOINTS[1]]


def test_http_errors_still_retry_with_backoff(monkeypatch):
    calls = []
    sleeps = []

    def fake_post(endpoint, **kwargs):
        calls.append(endpoint)
        raise requests.exceptions.HTTPError("429 Too Many Requests")

    monkeypatch.setattr(roads.requests, "post", fake_post)
    monkeypatch.setattr(roads.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError) as exc:
        roads._post_overpass("q", max_retries=3)

    # Load-type errors retry per endpoint (3 attempts each), with backoff
    # between attempts but no pointless sleep after an endpoint's last try.
    assert len(calls) == 3 * len(constants.OVERPASS_ENDPOINTS)
    assert sleeps == [1, 2] * len(constants.OVERPASS_ENDPOINTS)
    # Every endpoint's failure is reported, not just the last.
    for endpoint in constants.OVERPASS_ENDPOINTS:
        assert endpoint in str(exc.value)


def test_no_dead_endpoints_configured():
    # overpass.nchc.org.tw was decommissioned (DNS gone) — keep it out.
    assert all("nchc" not in ep for ep in constants.OVERPASS_ENDPOINTS)
