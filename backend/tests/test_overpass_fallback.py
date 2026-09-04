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


def test_429_honors_retry_after(monkeypatch):
    sleeps = []

    class _Resp429:
        status_code = 429
        headers = {"Retry-After": "17"}

    def fake_post(endpoint, **kwargs):
        err = requests.exceptions.HTTPError("429 Too Many Requests")
        err.response = _Resp429()
        raise err

    monkeypatch.setattr(roads.requests, "post", fake_post)
    monkeypatch.setattr(roads.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError):
        roads._post_overpass("q", max_retries=2)

    # One sleep per endpoint (between its two attempts), at the server's pace.
    assert sleeps == [17] * len(constants.OVERPASS_ENDPOINTS)


def test_water_fetch_is_a_single_overpass_request(monkeypatch):
    from core import water

    calls = []

    def fake_post_overpass(query, **kwargs):
        calls.append(query)
        return {
            "elements": [
                {
                    "type": "way",
                    "tags": {"natural": "water", "water": "lake"},
                    "geometry": [
                        {"lon": 0.0, "lat": 0.0},
                        {"lon": 1.0, "lat": 0.0},
                        {"lon": 1.0, "lat": 1.0},
                        {"lon": 0.0, "lat": 0.0},
                    ],
                },
                {
                    "type": "way",
                    "tags": {"waterway": "stream"},
                    "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 2.0, "lat": 2.0}],
                },
            ]
        }

    monkeypatch.setattr(water, "_post_overpass", fake_post_overpass)

    from shapely.geometry import box

    lakes, streams = water.fetch_water_overpass(box(0, 0, 1, 1))
    assert len(calls) == 1  # lakes + streams share one request (F-26)
    assert len(lakes) == 1 and len(streams) == 1
    assert lakes[0].area > 0
    wclass, line = streams[0]  # (waterway_class, geometry) since F-28
    assert wclass == "stream" and line.length > 0
    # Default: no name filter in the waterway block.
    assert '["name"]' not in calls[0]

    # F-31: named_streams_only filters in the query itself, waterways only.
    water.fetch_water_overpass(box(0, 0, 1, 1), named_streams_only=True)
    assert '"waterway"' in calls[1] and '["name"]' in calls[1]
    lake_block = [ln for ln in calls[1].splitlines() if '"natural"="water"' in ln]
    assert lake_block and all('["name"]' not in ln for ln in lake_block)
