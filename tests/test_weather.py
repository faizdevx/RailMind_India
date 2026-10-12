from datetime import datetime
from unittest.mock import MagicMock

import pytest

from railmind.analysis import weather_causes
from railmind.weather import openmeteo as om
from railmind.weather import service as wxs


def test_conditions_flags():
    c = om.conditions(temp=10, rh=98, dew=9.5, precip=0, gust=5, code=0, vis=None)
    assert c["fog"] and "proxy" in c["fog_basis"]
    c = om.conditions(temp=25, rh=60, dew=15, precip=9.0, gust=60, code=65, vis=8000)
    assert c["rain"] == "heavy" and c["strong_wind"] and not c["fog"]
    c = om.conditions(temp=44, rh=20, dew=5, precip=0, gust=10, code=0, vis=20000)
    assert c["heat"] and om.severity(c) == 1
    c = om.conditions(None, None, None, None, None, None, None)      # None-safe
    assert c["rain"] == "none" and not c["fog"]
    assert om.conditions(10, 90, 8, 0, 5, 45, None)["fog"]           # WMO fog code


def _resp(js, code=200):
    r = MagicMock(); r.status_code = code; r.json.return_value = js; r.text = "x"
    return r


def _payload(n):
    return [{"hourly": {"time": ["2026-01-10T06:00"], "temperature_2m": [8], "relative_humidity_2m": [97],
                        "dew_point_2m": [7.5], "precipitation": [0.0], "wind_gusts_10m": [4],
                        "weather_code": [0], "visibility": [300]}} for _ in range(n)]


def test_client_batches_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(om, "CACHE", tmp_path)
    s = MagicMock(); s.get.return_value = _resp(_payload(2))
    c = om.OpenMeteo(session=s, pause=0)
    h = c.hourly([(28.6, 77.2), (19.0, 72.8)], "2026-01-10", "2026-01-10")
    assert len(h) == 2 and s.get.call_count == 1
    c.hourly([(28.6, 77.2), (19.0, 72.8)], "2026-01-10", "2026-01-10")
    assert s.get.call_count == 1                                   # served from disk cache


def test_client_http_error_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(om, "CACHE", tmp_path)
    s = MagicMock(); s.get.return_value = _resp({}, 500)
    with pytest.raises(om.WeatherUnavailable):
        om.OpenMeteo(session=s, pause=0).hourly([(1.0, 2.0)], "2026-01-10", "2026-01-10")


def test_route_weather_aligns_and_flags_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(om, "CACHE", tmp_path)
    monkeypatch.setitem(wxs._state, "coords", {"AAA": (10.0, 20.0)})
    s = MagicMock(); s.get.return_value = _resp(_payload(1))
    out = wxs.route_weather([{"code": "AAA", "time": "2026-01-10 06:20"}, {"code": "ZZZ", "time": "2026-01-10 07:00"}],
                            datetime(2026, 1, 11), client=om.OpenMeteo(session=s, pause=0))
    assert out[0]["fog"] and out[1] is None                         # unknown station -> None, not invented


def test_adjustments_zero_without_model(monkeypatch):
    monkeypatch.setitem(wxs._state, "model", None)
    assert wxs.adjustments([{"code": "A", "sched_run": 30}, {"code": "B"}], [{"time": "2026-01-10T06:00"}, None],
                           [0, 0], 1) == [0.0]


def _wb(cond):
    return {"rows": [{"code": "A", "time": "2026-01-10T06:00", "phase": "passed", "cond": cond,
                      "severity": om.severity(cond) if cond else None, "text": "x"}],
            "adjust_min": [], "ahead_codes": []}


def test_weather_cause_benign_does_not_blame_weather():
    obs = {"sections": [{"from": "A", "to": "B", "delta": 12}]}
    benign = om.conditions(25, 50, 10, 0, 5, 0, 20000)
    out = weather_causes(_wb(benign), obs, None)
    assert out and "not supported" in out[0]["text"]


def test_weather_cause_adverse_is_context_unless_learned():
    obs = {"sections": [{"from": "A", "to": "B", "delta": 12}]}
    fog = om.conditions(8, 98, 7.8, 0, 3, 45, 200)
    out = weather_causes(_wb(fog), obs, {"effects_by_condition": {}})
    assert out[0]["label"] == "CONTEXT" and "does not prove" in out[0]["text"]
    eff = {"effects_by_condition": {"fog": {"significant": True, "mean_extra_delay_min": 3.0, "ci95": [2.0, 4.0],
                                            "n": 5000}}}
    out = weather_causes(_wb(fog), obs, eff)
    assert out[0]["label"] == "LEARNED" and "+3.0" in out[0]["text"]
