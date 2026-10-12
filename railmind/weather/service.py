"""Weather context + learned weather adjustment for one train's route.

Used three ways:
  1. context  - conditions at each stop at its (expected) time, flagged as adverse or benign
  2. evidence - passed sections with delay gain are checked against the weather there
  3. forecast - learned residual model adds expected weather-related delay change per upcoming
                section, and the same minutes feed the simulation run times
The residual model is used only if it beat the weather-free baseline on held-out dates.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from railmind.weather.learn import EFFECTS, FEATURES, MODEL, load_coords
from railmind.weather.openmeteo import ATTRIBUTION, OpenMeteo, WeatherUnavailable, at_time, severity

_state: dict = {}


def coords() -> dict:
    if "coords" not in _state:
        try:
            _state["coords"] = load_coords()
        except FileNotFoundError:
            _state["coords"] = {}
    return _state["coords"]


def effects() -> Optional[dict]:
    if "effects" not in _state:
        _state["effects"] = json.loads(EFFECTS.read_text()) if EFFECTS.exists() else None
    return _state["effects"]


def residual_model():
    if "model" not in _state:
        if MODEL.exists():
            with open(MODEL, "rb") as fh:
                _state["model"] = pickle.load(fh)
        else:
            _state["model"] = None
    m = _state["model"]
    return m["model"] if m and m.get("accepted") else None


def _iso(x) -> Optional[str]:
    if not x:
        return None
    try:
        s = str(x).replace(" ", "T")
        return datetime.fromisoformat(s[:19]).strftime("%Y-%m-%dT%H:%M")
    except ValueError:
        return None


def route_weather(items: list, today: datetime, client: Optional[OpenMeteo] = None) -> list:
    """items: [{'code', 'time'}] -> aligned list of condition dicts (or None)."""
    cs = coords()
    client = client or OpenMeteo(pause=0.2)
    out: list = [None] * len(items)
    want: dict = {}
    for i, it in enumerate(items):
        t = _iso(it.get("time"))
        if t and it["code"] in cs:
            want.setdefault(it["code"], []).append((i, t))
    if not want:
        return out
    archive_codes, fc_codes = {}, {}
    for code, lst in want.items():
        for i, t in lst:
            d = datetime.fromisoformat(t[:10])
            bucket = fc_codes if (today.date() - d.date()).days <= 2 else archive_codes
            bucket.setdefault(code, []).append((i, t))
    for forecast, group in ((False, archive_codes), (True, fc_codes)):
        if not group:
            continue
        codes = list(group)
        dates = [t[:10] for lst in group.values() for _, t in lst]
        try:
            hs = client.hourly([cs[c] for c in codes], min(dates), max(dates), forecast=forecast, batch=25)
        except WeatherUnavailable:
            continue
        for c, h in zip(codes, hs):
            for i, t in group[c]:
                out[i] = at_time(h, t) if h else None
    return out


def adjustments(stops: list, weathers: list, delays_at_stop: list, month: int) -> list:
    """Expected weather-related delay change (min) for each section stops[k] -> stops[k+1].
    stops: [{'code','sched_run'}]; weathers aligned with stops; model-based, clipped."""
    model = residual_model()
    out = [0.0] * max(0, len(stops) - 1)
    if model is None:
        return out
    X, idx = [], []
    for k in range(len(stops) - 1):
        w = weathers[k]
        if not w:
            continue
        dew_gap = None if w.get("temp_c") is None or w.get("dew_c") is None else w["temp_c"] - w["dew_c"]
        hour = int(w["time"][11:13]) if w.get("time") else 12
        t, rh = w.get("temp_c"), w.get("rh")
        X.append([np.nan if w.get("precip_mm") is None else w["precip_mm"],
                  np.nan if w.get("gust_kmh") is None else w["gust_kmh"],
                  np.nan if rh is None else rh, np.nan if t is None else t,
                  np.nan if dew_gap is None else dew_gap,
                  hour, month,
                  delays_at_stop[k], stops[k].get("sched_run", 60)])
        idx.append(k)
    if X:
        p = np.clip(model.predict(np.array(X, dtype=float)), -5.0, 15.0)
        for k, v in zip(idx, p):
            out[k] = float(round(v, 1))
    return out


def summary_text(w: dict) -> str:
    bits = []
    if w["rain"] != "none":
        bits.append(f"{w['rain']} rain ({w['precip_mm']:.1f} mm/h)")
    if w["fog"]:
        bits.append("fog (" + (w.get("fog_basis") or "") + ")")
    if w["strong_wind"]:
        bits.append(f"strong gusts ({w['gust_kmh']:.0f} km/h)")
    if w["heat"]:
        bits.append(f"extreme heat ({w['temp_c']:.0f} C)")
    if w["cold"]:
        bits.append(f"cold ({w['temp_c']:.0f} C)")
    return ", ".join(bits) or "benign"


__all__ = ["route_weather", "adjustments", "effects", "summary_text", "severity", "ATTRIBUTION", "FEATURES"]
