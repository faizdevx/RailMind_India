"""Open-Meteo client (free, no API key; non-commercial use; data CC-BY 4.0).

Docs/terms checked 2026-10-12: https://open-meteo.com/en/terms
  free tier limits: 600/min, 5 000/h, 10 000/day, 300 000/month.
Attribution: "Weather data by Open-Meteo.com" (CC BY 4.0).
Archive API = reanalysis (past dates); forecast API = model forecast (now/future).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "weather_cache"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
          "wind_gusts_10m", "weather_code", "visibility"]
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"
BATCH = 5


class WeatherUnavailable(Exception):
    pass


def _key(url, params) -> str:
    return hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()


class OpenMeteo:
    def __init__(self, session=None, cache: bool = True, timeout: float = 30.0, pause: float = 10.0):
        self.http = session or requests.Session()
        self.cache = cache
        self.timeout = timeout
        self.pause = pause
        self.calls = 0

    def _get(self, url: str, params: dict):
        if self.cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            f = CACHE / (_key(url, params) + ".json")
            if f.exists():
                return json.loads(f.read_text())
        for attempt in range(6):
            try:
                r = self.http.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                raise WeatherUnavailable(f"weather request failed: {type(e).__name__}")
            if r.status_code == 429:
                time.sleep(65)          # per-minute weighted quota window
                continue
            if r.status_code >= 400:
                raise WeatherUnavailable(f"weather API HTTP {r.status_code}: {r.text[:120]}")
            self.calls += 1
            data = r.json()
            if self.cache:
                f.write_text(json.dumps(data))
            time.sleep(self.pause)
            return data
        raise WeatherUnavailable("weather API rate limited (HTTP 429)")

    def hourly(self, coords: list, start: str, end: str, forecast: bool = False, batch: int = BATCH) -> list:
        """coords: [(lat, lon)]. Returns one dict per location:
        {"time": [...ISO local...], <var>: [...], ...}. Times are Asia/Kolkata."""
        out: list = []
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            params = {"latitude": ",".join(f"{c[0]:.3f}" for c in chunk),
                      "longitude": ",".join(f"{c[1]:.3f}" for c in chunk),
                      "hourly": ",".join(HOURLY), "timezone": "Asia/Kolkata",
                      "start_date": start, "end_date": end}
            data = self._get(FORECAST if forecast else ARCHIVE, params)
            out.extend(data if isinstance(data, list) else [data])
        return [loc.get("hourly", {}) for loc in out]


def conditions(temp, rh, dew, precip, gust, code, vis) -> dict:
    """Operational weather flags from one hourly record. None-safe."""
    def num(x):
        return None if x is None else float(x)
    temp, rh, dew, precip, gust, vis = map(num, (temp, rh, dew, precip, gust, vis))
    fog = False
    why = None
    if vis is not None and vis < 1000:
        fog, why = True, f"visibility {vis:.0f} m"
    elif code in (45, 48):
        fog, why = True, "WMO fog code"
    elif rh is not None and temp is not None and dew is not None and rh >= 96 and temp - dew <= 1.0 \
            and temp <= 20 and (gust is None or gust < 25):
        fog, why = True, "humidity/dew-point proxy (visibility not available)"
    rain = "none"
    if precip is not None:
        rain = "heavy" if precip >= 7.6 else "moderate" if precip >= 2.5 else "light" if precip >= 0.2 else "none"
    return {"rain": rain, "fog": fog, "fog_basis": why,
            "strong_wind": gust is not None and gust >= 50.0,
            "heat": temp is not None and temp >= 42.0,
            "cold": temp is not None and temp <= 3.0,
            "precip_mm": precip, "gust_kmh": gust, "temp_c": temp, "rh": rh, "visibility_m": vis,
            "dew_c": dew}


def severity(c: dict) -> int:
    return (2 if c["rain"] == "heavy" else 1 if c["rain"] == "moderate" else 0) \
        + (2 if c["fog"] else 0) + (1 if c["strong_wind"] else 0) + (1 if c["heat"] else 0)


def at_time(h: dict, iso_local: str) -> Optional[dict]:
    """Conditions at the hour containing iso_local ('YYYY-MM-DDTHH:MM')."""
    t = iso_local[:13] + ":00"
    try:
        i = h["time"].index(t)
    except (KeyError, ValueError):
        return None
    g = lambda k: (h.get(k) or [None] * len(h["time"]))[i]  # noqa: E731
    c = conditions(g("temperature_2m"), g("relative_humidity_2m"), g("dew_point_2m"), g("precipitation"),
                   g("wind_gusts_10m"), g("weather_code"), g("visibility"))
    c["time"] = t
    return c
