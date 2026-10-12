"""Live-status provider adapters. No data is ever invented here."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import requests

from railmind.config import Config


class ProviderUnavailable(Exception):
    """Provider not configured / not reachable / no authorised access."""

    def __init__(self, reason: str, code: str = "unavailable", http: int = 503):
        super().__init__(reason)
        self.reason, self.code, self.http = reason, code, http


@dataclass
class StationStop:
    seq: Optional[int] = None
    passed: Optional[bool] = None
    provenance: Optional[str] = None      # provider flag, e.g. "predicted" (estimate, not confirmed)
    name: Optional[str] = None
    code: Optional[str] = None
    sched_arr: Optional[str] = None
    actual_arr: Optional[str] = None
    sched_dep: Optional[str] = None
    actual_dep: Optional[str] = None
    delay_min: Optional[int] = None


@dataclass
class TrainStatus:
    train_no: str
    provider: str
    retrieved_utc: str
    train_name: Optional[str] = None                 # provider-reported
    current_station: Optional[str] = None            # provider-reported
    current_station_code: Optional[str] = None
    previous_station: Optional[str] = None
    next_station: Optional[str] = None
    delay_min: Optional[int] = None                  # provider-reported
    last_update_text: Optional[str] = None           # provider's own timestamp text, if any
    stops: list = field(default_factory=list)
    raw: Optional[dict] = None                       # untouched provider JSON
    stops_meta: Optional[dict] = None
    missing_fields: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _first(d: dict, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, "", "NA", "N/A"):
            return d[k]
    return None


def _int(v):
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


class RateLimiter:
    def __init__(self, per_min: int, per_day: int):
        self.per_min, self.per_day = per_min, per_day
        self.calls: list = []
        self.lock = threading.Lock()

    def acquire(self) -> None:
        now = time.time()
        with self.lock:
            self.calls = [c for c in self.calls if now - c < 86400]
            if len([c for c in self.calls if now - c < 60]) >= self.per_min:
                raise ProviderUnavailable("Local per-minute provider budget reached; retry shortly.",
                                          "rate_limited_local", 429)
            if len(self.calls) >= self.per_day:
                raise ProviderUnavailable("Local daily provider budget reached "
                                          "(RAILMIND_MAX_CALLS_PER_DAY).", "quota_local", 429)
            self.calls.append(now)

    def usage(self) -> dict:
        now = time.time()
        with self.lock:
            return {"last_minute": len([c for c in self.calls if now - c < 60]),
                    "last_24h": len([c for c in self.calls if now - c < 86400]),
                    "max_per_minute": self.per_min, "max_per_day": self.per_day}


class BaseProvider:
    name = "none"
    verified = False

    def configured(self) -> tuple:
        return False, "No provider configured. Set RAILMIND_PROVIDER and credentials in .env."

    def fetch(self, train_no: str) -> TrainStatus:
        ok, why = self.configured()
        raise ProviderUnavailable(why, "not_configured")

    def describe(self) -> dict:
        ok, why = self.configured()
        return {"name": self.name, "configured": ok, "detail": why,
                "verified_against_live_responses": self.verified}


class NoProvider(BaseProvider):
    pass


class RapidApiIrctcProvider(BaseProvider):
    """Adapter for a RapidAPI-hosted IRCTC status API (default host irctc1.p.rapidapi.com).

    UNVERIFIED: written from public descriptions only. Endpoint path and response
    keys are configurable and parsed defensively; the untouched provider JSON is
    always returned in `raw`. Fields that cannot be found are listed in
    `missing_fields` rather than guessed.
    """
    name = "rapidapi_irctc"
    verified = False

    def __init__(self, cfg=Config, session=None):
        self.cfg = cfg
        self.http = session or requests.Session()
        self.limiter = RateLimiter(cfg.MAX_CALLS_PER_MINUTE, cfg.MAX_CALLS_PER_DAY)

    def configured(self):
        if not self.cfg.PROVIDER_API_KEY:
            return False, ("RAILMIND_PROVIDER_API_KEY is empty. An authorised API key for the "
                           "configured provider is required.")
        return True, f"Using {self.cfg.PROVIDER_HOST}{self.cfg.PROVIDER_STATUS_PATH}"

    def fetch(self, train_no: str) -> TrainStatus:
        ok, why = self.configured()
        if not ok:
            raise ProviderUnavailable(why, "not_configured")
        self.limiter.acquire()
        url = f"https://{self.cfg.PROVIDER_HOST}{self.cfg.PROVIDER_STATUS_PATH}"
        headers = {"x-rapidapi-key": self.cfg.PROVIDER_API_KEY,
                   "x-rapidapi-host": self.cfg.PROVIDER_HOST}
        try:
            r = self.http.get(url, params={"trainNo": train_no, "startDay": "1"},
                              headers=headers, timeout=self.cfg.HTTP_TIMEOUT)
        except requests.RequestException as e:
            raise ProviderUnavailable(f"Provider request failed: {type(e).__name__}", "network", 502)
        if r.status_code == 429:
            raise ProviderUnavailable("Provider rate limit / quota exceeded (HTTP 429).", "rate_limited", 429)
        if r.status_code in (401, 403):
            raise ProviderUnavailable(f"Provider rejected credentials (HTTP {r.status_code}).", "auth", 502)
        if r.status_code >= 400:
            raise ProviderUnavailable(f"Provider returned HTTP {r.status_code}.", "provider_error", 502)
        try:
            body = r.json()
        except ValueError:
            raise ProviderUnavailable("Provider returned non-JSON response.", "bad_response", 502)
        return self.parse(train_no, body)

    def parse(self, train_no: str, body) -> TrainStatus:
        now = datetime.now(timezone.utc).isoformat()
        data = body.get("data", body) if isinstance(body, dict) else {}
        if not isinstance(data, dict):
            data = {}
        st = TrainStatus(train_no=train_no, provider=self.name, retrieved_utc=now, raw=body)
        st.train_name = _first(data, "train_name", "trainName", "name")
        st.current_station = _first(data, "current_station_name", "currentStationName", "current_station")
        st.current_station_code = _first(data, "current_station_code", "currentStationCode")
        st.previous_station = _first(data, "previous_station_name", "prev_station_name")
        st.next_station = _first(data, "next_station_name", "upcoming_station_name")
        st.delay_min = _int(_first(data, "delay", "delay_minutes", "current_delay"))
        st.last_update_text = _first(data, "update_time", "last_updated", "status_as_of", "ta")
        rows = _first(data, "upcoming_stations", "route", "stations", "previous_stations") or []
        if isinstance(rows, list):
            for s in rows:
                if not isinstance(s, dict):
                    continue
                st.stops.append(StationStop(
                    name=_first(s, "station_name", "stationName", "name"),
                    code=_first(s, "station_code", "stationCode", "code"),
                    sched_arr=_first(s, "sta", "scheduled_arrival", "arrival_time", "std_arrival"),
                    actual_arr=_first(s, "eta", "actual_arrival"),
                    sched_dep=_first(s, "std", "scheduled_departure", "departure_time"),
                    actual_dep=_first(s, "etd", "actual_departure"),
                    delay_min=_int(_first(s, "delay", "arrival_delay"))))
        for k in ("train_name", "current_station", "delay_min", "previous_station", "next_station"):
            if getattr(st, k) is None:
                st.missing_fields.append(k)
        if not st.stops:
            st.missing_fields.append("route/stops")
        return st


class RailRadarProvider(BaseProvider):
    """RailRadar live running status (https://railradar.in/docs/live-train-status).

    Endpoint, auth and top-level fields follow the vendor's public docs
    (fetched 2026-10-12). NOT verified against a live response: no key was
    available. Free sandbox reportedly 100 requests/month - budget accordingly.
    Per-stop schedule/actual key names are not documented in the page text we
    could read, so they are parsed defensively and `raw` is always kept.
    """
    name = "railradar"
    verified = True      # parsed against one real response (train 12301, 2026-10-12); see docs

    def __init__(self, cfg=Config, session=None):
        self.cfg = cfg
        self.http = session or requests.Session()
        self.limiter = RateLimiter(cfg.MAX_CALLS_PER_MINUTE, cfg.MAX_CALLS_PER_DAY)
        self.host = "api.railradar.in" if cfg.PROVIDER_HOST.startswith("irctc1") else cfg.PROVIDER_HOST
        self.path = "/v1/trains/{number}/live"

    def configured(self):
        if not self.cfg.PROVIDER_API_KEY:
            return False, ("RAILMIND_PROVIDER_API_KEY is empty. Create a RailRadar account/API key "
                           "(railradar.in) and set it in .env.")
        return True, f"Using https://{self.host}{self.path}"

    def fetch(self, train_no: str) -> TrainStatus:
        ok, why = self.configured()
        if not ok:
            raise ProviderUnavailable(why, "not_configured")
        self.limiter.acquire()
        url = f"https://{self.host}{self.path.format(number=train_no)}"
        try:
            r = self.http.get(url, headers={"Authorization": f"Bearer {self.cfg.PROVIDER_API_KEY}"},
                              params={"haltsOnly": "true"}, timeout=self.cfg.HTTP_TIMEOUT)
        except requests.RequestException as e:
            raise ProviderUnavailable(f"Provider request failed: {type(e).__name__}", "network", 502)
        if r.status_code == 429:
            raise ProviderUnavailable("Provider rate limit / quota exceeded (HTTP 429).", "rate_limited", 429)
        if r.status_code in (401, 403):
            raise ProviderUnavailable(f"Provider rejected credentials (HTTP {r.status_code}).", "auth", 502)
        if r.status_code == 404:
            raise ProviderUnavailable("Train not found at provider (HTTP 404).", "not_found", 404)
        if r.status_code >= 400:
            raise ProviderUnavailable(f"Provider returned HTTP {r.status_code}.", "provider_error", 502)
        try:
            body = r.json()
        except ValueError:
            raise ProviderUnavailable("Provider returned non-JSON response.", "bad_response", 502)
        return self.parse(train_no, body)

    def parse(self, train_no: str, body) -> TrainStatus:
        now = datetime.now(timezone.utc).isoformat()
        d = body.get("data", body) if isinstance(body, dict) else {}
        if not isinstance(d, dict):
            d = {}
        loc = d.get("currentLocation") or {}
        st = TrainStatus(train_no=train_no, provider=self.name, retrieved_utc=now, raw=body)
        st.train_name = _first(d, "trainName")
        st.current_station_code = _first(loc, "stationCode")
        st.current_station = st.current_station_code
        st.delay_min = _int(_first(d, "delayMinutes"))
        st.last_update_text = _first(d, "lastUpdatedAt")
        prev, nxt = d.get("previousHalt") or {}, d.get("nextHalt") or {}
        st.previous_station = _first(prev, "stationName", "stationCode")
        st.next_station = _first(nxt, "stationName", "stationCode")
        for r_ in d.get("route") or []:
            if not isinstance(r_, dict):
                continue
            st.stops.append(StationStop(
                seq=_int(_first(r_, "sequence")), name=_first(r_, "stationName"),
                code=_first(r_, "stationCode"),
                sched_arr=_first(r_, "scheduledArrival", "scheduledArrivalTime", "sta"),
                actual_arr=_first(r_, "actualArrival", "actualArrivalTime", "eta"),
                sched_dep=_first(r_, "scheduledDeparture", "scheduledDepartureTime", "std"),
                actual_dep=_first(r_, "actualDeparture", "actualDepartureTime", "etd"),
                delay_min=_int(_first(r_, "delayArrival", "delayDeparture")),
                passed=(r_.get("status") in ("departed", "at-station", "arrived")) if r_.get("status") else None,
                provenance=r_.get("provenance")))
        info = d.get("train") or {}
        st.stops_meta = {"is_live": d.get("isLive"), "status": d.get("status"),
                         "tracking_mode": d.get("trackingMode"),
                         "location_status": loc.get("status"),
                         "current_station_name": loc.get("stationName"),
                         "category": info.get("category"), "type": info.get("type"),
                         "segment_progress": loc.get("segmentProgress"),
                         "is_actual_position": loc.get("isActualPosition")}
        for k in ("train_name", "current_station", "delay_min"):
            if getattr(st, k) is None:
                st.missing_fields.append(k)
        if not st.stops:
            st.missing_fields.append("route/stops")
        return st


def build_provider(cfg=Config) -> BaseProvider:
    if cfg.PROVIDER == "railradar":
        return RailRadarProvider(cfg)
    if cfg.PROVIDER in ("rapidapi_irctc", "rapidapi"):
        return RapidApiIrctcProvider(cfg)
    return NoProvider()


TRAIN_NO_RE = re.compile(r"^\d{5}$")


def valid_train_no(s: str) -> bool:
    return bool(TRAIN_NO_RE.match(s or ""))
