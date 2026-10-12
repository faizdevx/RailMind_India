"""Service layer: cached/rate-limited status fetch and simulation comparison."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from railmind.config import Config
from railmind.providers.base import (BaseProvider, ProviderUnavailable, TrainStatus, build_provider)
from railmind.rl.train import META_PATH, load_model

log = logging.getLogger("railmind")


class StatusService:
    def __init__(self, provider: BaseProvider | None = None, cfg=Config):
        self.provider = provider or build_provider(cfg)
        self.cfg = cfg
        self.cache: dict = {}
        self.last_error: dict = {}
        self.lock = threading.Lock()

    def get(self, train_no: str, force: bool = False) -> dict:
        now = time.time()
        with self.lock:
            hit = self.cache.get(train_no)
        if hit and now - hit["fetched_at"] < self.cfg.MIN_SECONDS_BETWEEN_CALLS_PER_TRAIN:
            return self._decorate(hit, from_cache=True)
        try:
            st: TrainStatus = self.provider.fetch(train_no)
        except ProviderUnavailable as e:
            log.warning("provider failure train=%s code=%s", train_no, e.code)
            self.last_error[train_no] = {"code": e.code, "reason": e.reason}
            if hit:       # serve last good value, clearly flagged
                d = self._decorate(hit, from_cache=True)
                d["warnings"].append(f"Refresh failed ({e.code}); showing last successful retrieval.")
                return d
            raise
        entry = {"status": st.as_dict(), "fetched_at": time.time()}
        with self.lock:
            self.cache[train_no] = entry
        return self._decorate(entry, from_cache=False)

    def _decorate(self, entry: dict, from_cache: bool) -> dict:
        age = time.time() - entry["fetched_at"]
        warns = []
        stale = age > self.cfg.STALE_AFTER_SECONDS
        if stale:
            warns.append(f"STALE: last successful retrieval {int(age // 60)} min ago.")
        s = entry["status"]
        if s["missing_fields"]:
            warns.append("Provider did not supply: " + ", ".join(s["missing_fields"]))
        return {"status": s, "age_seconds": round(age), "stale": stale, "from_cache": from_cache,
                "warnings": warns,
                "provenance": {"observed": ["train_name", "current_station", "previous_station",
                                            "next_station", "delay_min", "stops"],
                               "calculated": ["age_seconds", "stale"]}}


def model_info() -> dict:
    import json
    if META_PATH.exists():
        try:
            return {"trained": True, **json.loads(META_PATH.read_text())}
        except ValueError:
            pass
    return {"trained": False}


_cache: dict = {}


def get_model():
    if "m" not in _cache:
        _cache["m"] = load_model()
    return _cache["m"]


def get_knowledge():
    if "kn" not in _cache:
        from railmind.data import history
        _cache["kn"] = history.load()
    return _cache["kn"]


def analyze_status(status: dict, n_mc: int = 15, single_line: bool = False, extra_delay: int = 0,
                   seed: int = 0, use_weather: bool = True) -> dict:
    from railmind.analysis import analyze
    kn = get_knowledge()
    if kn is None:
        raise ProviderUnavailable("Historical knowledge not built. Run scripts/download_data.py and "
                                  "python -m railmind.data.history", "no_knowledge", 503)
    out = analyze(kn, status, model=get_model(), n_mc=n_mc, single_line=single_line,
                  extra_delay=extra_delay, seed=seed, use_weather=use_weather)
    out["id"] = uuid.uuid4().hex[:12]
    out["created_utc"] = datetime.now(timezone.utc).isoformat()
    out["model"] = model_info()
    out["knowledge"] = kn.describe()
    return out


class SimStore:
    def __init__(self):
        self.items: dict = {}
        self.lock = threading.Lock()

    def put(self, sim: dict):
        with self.lock:
            self.items[sim["id"]] = sim
            while len(self.items) > 50:
                self.items.pop(next(iter(self.items)))

    def get(self, sid: str):
        return self.items.get(sid)


class Jobs:
    """Single background job runner for training / evaluation."""

    def __init__(self):
        self.state: dict = {"training": {"status": "idle"}, "evaluation": {"status": "idle"}}
        self.lock = threading.Lock()

    def start(self, kind: str, fn, *a):
        with self.lock:
            if self.state[kind]["status"] == "running":
                return False
            self.state[kind] = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}

        def run():
            try:
                res = fn(*a)
                self.state[kind] = {"status": "completed", "result": res,
                                    "finished_utc": datetime.now(timezone.utc).isoformat()}
                _cache.clear()
            except Exception as e:      # noqa: BLE001
                log.exception("%s failed", kind)
                self.state[kind] = {"status": "failed", "error": str(e)}
        threading.Thread(target=run, daemon=True).start()
        return True
