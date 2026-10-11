"""Service layer: cached/rate-limited status fetch and simulation comparison."""
from __future__ import annotations

import logging
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone

from railmind.config import Config
from railmind.providers.base import (BaseProvider, ProviderUnavailable, TrainStatus, build_provider)
from railmind.rl.env import run_ppo
from railmind.rl.train import META_PATH, load_model
from railmind.sim.corridor import POLICIES, WEIGHTS, perturb, run_policy
from railmind.sim.snapshot import build_scenario

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


_model_cache: dict = {}


def get_model():
    if "m" not in _model_cache:
        _model_cache["m"] = load_model()
    return _model_cache["m"]


def explain_events(c, name: str) -> list:
    """Templated, factual explanations of holds in a finished run (SIMULATED)."""
    out = []
    seen = set()
    for e in c.events:
        out.append({**e, "label": "SIMULATED"})
    return out


def bottleneck(c) -> dict:
    ws = c.wait_by_section
    if not ws or max(ws) == 0:
        return {"label": "SIMULATED", "text": "No queueing observed in this simulated run."}
    s = max(range(len(ws)), key=lambda k: ws[k])
    names = c.sc.station_names
    return {"label": "SIMULATED SCENARIO", "section": s, "wait_minutes": ws[s],
            "text": f"Most train-minutes spent waiting for section {names[s]} - {names[s + 1]} "
                    f"({ws[s]} train-min). Cause within the simulation: single-track block/loop capacity; "
                    "real-world cause NOT established (HYPOTHESIS)."}


def simulate(status: dict, seed: int = 7, load: float = 1.0, extra_delay: int = 0,
             n_runs: int = 20, jitter: int = 2) -> dict:
    sc = build_scenario(status, seed=seed, load=load, extra_delay=extra_delay, jitter=jitter)
    model = get_model()
    policies = dict(POLICIES)
    runs: dict = {}
    for k, p in policies.items():
        runs[k] = run_policy(sc, p)
    if model is not None:
        runs["ppo"] = run_ppo(model, sc)
    # uncertainty: jitter run times over n_runs and compare on identical perturbed scenarios
    saved = {k: [] for k in runs if k != "fcfs"}
    obs_delay = {k: [] for k in runs}
    for r in range(n_runs):
        ps = perturb(sc, 1000 + r, jitter)
        base = run_policy(ps, POLICIES["fcfs"]).summary()["per_train"][0]["delay"]
        obs_delay["fcfs"].append(base)
        for k in saved:
            c = run_policy(ps, POLICIES[k]) if k in POLICIES else run_ppo(model, ps)
            d = c.summary()["per_train"][0]["delay"]
            obs_delay[k].append(d)
            saved[k].append(base - d)

    def ms(v):
        return {"mean": round(statistics.fmean(v), 1), "std": round(statistics.pstdev(v), 1)} if v else None

    out = {"id": uuid.uuid4().hex[:12], "created_utc": datetime.now(timezone.utc).isoformat(),
           "label": "SIMULATED / PREDICTED - NOT REAL RAILWAY CONTROL",
           "snapshot": {"train_no": status["train_no"], "provider": status["provider"],
                        "retrieved_utc": status["retrieved_utc"],
                        "observed_delay_min": status.get("delay_min"),
                        "current_station": status.get("current_station")},
           "assumptions": sc.assumptions, "station_names": sc.station_names,
           "scenario": {"seed": seed, "load": load, "extra_delay": extra_delay,
                        "uncertainty_runs": n_runs, "jitter_min": jitter},
           "trains": [{"id": t.id, "name": t.name, "source": t.source,
                       "class": ["freight", "passenger", "express"][t.prio],
                       "direction": t.direction} for t in sc.trains],
           "policies": {}, "model": model_info()}
    for k, c in runs.items():
        s = c.summary()
        out["policies"][k] = {"summary": s, "tracks": c.tracks, "events": c.events,
                              "bottleneck": bottleneck(c),
                              "observed_train_delay_mean_std": ms(obs_delay[k]),
                              "minutes_saved_vs_fcfs_mean_std": ms(saved.get(k, [])) if k != "fcfs" else None}
    if "ppo" in runs:
        a = out["policies"]["fcfs"]["summary"]["per_train"]
        b = out["policies"]["ppo"]["summary"]["per_train"]
        out["ppo_vs_fcfs"] = {
            "helped": [x["name"] for x, y in zip(a, b) if x["delay"] - y["delay"] > 1],
            "harmed": [x["name"] for x, y in zip(a, b) if y["delay"] - x["delay"] > 1],
            "constraint_violations_executed": 0,
            "attempted_violations_ppo": out["policies"]["ppo"]["summary"]["attempted_violations"]}
    else:
        out["ppo_note"] = "RL model NOT TRAINED - baselines only."
    out["interpretation"] = ("Predicted minutes saved exist only inside this simulation on a simplified "
                             "corridor with synthetic background traffic. They do not demonstrate "
                             "real-world punctuality improvement.")
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
                _model_cache.clear()
            except Exception as e:      # noqa: BLE001
                log.exception("%s failed", kind)
                self.state[kind] = {"status": "failed", "error": str(e)}
        threading.Thread(target=run, daemon=True).start()
        return True
