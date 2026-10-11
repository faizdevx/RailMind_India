"""Build a labelled counterfactual scenario from a real provider snapshot.

Only what the provider reported is treated as observed (train name, delay,
station names/schedule if supplied). Everything else is an explicit assumption.
"""
from __future__ import annotations

import random
import re
from typing import Optional

from railmind.sim.corridor import (N_STATIONS, Scenario, TrainSpec, make_scenario)

EXPRESS_WORDS = ("rajdhani", "shatabdi", "vande", "duronto", "tejas", "garib")


def _hhmm(s) -> Optional[int]:
    m = re.search(r"(\d{1,2}):(\d{2})", str(s or ""))
    return int(m[1]) * 60 + int(m[2]) if m else None


def build_scenario(status: dict, seed: int = 7, load: float = 1.0, extra_delay: int = 0,
                   n_background: int = 5, jitter: int = 2) -> Scenario:
    """status: TrainStatus.as_dict() from a verified provider response."""
    base = make_scenario(seed, n_trains=n_background + 1, load=load, jitter=jitter)
    assumptions = [
        "SIMULATED corridor of 6 stations, single track, crossing loops (capacity 3). "
        "Not the real infrastructure: block layout, signalling, gradients, speed limits unknown.",
        "Background trains are SYNTHETIC (SYNTHETIC DATA - NOT REAL INDIAN RAILWAYS DATA) "
        "- real conflicting traffic is not supplied by the provider.",
        "The observed train is placed at corridor station 1 heading toward station 6.",
        "Train class (express/passenger) is inferred from the train name; real stopping pattern unknown.",
    ]
    names = [f"Stn {i + 1}" for i in range(N_STATIONS)]
    stops = status.get("stops") or []
    cur = status.get("current_station")
    route_names = [s.get("name") for s in stops if s.get("name")]
    if cur and cur not in route_names:
        route_names.insert(0, cur)
    if len(route_names) >= 2:
        for i, n in enumerate(route_names[:N_STATIONS]):
            names[i] = n
        assumptions.append(f"Station names 1-{min(len(route_names), N_STATIONS)} taken from provider "
                           "route; remaining names are placeholders.")
    else:
        assumptions.append("Provider returned no route; all station names are placeholders.")

    # section run times from provider schedule when >=2 parseable scheduled arrivals
    arr = [_hhmm(s.get("sched_arr")) for s in stops][: N_STATIONS]
    diffs = []
    for a, b in zip(arr, arr[1:]):
        if a is not None and b is not None:
            d = (b - a) % 1440
            if 3 <= d <= 120:
                diffs.append(d)
    sc = base
    if len(diffs) >= 2:
        for s in range(N_STATIONS - 1):
            d = diffs[s] if s < len(diffs) else diffs[-1]
            sc.nominal_run[s][2] = d
            sc.nominal_run[s][1] = d + 2
            sc.actual_run[s][2] = max(3, d + random.Random(seed + s).randint(-jitter, jitter))
            sc.actual_run[s][1] = max(3, sc.actual_run[s][2] + 2)
        assumptions.append("Section run times for the observed train derived from provider scheduled "
                           "arrival times; the rest are synthetic.")
    else:
        assumptions.append("No usable scheduled times from provider; section run times are synthetic defaults.")

    name = status.get("train_name") or f"Train {status.get('train_no')}"
    prio = 2 if any(w in name.lower() for w in EXPRESS_WORDS) else 1
    delay = max(0, min(int(status.get("delay_min") or 0), 180)) + max(0, int(extra_delay))
    if status.get("delay_min") is None:
        assumptions.append("Provider reported no delay value: observed train starts on time (delay 0).")
    synthetic = str(status.get("provider", "")).startswith("SYNTHETIC")
    real = TrainSpec(0, f"{status.get('train_no')} {name}", 1, prio, 0, 0, delay,
                     source="SYNTHETIC" if synthetic else "OBSERVED_SNAPSHOT",
                     provider_ref={"provider": status.get("provider"),
                                   "retrieved_utc": status.get("retrieved_utc"),
                                   "reported_delay_min": status.get("delay_min")})
    if synthetic:
        assumptions.insert(0, "Primary train is SYNTHETIC too: no real data in this run "
                              "(SYNTHETIC DATA - NOT REAL INDIAN RAILWAYS DATA).")
    sc.trains[0] = real
    # keep background trains from starting before the observed one is in scenario
    sc.station_names = names
    sc.label = "SIMULATED / PREDICTED - counterfactual built from a provider snapshot"
    sc.assumptions = assumptions
    return sc
