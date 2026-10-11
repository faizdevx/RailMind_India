"""Single-track corridor simulator (SIMULATED - not real railway control).

A short line of stations joined by single-track sections. Stations have
crossing loops of limited capacity. Safety constraints (block occupancy,
headway, loop capacity) are enforced here, never by reward shaping.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

N_STATIONS = 6
STATION_CAP = 3          # loop capacity at intermediate stations
HEADWAY = 2              # min minutes between successive entries into a section
MAX_POLICY_HOLD = 20     # a ready+safe train held this long is released (all policies)
MAX_MINUTES = 400
DEADLOCK_WINDOW = 45     # minutes without any movement before declaring deadlock
CLASS_NAMES = {0: "freight", 1: "passenger", 2: "express"}
WEIGHTS = {0: 1.0, 1: 1.5, 2: 2.0}
DWELL = {0: 0, 1: 2, 2: 1}
SYNTHETIC_LABEL = "SYNTHETIC DATA - NOT REAL INDIAN RAILWAYS DATA"


@dataclass
class TrainSpec:
    id: int
    name: str
    direction: int            # +1 toward higher station index, -1 toward lower
    prio: int                 # 0 freight, 1 passenger, 2 express
    origin: int               # station index where the train starts
    sched_start: int          # scheduled departure minute from origin
    start_time: int           # actual earliest departure (sched_start + initial delay)
    source: str = "SYNTHETIC"  # SYNTHETIC | OBSERVED_SNAPSHOT
    provider_ref: Optional[dict] = None

    @property
    def dest(self) -> int:
        return N_STATIONS - 1 if self.direction > 0 else 0


@dataclass
class Scenario:
    trains: list
    nominal_run: list         # nominal_run[section][prio] minutes (used for schedule)
    actual_run: list          # realised run times (nominal + jitter), same shape
    station_names: list
    seed: Optional[int] = None
    label: str = SYNTHETIC_LABEL
    assumptions: list = field(default_factory=list)

    def sections_between(self, a: int, b: int):
        return range(min(a, b), max(a, b))


def schedule_for(sc: Scenario, t: TrainSpec) -> dict:
    """Nominal timetable {station: (sched_arr, sched_dep)} computed from nominal run times."""
    out = {t.origin: (t.sched_start, t.sched_start)}
    cur = t.sched_start
    st = t.origin
    while st != t.dest:
        sec = st if t.direction > 0 else st - 1
        arr = cur + sc.nominal_run[sec][t.prio]
        nxt = st + t.direction
        dep = arr if nxt == t.dest else arr + DWELL[t.prio]
        out[nxt] = (arr, dep)
        cur, st = dep, nxt
    return out


def make_scenario(seed: int, n_trains: int = 6, load: float = 1.0, jitter: int = 2,
                  max_initial_delay: int = 25) -> Scenario:
    rng = random.Random(seed)
    base = {0: (14, 18), 1: (10, 13), 2: (7, 10)}
    nominal, actual = [], []
    for _ in range(N_STATIONS - 1):
        nom = {p: rng.randint(*base[p]) for p in base}
        nominal.append(nom)
        actual.append({p: max(3, nom[p] + rng.randint(-jitter, jitter)) for p in base})
    trains = []
    spread = max(10, int(45 / max(load, 0.25)))
    for i in range(n_trains):
        d = 1 if i % 2 == 0 else -1
        prio = rng.choice([0, 1, 1, 2])
        sched = rng.randint(0, spread)
        delayed = rng.random() < 0.5
        delay = rng.randint(3, max_initial_delay) if delayed else 0
        origin = 0 if d > 0 else N_STATIONS - 1
        trains.append(TrainSpec(i, f"SYN-{i + 1:02d}", d, prio, origin, sched, sched + delay))
    return Scenario(trains, nominal, actual,
                    [f"Stn {i + 1}" for i in range(N_STATIONS)], seed=seed)


def perturb(sc: Scenario, seed: int, jitter: int = 2) -> Scenario:
    """Same scenario, different realised section run times (for uncertainty)."""
    rng = random.Random(seed)
    import copy
    new = copy.deepcopy(sc)
    new.actual_run = [{p: max(3, sc.nominal_run[s][p] + rng.randint(-jitter, jitter))
                       for p in sc.nominal_run[s]} for s in range(len(sc.nominal_run))]
    new.seed = seed
    return new


class Corridor:
    def __init__(self, sc: Scenario):
        self.sc = sc
        self.t = 0
        n = len(sc.trains)
        self.status = ["pending"] * n          # pending|at_station|running|done
        self.station = [t.origin for t in sc.trains]
        self.ready_at = [t.start_time for t in sc.trains]
        self.hold = [0] * n
        self.wait = [0] * n
        self.arr_t: list = [None] * n          # arrival time at next station if running
        self.dep_t = [0] * n
        self.final_arrival: list = [None] * n
        self.sched = [schedule_for(sc, t) for t in sc.trains]
        self.section_occ: dict = {}            # section -> train id
        self.last_exit = {s: -10 ** 6 for s in range(N_STATIONS - 1)}
        self.wait_by_section = [0] * (N_STATIONS - 1)
        self.events: list = []
        self.tracks: dict = {t.id: [] for t in sc.trains}
        self.forced = 0
        self.attempted_violations = 0
        self.releases = 0
        self.idle = 0
        self.deadlocked = False
        self.begin_minute()

    # ---- queries -------------------------------------------------------
    def n(self) -> int:
        return len(self.sc.trains)

    def section_of(self, i: int) -> int:
        t = self.sc.trains[i]
        return self.station[i] if t.direction > 0 else self.station[i] - 1

    def next_station(self, i: int) -> int:
        return self.station[i] + self.sc.trains[i].direction

    def occupancy(self, st: int) -> int:
        inbound = sum(1 for j in range(self.n()) if self.status[j] == "running"
                      and self.next_station(j) == st)
        here = sum(1 for j in range(self.n()) if self.status[j] == "at_station"
                   and self.station[j] == st)
        return inbound + here

    def is_ready(self, i: int) -> bool:
        if self.status[i] != "at_station":
            return False
        sd = self.sched[i][self.station[i]][1]
        return self.t >= self.ready_at[i] and self.t >= sd

    def check_release(self, i: int):
        """Return (ok, reason). Hard safety rules; independent of any policy."""
        if not self.is_ready(i):
            return False, "not ready"
        s = self.section_of(i)
        if s in self.section_occ:
            return False, "block occupied"
        if self.t - self.last_exit[s] < HEADWAY:
            return False, "headway"
        nxt = self.next_station(i)
        if nxt != self.sc.trains[i].dest and self.occupancy(nxt) >= STATION_CAP:
            return False, "loop full"
        return True, "ok"

    def ready_trains(self):
        return [i for i in range(self.n()) if self.is_ready(i)]

    def finished(self) -> bool:
        return all(s == "done" for s in self.status)

    # ---- transitions ---------------------------------------------------
    def release(self, i: int, note: str = "") -> bool:
        ok, why = self.check_release(i)
        if not ok:
            self.attempted_violations += 1 if why != "not ready" else 0
            return False
        tr = self.sc.trains[i]
        s = self.section_of(i)
        run = self.sc.actual_run[s][tr.prio]
        self.section_occ[s] = i
        self.dep_t[i] = self.t
        self.arr_t[i] = self.t + run
        self.status[i] = "running"
        self.hold[i] = 0
        self.releases += 1
        self.idle = 0
        self.events.append({"t": self.t, "train": tr.name, "kind": "DEPART", "text":
                            f"{tr.name} departs {self.sc.station_names[self.station[i]]} "
                            f"-> {self.sc.station_names[self.next_station(i)]}" + (f" ({note})" if note else "")})
        return True

    def begin_minute(self):
        for i, tr in enumerate(self.sc.trains):
            if self.status[i] == "pending" and self.t >= tr.start_time:
                self.status[i] = "at_station"
            if self.status[i] == "running" and self.arr_t[i] <= self.t:
                s = self.section_of(i)
                self.section_occ.pop(s, None)
                self.last_exit[s] = self.t
                self.station[i] = self.next_station(i)
                if self.station[i] == tr.dest:
                    self.status[i] = "done"
                    self.final_arrival[i] = self.t
                    late = self.t - self.sched[i][tr.dest][0]
                    self.events.append({"t": self.t, "train": tr.name, "kind": "ARRIVE",
                                        "text": f"{tr.name} arrives {self.sc.station_names[tr.dest]} "
                                                f"({late:+d} min vs schedule)"})
                else:
                    self.status[i] = "at_station"
                    self.ready_at[i] = self.t + DWELL[tr.prio]
                    self.events.append({"t": self.t, "train": tr.name, "kind": "ARRIVE",
                                        "text": f"{tr.name} arrives {self.sc.station_names[self.station[i]]}"})
        for i in range(self.n()):          # bounded-hold rule, same for every policy
            if self.hold[i] >= MAX_POLICY_HOLD and self.check_release(i)[0]:
                if self.release(i, note="forced after max hold"):
                    self.forced += 1

    def advance(self):
        for i in range(self.n()):
            if self.is_ready(i) and self.status[i] == "at_station":
                self.hold[i] += 1
                self.wait[i] += 1
                self.wait_by_section[self.section_of(i)] += 1
        self._record()
        moved = any(s == "running" for s in self.status)
        self.idle = 0 if moved else self.idle + 1
        if self.idle >= DEADLOCK_WINDOW and not self.finished() and not any(
                s == "pending" for s in self.status):
            self.deadlocked = True
        self.t += 1
        self.begin_minute()

    def _record(self):
        for i, tr in enumerate(self.sc.trains):
            if self.status[i] == "pending":
                continue
            if self.status[i] == "running":
                run = max(1, self.arr_t[i] - self.dep_t[i])
                pos = self.station[i] + tr.direction * (self.t - self.dep_t[i]) / run
            else:
                pos = float(self.station[i])
            self.tracks[tr.id].append([self.t, round(pos, 3)])

    def over(self) -> bool:
        return self.finished() or self.deadlocked or self.t >= MAX_MINUTES

    # ---- results -------------------------------------------------------
    def summary(self) -> dict:
        per = []
        total_w = 0.0
        mx = 0
        for i, tr in enumerate(self.sc.trains):
            sa = self.sched[i][tr.dest][0]
            fa = self.final_arrival[i]
            if fa is None:                       # unfinished: lower bound at stop time
                delay, done = self.t - sa, False
            else:
                delay, done = fa - sa, True
            d = max(0, delay)
            total_w += WEIGHTS[tr.prio] * d
            mx = max(mx, d)
            per.append({"id": tr.id, "name": tr.name, "class": CLASS_NAMES[tr.prio],
                        "source": tr.source, "sched_arrival": sa, "arrival": fa,
                        "delay": delay, "completed": done, "wait_min": self.wait[i],
                        "initial_delay": tr.start_time - tr.sched_start})
        return {"per_train": per, "weighted_delay": round(total_w, 2),
                "mean_delay": round(sum(max(0, p["delay"]) for p in per) / len(per), 2),
                "max_delay": mx, "completed": sum(p["completed"] for p in per),
                "n_trains": len(per), "minutes": self.t, "forced_releases": self.forced,
                "attempted_violations": self.attempted_violations,
                "deadlocked": self.deadlocked, "total_wait": sum(self.wait)}


# ---- baseline policies --------------------------------------------------
def policy_fcfs(c: Corridor) -> list:
    """First-come-first-served: release ready trains in order of readiness."""
    ids = c.ready_trains()
    return sorted(ids, key=lambda i: (c.ready_at[i], i))


def policy_scheduled(c: Corridor) -> list:
    """Scheduled-order: a train yields to any other train scheduled earlier
    for the same section that has not yet entered it."""
    out = []
    for i in c.ready_trains():
        s = c.section_of(i)
        key = (c.sched[i][c.station[i]][1], i)
        blocked = False
        for j in range(c.n()):
            if j == i or c.status[j] in ("done", "running"):
                continue
            tj = c.sc.trains[j]
            up = s if tj.direction > 0 else s + 1       # upstream end of section s for j
            before = c.station[j] <= s if tj.direction > 0 else c.station[j] >= s + 1
            sj = c.sched[j].get(up)
            if before and sj is not None and (sj[1], j) < key:
                blocked = True
                break
        if not blocked:
            out.append(i)
    return sorted(out, key=lambda i: (c.sched[i][c.station[i]][1], i))


POLICIES: dict = {"fcfs": policy_fcfs, "scheduled": policy_scheduled}


def run_policy(sc: Scenario, policy: Callable[[Corridor], list], explain: bool = False) -> Corridor:
    c = Corridor(sc)
    while not c.over():
        for i in policy(c):
            if c.check_release(i)[0]:        # baselines only request safe releases
                c.release(i)
        c.advance()
    return c
