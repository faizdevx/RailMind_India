"""Corridor simulator (SIMULATED - not real railway control).

A line of stations joined by sections. Station loops have limited capacity.
Safety constraints (block occupancy, headway, loop capacity) are enforced
here, never by reward shaping.

Topology assumption (documented in the UI): by default sections are double
line (one block per direction); set Scenario.single_line=True for a single
track shared by both directions.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

N_STATIONS = 6               # synthetic scenarios only
STATION_CAP = 3              # loop capacity at intermediate stations
HEADWAY = 2                  # min minutes between successive entries into a block
MAX_POLICY_HOLD = 20         # a ready+safe train held this long is released (all policies)
MAX_MINUTES = 1000
DEADLOCK_WINDOW = 45
CLASS_NAMES = {0: "freight/other", 1: "passenger", 2: "express"}
WEIGHTS = {0: 1.0, 1: 1.5, 2: 2.0}
DWELL = {0: 0, 1: 2, 2: 1}
SYNTHETIC_LABEL = "SYNTHETIC DATA - NOT REAL INDIAN RAILWAYS DATA"


@dataclass
class TrainSpec:
    id: int
    name: str
    direction: int            # +1 toward higher station index, -1 toward lower
    prio: int                 # 0 other, 1 passenger, 2 express/superfast
    origin: int               # first corridor station this train touches
    dest: int                 # last corridor station this train touches
    sched_start: int          # scheduled departure minute from origin (t=0 is "now")
    start_time: int           # earliest actual departure (sched_start + initial delay)
    source: str = "SYNTHETIC"  # SYNTHETIC | OBSERVED_SNAPSHOT | HISTORICAL_SCHEDULE
    code: str = ""
    delay_source: str = ""
    progress: Optional[float] = None   # 0..1 if already running in first section


@dataclass
class Scenario:
    trains: list
    n_stations: int
    station_names: list
    sched: dict               # tid -> {station: (sched_arr, sched_dep)}
    nominal_run: dict         # tid -> {section: minutes}
    actual_run: dict          # tid -> {section: minutes} realised
    single_line: bool = False
    seed: Optional[int] = None
    label: str = SYNTHETIC_LABEL
    assumptions: list = field(default_factory=list)


def _syn_schedule(trains, nominal_run, n_stations):
    out = {}
    for t in trains:
        d = {t.origin: (t.sched_start, t.sched_start)}
        cur, st = t.sched_start, t.origin
        while st != t.dest:
            sec = min(st, st + t.direction)
            arr = cur + nominal_run[t.id][sec]
            nxt = st + t.direction
            dep = arr if nxt == t.dest else arr + DWELL[t.prio]
            d[nxt] = (arr, dep)
            cur, st = dep, nxt
        out[t.id] = d
    return out


def make_scenario(seed: int, n_trains: int = 6, load: float = 1.0, jitter: int = 2,
                  max_initial_delay: int = 25, single_line: bool = True) -> Scenario:
    rng = random.Random(seed)
    base = {0: (14, 18), 1: (10, 13), 2: (7, 10)}
    nom_sec = [{p: rng.randint(*base[p]) for p in base} for _ in range(N_STATIONS - 1)]
    act_sec = [{p: max(3, nom_sec[s][p] + rng.randint(-jitter, jitter)) for p in base}
               for s in range(N_STATIONS - 1)]
    trains = []
    spread = max(10, int(45 / max(load, 0.25)))
    for i in range(n_trains):
        d = 1 if i % 2 == 0 else -1
        prio = rng.choice([0, 1, 1, 2])
        sched = rng.randint(0, spread)
        delay = rng.randint(3, max_initial_delay) if rng.random() < 0.5 else 0
        o, de = (0, N_STATIONS - 1) if d > 0 else (N_STATIONS - 1, 0)
        trains.append(TrainSpec(i, f"SYN-{i + 1:02d}", d, prio, o, de, sched, sched + delay))
    nominal = {t.id: {s: nom_sec[s][t.prio] for s in range(N_STATIONS - 1)} for t in trains}
    actual = {t.id: {s: act_sec[s][t.prio] for s in range(N_STATIONS - 1)} for t in trains}
    return Scenario(trains, N_STATIONS, [f"Stn {i + 1}" for i in range(N_STATIONS)],
                    _syn_schedule(trains, nominal, N_STATIONS), nominal, actual,
                    single_line=single_line, seed=seed)


def perturb(sc: Scenario, seed: int, jitter: int = 2) -> Scenario:
    """Same scenario, different realised section run times (for uncertainty)."""
    rng = random.Random(seed)
    new = copy.deepcopy(sc)
    for tid, secs in sc.nominal_run.items():
        new.actual_run[tid] = {s: max(3, m + rng.randint(-jitter, jitter)) for s, m in secs.items()}
    new.seed = seed
    return new


class Corridor:
    def __init__(self, sc: Scenario):
        self.sc = sc
        self.t = 0
        n = len(sc.trains)
        self.status = ["pending"] * n
        self.station = [t.origin for t in sc.trains]
        self.ready_at = [t.start_time for t in sc.trains]
        self.hold = [0] * n
        self.wait = [0] * n
        self.arr_t: list = [None] * n
        self.dep_t = [0] * n
        self.final_arrival: list = [None] * n
        self.sched = [sc.sched[t.id] for t in sc.trains]
        self.in_section: dict = {}      # section -> list of train ids currently running in it
        self.last_entry: dict = {}      # (section, direction) -> last entry minute
        self.wait_by_section = [0] * max(1, sc.n_stations - 1)
        self.events: list = []
        self.tracks: dict = {t.id: [] for t in sc.trains}
        self.arrivals: dict = {t.id: {} for t in sc.trains}
        self.departures: dict = {t.id: {} for t in sc.trains}
        self.forced = 0
        self.attempted_violations = 0
        self.releases = 0
        self.idle = 0
        self.deadlocked = False
        self._init_running()
        self.begin_minute()

    def _init_running(self):
        for i, tr in enumerate(self.sc.trains):
            if tr.progress is not None and tr.origin != tr.dest:
                s = self.section_of(i)
                run = self.sc.actual_run[tr.id][s]
                self.status[i] = "running"
                self.dep_t[i] = -int(round(tr.progress * run))
                self.arr_t[i] = max(1, self.dep_t[i] + run)
                self.in_section.setdefault(s, []).append(i)

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
        d = self.sc.trains[i].direction
        if self.sc.single_line and any(self.sc.trains[j].direction != d
                                       for j in self.in_section.get(s, [])):
            return False, "block occupied"      # opposing train in single-line section
        if self.t - self.last_entry.get((s, d), -10 ** 6) < HEADWAY:
            return False, "headway"
        nxt = self.next_station(i)
        if nxt != self.sc.trains[i].dest and self.occupancy(nxt) >= STATION_CAP:
            return False, "loop full"
        return True, "ok"

    def blocker(self, i: int) -> Optional[int]:
        """An opposing train occupying the single-line section, if any."""
        s, d = self.section_of(i), self.sc.trains[i].direction
        return next((j for j in self.in_section.get(s, []) if self.sc.trains[j].direction != d), None)

    def ready_trains(self):
        return [i for i in range(self.n()) if self.is_ready(i)]

    def finished(self) -> bool:
        return all(s == "done" for s in self.status)

    # ---- transitions ---------------------------------------------------
    def _name(self, i):
        return self.sc.trains[i].name

    def _st(self, k):
        return self.sc.station_names[k]

    def release(self, i: int, note: str = "") -> bool:
        ok, why = self.check_release(i)
        if not ok:
            self.attempted_violations += 1 if why != "not ready" else 0
            return False
        tr = self.sc.trains[i]
        s = self.section_of(i)
        run = self.sc.actual_run[tr.id][s]
        d = tr.direction
        arr = self.t + run
        ahead = [self.arr_t[j] for j in self.in_section.get(s, []) if self.sc.trains[j].direction == d]
        if ahead:                               # cannot overtake inside a section: follow at headway
            arr = max(arr, max(ahead) + HEADWAY)
        self.in_section.setdefault(s, []).append(i)
        self.last_entry[(s, d)] = self.t
        self.dep_t[i] = self.t
        self.arr_t[i] = arr
        self.departures[tr.id][self.station[i]] = self.t
        self.status[i] = "running"
        self.hold[i] = 0
        self.releases += 1
        self.idle = 0
        self.events.append({"t": self.t, "train": tr.name, "kind": "DEPART", "text":
                            f"{tr.name} departs {self._st(self.station[i])} -> "
                            f"{self._st(self.next_station(i))}" + (f" ({note})" if note else "")})
        return True

    def begin_minute(self):
        for i, tr in enumerate(self.sc.trains):
            if self.status[i] == "pending" and self.t >= tr.start_time:
                self.status[i] = "at_station"
            if self.status[i] == "running" and self.arr_t[i] <= self.t:
                self.in_section[self.section_of(i)].remove(i)
                self.station[i] = self.next_station(i)
                self.arrivals[tr.id][self.station[i]] = self.t
                sa = self.sched[i].get(self.station[i])
                late = (self.t - sa[0]) if sa else None
                if self.station[i] == tr.dest:
                    self.status[i] = "done"
                    self.final_arrival[i] = self.t
                    self.events.append({"t": self.t, "train": tr.name, "kind": "ARRIVE",
                                        "text": f"{tr.name} arrives {self._st(tr.dest)}"
                                                + (f" ({late:+d} min vs schedule)" if late is not None else "")})
                else:
                    self.status[i] = "at_station"
                    sd = sa[1] - sa[0] if sa else DWELL[tr.prio]
                    self.ready_at[i] = self.t + max(0, min(sd, DWELL[tr.prio] + 1))
                    self.events.append({"t": self.t, "train": tr.name, "kind": "ARRIVE",
                                        "text": f"{tr.name} arrives {self._st(self.station[i])}"
                                                + (f" ({late:+d} min)" if late is not None else "")})
        for i in range(self.n()):
            if self.hold[i] >= MAX_POLICY_HOLD and self.check_release(i)[0]:
                if self.release(i, note="forced after max hold"):
                    self.forced += 1

    def advance(self):
        for i in range(self.n()):
            if self.is_ready(i):
                self.hold[i] += 1
                self.wait[i] += 1
                self.wait_by_section[max(0, self.section_of(i))] += 1
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
            if fa is None:
                delay, done = self.t - sa, False
            else:
                delay, done = fa - sa, True
            d = max(0, delay)
            total_w += WEIGHTS[tr.prio] * d
            mx = max(mx, d)
            stops = {}
            for st, at in self.arrivals[tr.id].items():
                if st in self.sched[i]:
                    stops[st] = {"station": self._st(st), "pred_arrival": at,
                                 "sched_arrival": self.sched[i][st][0],
                                 "delay": at - self.sched[i][st][0]}
            per.append({"id": tr.id, "name": tr.name, "code": tr.code, "class": CLASS_NAMES[tr.prio],
                        "source": tr.source, "sched_arrival": sa, "arrival": fa,
                        "delay": delay, "completed": done, "wait_min": self.wait[i],
                        "initial_delay": tr.start_time - tr.sched_start, "stops": stops})
        return {"per_train": per, "weighted_delay": round(total_w, 2),
                "mean_delay": round(sum(max(0, p["delay"]) for p in per) / max(1, len(per)), 2),
                "max_delay": mx, "completed": sum(p["completed"] for p in per),
                "n_trains": len(per), "minutes": self.t, "forced_releases": self.forced,
                "attempted_violations": self.attempted_violations,
                "deadlocked": self.deadlocked, "total_wait": sum(self.wait)}


# ---- baseline policies --------------------------------------------------
def policy_fcfs(c: Corridor) -> list:
    """First-come-first-served: release ready trains in order of readiness."""
    return sorted(c.ready_trains(), key=lambda i: (c.ready_at[i], i))


def policy_scheduled(c: Corridor) -> list:
    """Scheduled-order: a train yields to any train scheduled earlier for the
    same block that has not yet entered it."""
    out = []
    for i in c.ready_trains():
        s = c.section_of(i)
        di = c.sc.trains[i].direction
        key = (c.sched[i][c.station[i]][1], i)
        blocked = False
        for j in range(c.n()):
            if j == i or c.status[j] in ("done", "running"):
                continue
            tj = c.sc.trains[j]
            if not c.sc.single_line and tj.direction != di:
                continue
            up = s if tj.direction > 0 else s + 1
            before = c.station[j] <= s if tj.direction > 0 else c.station[j] >= s + 1
            if tj.direction > 0 and not (c.station[j] <= s < tj.dest):
                continue
            if tj.direction < 0 and not (tj.dest <= s < c.station[j]):
                continue
            sj = c.sched[j].get(up)
            if before and sj is not None and (sj[1], j) < key:
                blocked = True
                break
        if not blocked:
            out.append(i)
    return sorted(out, key=lambda i: (c.sched[i][c.station[i]][1], i))


POLICIES: dict = {"fcfs": policy_fcfs, "scheduled": policy_scheduled}


def run_policy(sc: Scenario, policy: Callable[[Corridor], list]) -> Corridor:
    c = Corridor(sc)
    while not c.over():
        for i in policy(c):
            if c.check_release(i)[0]:
                c.release(i)
        c.advance()
    return c
