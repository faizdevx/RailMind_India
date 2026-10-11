"""Gymnasium dispatching environment over the synthetic corridor.

Action space: Discrete(n_trains + 1)
  k < n  : request release of train k now (only executed if hard safety checks pass)
  k == n : advance one minute (hold everything not yet released)
Safety (block occupancy, headway, loop capacity) lives in Corridor.check_release,
not in the reward. Unsafe requests are no-ops and are counted + penalised.
There is no cancel action; unfinished trains at the horizon are charged their lateness.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from railmind.sim.corridor import (Corridor, MAX_MINUTES, MAX_POLICY_HOLD, N_STATIONS,
                                   WEIGHTS, make_scenario)

N_TRAINS = 6
FEATS = 12
TRAIN_SEED_BASE = 0
HELDOUT_SEED_BASE = 100_000


class RailDispatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_trains: int = N_TRAINS, seed: int = 0, scenario=None):
        super().__init__()
        self.n = n_trains
        self.action_space = spaces.Discrete(self.n + 1)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.n * FEATS + 1,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._fixed = scenario
        self.c: Corridor | None = None
        self._acts_this_min = 0

    # -- observation --------------------------------------------------
    def _obs(self) -> np.ndarray:
        c = self.c
        out = []
        for i, tr in enumerate(c.sc.trains):
            st = c.status[i]
            ready = c.is_ready(i)
            safe = c.check_release(i)[0]
            sd = c.sched[i].get(c.station[i], (0, 0))[1]
            late = np.clip((c.t - sd) / 60.0, -1, 4) if st != "done" else 0.0
            remaining = abs(tr.dest - c.station[i]) / (N_STATIONS - 1)
            out += [st == "pending", st == "at_station", st == "running", st == "done",
                    tr.direction, tr.prio / 2.0, c.station[i] / (N_STATIONS - 1),
                    float(ready), float(safe), c.hold[i] / MAX_POLICY_HOLD, late, remaining]
        out.append(c.t / MAX_MINUTES)
        return np.asarray(out, dtype=np.float32)

    # -- gym API ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        sc = (options or {}).get("scenario") or self._fixed
        if sc is None:
            sc = make_scenario(int(self._rng.integers(TRAIN_SEED_BASE, HELDOUT_SEED_BASE)), self.n)
        self.c = Corridor(sc)
        self._acts_this_min = 0
        return self._obs(), {}

    def step(self, action: int):
        c = self.c
        reward = 0.0
        advance = action == self.n
        if not advance:
            ok = c.release(int(action))
            if not ok and c.check_release(int(action))[1] != "not ready":
                reward -= 0.2
            elif not ok:
                reward -= 0.05          # requesting a train that cannot move
            self._acts_this_min += 1
            if self._acts_this_min > self.n + 1:
                advance = True
        if advance:
            late = 0.0
            for i, tr in enumerate(c.sc.trains):
                if c.status[i] == "at_station" and c.t > c.sched[i][c.station[i]][1]:
                    late += WEIGHTS[tr.prio]
                elif c.status[i] == "running" and c.t > c.sched[i][c.next_station(i)][0]:
                    late += WEIGHTS[tr.prio]
            reward -= late / 20.0
            c.advance()
            self._acts_this_min = 0
        terminated = c.finished() or c.deadlocked
        truncated = (not terminated) and c.t >= MAX_MINUTES
        if terminated or truncated:
            s = c.summary()
            reward -= s["weighted_delay"] / 30.0 + 0.5 * s["max_delay"] / 30.0
            if c.deadlocked or truncated:
                reward -= 10.0
        return self._obs(), reward, terminated, truncated, {}


def run_ppo(model, scenario, deterministic: bool = True) -> Corridor:
    """Roll a trained PPO policy on a scenario; returns the finished Corridor."""
    env = RailDispatchEnv(scenario=scenario)
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=deterministic)
        obs, _, term, trunc, _ = env.step(int(a))
        done = term or trunc
    return env.c
