"""Evaluate FCFS, scheduled-order and PPO on identical held-out SYNTHETIC scenarios."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from railmind.data import history as H
from railmind.data.scenario import sample_scenario
from railmind.rl.env import run_ppo
from railmind.rl.train import MODEL_DIR, load_model
from railmind.sim.corridor import POLICIES, run_policy

EVAL_PATH = MODEL_DIR / "ppo_corridor_v2_eval.json"
TEST_SEED = 777_000


def compare_on(sc, model) -> dict:
    out = {n: run_policy(sc, p).summary() for n, p in POLICIES.items()}
    if model is not None:
        out["ppo"] = run_ppo(model, sc).summary()
    return out


def evaluate(n_scenarios: int = 100, model=None, use_cut: bool = False) -> dict:
    model = model or load_model()
    kn = H.load(cut=use_cut)
    rng = np.random.default_rng(TEST_SEED)
    rows = {k: [] for k in ("fcfs", "scheduled", "ppo")}
    tgt = {k: [] for k in rows}
    harmed = helped = 0
    for _ in range(n_scenarios):
        sc = sample_scenario(kn, rng, "test")
        res = compare_on(sc, model)
        for k, v in res.items():
            rows[k].append(v)
            tgt[k].append(v["per_train"][0]["delay"])
        if "ppo" in res:
            for a, b in zip(res["fcfs"]["per_train"], res["ppo"]["per_train"]):
                d = a["delay"] - b["delay"]
                helped += d > 1
                harmed += d < -1
    seeds = [TEST_SEED]

    def agg(lst):
        if not lst:
            return None
        w = np.array([r["weighted_delay"] for r in lst])
        m = np.array([r["max_delay"] for r in lst])
        return {"weighted_delay_mean": round(float(w.mean()), 2),
                "weighted_delay_std": round(float(w.std()), 2),
                "mean_train_delay_mean": round(float(np.mean([r["mean_delay"] for r in lst])), 2),
                "max_delay_mean": round(float(m.mean()), 2),
                "completed_fraction": round(float(np.mean([r["completed"] / r["n_trains"] for r in lst])), 3),
                "deadlocks": int(sum(r["deadlocked"] for r in lst)),
                "forced_releases_mean": round(float(np.mean([r["forced_releases"] for r in lst])), 2),
                "attempted_violations_total": int(sum(r["attempted_violations"] for r in lst))}

    result = {"evaluated_utc": datetime.now(timezone.utc).isoformat(), "n_scenarios": n_scenarios,
              "seeds": f"rng seed {TEST_SEED}; target trains held out (train number % 5 == 0), never used as "
                       "training targets",
              "data": f"scenarios sampled from historical timetables/delays {kn.date_from}..{kn.date_to}",
              "target_train_mean_delay": {k: round(float(np.mean(v)), 2) for k, v in tgt.items() if v},
              "metrics": {k: agg(v) for k, v in rows.items()},
              "ppo_trained": model is not None,
              "ppo_vs_fcfs_train_outcomes": {"helped_gt_1min": int(helped), "harmed_gt_1min": int(harmed)},
              "note": "Simulated corridor with assumed topology. Not validated against real railway operations."}
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--cut", action="store_true")
    a = ap.parse_args()
    r = evaluate(a.scenarios, use_cut=a.cut)
    MODEL_DIR.mkdir(exist_ok=True)
    EVAL_PATH.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
