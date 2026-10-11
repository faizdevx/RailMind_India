"""Evaluate FCFS, scheduled-order and PPO on identical held-out SYNTHETIC scenarios."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from railmind.rl.env import HELDOUT_SEED_BASE, run_ppo
from railmind.rl.train import MODEL_DIR, load_model
from railmind.sim.corridor import POLICIES, make_scenario, run_policy

EVAL_PATH = MODEL_DIR / "ppo_corridor_v1_eval.json"


def compare_on(sc, model) -> dict:
    out = {n: run_policy(sc, p).summary() for n, p in POLICIES.items()}
    if model is not None:
        out["ppo"] = run_ppo(model, sc).summary()
    return out


def evaluate(n_scenarios: int = 100, model=None) -> dict:
    model = model or load_model()
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_scenarios)]
    rows = {k: [] for k in ("fcfs", "scheduled", "ppo")}
    harmed = helped = 0
    for sd in seeds:
        res = compare_on(make_scenario(sd), model)
        for k, v in res.items():
            rows[k].append(v)
        if "ppo" in res:
            for a, b in zip(res["fcfs"]["per_train"], res["ppo"]["per_train"]):
                d = a["delay"] - b["delay"]
                helped += d > 1
                harmed += d < -1

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
              "seeds": f"{seeds[0]}..{seeds[-1]} (held out from training seeds 0..{HELDOUT_SEED_BASE - 1})",
              "data": "SYNTHETIC DATA - NOT REAL INDIAN RAILWAYS DATA",
              "metrics": {k: agg(v) for k, v in rows.items()},
              "ppo_trained": model is not None,
              "ppo_vs_fcfs_train_outcomes": {"helped_gt_1min": int(helped), "harmed_gt_1min": int(harmed)},
              "note": "Simulated, synthetic scenarios only. Not validated against real railway operations."}
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=100)
    a = ap.parse_args()
    r = evaluate(a.scenarios)
    MODEL_DIR.mkdir(exist_ok=True)
    EVAL_PATH.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
