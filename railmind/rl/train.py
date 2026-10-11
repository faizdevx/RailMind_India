"""Train PPO on labelled SYNTHETIC scenarios. Usage: python -m railmind.rl.train --timesteps 200000"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch
from datetime import datetime, timezone
from pathlib import Path

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from railmind.rl.env import RailDispatchEnv, run_ppo
from railmind.sim.corridor import POLICIES, make_scenario, run_policy

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "ppo_corridor_v1.zip"
META_PATH = MODEL_DIR / "ppo_corridor_v1.json"


VAL_SEED_BASE = 50_000          # model-selection seeds; disjoint from train and held-out test seeds


def _expert_action(c, n):
    from railmind.sim.corridor import policy_fcfs
    for i in policy_fcfs(c):
        if c.check_release(i)[0]:
            return i
    return n


def behaviour_clone(model, n_episodes: int, seed: int, epochs: int = 8) -> dict:
    """Warm start: imitate the FCFS rule on synthetic training scenarios, then PPO fine-tunes."""
    env = RailDispatchEnv(seed=seed)
    X, Y = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a = _expert_action(env.c, env.n)
            X.append(obs); Y.append(a)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
    X = torch.as_tensor(np.array(X)); Y = torch.as_tensor(np.array(Y))
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    loss = None
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for k in range(0, len(X), 512):
            idx = perm[k:k + 512]
            _, logp, _ = model.policy.evaluate_actions(X[idx], Y[idx])
            loss = -logp.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return {"bc_samples": int(len(X)), "bc_episodes": n_episodes, "bc_final_loss": float(loss)}


def _val_score(model, n=40) -> float:
    tot = 0.0
    for i in range(n):
        tot += run_ppo(model, make_scenario(VAL_SEED_BASE + i)).summary()["weighted_delay"]
    return tot / n


def train(timesteps: int = 200_000, seed: int = 0, progress=None) -> dict:
    cfg = {"algo": "PPO", "policy": "MlpPolicy", "learning_rate": 5e-5, "n_steps": 4096,
           "batch_size": 256, "gamma": 1.0, "ent_coef": 0.0, "net_arch": [128, 128],
           "total_timesteps": timesteps, "seed": seed, "warm_start": "behaviour cloning of FCFS rule",
           "bc_episodes": 300, "model_selection": f"best of BC-only / PPO-final on validation seeds {VAL_SEED_BASE}+"}
    env = Monitor(RailDispatchEnv(seed=seed))
    model = PPO("MlpPolicy", env, learning_rate=cfg["learning_rate"], n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"], gamma=cfg["gamma"], ent_coef=cfg["ent_coef"],
                policy_kwargs={"net_arch": cfg["net_arch"]}, seed=seed, device="cpu", verbose=0)
    t0 = time.time()
    bc = behaviour_clone(model, cfg["bc_episodes"], seed)
    bc_val = _val_score(model)
    model.save(str(MODEL_DIR / "_bc_tmp"))
    model.learn(total_timesteps=timesteps)
    ppo_val = _val_score(model)
    chosen = "ppo_finetuned"
    if bc_val < ppo_val:
        model = PPO.load(str(MODEL_DIR / "_bc_tmp"), device="cpu")
        chosen = "bc_warm_start_only"
    (MODEL_DIR / "_bc_tmp.zip").unlink(missing_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH.with_suffix("")))
    meta = {"model_version": "ppo_corridor_v1", "trained_utc": datetime.now(timezone.utc).isoformat(),
            "config": cfg, "bc": bc, "validation_weighted_delay": {"bc_only": round(bc_val, 2),
            "ppo_finetuned": round(ppo_val, 2)}, "selected_checkpoint": chosen,
            "train_seconds": round(time.time() - t0, 1),
            "stable_baselines3": sb3.__version__, "data": "SYNTHETIC DATA - NOT REAL INDIAN RAILWAYS DATA",
            "validated_against_real_operations": False}
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def load_model():
    if not MODEL_PATH.exists():
        return None
    return PPO.load(str(MODEL_PATH.with_suffix("")), device="cpu")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(train(a.timesteps, a.seed), indent=2))
