"""Train PPO on scenarios sampled from learned historical data. python -m railmind.rl.train --timesteps 300000"""
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
from railmind.data import history as H
from railmind.data.scenario import sample_scenario

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "ppo_corridor_v2.zip"
META_PATH = MODEL_DIR / "ppo_corridor_v2.json"


VAL_SEED_BASE = 50_000          # model-selection seeds; disjoint from train and held-out test seeds


def _sampler(kn):
    return lambda rng: sample_scenario(kn, rng, "train")


def _expert_action(c, n):
    from railmind.sim.corridor import policy_fcfs
    for i in policy_fcfs(c):
        if c.check_release(i)[0]:
            return i
    return n


def behaviour_clone(model, n_episodes: int, seed: int, sampler, epochs: int = 8) -> dict:
    """Warm start: imitate the FCFS rule on training scenarios, then PPO fine-tunes."""
    env = RailDispatchEnv(seed=seed, sampler=sampler)
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


def _val_scenarios(kn, n=40):
    rng = np.random.default_rng(VAL_SEED_BASE)
    return [sample_scenario(kn, rng, "train") for _ in range(n)]


def _val_score(model, scs) -> float:
    return float(np.mean([run_ppo(model, sc).summary()["weighted_delay"] for sc in scs if sc]))


def train(timesteps: int = 200_000, seed: int = 0, progress=None, use_cut: bool = False) -> dict:
    kn = H.load(cut=use_cut)
    if kn is None:
        raise FileNotFoundError("Knowledge missing. Run scripts/download_data.py then "
                                "python -m railmind.data.history")
    sampler = _sampler(kn)
    val = _val_scenarios(kn)
    cfg = {"algo": "PPO", "policy": "MlpPolicy", "learning_rate": 5e-5, "n_steps": 4096,
           "batch_size": 256, "gamma": 1.0, "ent_coef": 0.0, "net_arch": [128, 128],
           "total_timesteps": timesteps, "seed": seed, "warm_start": "behaviour cloning of FCFS rule",
           "bc_episodes": 300, "model_selection": f"best of BC-only / PPO-final on validation seeds {VAL_SEED_BASE}+"}
    env = Monitor(RailDispatchEnv(seed=seed, sampler=sampler))
    model = PPO("MlpPolicy", env, learning_rate=cfg["learning_rate"], n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"], gamma=cfg["gamma"], ent_coef=cfg["ent_coef"],
                policy_kwargs={"net_arch": cfg["net_arch"]}, seed=seed, device="cpu", verbose=0)
    t0 = time.time()
    bc = behaviour_clone(model, cfg["bc_episodes"], seed, sampler)
    bc_val = _val_score(model, val)
    model.save(str(MODEL_DIR / "_bc_tmp"))
    model.learn(total_timesteps=timesteps)
    ppo_val = _val_score(model, val)
    chosen = "ppo_finetuned"
    if bc_val < ppo_val:
        model = PPO.load(str(MODEL_DIR / "_bc_tmp"), device="cpu")
        chosen = "bc_warm_start_only"
    (MODEL_DIR / "_bc_tmp.zip").unlink(missing_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH.with_suffix("")))
    meta = {"model_version": "ppo_corridor_v2", "trained_utc": datetime.now(timezone.utc).isoformat(),
            "config": cfg, "bc": bc, "validation_weighted_delay": {"bc_only": round(bc_val, 2),
            "ppo_finetuned": round(ppo_val, 2)}, "selected_checkpoint": chosen,
            "train_seconds": round(time.time() - t0, 1),
            "stable_baselines3": sb3.__version__, "data": f"historical NTES-derived delays + timetables ({kn.date_from}..{kn.date_to}); sampled scenarios",
            "knowledge": kn.describe(),
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
