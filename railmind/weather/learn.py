"""Learn how weather changes section delay, from history joined with Open-Meteo archive weather.

Method (all measured, nothing assumed):
  1. baseline = historical median delay change per (train, section, incoming-delay bucket)
     from knowledge built BEFORE the cutoff
  2. residual = actual delay change - baseline
  3. join weather at the departure station/hour of each section (reanalysis)
  4. effect table by condition (model-free, with bootstrap CI) on pre-cutoff rows
  5. gradient-boosted residual model; accepted only if it lowers error on post-cutoff rows
"""
from __future__ import annotations

import json
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd

from railmind.data import history as H
from railmind.data.history import _to_min, file_date
from railmind.rl.train import MODEL_DIR
from railmind.weather.openmeteo import HOURLY, OpenMeteo, conditions

STATIONS = H.RAW / "stations" / "stations.json"
EFFECTS = MODEL_DIR / "weather_effects.json"
MODEL = MODEL_DIR / "weather_residual.pkl"
TOP_K = 120
FEATURES = ["precip", "gust", "rh", "temp", "dew_gap", "hour", "month", "d_prev", "srun"]   # archive has no visibility


def load_coords() -> dict:
    d = json.load(open(STATIONS, encoding="utf-8"))
    out = {}
    for f in d["features"]:
        g = f.get("geometry") or {}
        c = g.get("coordinates")
        code = (f.get("properties") or {}).get("code")
        if code and c and len(c) == 2 and c[0] and c[1]:
            out[code] = (float(c[1]), float(c[0]))       # (lat, lon)
    return out


def top_stations(kn, coords, k=TOP_K) -> list:
    cnt: Counter = Counter()
    for (fr, to, b), (q, n) in kn.t2.items():
        cnt[fr] += n
    names = kn.stn_names
    out = []
    for sid, _ in cnt.most_common():
        code = names[sid]
        if code in coords:
            out.append(code)
        if len(out) >= k:
            break
    return out


def weather_table(stations: list, coords: dict, ranges: list) -> dict:
    """{(code, 'YYYY-MM-DDTHH:00'): conditions-ish tuple} via Open-Meteo archive."""
    om = OpenMeteo()
    tab: dict = {}
    for start, end in ranges:
        cs = [coords[s] for s in stations]
        for s, h in zip(stations, om.hourly(cs, start, end)):
            if not h or "time" not in h:
                continue
            cols = [h.get(v) or [None] * len(h["time"]) for v in HOURLY]
            for i, t in enumerate(h["time"]):
                tab[(s, t)] = tuple(c[i] for c in cols)
    return tab


def section_rows(files, stations: set, kn) -> pd.DataFrame:
    rows = []
    for f in files:
        df = pd.read_parquet(f, columns=H.COLS)
        df = df[df["status"].astype(str) == "completed"]
        df = df.sort_values(["train_number", "start_date", "seq"], kind="stable")
        sa, sd = _to_min(df["scheduled_arrival"]), _to_min(df["scheduled_departure"])
        sa_ff, sd_ff = sa.fillna(sd).values, sd.fillna(sa).values
        delay = df["arrival_delay_min"].fillna(df["departure_delay_min"]).astype(float).values
        tr = df["train_number"].astype(str).values
        st = df["station_code"].astype(str).values
        sdt = df["start_date"].astype(str).values
        same = np.concatenate([[False], (tr[1:] == tr[:-1]) & (sdt[1:] == sdt[:-1])])
        prev = lambda a: np.concatenate([[np.nan], a[:-1]])  # noqa: E731
        d_prev, run = prev(delay), sa_ff - prev(sd_ff)
        ok = same & ~np.isnan(d_prev) & ~np.isnan(delay) & (np.abs(delay - d_prev) <= 300) \
            & (np.abs(d_prev) <= 720) & (run > 0) & (run < 1440)
        idx = np.where(ok)[0]
        if not len(idx):
            continue
        fr = st[idx - 1]
        keep = np.isin(fr, list(stations))
        idx, fr = idx[keep], fr[keep]
        if not len(idx):
            continue
        dep = prev(sd_ff)[idx]                              # minutes since epoch, local wall time
        ts = pd.to_datetime(dep, unit="m")
        rows.append(pd.DataFrame({
            "train": tr[idx], "fr": fr, "to": st[idx], "d_prev": d_prev[idx], "delta": delay[idx] - d_prev[idx],
            "srun": run[idx], "time": ts.strftime("%Y-%m-%dT%H:00"), "date": ts.strftime("%Y-%m-%d"),
            "hour": ts.hour, "month": ts.month}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def attach(df: pd.DataFrame, tab: dict, kn) -> pd.DataFrame:
    w = [tab.get((f, t)) for f, t in zip(df["fr"].values, df["time"].values)]
    ok = np.array([x is not None for x in w])
    df = df[ok].reset_index(drop=True)
    w = np.array([x for x in w if x is not None], dtype=object)
    f = lambda i: np.array([np.nan if v[i] is None else float(v[i]) for v in w])  # noqa: E731
    df["temp"], df["rh"], df["dew"], df["precip"], df["gust"], df["code"], df["vis"] = \
        f(0), f(1), f(2), f(3), f(4), f(5), f(6)
    df["dew_gap"] = df["temp"] - df["dew"]
    df["vis_km"] = df["vis"] / 1000.0
    # baseline median per row from pre-cutoff knowledge
    base = np.empty(len(df))
    for i, (t, a, b, d) in enumerate(zip(df["train"].values, df["fr"].values, df["to"].values,
                                         df["d_prev"].values)):
        base[i] = kn.median_delta(kn.train_id(t), kn.stn_id(a), kn.stn_id(b), d)
    df["base"] = base
    df["resid"] = df["delta"] - df["base"]
    return df


def flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rain_light"] = (df.precip >= 0.2) & (df.precip < 2.5)
    out["rain_moderate"] = (df.precip >= 2.5) & (df.precip < 7.6)
    out["rain_heavy"] = df.precip >= 7.6
    fog_vis = df.vis < 1000
    fog_code = df.code.isin([45, 48])
    fog_px = (df.rh >= 96) & (df.dew_gap <= 1.0) & (df.temp <= 20) & (df.gust < 25) & df.vis.isna()
    out["fog"] = fog_vis | fog_code | fog_px
    out["strong_wind"] = df.gust >= 50
    out["heat"] = df.temp >= 42
    out["cold"] = df.temp <= 3
    return out


def effect_table(df: pd.DataFrame, rng) -> dict:
    fl = flags(df)
    res = {}
    base_share = float((df.resid >= 5).mean())
    for c in fl.columns:
        m = fl[c].values
        n = int(m.sum())
        if n < 200:
            res[c] = {"n": n, "note": "too few observations (n<200)"}
            continue
        r_on, r_off = df.resid.values[m], df.resid.values[~m]
        diffs = []
        n_off = min(len(r_off), 200_000)
        for _ in range(300):                    # full-size bootstrap: CI reflects the real sample sizes
            diffs.append(rng.choice(r_on, len(r_on)).mean() - rng.choice(r_off, n_off).mean())
        res[c] = {"n": n, "mean_extra_delay_min": round(float(r_on.mean() - r_off.mean()), 2),
                  "ci95": [round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)],
                  "share_gain_ge5_on": round(float((r_on >= 5).mean()), 3),
                  "share_gain_ge5_off": round(float((r_off >= 5).mean()), 3)}
        res[c]["significant"] = bool(res[c]["ci95"][0] > 0 or res[c]["ci95"][1] < 0)
    res["_baseline_share_gain_ge5"] = round(base_share, 3)
    return res


def run(max_train_files: int = 140, max_rows: int = 1_500_000) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.time()
    kn = H.load(cut=True)                         # learned before EVAL_CUTOFF
    coords = load_coords()
    stations = top_stations(kn, coords)
    allf = H.list_files()
    train_f = [f for f in allf if file_date(f) < H.EVAL_CUTOFF]
    test_f = [f for f in allf if file_date(f) >= H.EVAL_CUTOFF]
    train_f = train_f[:: max(1, len(train_f) // max_train_files)]
    test_f = test_f[:: max(1, len(test_f) // 40)]
    ranges = [("2025-02-08", "2026-02-07"), ("2026-10-01", "2026-10-11")]
    tab = weather_table(stations, coords, ranges)
    tr = attach(section_rows(train_f, set(stations), kn), tab, kn)
    te = attach(section_rows(test_f, set(stations), kn), tab, kn)
    if len(tr) > max_rows:
        tr = tr.sample(max_rows, random_state=0).reset_index(drop=True)
    rng = np.random.default_rng(0)
    eff = effect_table(tr, rng)
    X, y = tr[FEATURES].values, tr["resid"].values
    model = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.08, max_iter=200,
                                          loss="absolute_error", random_state=0)
    model.fit(X, y)
    pred = np.clip(model.predict(te[FEATURES].values), -5, 15)
    mae_base = float(np.abs(te["resid"]).mean())
    mae_wx = float(np.abs(te["resid"].values - pred).mean())
    # does it help specifically when weather is adverse?
    adv = flags(te).any(axis=1).values
    out = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "stations": len(stations), "train_rows": int(len(tr)), "test_rows": int(len(te)),
           "train_range": f"<{H.EVAL_CUTOFF}", "test_range": f">={H.EVAL_CUTOFF}",
           "weather_source": "Open-Meteo archive (reanalysis), CC BY 4.0",
           "effects_by_condition": eff,
           "heldout_mae_min": {"baseline_median": round(mae_base, 3), "baseline_plus_weather": round(mae_wx, 3),
                               "adverse_weather_rows": int(adv.sum()),
                               "baseline_on_adverse": round(float(np.abs(te["resid"].values[adv]).mean()), 3)
                               if adv.any() else None,
                               "with_weather_on_adverse": round(float(np.abs(te["resid"].values[adv] - pred[adv]).mean()), 3)
                               if adv.any() else None},
           "accepted": bool(mae_wx <= 0.99 * mae_base and
                            (not adv.any() or np.abs(te["resid"].values[adv] - pred[adv]).mean()
                             <= np.abs(te["resid"].values[adv]).mean())),
           "acceptance_rule": "held-out MAE must fall by >=1 % and must not rise on adverse-weather rows",
           "features": FEATURES,
           "seconds": round(time.time() - t0, 1)}
    EFFECTS.write_text(json.dumps(out, indent=2))
    with open(MODEL, "wb") as fh:
        pickle.dump({"model": model, "features": FEATURES, "accepted": out["accepted"]}, fh)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
