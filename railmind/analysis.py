"""Per-train analysis: observed past + learned patterns -> predictions, causes, alternatives.

Every statement carries an evidence label:
  OBSERVED   - reported by the provider (or recorded replay)
  LEARNED    - computed from historical data (comparison/percentile)
  SIMULATED  - result of the counterfactual corridor model (assumed topology)
  HYPOTHESIS - plausible but NOT supported by available data
"""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from railmind.data import history as H
from railmind.data.predict import percentile_of, rollout
from railmind.data.scenario import apply_section_extra, build_scenario, find_stop_index
from railmind.rl.env import run_ppo
from railmind.sim.corridor import POLICIES, Scenario, run_policy
from railmind.weather import service as wxs
from railmind.weather.openmeteo import ATTRIBUTION, WeatherUnavailable


def ist_now() -> datetime:
    return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).replace(tzinfo=None)


def _ts(x) -> Optional[datetime]:
    if not x:
        return None
    try:
        t = pd.Timestamp(x)
        return None if pd.isna(t) else t.tz_localize(None).to_pydatetime() if t.tzinfo else t.to_pydatetime()
    except (ValueError, TypeError):
        return None


def _mins(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    return None if a is None or b is None else (b - a).total_seconds() / 60.0


def _decisions_diff(base, other, names) -> list:
    out = []
    for tid, deps in other.departures.items():
        for st, t_o in deps.items():
            t_b = base.departures.get(tid, {}).get(st)
            if t_b is not None and abs(t_o - t_b) >= 2:
                out.append({"train": base.sc.trains[tid].name, "station": names[st],
                            "baseline_dep_min": t_b, "policy_dep_min": t_o, "shift_min": t_o - t_b})
    return sorted(out, key=lambda r: -abs(r["shift_min"]))[:8]


def observed_facts(kn, status: dict) -> dict:
    stops = [s for s in status.get("stops") or [] if s.get("code")]
    tno = status["train_no"]
    cur = status.get("current_station_code") or status.get("current_station")
    cur_idx = next((i for i, s in enumerate(stops) if s["code"] == cur), None)
    passed = [s for i, s in enumerate(stops)
              if s.get("delay_min") is not None and (cur_idx is None or i <= cur_idx)]
    sections, dwell = [], []
    for a, b in zip(passed[:-1], passed[1:]):
        delta = max(0, b["delay_min"]) - max(0, a["delay_min"])   # earliness burn-off is not a delay cause
        pc = percentile_of(kn, tno, a["code"], b["code"], a["delay_min"], delta) if kn else None
        sections.append({"from": a["code"], "to": b["code"], "delay_from": a["delay_min"],
                         "delay_to": b["delay_min"], "delta": delta, "history": pc})
    for s in passed:
        sa, sd, aa, ad = (_ts(s.get(k)) for k in ("sched_arr", "sched_dep", "actual_arr", "actual_dep"))
        ds, da = _mins(sa, sd), _mins(aa, ad)
        arr_d, dep_d = _mins(sa, aa), _mins(sd, ad)
        if ds is not None and da is not None and arr_d is not None and dep_d is not None:
            # delay added at the halt: lateness leaving minus lateness arriving (early arrivals waiting for
            # the booked departure time are not a cause of delay)
            dwell.append({"station": s["code"], "sched_dwell": ds, "actual_dwell": da,
                          "excess": max(0.0, dep_d) - max(0.0, arr_d)})
    return {"cur_idx": cur_idx, "passed": passed, "sections": sections, "dwell": dwell, "stops": stops}


def build_causes(kn, status, obs, sim_info) -> list:
    out = []
    d0 = status.get("delay_min")
    passed = obs["passed"]
    if passed and passed[0].get("delay_min") is not None and passed[0]["delay_min"] >= 10:
        out.append({"label": "OBSERVED", "cause": "Late start / origin delay",
                    "text": f"Delay already {passed[0]['delay_min']} min at first recorded stop "
                            f"{passed[0]['code']}. Reason for the late start is not in the data."})
    secs = sorted([s for s in obs["sections"] if s["delta"] >= 4], key=lambda s: -s["delta"])[:3]
    for s in secs:
        h = s["history"]
        if h and h["n"] >= 8:
            rel = (f"unusual for this section (p{int(h['pct'])}; typical change {h['median']:+.0f} min, "
                   f"p90 {h['p90']:+.0f})") if h["pct"] >= 85 else \
                  f"within the usual range (p{int(h['pct'])}; typical {h['median']:+.0f} min)"
            lab = "LEARNED"
        else:
            rel, lab = "no reliable history for this section", "OBSERVED"
        out.append({"label": lab, "cause": f"Delay gained between {s['from']} and {s['to']}",
                    "text": f"+{s['delta']:.0f} min late-time added ({s['delay_from']:.0f} -> {s['delay_to']:.0f}); {rel}. "
                            "Physical reason (signal, block, speed restriction, wait) is not in the data."})
    for dw in sorted(obs["dwell"], key=lambda x: -x["excess"])[:2]:
        if dw["excess"] >= 5:
            out.append({"label": "OBSERVED", "cause": "Extended station halt",
                        "text": f"{dw['station']}: halted {dw['actual_dwell']:.0f} min vs scheduled "
                                f"{dw['sched_dwell']:.0f} min (+{dw['excess']:.0f})."})
    raw = status.get("raw") or {}
    d = (raw.get("data") if isinstance(raw, dict) else None) or {}
    for ex in (d.get("exceptions") or []) if isinstance(d, dict) else []:
        if isinstance(ex, dict):
            out.append({"label": "OBSERVED", "cause": f"Provider exception: {ex.get('type')}",
                        "text": "Provider reports " + str(ex.get("type")) + "."})
    if sim_info.get("congestion_minutes") is not None:
        cm = sim_info["congestion_minutes"]
        out.append({"label": "SIMULATED", "cause": "Interaction with nearby scheduled trains",
                    "text": (f"In the simulated corridor the target train loses ~{cm:.0f} min versus running "
                             "alone, because of headway/loop/block conflicts with nearby trains. "
                             "Real conflicts depend on live positions and track layout not in the data. Caution: on held-out "
                             "replays the simulated interactions did not improve accuracy over the learned forecast."
                             if cm >= 1 else
                             "Simulated corridor shows no meaningful loss from nearby trains.")})
    out.append({"label": "HYPOTHESIS", "cause": "Maintenance restrictions, signal/equipment faults, "
                                                "late incoming rake",
                "text": "No feed for these is integrated, and rake linkage is not in the data. They may explain "
                        "delay but cannot be attributed from the available evidence."})
    return out


def _stop_time(s: dict) -> Optional[str]:
    for k in ("actual_dep", "actual_arr", "sched_dep", "sched_arr"):
        t = _ts(s.get(k))
        if t is not None:
            return t.strftime("%Y-%m-%dT%H:%M")
    return None


def weather_block(obs: dict, ahead: list, tt_sched: dict, d0: float, roll: list, now: datetime) -> dict:
    """Weather at passed stops (context for observed delay) and ahead stops (expected time)."""
    passed = [s for s in obs["passed"] if s.get("code")]
    items = [{"code": s["code"], "time": _stop_time(s)} for s in passed]
    n_p = len(items)
    for c in ahead:
        s = tt_sched.get(c, {})
        t = _ts(s.get("actual_arr")) or _ts(s.get("sched_arr"))
        if t is not None and s.get("actual_arr") is None:
            t = t + timedelta(minutes=max(0.0, d0))
        items.append({"code": c, "time": t.strftime("%Y-%m-%dT%H:%M") if t else None})
    try:
        ws = wxs.route_weather(items, now)
    except WeatherUnavailable:
        ws = [None] * len(items)
    rows = []
    for i, (it, w) in enumerate(zip(items, ws)):
        rows.append({"code": it["code"], "time": it["time"], "phase": "passed" if i < n_p else "ahead",
                     "cond": w, "severity": wxs.severity(w) if w else None,
                     "text": wxs.summary_text(w) if w else "weather unavailable"})
    ahead_w = ws[n_p:]
    stops_a = []
    for k, c in enumerate(ahead[:-1]):
        a, b = tt_sched.get(c, {}), tt_sched.get(ahead[k + 1], {})
        run = _mins(_ts(a.get("sched_dep")) or _ts(a.get("sched_arr")), _ts(b.get("sched_arr")))
        stops_a.append({"code": c, "sched_run": run if run and run > 0 else 60})
    stops_a.append({"code": ahead[-1], "sched_run": 60})
    delays = [r["p50"] for r in roll] if roll else []
    dl = [d0] + delays[:len(stops_a) - 1]
    dl += [dl[-1]] * (len(stops_a) - len(dl))
    adj = wxs.adjustments(stops_a, ahead_w, dl, now.month)
    return {"rows": rows, "ahead_codes": ahead, "adjust_min": adj, "n_passed": n_p}


def weather_causes(wb: dict, obs: dict, eff: Optional[dict]) -> list:
    out = []
    by_code = {r["code"]: r for r in wb["rows"] if r["phase"] == "passed"}
    sig = {k: v for k, v in (eff or {}).get("effects_by_condition", {}).items()
           if isinstance(v, dict) and v.get("significant")}
    gains = [s for s in obs["sections"] if s["delta"] >= 4]
    adverse = []
    for s in gains:
        w = by_code.get(s["from"])
        if w and w["cond"] and w["severity"]:
            adverse.append((s, w))
    seen = {r["code"] for r in wb["rows"] if r["phase"] == "passed" and r["cond"]}
    if adverse:
        for s, w in adverse[:3]:
            c = w["cond"]
            learned = []
            for key, on in (("rain_heavy", c["rain"] == "heavy"), ("rain_moderate", c["rain"] == "moderate"),
                            ("rain_light", c["rain"] == "light"), ("fog", c["fog"]),
                            ("strong_wind", c["strong_wind"]), ("heat", c["heat"]), ("cold", c["cold"])):
                if on and key in sig:
                    e = sig[key]
                    learned.append(f"{key.replace('_', ' ')}: {e['mean_extra_delay_min']:+.1f} min per section "
                                   f"historically (95% CI {e['ci95'][0]:+.1f}..{e['ci95'][1]:+.1f}, n={e['n']})")
            out.append({"label": "LEARNED" if learned else "CONTEXT",
                        "cause": f"Adverse weather near {s['from']} while delay grew +{s['delta']:.0f} min",
                        "text": f"{w['text']} at {w['time']}. "
                                + ("; ".join(learned) + ". " if learned else
                                   "No statistically reliable learned effect for this condition. ")
                                + "Co-occurrence only; it supports, but does not prove, weather as the cause."})
    elif gains and seen:
        out.append({"label": "CONTEXT", "cause": "Weather at delay-gaining sections",
                    "text": "Reanalysis weather at the stops where delay grew was benign (no heavy rain, fog, "
                            "strong gusts, extreme heat/cold). Weather is not supported as the explanation "
                            "for those sections."})
    ahead_bad = [r for r in wb["rows"] if r["phase"] == "ahead" and r["severity"]]
    if ahead_bad:
        r = ahead_bad[0]
        tot = sum(wb["adjust_min"]) if wb["adjust_min"] else 0
        out.append({"label": "PREDICTED", "cause": "Adverse weather ahead",
                    "text": f"{len(ahead_bad)} upcoming stop(s) with adverse conditions, first {r['code']} "
                            f"around {r['time']}: {r['text']}. Learned model adds about {tot:+.0f} min over the "
                            "forecast horizon (0 if the model was not accepted)."})
    return out


def analyze(kn, status: dict, model=None, now: Optional[datetime] = None, n_mc: int = 15,
            single_line: bool = False, extra_delay: int = 0, seed: int = 0,
            use_weather: bool = True) -> dict:
    tno = status["train_no"]
    d0 = float(status.get("delay_min") or 0)
    raw = status.get("raw") or {}
    if now is None:
        now = _ts(raw.get("now_ist")) if isinstance(raw, dict) else None
    if now is None and "replay" not in str(status.get("provider", "")).lower():
        now = ist_now()                  # live: snapshot time is retrieval time
    now = now or ist_now()
    obs = observed_facts(kn, status)
    cur_code = status.get("current_station_code") or status.get("current_station")
    sm = status.get("stops_meta") or {}
    names = {s_["code"]: s_.get("name") for s_ in (status.get("stops") or []) if s_.get("code") and s_.get("name")}
    names = {c_: n_ for c_, n_ in names.items() if n_ != c_}
    anchor = cur_code
    if kn is not None and kn.timetable(tno) and find_stop_index(kn, tno, cur_code) is None:
        # provider position is not a timetable stop (e.g. a non-halt): anchor to the last known halt before it
        codes_route = [s_["code"] for s_ in obs["stops"]]
        cur_i = obs["cur_idx"] if obs["cur_idx"] is not None else 0
        for c_ in reversed(codes_route[: cur_i + 1]):
            if find_stop_index(kn, tno, c_) is not None:
                anchor = c_
                break
    known = kn is not None and kn.train_id(tno) >= 0 and kn.timetable(tno) is not None
    res = {"train_no": tno, "train_name": status.get("train_name"), "provider": status["provider"],
           "retrieved_utc": status.get("retrieved_utc"), "now_ist": now.isoformat(timespec="minutes"),
           "observed_delay_min": status.get("delay_min"), "current_station": cur_code,
           "anchor_station": anchor, "station_names": names, "tracking": sm,
           "observed": {"sections": obs["sections"], "dwell": obs["dwell"]},
           "learned_model_available": bool(known), "limitations": [
               "Predictions are statistical (history) and simulated (assumed topology); not railway control.",
               "Station names come from the live provider when supplied; historical data has codes only.",
               "Simulated improvement does not prove real-world punctuality improvement."]}
    if not known:
        res["warning"] = (f"Train {tno} not found in learned history: no learned pattern, no simulation. "
                          "Only observed facts shown.")
        res["causes"] = build_causes(None, status, obs, {})
        return res
    # --- learned roll-forward (data-driven "how will it run further") -----------------
    route = [s["code"] for s in obs["stops"]] or [s[0] for s in kn.timetable(tno)["stops"]]
    ci = obs["cur_idx"] if obs["cur_idx"] is not None else (route.index(cur_code) if cur_code in route else 0)
    if anchor in route:
        ci = route.index(anchor)
    ahead = route[ci: ci + 9]
    tt_sched = {s["code"]: s for s in obs["stops"]}
    roll = rollout(kn, tno, ahead, d0 + extra_delay, n=400, seed=seed) if len(ahead) > 1 else []
    for r in roll:
        s = tt_sched.get(r["station"], {})
        r["sched_arr"] = s.get("sched_arr")
        r["name"] = names.get(r["station"])
        r["provider_estimate"] = s.get("delay_min")
        r["provenance"] = s.get("provenance")
    wb = None
    if use_weather:
        wb = weather_block(obs, ahead, tt_sched, d0 + extra_delay, roll, now)
        cum = 0.0
        for k, r in enumerate(roll):
            cum += wb["adjust_min"][k] if k < len(wb["adjust_min"]) else 0.0
            r["wx_adjust_cum"] = round(cum, 1)
            r["p50_wx"] = round(r["p50"] + cum, 1)
        wmap = {r["code"]: r for r in wb["rows"] if r["phase"] == "ahead"}
        for r in roll:
            w = wmap.get(r["station"])
            r["weather"] = w["text"] if w else None
            r["weather_severity"] = w["severity"] if w else None
    res["learned_forecast"] = {"method": "Monte-Carlo roll-forward of historical delay change per "
                                         "(train, section, incoming-delay bucket)",
                               "stops": roll, "label": "LEARNED / PREDICTED"}
    # --- counterfactual corridor ---------------------------------------------------
    cat_hint = sm.get("category") or sm.get("type")
    sc = build_scenario(kn, tno, anchor, d0, now, seed=seed, single_line=single_line,
                        extra_delay=extra_delay, category_hint=cat_hint) if anchor else None
    sim_info: dict = {}
    names_map = names
    sec_extra: dict = {}
    if sc is not None and wb is not None:
        by_from = {c_: wb["adjust_min"][k] for k, c_ in enumerate(wb["ahead_codes"][:-1])
                   if k < len(wb["adjust_min"])}
        sec_extra = {sec: by_from.get(sc.station_names[sec], 0.0) for sec in range(sc.n_stations - 1)}
        apply_section_extra(sc, sec_extra)
    if sc is None:
        res["simulation"] = {"available": False, "reason": "Current station not on the learned timetable "
                                                          "or too close to route end."}
    else:
        det = {k: run_policy(sc, p) for k, p in POLICIES.items()}
        if model is not None:
            det["ppo"] = run_ppo(model, sc)
        alone = Scenario(sc.trains[:1], sc.n_stations, sc.station_names, {0: sc.sched[0]},
                         {0: sc.nominal_run[0]}, {0: sc.actual_run[0]}, sc.single_line)
        alone_run = run_policy(alone, POLICIES["fcfs"]).summary()["per_train"][0]
        alone_delay = alone_run["delay"]
        alone_stops = {sc.station_names[k]: v["delay"] for k, v in alone_run["stops"].items()}
        fcfs_delay = det["fcfs"].summary()["per_train"][0]["delay"]
        sim_info["congestion_minutes"] = max(0, fcfs_delay - alone_delay)
        # Monte-Carlo over sampled run times / nearby-train delays; identical scenario per seed for all policies
        mc = {k: [] for k in det}
        stops_mc = {k: {} for k in det}
        for r in range(n_mc):
            s2 = build_scenario(kn, tno, anchor, d0, now, seed=1000 + r, sample=True,
                                single_line=single_line, extra_delay=extra_delay, category_hint=cat_hint)
            if sec_extra:
                apply_section_extra(s2, sec_extra)
            for k in det:
                c = run_policy(s2, POLICIES[k]) if k in POLICIES else run_ppo(model, s2)
                p0 = c.summary()["per_train"][0]
                mc[k].append(p0["delay"])
                for st, v in p0["stops"].items():
                    stops_mc[k].setdefault(st, []).append(v["delay"])

        def ms(v):
            return {"mean": round(statistics.fmean(v), 1), "std": round(statistics.pstdev(v), 1),
                    "p10": round(float(np.percentile(v, 10)), 1), "p90": round(float(np.percentile(v, 90)), 1)}
        names = [names_map.get(c_, c_) for c_ in sc.station_names]
        sc.station_names = names
        pols = {}
        for k, c in det.items():
            s = c.summary()
            p0 = s["per_train"][0]
            saved = [a - b for a, b in zip(mc["fcfs"], mc[k])]
            pols[k] = {"label": "SIMULATED / PREDICTED", "summary": s, "tracks": c.tracks,
                       "events": c.events, "target_delay_end": p0["delay"],
                       "target_delay_mc": ms(mc[k]),
                       "minutes_saved_vs_fcfs_mc": ms(saved) if k != "fcfs" else None,
                       "target_stops_median": {names[st]: round(float(np.median(v)), 1)
                                               for st, v in sorted(stops_mc[k].items())},
                       "constraint_violations_executed": 0,
                       "attempted_violations": s["attempted_violations"]}
        best = "ppo" if "ppo" in det else "scheduled"
        diff = _decisions_diff(det["fcfs"], det[best], names)
        fa = det["fcfs"].summary()["per_train"]
        ba = det[best].summary()["per_train"]
        helped = [a["name"] for a, b in zip(fa, ba) if a["delay"] - b["delay"] > 1]
        harmed = [a["name"] for a, b in zip(fa, ba) if b["delay"] - a["delay"] > 1]
        ws = det["fcfs"].wait_by_section
        bn = int(np.argmax(ws)) if ws and max(ws) > 0 else None
        res["simulation"] = {
            "available": True, "label": "SIMULATED / PREDICTED - NOT REAL RAILWAY CONTROL",
            "station_names": names, "assumptions": sc.assumptions,
            "trains": [{"id": t.id, "name": t.name, "source": t.source, "direction": t.direction,
                        "delay_source": t.delay_source} for t in sc.trains],
            "policies": pols, "best_policy": best,
            "compared_policy_note": ("PPO (trained on sampled historical scenarios, not validated against real "
                                     "operations)" if best == "ppo" else
                                     "PPO model not trained; scheduled-order shown as alternative"),
            "interventions": diff, "helped": helped, "harmed": harmed,
            "bottleneck": ({"section": f"{names[bn]}-{names[bn + 1]}", "wait_train_min": ws[bn],
                            "label": "SIMULATED SCENARIO"} if bn is not None else None),
            "alone_delay": alone_delay, "alone_stops": alone_stops, "mc_runs": n_mc}
    res["causes"] = build_causes(kn, status, obs, sim_info)
    if wb is not None:
        eff = wxs.effects()
        res["causes"][-1:-1] = weather_causes(wb, obs, eff)          # before the generic hypothesis
        res["weather"] = {"attribution": ATTRIBUTION, "rows": wb["rows"],
                          "model_accepted": bool(wxs.residual_model() is not None),
                          "sim_extra_min_by_section": sec_extra,
                          "learned": ({k: eff[k] for k in ("train_rows", "test_rows", "heldout_mae_min",
                                                          "accepted", "train_range", "test_range")}
                                      if eff else None)}
    truth = (raw.get("truth_after_now") if isinstance(raw, dict) else None) or []
    if truth:
        fc = {r["station"]: r for r in roll}
        sp = (res.get("simulation") or {}).get("policies", {})
        rows = []
        for t in truth:
            f = fc.get(t["station"])
            if t["delay"] is None or f is None:
                continue
            row = {"station": t["station"], "actual_delay": t["delay"], "forecast_p50": f["p50"],
                   "forecast_p10": f["p10"], "forecast_p90": f["p90"], "persistence": d0}
            for k, p in sp.items():
                v = p.get("target_stops_median", {}).get(t["station"])
                if v is not None:
                    row[f"sim_{k}"] = v
            if t["station"] in (res.get("simulation") or {}).get("alone_stops", {}):
                row["sim_alone"] = res["simulation"]["alone_stops"][t["station"]]
            rows.append(row)
        res["replay_check"] = {"label": "RECORDED OUTCOME (evaluation only; never given to the model)",
                               "rows": rows}
    return res
