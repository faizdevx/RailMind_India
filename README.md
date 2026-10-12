# RailMind India

Dashboard that, for one Indian Railways train, (1) takes its observed status, (2) compares it with
delay patterns **learned from ~16 million historical per-stop observations**, (3) explains where the
delay was gained, with evidence labels, (4) forecasts the delay ahead with uncertainty, and (5) runs a
**simulated** what-if around nearby trains with FCFS / scheduled-order baselines and a PPO policy.

> Not railway control software. It never sends commands to infrastructure. Simulated gains do not
> prove real-world punctuality gains.

![RailMind dashboard - live railradar data, train 12626, 2026-10-12 17:04 IST](docs/screenshots/railmind-live-dashboard.png)

**Screenshot:** genuine capture of the running app using the **live railradar provider** for train
12626 (Kerala Express) at 2026-10-12 17:04 IST: provider-reported position "departed Kalamasseri",
delay 47 min, tracking mode `real-time`. Rows marked *provider-predicted* are the provider's own
estimates, not confirmed observations. Left: observed facts and evidence-labelled causes; right:
learned forecast, simulated what-if and measured accuracy of each method.

## What is real, what is not

| Item | Status |
|---|---|
| Historical data | **Real**: Hugging Face [`pjay-in/indian-railways-delays`](https://huggingface.co/datasets/pjay-in/indian-railways-delays) (cc-by-4.0). 370 daily files, 2025-02-08 .. 2026-10-11, 16.3 M section observations, 9 422 trains. An observed NTES-collected config (2026-10) plus a 2025-02..2026-02 config whose "actual" times are scheduled + reported delay. Station *names* are not in the data (codes only). |
| Live status | **Verified once** (2026-10-12) with the `railradar` adapter against real responses for trains 12301 and 12626 (key kept only in the git-ignored `.env`). Free sandbox reportedly 100 requests/month; local budget caps and a 120 s cache apply. The `rapidapi_irctc` adapter remains unverified. NTES has no documented public API or data terms, so it is not scraped. |
| Replay | `source=replay` re-plays the latest recorded run of a train, truncated at a stop ("now"). Labelled `HISTORICAL_REPLAY (not live)`. The recorded future is shown only afterwards, for evaluation. |
| Nearby trains | From historical timetables of the same stations (weekday matched). Whether they run today and where they are right now is **not known**; their initial delays are sampled from history. |
| Track layout | **Assumed** (double line, loop capacity 3, 2-min headway; single line optional). Not in any data. |

## How it works

1. `railmind/data/history.py` learns, per (train, section, incoming-delay bucket), the empirical
   distribution of delay change, with pooled fall-backs (train+section, section, global).
2. `railmind/data/predict.py` rolls delay forward by Monte Carlo: p10/p50/p90 per stop, with the
   evidence level shown.
3. `railmind/analysis.py` compares each observed section and halt with its historical percentile and
   labels causes `OBSERVED`, `LEARNED`, `SIMULATED` or `HYPOTHESIS`. Weather, maintenance, signal faults
   and rake linkage have no data feed and are only ever `HYPOTHESIS`.
4. `railmind/data/scenario.py` builds a corridor of at most 6 stops with up to 5 nearby trains;
   `railmind/sim/corridor.py` enforces headway, no overtaking inside a section, loop capacity and
   (single line) exclusive sections, independently of reward. Holds are capped at 20 min.
5. `railmind/rl/`: Gymnasium env + PPO (Stable-Baselines3). Warm start by imitating FCFS, then PPO;
   checkpoint chosen on validation scenarios. Target trains with number % 5 == 0 are held out.

## Install and run

```bash
pip install -r requirements.txt
python scripts/download_data.py            # ~172 MB, cc-by-4.0, to data/raw (git-ignored)
python -m railmind.data.history            # learn patterns -> data/processed (~2-3 min)
python app.py                              # http://127.0.0.1:5000  (RAILMIND_PORT to change)
```

Pick **Historical replay** or **Live provider**, enter a train number (the datalist offers known
trains), press *Fetch & analyse*. Controls: extra delay, single/double line, MC runs, seed, play, pause,
reset, speed.

### Configuring a live provider
Create an account and key with your chosen provider, then in `.env` (copy `.env.example`):
`RAILMIND_PROVIDER=railradar`, `RAILMIND_PROVIDER_API_KEY=...`. Budgets:
`RAILMIND_MAX_CALLS_PER_MINUTE`, `RAILMIND_MAX_CALLS_PER_DAY`, `RAILMIND_MIN_REFRESH_SECONDS`
(responses are cached). Verified response fields: `trainName`, `delayMinutes`, `currentLocation`,
`route[]` (`scheduledArrival`, `actualArrival`, `delayArrival`, `status`, `provenance`), `trackingMode`.
`.env` is git-ignored; never commit the key.

### API
`GET /api/health` - `GET /api/trains/<no>/status` - `GET /api/replay/trains` -
`POST /api/analysis` (`train_no`, `source` live|replay, `at`, `extra_delay`, `single_line`, `mc_runs`,
`seed`) - `GET /api/analysis/<id>` - `GET /api/model/backtest` - `POST|GET /api/training` -
`POST|GET /api/evaluation`.

### Reproduce the measurements
```bash
python -m railmind.data.history --cutoff 2025-12-01   # learn only from before the cutoff
python -m railmind.data.predict                       # forecast backtest on later dates
python -m railmind.rl.train --timesteps 300000        # PPO, about 5 min on CPU
python -m railmind.rl.evaluate --scenarios 150        # held-out target trains
python -m railmind.data.sim_check                     # simulation vs recorded outcomes
```

## Measured results (committed runs in `models/`)

**Delay forecast.** Learned before 2025-12-01, tested on 6 000 runs from later dates (74 files,
2025-12 .. 2026-10). Mean absolute error of the median forecast, minutes:

| Stops ahead | Learned | Persistence (current delay) | 10-90 % interval coverage |
|---|---|---|---|
| 1 | 3.77 | 5.52 | 81.7 % |
| 3 | 7.32 | 11.08 | 85.1 % |
| 5 | 10.08 | 15.96 | 84.3 % |

**Simulation vs recorded outcomes.** 150 replays of observed 2026-10 runs, 488 stop predictions, MAE in
minutes: persistence 18.29, learned forecast **15.62**, simulation alone 16.76, simulation with nearby
trains 19.88. **The corridor simulation is not more accurate than the learned forecast**, and its
nearby-train interaction currently adds error. Treat it as an exploratory what-if, not a forecast.

**RL (PPO v2).** 150 held-out scenarios, weighted delay (mean; std about 755): FCFS 553.9,
scheduled-order 557.6, PPO 550.7. Target-train mean delay 81.0 / 79.3 / 80.1 min. PPO helped 28 and
harmed 42 train-instances (more than 1 min). Differences are well inside noise: **no demonstrated RL
improvement**. PPO requests blocked as unsafe: 4 336 (none executed). Not validated against real
operations.

## Live example (2026-10-12 17:04 IST, train 12626)
Provider: 47 min late, departed Kalamasseri. Observed history: delay already 112 min at Mathura, peaked
at 143 min (Nagpur..Chandrapur section +32 and +30, p90 and p80 of history), then recovered. Learned
forecast median +46..+50 min for the next 7 halts (some stops use pooled or global fall-back evidence,
shown in the table). Simulated alternatives: no decision differs from FCFS (no conflicts found).
The observed halts at Gudur (+20) and Katpadi (+13) are reported facts; their cause is not in the data.

## Limitations
- Live adapter checked on two trains only; some forecast stops fall back to pooled statistics.
- In the "derived" config delay is one number per stop; dwell/run split is inferred.
- No weather, maintenance, signalling or rake-link data; causes beyond observed deltas are hypotheses.
- Simulation spread is large for late trains (Monte-Carlo std ~64 min in the live example).
- Corridor at most 6 stops / about 5 h; train class inferred from category; no freight in the data.
- Sources and provider research: [docs/SOURCES.md](docs/SOURCES.md).

## Tests
`python -m pytest -q`: 36 passed (safety invariants, headway and no-overtake, determinism, provider
parsing and errors, rate limiting, API validation and honest unavailable states, forecast roll-forward).
