"""HISTORICAL REPLAY provider - NOT LIVE.

Replays the most recent *recorded* run of a train from the downloaded NTES-derived
dataset, truncated at a chosen stop as if that moment were "now". Used to exercise
and validate the analysis pipeline without a live API key. Everything it returns is
labelled HISTORICAL_REPLAY; it must never be shown as live.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd

from railmind.data.history import RAW, _to_min
from railmind.providers.base import BaseProvider, ProviderUnavailable, StationStop, TrainStatus

NAME = "HISTORICAL_REPLAY (not live)"


@lru_cache(maxsize=1)
def _observed() -> pd.DataFrame:
    files = sorted((RAW / "data").rglob("*.parquet"))
    if not files:
        raise ProviderUnavailable("No observed dataset in data/raw/data. Run scripts/download_data.py.",
                                  "no_data", 503)
    cols = ["train_number", "train_name", "category", "start_date", "status", "seq", "station_code",
            "scheduled_arrival", "scheduled_departure", "actual_arrival", "actual_departure",
            "arrival_delay_min", "departure_delay_min"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return df[df["status"].astype(str) == "completed"]


def _fmt(ts) -> str | None:
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%d %H:%M")


class ReplayProvider(BaseProvider):
    name = NAME
    verified = True      # verified to replay recorded data; says nothing about live access

    def configured(self):
        ok = (RAW / "data").exists()
        return ok, "Replaying recorded NTES-derived runs (offline)." if ok else "dataset missing"

    def list_trains(self, limit: int = 25) -> list:
        df = _observed()
        g = df.groupby("train_number").size()
        return [str(t) for t in g[(g >= 12) & (g.index.astype(str).str[0].isin(list("123456")))].index[::7][:limit]]

    def fetch(self, train_no: str, at: float = 0.4) -> TrainStatus:
        df = _observed()
        d = df[df["train_number"].astype(str) == train_no]
        if d.empty:
            raise ProviderUnavailable(f"No recorded run for train {train_no} in the observed dataset.",
                                      "not_found", 404)
        last = d["start_date"].astype(str).max()
        d = d[d["start_date"].astype(str) == last].sort_values("seq")
        if len(d) < 4:
            raise ProviderUnavailable("Recorded run too short to replay.", "too_short", 404)
        k = min(len(d) - 2, max(1, int(at * len(d))))
        delay = d["arrival_delay_min"].fillna(d["departure_delay_min"]).astype(float).tolist()
        cur = d.iloc[k]
        now_ts = cur["actual_departure"] if pd.notna(cur["actual_departure"]) else cur["actual_arrival"]
        st = TrainStatus(train_no=train_no, provider=NAME,
                         retrieved_utc=datetime.now(timezone.utc).isoformat(),
                         train_name=str(cur["train_name"]),
                         current_station=str(cur["station_code"]), current_station_code=str(cur["station_code"]),
                         previous_station=str(d.iloc[k - 1]["station_code"]),
                         next_station=str(d.iloc[k + 1]["station_code"]),
                         delay_min=int(delay[k]) if delay[k] == delay[k] else None,
                         last_update_text=f"replay of run starting {last}; 'now' = {_fmt(now_ts)} IST")
        for i, r in enumerate(d.itertuples()):
            passed = i <= k
            st.stops.append(StationStop(
                seq=int(r.seq), passed=passed, code=str(r.station_code), name=str(r.station_code),
                sched_arr=_fmt(r.scheduled_arrival), sched_dep=_fmt(r.scheduled_departure),
                actual_arr=_fmt(r.actual_arrival) if passed else None,
                actual_dep=_fmt(r.actual_departure) if passed else None,
                delay_min=int(delay[i]) if passed and delay[i] == delay[i] else None))
        st.raw = {"replay": True, "run_start_date": last, "now_index": k,
                  "now_ist": _fmt(now_ts), "category": str(cur["category"]),
                  # what actually happened after "now" - evaluation only, never fed to the model
                  "truth_after_now": [{"station": str(r.station_code),
                                       "delay": (None if delay[i] != delay[i] else float(delay[i]))}
                                      for i, r in enumerate(d.itertuples()) if i > k]}
        st.stops_meta = {"is_live": False, "status": "replay", "segment_progress": None,
                         "is_actual_position": True}
        return st
