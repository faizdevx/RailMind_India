from unittest.mock import MagicMock

import numpy as np
import pytest

from railmind.config import Config
from railmind.providers.base import (NoProvider, ProviderUnavailable, RapidApiIrctcProvider,
                                     valid_train_no)
from railmind.sim.corridor import (Corridor, HEADWAY, POLICIES, STATION_CAP, make_scenario,
                                   perturb, run_policy)
from railmind.data.history import BUCKET_EDGES, Knowledge, QLEVELS
from railmind.data.predict import rollout
from railmind.rl.env import RailDispatchEnv
from app import create_app


class Cfg(Config):
    PROVIDER_API_KEY = "test-key"      # test fixture only, never a real credential
    MAX_CALLS_PER_MINUTE = 100
    MAX_CALLS_PER_DAY = 100


def test_train_no_validation():
    assert valid_train_no("12301")
    for bad in ("1230", "abcde", "", "123456", "12 01"):
        assert not valid_train_no(bad)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("name", list(POLICIES))
def test_safety_invariants_hold(seed, name):
    sc = make_scenario(seed)           # single line: opposing trains must never share a section
    c = Corridor(sc)
    pol = POLICIES[name]
    while not c.over():
        for i in pol(c):
            if c.check_release(i)[0]:
                c.release(i)
        for s_, ids in c.in_section.items():
            dirs = {sc.trains[j].direction for j in ids}
            assert len(dirs) <= 1, "opposing trains in one single-line section"
        for st in range(1, sc.n_stations - 1):
            assert c.occupancy(st) <= STATION_CAP
        c.advance()
    assert c.summary()["completed"] == c.n() or c.deadlocked or c.t >= 1000


def test_same_direction_headway_enforced():
    sc = make_scenario(1, single_line=False)
    c = Corridor(sc)
    i = next(i for i in range(c.n()) if sc.trains[i].direction > 0)
    s_ = c.section_of(i)
    c.last_entry[(s_, 1)] = c.t
    c.status[i] = "at_station"; c.ready_at[i] = 0
    c.sched[i][c.station[i]] = (0, 0)
    assert c.check_release(i) == (False, "headway")


def test_cannot_overtake_inside_section():
    sc = make_scenario(2, single_line=False)
    c = Corridor(sc)
    ups = [i for i in range(c.n()) if sc.trains[i].direction > 0 and sc.trains[i].origin == 0]
    if len(ups) >= 2:
        a, b = ups[:2]
        for i in (a, b):
            c.status[i] = "at_station"; c.ready_at[i] = 0; c.sched[i][0] = (0, 0)
        c.sc.actual_run[a][0] = 30
        c.sc.actual_run[b][0] = 5          # b is much faster but entered later
        assert c.release(a)
        c.t += HEADWAY
        assert c.release(b)
        assert c.arr_t[b] >= c.arr_t[a] + HEADWAY


def test_same_scenario_same_result():
    a = run_policy(make_scenario(5), POLICIES["fcfs"]).summary()
    b = run_policy(make_scenario(5), POLICIES["fcfs"]).summary()
    assert a == b


def test_perturb_keeps_trains():
    sc = make_scenario(2)
    p = perturb(sc, 9)
    assert [t.start_time for t in sc.trains] == [t.start_time for t in p.trains]


def test_env_api_and_unsafe_action_does_not_move_train():
    env = RailDispatchEnv(seed=0)
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    for _ in range(50):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    running = [i for i in range(env.c.n()) if env.c.status[i] == "running"]
    secs = [env.c.section_of(i) for i in running]
    assert len(secs) == len(set(secs))


def test_no_cancel_every_train_finishes_or_is_charged():
    c = run_policy(make_scenario(11), POLICIES["scheduled"])
    s = c.summary()
    assert s["completed"] == s["n_trains"] or s["deadlocked"] or c.t >= 400
    assert len(s["per_train"]) == 6


def test_provider_not_configured_raises():
    with pytest.raises(ProviderUnavailable):
        NoProvider().fetch("12301")
    class NoKey(Config):
        PROVIDER_API_KEY = ""
    with pytest.raises(ProviderUnavailable) as e:
        RapidApiIrctcProvider(NoKey).fetch("12301")
    assert e.value.code == "not_configured"


def _resp(code, js):
    r = MagicMock(); r.status_code = code; r.json.return_value = js
    return r


def test_provider_parser_and_missing_fields():
    s = MagicMock(); s.get.return_value = _resp(200, {"data": {"train_name": "T", "delay": "12"}})
    st = RapidApiIrctcProvider(Cfg, s).fetch("12301")
    assert st.train_name == "T" and st.delay_min == 12
    assert "current_station" in st.missing_fields      # not guessed
    assert st.raw["data"]["delay"] == "12"


@pytest.mark.parametrize("code,expect", [(429, "rate_limited"), (401, "auth"), (500, "provider_error")])
def test_provider_http_errors(code, expect):
    s = MagicMock(); s.get.return_value = _resp(code, {})
    with pytest.raises(ProviderUnavailable) as e:
        RapidApiIrctcProvider(Cfg, s).fetch("12301")
    assert e.value.code == expect


def test_local_rate_limit():
    class Tight(Cfg):
        MAX_CALLS_PER_MINUTE = 1
    s = MagicMock(); s.get.return_value = _resp(200, {"data": {}})
    p = RapidApiIrctcProvider(Tight, s)
    p.fetch("12301")
    with pytest.raises(ProviderUnavailable) as e:
        p.fetch("12301")
    assert e.value.code == "rate_limited_local"


@pytest.fixture
def client():
    return create_app(NoProvider()).test_client()


def test_api_health_and_unavailable_state(client):
    h = client.get("/api/health").get_json()
    assert h["provider"]["configured"] is False
    r = client.get("/api/trains/12301/status")
    assert r.status_code == 503 and "error" in r.get_json()


def test_api_validation(client):
    assert client.get("/api/trains/abc/status").status_code == 400
    assert client.get("/api/trains/search?q=1").status_code == 400
    assert client.post("/api/analysis", json={"train_no": "12301"}).status_code == 503
    assert client.post("/api/analysis", json={"train_no": "abc", "source": "replay"}).status_code == 400
    assert client.post("/api/analysis", json={"train_no": "12301", "source": "replay", "mc_runs": 999}).status_code == 400
    assert client.post("/api/training", json={"timesteps": 5}).status_code == 400


def _fake_kn():
    t3 = {(b,): (np.linspace(-2, 8, 11).astype("float32"), 1000) for b in range(4)}
    return Knowledge({"t1": {}, "t2": {}, "t3": t3, "lv1": {}, "lv2": {}, "train_ids": {"12301": 0},
                      "stn_ids": {"A": 0, "B": 1, "C": 2}, "timetables": {}, "meta": {}, "weekdays": {},
                      "by_station": {}, "n_rows": 0, "n_files": 0, "date_from": "", "date_to": "",
                      "cutoff": None, "built_utc": ""})


def test_rollout_uses_learned_distribution_and_flags_evidence_level():
    r = rollout(_fake_kn(), "12301", ["A", "B", "C"], 10.0, n=200, seed=1)
    assert len(r) == 2 and r[0]["evidence_level"] == 3      # pooled global fallback, flagged
    assert r[0]["p10"] <= r[0]["p50"] <= r[0]["p90"]
    assert r[1]["p50"] > r[0]["p50"] - 3                    # delays accumulate on average
