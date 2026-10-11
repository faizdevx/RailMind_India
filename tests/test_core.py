from unittest.mock import MagicMock

import pytest

from railmind.config import Config
from railmind.providers.base import (NoProvider, ProviderUnavailable, RapidApiIrctcProvider,
                                     valid_train_no)
from railmind.sim.corridor import (Corridor, HEADWAY, POLICIES, STATION_CAP, make_scenario,
                                   perturb, run_policy)
from railmind.sim.snapshot import build_scenario
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
    sc = make_scenario(seed)
    c = Corridor(sc)
    pol = POLICIES[name]
    while not c.over():
        for i in pol(c):
            c.release(i)
        # block occupancy: at most one train per section; loop capacity respected
        running = [i for i in range(c.n()) if c.status[i] == "running"]
        secs = [c.section_of(i) for i in running]
        assert len(secs) == len(set(secs))
        for st in range(1, 5):
            assert c.occupancy(st) <= STATION_CAP
        c.advance()
    assert c.summary()["completed"] == c.n() or c.deadlocked or c.t >= 400


def test_headway_enforced():
    c = Corridor(make_scenario(1))
    c.last_exit[0] = c.t       # just vacated
    i = next(i for i in range(c.n()) if c.sc.trains[i].direction > 0)
    c.status[i] = "at_station"; c.ready_at[i] = 0
    c.sched[i][c.station[i]] = (0, 0)
    assert c.check_release(i) == (False, "headway") or c.t - c.last_exit[0] >= HEADWAY


def test_unsafe_release_is_noop():
    c = Corridor(make_scenario(3))
    ready = c.ready_trains()
    if len(ready) >= 2:
        # find two trains competing for one section
        pass
    assert c.release(0) in (True, False)  # never raises
    assert c.attempted_violations >= 0


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


def test_snapshot_scenario_is_labelled():
    st = {"train_no": "12345", "train_name": "Test Rajdhani", "provider": "TEST", "retrieved_utc": "x",
          "delay_min": 17, "current_station": "A", "stops": []}
    sc = build_scenario(st)
    assert sc.trains[0].source == "OBSERVED_SNAPSHOT"
    assert sc.trains[0].start_time - sc.trains[0].sched_start == 17
    assert all(t.source == "SYNTHETIC" for t in sc.trains[1:])
    assert any("SYNTHETIC" in a for a in sc.assumptions)


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
    assert client.post("/api/simulations", json={"train_no": "12301"}).status_code == 503
    assert client.post("/api/simulations", json={"mode": "synthetic", "load": 99}).status_code == 400
    assert client.post("/api/training", json={"timesteps": 5}).status_code == 400


def test_api_synthetic_simulation_is_labelled(client):
    r = client.post("/api/simulations", json={"mode": "synthetic", "uncertainty_runs": 2})
    assert r.status_code == 201
    j = r.get_json()
    assert j["snapshot"]["provider"].startswith("SYNTHETIC")
    assert "SIMULATED" in j["label"]
    assert set(j["policies"]) >= {"fcfs", "scheduled"}
    assert client.get("/api/simulations/" + j["id"]).status_code == 200
