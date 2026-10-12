"""RailMind India - Flask application. Run: python app.py"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from railmind.config import Config
from railmind.providers.base import ProviderUnavailable, build_provider, valid_train_no
from railmind.providers.replay import ReplayProvider
from railmind.services import (Jobs, SimStore, StatusService, analyze_status, get_knowledge, get_model,
                               model_info)
from railmind.sim.corridor import SYNTHETIC_LABEL

log = logging.getLogger("railmind")

def create_app(provider=None) -> Flask:
    logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    prov = provider or build_provider()
    status_svc = StatusService(prov)
    sims, jobs = SimStore(), Jobs()
    replay = ReplayProvider()

    def err(msg, status=400, **kw):
        return jsonify({"error": msg, **kw}), status

    @app.get("/vendor/plotly.min.js")
    def plotly_js():
        import os
        import plotly
        from flask import send_from_directory
        d = os.path.join(os.path.dirname(plotly.__file__), "package_data")
        return send_from_directory(d, "plotly.min.js", max_age=86400)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "time_utc": datetime.now(timezone.utc).isoformat(),
                        "provider": prov.describe(), "model": {k: v for k, v in model_info().items() if k in ("trained", "model_version", "selected_checkpoint", "trained_utc")},
                        "knowledge": (lambda k: k.describe() if k else None)(get_knowledge()),
                        "replay": replay.describe(),
                        "rate_limit": getattr(getattr(prov, "limiter", None), "usage", lambda: None)()})

    @app.get("/api/trains/search")
    def search():
        q = (request.args.get("q") or "").strip()
        if not valid_train_no(q):
            return err("Train number must be exactly 5 digits.", 400)
        ok, why = prov.configured()
        if not ok:
            return err("Live provider not configured: " + why, 503, code="not_configured",
                       provider=prov.describe())
        # No local train directory exists (none is fabricated): validate by fetching status.
        try:
            d = status_svc.get(q)
        except ProviderUnavailable as e:
            return err(e.reason, e.http, code=e.code)
        s = d["status"]
        return jsonify({"results": [{"train_no": q, "train_name": s["train_name"]}],
                        "provider": s["provider"]})

    @app.get("/api/trains/<train_no>/status")
    def status(train_no):
        if not valid_train_no(train_no):
            return err("Train number must be exactly 5 digits.", 400)
        try:
            return jsonify(status_svc.get(train_no, force=request.args.get("force") == "1"))
        except ProviderUnavailable as e:
            return err(e.reason, e.http, code=e.code, provider=prov.describe())

    def _status_for(body):
        """Resolve a status snapshot: live provider, or recorded replay (clearly labelled)."""
        tn = str(body.get("train_no", ""))
        if not valid_train_no(tn):
            raise ValueError("train_no must be exactly 5 digits.")
        if body.get("source") == "replay":
            try:
                at = float(body.get("at", 0.4))
            except (TypeError, ValueError):
                raise ValueError("at must be a number between 0.1 and 0.9.")
            if not 0.1 <= at <= 0.9:
                raise ValueError("at must be between 0.1 and 0.9.")
            return replay.fetch(tn, at=at).as_dict(), True
        return status_svc.get(tn)["status"], False

    @app.get("/api/replay/trains")
    def replay_trains():
        try:
            return jsonify({"label": "HISTORICAL REPLAY - NOT LIVE", "trains": replay.list_trains(40)})
        except ProviderUnavailable as e:
            return err(e.reason, e.http, code=e.code)

    @app.post("/api/analysis")
    def analysis():
        body = request.get_json(silent=True) or {}
        try:
            mc = int(body.get("mc_runs", 15))
            extra = int(body.get("extra_delay", 0))
            seed = int(body.get("seed", 0))
            single = bool(body.get("single_line", False))
            use_wx = bool(body.get("use_weather", True))
        except (TypeError, ValueError):
            return err("mc_runs/extra_delay/seed must be numeric.")
        if not (1 <= mc <= 60 and 0 <= extra <= 180 and 0 <= seed < 10**6):
            return err("Parameter out of range: mc_runs 1-60, extra_delay 0-180.")
        try:
            snap, is_replay = _status_for(body)
        except ValueError as e:
            return err(str(e))
        except ProviderUnavailable as e:
            return err("No verified snapshot: " + e.reason, e.http, code=e.code)
        try:
            out = analyze_status(snap, n_mc=mc, single_line=single, extra_delay=extra, seed=seed,
                                 use_weather=use_wx)
        except ProviderUnavailable as e:
            return err(e.reason, e.http, code=e.code)
        out["source"] = "HISTORICAL_REPLAY (not live)" if is_replay else "LIVE_PROVIDER"
        sims.put(out)
        return jsonify(out), 201

    @app.get("/api/analysis/<sid>")
    def get_analysis(sid):
        s = sims.get(sid)
        return jsonify(s) if s else err("Unknown analysis id.", 404)

    @app.get("/api/weather/effects")
    def weather_effects():
        from railmind.weather import service as wxs
        from railmind.weather.openmeteo import ATTRIBUTION
        return jsonify({"effects": wxs.effects(), "attribution": ATTRIBUTION,
                        "terms": "Open-Meteo free API: non-commercial use, CC BY 4.0"})

    @app.get("/api/model/backtest")
    def backtest_report():
        import json
        from railmind.data.predict import REPORT
        from railmind.data.sim_check import OUT
        rd = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731
        return jsonify({"delay_forecast_backtest": rd(REPORT), "simulation_vs_recorded": rd(OUT)})

    @app.post("/api/training")
    def start_training():
        body = request.get_json(silent=True) or {}
        try:
            steps, seed = int(body.get("timesteps", 100_000)), int(body.get("seed", 0))
        except (TypeError, ValueError):
            return err("timesteps and seed must be integers.")
        if not (1000 <= steps <= 2_000_000):
            return err("timesteps must be 1000-2000000.")
        from railmind.rl.train import train
        if not jobs.start("training", train, steps, seed):
            return err("Training already running.", 409)
        return jsonify({"status": "started"}), 202

    @app.get("/api/training")
    def training_status():
        return jsonify({"job": jobs.state["training"], "model": model_info()})

    @app.post("/api/evaluation")
    def start_eval():
        body = request.get_json(silent=True) or {}
        try:
            n = int(body.get("scenarios", 50))
        except (TypeError, ValueError):
            return err("scenarios must be an integer.")
        if not (5 <= n <= 500):
            return err("scenarios must be 5-500.")
        from railmind.rl.evaluate import EVAL_PATH, evaluate
        import json

        def job():
            r = evaluate(n, get_model())
            EVAL_PATH.write_text(json.dumps(r, indent=2))
            return r
        if not jobs.start("evaluation", job):
            return err("Evaluation already running.", 409)
        return jsonify({"status": "started"}), 202

    @app.get("/api/evaluation")
    def eval_status():
        from railmind.rl.evaluate import EVAL_PATH
        import json
        last = json.loads(EVAL_PATH.read_text()) if EVAL_PATH.exists() else None
        return jsonify({"job": jobs.state["evaluation"], "last_saved": last})

    @app.errorhandler(404)
    def nf(e):
        return err("Not found.", 404)

    @app.errorhandler(Exception)
    def boom(e):
        log.exception("unhandled")
        return err("Internal error.", 500)

    return app


if __name__ == "__main__":
    create_app().run(host=Config.HOST, port=Config.PORT, debug=False)
