"""RailMind India - Flask application. Run: python app.py"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from railmind.config import Config
from railmind.providers.base import ProviderUnavailable, build_provider, valid_train_no
from railmind.services import Jobs, SimStore, StatusService, get_model, model_info, simulate
from railmind.sim.corridor import SYNTHETIC_LABEL

log = logging.getLogger("railmind")

SYNTHETIC_STATUS = {
    "train_no": "SYNTH", "provider": "SYNTHETIC (development only)",
    "retrieved_utc": None, "train_name": "Synthetic Express", "current_station": None,
    "delay_min": 15, "stops": [], "missing_fields": [], "label": SYNTHETIC_LABEL}


def create_app(provider=None) -> Flask:
    logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    prov = provider or build_provider()
    status_svc = StatusService(prov)
    sims, jobs = SimStore(), Jobs()

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
                        "provider": prov.describe(), "model": model_info(),
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

    @app.post("/api/simulations")
    def start_sim():
        body = request.get_json(silent=True) or {}
        try:
            seed = int(body.get("seed", 7))
            load = float(body.get("load", 1.0))
            extra = int(body.get("extra_delay", 0))
            runs = int(body.get("uncertainty_runs", 20))
        except (TypeError, ValueError):
            return err("seed/load/extra_delay/uncertainty_runs must be numeric.")
        if not (0.25 <= load <= 2.0 and 0 <= extra <= 120 and 1 <= runs <= 50 and 0 <= seed < 10**6):
            return err("Parameter out of range: load 0.25-2, extra_delay 0-120, uncertainty_runs 1-50.")
        if body.get("mode") == "synthetic":
            snap = dict(SYNTHETIC_STATUS, retrieved_utc=datetime.now(timezone.utc).isoformat())
        else:
            tn = str(body.get("train_no", ""))
            if not valid_train_no(tn):
                return err("train_no must be exactly 5 digits (or mode='synthetic').")
            try:
                snap = status_svc.get(tn)["status"]
            except ProviderUnavailable as e:
                return err("No verified snapshot: " + e.reason, e.http, code=e.code)
        sim = simulate(snap, seed=seed, load=load, extra_delay=extra, n_runs=runs)
        sims.put(sim)
        return jsonify(sim), 201

    @app.get("/api/simulations/<sid>")
    def get_sim(sid):
        s = sims.get(sid)
        return jsonify(s) if s else err("Unknown simulation id.", 404)

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
        return jsonify({"status": "started", "label": SYNTHETIC_LABEL}), 202

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
