"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const state = { a: null, t: 0, tmax: 0, timer: null, health: null };
const COLORS = ["#19c3d6", "#ffb020", "#3ddc97", "#ff5d5d", "#b38cff", "#ff8fd0", "#9fb4cc"];
const fx = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? "n/a" : Number(v).toFixed(d);
const sgn = (v) => (v > 0 ? "+" : "") + fx(v);
const TAG = { OBSERVED: "obs", LEARNED: "calc", SIMULATED: "sim", HYPOTHESIS: "hyp", CONTEXT: "calc", PREDICTED: "sim" };

async function api(url, opts) {
  const r = await fetch(url, opts);
  let body = null;
  try { body = await r.json(); } catch (e) { /* non-json */ }
  if (!r.ok) { const e = new Error((body && body.error) || ("HTTP " + r.status)); e.body = body; throw e; }
  return body;
}

async function loadHealth() {
  try {
    const h = await api("/api/health");
    state.health = h;
    const p = h.provider;
    $("health").className = "pill " + (p.configured ? "ok" : "bad");
    $("health").textContent = p.configured ? "live provider: " + p.name : "live provider NOT configured";
    $("providerBox").className = "box " + (p.configured ? "" : "warn");
    $("providerBox").innerHTML = p.configured
      ? `<b>Live provider:</b> ${esc(p.name)} <span class="tag ${p.verified_against_live_responses ? "obs" : "calc"}">${p.verified_against_live_responses ? "verified" : "adapter not verified against live responses"}</span><br><small>${esc(p.detail)}</small>`
      : `<b>Live data unavailable.</b> ${esc(p.detail)}<br><small>No live trains are shown and none are substituted. Choose "Historical replay" to exercise the analysis on recorded runs, or configure a provider (README).</small>`;
    const k = h.knowledge;
    $("prov").innerHTML = k ? `Learned from ${k.sections_rows.toLocaleString()} section observations, ${k.trains} trains, ${k.files} daily files (${esc(k.date_from)}..${esc(k.date_to)}).`
      : "<b>Historical knowledge not built.</b> Run scripts/download_data.py and python -m railmind.data.history.";
    api("/api/model/backtest").then((b) => {
      const d = b.delay_forecast_backtest, v = b.simulation_vs_recorded;
      let t = "";
      if (d) { const h1 = d.by_horizon_stops_ahead["1"], h5 = d.by_horizon_stops_ahead["5"];
        t += `<br>Forecast backtest on held-out dates (${d.queries} runs): MAE ${h1.mae_model_p50_min} min vs persistence ${h1.mae_persistence_min} at 1 stop ahead; ${h5.mae_model_p50_min} vs ${h5.mae_persistence_min} at 5 stops; 10-90% interval covers ${(h5.p10_p90_coverage * 100).toFixed(0)}%.`; }
      if (v) t += `<br>Simulation vs recorded outcomes (${v.stop_predictions} stop predictions, MAE min): persistence ${v.mae_min.persistence}, learned forecast ${v.mae_min.learned_p50}, simulation alone ${v.mae_min.sim_alone}, simulation with nearby trains ${v.mae_min.sim_fcfs}. <b>The corridor simulation is exploratory and not more accurate than the learned forecast.</b>`;
      $("prov").insertAdjacentHTML("beforeend", t);
    }).catch(() => {});
    const m = h.model;
    state.modelTrained = !!m.trained;
    $("rlTitle").innerHTML = (m.trained ? "RL-proposed (PPO " + esc(m.model_version) + ")" : "Scheduled-order (PPO NOT TRAINED)") + ' <span class="tag sim">SIMULATED</span>';
    if (!p.configured) { $("source").value = "replay"; onSource(); }
    loadReplayList();
  } catch (e) {
    $("health").className = "pill bad"; $("health").textContent = "backend unreachable";
  }
}

async function loadReplayList() {
  try {
    const r = await api("/api/replay/trains");
    $("replayList").innerHTML = r.trains.map((t) => `<option value="${esc(t)}">`).join("");
    if (!$("trainNo").value && r.trains.length) $("trainNo").value = r.trains[3] || r.trains[0];
  } catch (e) { /* dataset missing */ }
}

function onSource() { $("atWrap").classList.toggle("hidden", $("source").value !== "replay"); }

function params() {
  const replay = $("source").value === "replay";
  return { train_no: $("trainNo").value.trim(), source: replay ? "replay" : "live", at: (+$("at").value) / 100,
           extra_delay: +$("extra").value, single_line: $("line").value === "1", mc_runs: +$("runs").value, seed: +$("seed").value, use_weather: $("wx").value === "1" };
}

async function run() {
  const p = params();
  if (!/^\d{5}$/.test(p.train_no)) { $("msg").innerHTML = '<div class="box err">Train number must be 5 digits.</div>'; return; }
  stop(); $("msg").innerHTML = '<div class="box">Fetching snapshot and analysing...</div>';
  try {
    state.a = await api("/api/analysis", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) });
    $("msg").innerHTML = "";
    const sim = state.a.simulation;
    state.t = 0;
    state.tmax = sim && sim.available ? Math.max(...Object.values(sim.policies).map((x) => x.summary.minutes)) : 0;
    renderAll(); draw();
  } catch (e) {
    $("msg").innerHTML = `<div class="box err"><b>Unavailable:</b> ${esc(e.message)}</div>`;
  }
}

function renderAll() {
  const a = state.a, replay = a.source.startsWith("HISTORICAL");
  const rows = (a.learned_forecast.stops || []).map((r) =>
    `<tr><td>${esc(r.name ? r.name + " (" + r.station + ")" : r.station)}</td><td>${esc(r.sched_arr || "-")}</td><td>${r.provider_estimate === null || r.provider_estimate === undefined ? "-" : sgn(r.provider_estimate) + (r.provenance === "predicted" ? ' <span class="tag calc">provider-predicted</span>' : ' <span class="tag obs">reported</span>')}</td><td>${sgn(r.p50)}</td><td>${r.p50_wx === undefined ? "-" : sgn(r.p50_wx) + (r.wx_adjust_cum ? " <small>(" + sgn(r.wx_adjust_cum) + ")</small>" : "")}</td><td>${sgn(r.p10)} .. ${sgn(r.p90)}</td><td>${r.weather ? (r.weather_severity ? '<span class="tag hyp">' + esc(r.weather) + '</span>' : esc(r.weather)) : "-"}</td><td>${r.evidence_level === 1 ? "train+section" : r.evidence_level === 2 ? "section pooled" : "global fallback"} (n=${r.support_n})</td></tr>`).join("");
  $("train").innerHTML = `<div class="card">
    <h4>${esc(a.train_no)} ${esc(a.train_name || "(name not supplied)")} <span class="tag ${replay ? "calc" : "obs"}">${replay ? "RECORDED REPLAY - NOT LIVE" : "OBSERVED"}</span></h4>
    <div class="kv">
      <div>Source</div><div>${esc(a.provider)}</div>
      <div>Current station</div><div>${esc(a.current_station || "n/a")} <span class="tag obs">reported</span></div>
      <div>Delay now</div><div>${fx(a.observed_delay_min)} min <span class="tag obs">reported</span></div>
      <div>Snapshot time (IST)</div><div>${esc(a.now_ist)}</div>
      <div>Retrieved (UTC)</div><div>${esc(a.retrieved_utc)}</div>
      <div>Position</div><div>${replay ? "recorded stop" : esc((a.tracking && a.tracking.location_status) || "provider-reported") + " at " + esc((a.station_names && a.station_names[a.current_station]) || a.current_station) + "; not independently confirmed"}</div>
      <div>Provider tracking</div><div>${esc((a.tracking && a.tracking.tracking_mode) || "n/a")}, live=${esc(String(a.tracking && a.tracking.is_live))}</div>
      <div>Anchor for model</div><div>${esc(a.anchor_station || "n/a")} <small>(last timetable halt used as corridor start)</small></div>
    </div>
    ${a.warning ? `<div class="box warn">${esc(a.warning)}</div>` : ""}
  </div>`;
  const secs = (a.observed.sections || []).map((s) =>
    `<tr><td>${esc(s.from)} &rarr; ${esc(s.to)}</td><td>${fx(s.delay_from)} &rarr; ${fx(s.delay_to)}</td><td>${sgn(s.delta)}</td><td>${s.history && s.history.n >= 8 ? "p" + s.history.pct + " (typ " + sgn(s.history.median) + ")" : "no history"}</td></tr>`).join("");
  $("causes").innerHTML = `<h3>Why is it running ${a.observed_delay_min > 0 ? "late" : "this way"}? <span class="tag obs">evidence-labelled</span></h3>` +
    (a.causes || []).map((c) => `<div class="box"><span class="tag ${TAG[c.label] || "calc"}">${esc(c.label)}</span> <b>${esc(c.cause)}</b><br>${esc(c.text)}</div>`).join("") +
    (secs ? `<h3>Observed delay change per section vs history</h3><table><tr><th>Section</th><th>Delay (min)</th><th>Change</th><th>Percentile in history</th></tr>${secs}</table>` : "");
  $("forecast").innerHTML = rows
    ? `<table><tr><th>Stop</th><th>Scheduled</th><th>Provider delay</th><th>Learned median</th><th>+ weather</th><th>10-90 %</th><th>Weather (expected time)</th><th>Evidence</th></tr>${rows}</table><p class="sub">${esc(a.learned_forecast.method)}.</p>`
    : "<p class='sub'>No forecast available.</p>";
  const W = a.weather;
  if (W) {
    const wr = W.rows.filter((r) => r.phase === "ahead" || r.severity).map((r) =>
      `<tr><td>${esc((a.station_names && a.station_names[r.code]) || r.code)} (${esc(r.code)})</td><td>${esc((r.time || "").replace("T", " "))}</td><td>${r.phase === "passed" ? "passed" : "expected"}</td><td>${r.cond ? esc(r.text) : "unavailable"}</td><td>${r.cond ? (r.cond.temp_c === null ? "-" : fx(r.cond.temp_c, 0) + " C") : "-"} / ${r.cond && r.cond.precip_mm !== null ? fx(r.cond.precip_mm, 1) + " mm" : "-"} / ${r.cond && r.cond.gust_kmh !== null ? fx(r.cond.gust_kmh, 0) + " km/h" : "-"}</td></tr>`).join("");
    const L = W.learned;
    $("weather").innerHTML = `<h3>Weather along the route <span class="tag calc">CONTEXT</span></h3>
      <p class="sub">${esc(W.attribution)}. Past stops: reanalysis; ahead: forecast at expected time. Learned weather model: ${W.model_accepted ? "ACCEPTED (beat weather-free baseline on held-out dates)" : "not accepted / not available - no weather adjustment applied"}${L ? "; held-out error " + fx(L.heldout_mae_min.baseline_median, 2) + " &rarr; " + fx(L.heldout_mae_min.baseline_plus_weather, 2) + " min per section" : ""}.</p>
      <table><tr><th>Stop</th><th>Time</th><th></th><th>Conditions</th><th>Temp / rain / gust</th></tr>${wr || '<tr><td colspan="5">no adverse conditions and no stops with weather</td></tr>'}</table>`;
  } else {
    $("weather").innerHTML = "";
  }
  const sim = a.simulation;
  if (!sim || !sim.available) {
    $("simMeta").className = "box warn"; $("simMeta").textContent = "Simulation unavailable: " + ((sim && sim.reason) || a.warning || "n/a");
    $("compare").innerHTML = ""; $("explain").innerHTML = ""; $("assumptions").innerHTML = (a.limitations || []).map((x) => `<li>${esc(x)}</li>`).join("");
    return;
  }
  $("simMeta").className = "box";
  $("simMeta").innerHTML = `<span class="tag sim">${esc(sim.label)}</span> Snapshot ${esc(a.now_ist)} IST (t=0). ${sim.trains.length - 1} nearby trains from timetables, ${sim.mc_runs} Monte-Carlo runs.`;
  const P = sim.policies, t0 = (k) => P[k] ? P[k].target_delay_mc : null;
  const cell = (m) => m ? `${fx(m.mean, 1)} &plusmn; ${fx(m.std, 1)} <small>(${fx(m.p10)}..${fx(m.p90)})</small>` : "n/a";
  const sv = (k) => P[k] && P[k].minutes_saved_vs_fcfs_mc ? `${fx(P[k].minutes_saved_vs_fcfs_mc.mean, 1)} &plusmn; ${fx(P[k].minutes_saved_vs_fcfs_mc.std, 1)}` : "-";
  $("compare").innerHTML = `<table><tr><th></th><th>Delay at corridor end (min)</th><th>Predicted saved vs FCFS</th><th>Violations executed</th></tr>
    <tr><td>Observed now (${replay ? "recorded" : "provider"})</td><td>${fx(a.observed_delay_min)} <span class="tag obs">OBSERVED</span></td><td>-</td><td>-</td></tr>
    <tr><td>Scheduled-order baseline</td><td>${cell(t0("scheduled"))}</td><td>${sv("scheduled")}</td><td>0</td></tr>
    <tr><td>FCFS baseline</td><td>${cell(t0("fcfs"))}</td><td>-</td><td>0</td></tr>
    <tr><td>${P.ppo ? "RL / PPO" : "RL / PPO <b>(not trained)</b>"}</td><td>${P.ppo ? cell(t0("ppo")) : "n/a"}</td><td>${P.ppo ? sv("ppo") : "n/a"}</td><td>${P.ppo ? "0 (attempted & blocked: " + P.ppo.attempted_violations + ")" : "-"}</td></tr></table>
    <p class="sub">Mean &plusmn; std (10th..90th percentile) over ${sim.mc_runs} runs on identical sampled scenarios. Alone-on-line delay in simulation: ${fx(sim.alone_delay)} min.</p>`;
  let ex = "";
  if (sim.bottleneck) ex += `<div class="box"><span class="tag sim">${esc(sim.bottleneck.label)}</span> Most simulated waiting in section ${esc(sim.bottleneck.section)} (${sim.bottleneck.wait_train_min} train-min).</div>`;
  ex += (sim.interventions.length ? sim.interventions.map((i) =>
    `<div class="box"><span class="tag sim">SIMULATED</span> ${esc(sim.best_policy.toUpperCase())}: <b>${esc(i.train)}</b> departs ${esc(i.station)} ${i.shift_min > 0 ? i.shift_min + " min later" : -i.shift_min + " min earlier"} than FCFS (t=${i.baseline_dep_min} &rarr; ${i.policy_dep_min}).</div>`).join("")
    : `<div class="box">No decision differs from FCFS by 2+ min in this scenario (${esc(sim.compared_policy_note)}).</div>`);
  ex += `<div class="box">Helped (&gt;1 min): ${esc(sim.helped.join(", ") || "none")}. Harmed (&gt;1 min): ${esc(sim.harmed.join(", ") || "none")}.</div>`;
  $("explain").innerHTML = ex;
  $("assumptions").innerHTML = sim.assumptions.concat(a.limitations || []).map((x) => `<li>${esc(x)}</li>`).join("");
  const rc = a.replay_check;
  $("replayCheck").innerHTML = rc && rc.rows.length ? `<h3>What actually happened <span class="tag obs">RECORDED</span></h3><p class="sub">${esc(rc.label)}</p><table><tr><th>Stop</th><th>Actual delay</th><th>Learned p50 (10..90)</th><th>Persistence</th><th>Sim FCFS median</th></tr>` +
    rc.rows.map((r) => `<tr><td>${esc(r.station)}</td><td>${fx(r.actual_delay)}</td><td>${fx(r.forecast_p50, 1)} (${fx(r.forecast_p10)}..${fx(r.forecast_p90)})</td><td>${fx(r.persistence)}</td><td>${fx(r.sim_fcfs, 1)}</td></tr>`).join("") + "</table>" : "";
}

function trace(tracks, names, tcur) {
  return Object.keys(tracks).map((id, k) => {
    const pts = tracks[id].filter((p) => p[0] <= tcur);
    return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), mode: "lines", name: names[id],
      line: { color: COLORS[k % COLORS.length], width: k === 0 ? 4 : 2, dash: names[id] && k > 0 ? "dot" : "solid" }, hovertemplate: "%{x} min<extra>" + names[id] + "</extra>" };
  });
}

function plot(div, p, tcur) {
  const s = state.a.simulation, names = {}; s.trains.forEach((t) => names[t.id] = t.name);
  const layout = { paper_bgcolor: "#0e141c", plot_bgcolor: "#0e141c", font: { color: "#dbe5f1", size: 11 }, margin: { l: 56, r: 10, t: 8, b: 36 },
    xaxis: { title: "simulated minutes from snapshot", range: [-30, state.tmax + 5], gridcolor: "#25303f" },
    yaxis: { tickmode: "array", tickvals: s.station_names.map((_, i) => i), ticktext: s.station_names, gridcolor: "#25303f", range: [-0.3, s.station_names.length - 0.7] },
    legend: { orientation: "h", y: -0.3, font: { size: 9 } },
    shapes: [{ type: "line", x0: tcur, x1: tcur, y0: 0, y1: 1, yref: "paper", line: { color: "#7aa2ff", dash: "dash" } }] };
  Plotly.react(div, trace(p.tracks, names, tcur), layout, { displayModeBar: false, responsive: true });
}

function draw() {
  const a = state.a; if (!a || !a.simulation || !a.simulation.available) return;
  const s = a.simulation;
  $("clock").textContent = "t = " + state.t + " min (simulated)";
  plot("plotBase", s.policies.fcfs, state.t);
  plot("plotRL", s.policies[s.best_policy], state.t);
  const ev = [];
  for (const [k, lbl] of [["fcfs", "FCFS"], [s.best_policy, s.best_policy.toUpperCase()]]) {
    s.policies[k].events.filter((e) => e.t <= state.t).forEach((e) => ev.push({ ...e, lbl }));
  }
  ev.sort((x, y) => x.t - y.t);
  $("events").innerHTML = ev.slice(-60).map((e) => `<div>[${String(e.t).padStart(4)}] <b>${esc(e.lbl)}</b> SIMULATED: ${esc(e.text)}</div>`).join("") || "no events yet";
}

function tick() { state.t = Math.min(state.tmax, state.t + (+$("speed").value)); draw(); if (state.t >= state.tmax) stop(); }
function play() { if (!state.a || state.timer) return; state.timer = setInterval(tick, 200); }
function stop() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }

$("source").onchange = onSource;
$("go").onclick = run;
$("rerun").onclick = run;
$("trainNo").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
$("play").onclick = play; $("pause").onclick = stop;
$("reset").onclick = () => { stop(); state.t = 0; draw(); };
onSource();
loadHealth();
