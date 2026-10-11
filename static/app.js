"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const state = { sim: null, t: 0, timer: null, tmax: 0, selected: [] };
const COLORS = ["#19c3d6", "#ffb020", "#3ddc97", "#ff5d5d", "#b38cff", "#ff8fd0", "#9fb4cc"];

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
    const p = h.provider;
    $("health").className = "pill " + (p.configured ? "ok" : "bad");
    $("health").textContent = p.configured ? "provider: " + p.name : "live provider NOT configured";
    $("providerBox").className = "box " + (p.configured ? "" : "warn");
    $("providerBox").innerHTML = p.configured
      ? `<b>Provider:</b> ${esc(p.name)} <span class="tag ${p.verified_against_live_responses ? "obs" : "calc"}">${p.verified_against_live_responses ? "verified" : "adapter not verified against live responses"}</span><br><small>${esc(p.detail)}</small>`
      : `<b>Live data unavailable.</b> ${esc(p.detail)}<br><small>No live trains are shown and none are substituted. See README: "Configuring a live provider".</small>`;
    const m = h.model;
    state.modelTrained = !!m.trained;
    $("rlTitle").innerHTML = (m.trained ? "RL-proposed (PPO, " + esc(m.model_version) + ", synthetic-trained)" : "RL-proposed (PPO) - NOT TRAINED") + ' <span class="tag sim">SIMULATED</span>';
  } catch (e) {
    $("health").className = "pill bad"; $("health").textContent = "backend unreachable";
  }
}

function renderTrain(d) {
  const s = d.status;
  const warn = (d.warnings || []).map((w) => `<div class="box warn">${esc(w)}</div>`).join("");
  const stops = (s.stops || []).slice(0, 8).map((x) =>
    `<tr><td>${esc(x.name)}</td><td>${esc(x.sched_arr || "-")}</td><td>${esc(x.actual_arr || "-")}</td><td>${esc(x.sched_dep || "-")}</td><td>${esc(x.actual_dep || "-")}</td><td>${x.delay_min ?? "-"}</td></tr>`).join("");
  return `<div class="card" data-no="${esc(s.train_no)}">
    <h4>${esc(s.train_no)} ${esc(s.train_name || "(name not supplied)")} <span class="tag obs">OBSERVED</span></h4>
    <div class="kv">
      <div>Current station</div><div>${esc(s.current_station || "not supplied")} <span class="tag obs">reported</span></div>
      <div>Previous / next</div><div>${esc(s.previous_station || "-")} / ${esc(s.next_station || "-")}</div>
      <div>Delay</div><div>${s.delay_min ?? "not supplied"} min <span class="tag obs">reported</span></div>
      <div>Provider</div><div>${esc(s.provider)}</div>
      <div>Retrieved (UTC)</div><div>${esc(s.retrieved_utc)}</div>
      <div>Data age</div><div>${d.age_seconds}s ${d.stale ? '<span class="badtxt">STALE</span>' : ""} <span class="tag calc">calculated</span></div>
      <div>Provider timestamp</div><div>${esc(s.last_update_text || "not supplied")}</div>
      <div>Route progress</div><div>${(s.stops || []).length ? "route supplied; progress not computed (position not confirmed)" : "unavailable: no route data"}</div>
    </div>
    ${stops ? `<table><tr><th>Station</th><th>Sch arr</th><th>Act arr</th><th>Sch dep</th><th>Act dep</th><th>Delay</th></tr>${stops}</table>` : ""}
    ${warn}
    <div class="controls"><button data-sim="${esc(s.train_no)}">Simulate this train</button></div>
  </div>`;
}

async function fetchTrain(no) {
  const box = $("trains");
  try {
    const d = await api("/api/trains/" + no + "/status");
    state.selected = state.selected.filter((x) => x.status.train_no !== no);
    state.selected.unshift(d);
  } catch (e) {
    const note = e.body && e.body.provider ? "" : "";
    box.insertAdjacentHTML("afterbegin", `<div class="box err"><b>Unavailable:</b> ${esc(e.message)} ${note}</div>`);
    return;
  }
  box.innerHTML = state.selected.map(renderTrain).join("");
  box.querySelectorAll("button[data-sim]").forEach((b) => b.onclick = () => runSim({ train_no: b.dataset.sim }));
}

function simParams() {
  return { seed: +$("seed").value, load: +$("load").value, extra_delay: +$("extra").value, uncertainty_runs: +$("runs").value };
}

async function runSim(extra) {
  stop();
  $("simMeta").className = "box"; $("simMeta").textContent = "Running simulation...";
  try {
    const sim = await api("/api/simulations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...simParams(), ...extra }) });
    state.sim = sim; state.t = 0;
    state.tmax = Math.max(...Object.values(sim.policies).map((p) => p.summary.minutes));
    renderSimStatic(); draw();
  } catch (e) {
    $("simMeta").className = "box err"; $("simMeta").textContent = "Simulation not started: " + e.message;
  }
}

function renderSimStatic() {
  const s = state.sim;
  const synth = s.snapshot.provider.startsWith("SYNTHETIC");
  $("simMeta").className = "box" + (synth ? " warn" : "");
  $("simMeta").innerHTML = (synth ? "<b>SYNTHETIC DATA - NOT REAL INDIAN RAILWAYS DATA</b><br>" : "") +
    `Snapshot: train ${esc(s.snapshot.train_no)} from ${esc(s.snapshot.provider)} at ${esc(s.snapshot.retrieved_utc)}; ${synth ? "assumed (synthetic)" : "observed"} delay ${s.snapshot.observed_delay_min ?? "n/a"} min.<br>
    <span class="tag sim">${esc(s.label)}</span> Sim id ${esc(s.id)}, seed ${s.scenario.seed}, load ${s.scenario.load}, runs ${s.scenario.uncertainty_runs}.`;
  $("assumptions").innerHTML = s.assumptions.map((a) => `<li>${esc(a)}</li>`).join("") + `<li>${esc(s.interpretation)}</li>`;
  // comparison table
  const P = s.policies, f = P.fcfs.summary.per_train, sc = P.scheduled.summary.per_train, pp = P.ppo ? P.ppo.summary.per_train : null;
  let rows = f.map((x, i) => {
    const obs = (x.source === "OBSERVED_SNAPSHOT" ? (s.snapshot.observed_delay_min ?? "n/a") : "-");
    const saved = pp ? (x.delay - pp[i].delay) : null;
    return `<tr><td>${esc(x.name)}<br><small>${esc(x.class)} ${x.source === "OBSERVED_SNAPSHOT" ? '<span class="tag obs">REAL</span>' : '<span class="tag sim">SYNTHETIC</span>'}</small></td>
      <td>${obs}</td><td>${sc[i].delay}</td><td>${x.delay}</td><td>${pp ? pp[i].delay : "n/a"}</td>
      <td class="${saved > 0 ? "good" : saved < 0 ? "badtxt" : ""}">${saved === null ? "n/a" : saved}</td></tr>`;
  }).join("");
  const real = P.fcfs.observed_train_delay_mean_std, saved = P.ppo && P.ppo.minutes_saved_vs_fcfs_mean_std;
  let h = `<table><tr><th>Train</th><th>Observed delay (provider, now)</th><th>Scheduled-order (pred.)</th><th>FCFS baseline (pred.)</th><th>RL / PPO (pred.)</th><th>Predicted min saved vs FCFS</th></tr>${rows}</table>`;
  h += `<p class="sub">Observed delay is the provider's current figure; predicted delays are lateness at corridor end inside the simulation - not directly comparable.
  Observed train, FCFS predicted: ${real.mean} &plusmn; ${real.std} min over ${s.scenario.uncertainty_runs} perturbed runs.
  ${saved ? `PPO minutes saved on observed train: ${saved.mean} &plusmn; ${saved.std} min (PREDICTED).` : '<b>PPO not trained: no RL comparison.</b>'}</p>`;
  const ppo = P.ppo ? P.ppo.summary : null;
  if (ppo) {
    const v = s.ppo_vs_fcfs;
    h += `<p class="sub">Weighted delay - scheduled: ${P.scheduled.summary.weighted_delay}, FCFS: ${P.fcfs.summary.weighted_delay}, PPO: ${ppo.weighted_delay}. Helped: ${esc(v.helped.join(", ") || "none")}. Harmed: ${esc(v.harmed.join(", ") || "none")}. Constraint violations executed: ${v.constraint_violations_executed} (attempted by PPO and blocked: ${v.attempted_violations_ppo}).</p>`;
  }
  $("compare").innerHTML = h;
  const bn = P.fcfs.bottleneck;
  $("explain").innerHTML = `<div class="box"><span class="tag hyp">${esc(bn.label)}</span> ${esc(bn.text)}</div>` +
    `<div class="box"><span class="tag hyp">HYPOTHESIS</span> Provider supplies no evidence on weather, maintenance restrictions or infrastructure faults; none are attributed.</div>` +
    (P.ppo ? `<div class="box"><b>RL interventions (SIMULATED):</b> compare departure order below with FCFS; the policy may hold a ready train so a higher-weight or opposing train clears the single-track block first. Holds are capped at 20 min and all safety rules are enforced by the simulator.</div>` : "");
}

function trace(tracks, names, tcur) {
  return Object.keys(tracks).map((id, k) => {
    const pts = tracks[id].filter((p) => p[0] <= tcur);
    const last = pts[pts.length - 1];
    return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), mode: "lines", name: names[id], line: { color: COLORS[k % COLORS.length], width: k === 0 ? 4 : 2, dash: names[id].includes("SYN") ? "dot" : "solid" }, hovertemplate: "%{x} min<extra>" + names[id] + "</extra>" };
  });
}

function plot(div, p, tcur) {
  const s = state.sim, names = {}; s.trains.forEach((t) => names[t.id] = t.name);
  const layout = { paper_bgcolor: "#0e141c", plot_bgcolor: "#0e141c", font: { color: "#dbe5f1", size: 11 }, margin: { l: 70, r: 10, t: 8, b: 36 },
    xaxis: { title: "simulated minutes", range: [0, state.tmax + 5], gridcolor: "#25303f" },
    yaxis: { tickmode: "array", tickvals: s.station_names.map((_, i) => i), ticktext: s.station_names, gridcolor: "#25303f", range: [-0.3, s.station_names.length - 0.7] },
    legend: { orientation: "h", y: -0.25, font: { size: 9 } },
    shapes: [{ type: "line", x0: tcur, x1: tcur, y0: 0, y1: 1, yref: "paper", line: { color: "#7aa2ff", dash: "dash" } }] };
  Plotly.react(div, trace(p.tracks, names, tcur), layout, { displayModeBar: false, responsive: true });
}

function draw() {
  const s = state.sim; if (!s) return;
  $("clock").textContent = "t = " + state.t + " min (simulated)";
  plot("plotBase", s.policies.fcfs, state.t);
  if (s.policies.ppo) plot("plotRL", s.policies.ppo, state.t);
  else Plotly.react("plotRL", [], { paper_bgcolor: "#0e141c", plot_bgcolor: "#0e141c", annotations: [{ text: "PPO NOT TRAINED - no RL result", showarrow: false, font: { color: "#ffb020" } }], xaxis: { visible: false }, yaxis: { visible: false } }, { displayModeBar: false });
  const ev = [];
  for (const [k, lbl] of [["fcfs", "FCFS"], ["ppo", "RL"]]) {
    if (!s.policies[k]) continue;
    s.policies[k].events.filter((e) => e.t <= state.t).forEach((e) => ev.push({ ...e, lbl }));
  }
  ev.sort((a, b) => a.t - b.t);
  $("events").innerHTML = ev.slice(-60).map((e) => `<div>[${String(e.t).padStart(3)}] <b>${e.lbl}</b> SIMULATED: ${esc(e.text)}</div>`).join("") || "no events yet";
}

function tick() {
  state.t = Math.min(state.tmax, state.t + (+$("speed").value));
  draw();
  if (state.t >= state.tmax) stop();
}
function play() { if (!state.sim || state.timer) return; state.timer = setInterval(tick, 250); }
function stop() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }

$("searchForm").onsubmit = (e) => { e.preventDefault(); const v = $("trainNo").value.trim(); if (!/^\d{5}$/.test(v)) { $("trains").insertAdjacentHTML("afterbegin", '<div class="box err">Train number must be 5 digits.</div>'); return; } fetchTrain(v); };
$("simSel").onclick = () => { if (!state.selected.length) { $("simMeta").className = "box warn"; $("simMeta").textContent = "Select (fetch) a real train first, or run the SYNTHETIC demo."; return; } runSim({ train_no: state.selected[0].status.train_no }); };
$("simSyn").onclick = () => runSim({ mode: "synthetic" });
$("play").onclick = play; $("pause").onclick = stop;
$("reset").onclick = () => { stop(); state.t = 0; draw(); };
loadHealth();
