/* RainCast frontend – vanilla JS, talks to the Python API in /api */
(() => {
  "use strict";

  const raw = window.__API_BASE__ || "";
  const API = raw.startsWith("__PORT_") ? "" : raw.replace(/\/$/, "");
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const state = { place: null, live: null, last: null, charts: {}, rainLevel: 0.25, storm: false };

  // ------------------------------------------------------------ fields
  const GROUPS = [
    ["Temperature", [
      ["MinTemp", "Min temp", -5, 45, 0.5, "°C", 24],
      ["MaxTemp", "Max temp", 0, 50, 0.5, "°C", 31],
      ["Temp9am", "Temp 9 am", -5, 45, 0.5, "°C", 26],
      ["Temp3pm", "Temp 3 pm", -5, 50, 0.5, "°C", 30],
    ]],
    ["Humidity & pressure", [
      ["Humidity9am", "Humidity 9 am", 0, 100, 1, "%", 75],
      ["Humidity3pm", "Humidity 3 pm", 0, 100, 1, "%", 62],
      ["Pressure9am", "Pressure 9 am", 980, 1040, 0.1, "hPa", 1010],
      ["Pressure3pm", "Pressure 3 pm", 980, 1040, 0.1, "hPa", 1008],
    ]],
    ["Wind", [
      ["WindGustSpeed", "Max gust", 0, 120, 1, "km/h", 35],
      ["WindGustDir", "Gust direction", 0, 359, 1, "°", 225],
      ["WindSpeed9am", "Wind 9 am", 0, 80, 1, "km/h", 10],
      ["WindSpeed3pm", "Wind 3 pm", 0, 80, 1, "km/h", 16],
    ]],
    ["Cloud & rain", [
      ["Cloud9am", "Cloud 9 am", 0, 8, 0.5, "okta", 5],
      ["Cloud3pm", "Cloud 3 pm", 0, 8, 0.5, "okta", 5],
      ["Rainfall", "Rain last 24 h", 0, 100, 0.1, "mm", 0],
    ]],
  ];
  const FIELD_META = Object.fromEntries(GROUPS.flatMap(([, f]) => f).map((f) => [f[0], f]));

  function buildForm() {
    const form = $("#manualForm");
    form.innerHTML = "";
    for (const [title, fields] of GROUPS) {
      const t = document.createElement("div");
      t.className = "fieldset-title";
      t.textContent = title;
      form.appendChild(t);
      for (const [key, label, min, max, step, unit, def] of fields) {
        const wrap = document.createElement("div");
        wrap.className = "field";
        wrap.innerHTML = `<label class="field-label" for="f_${key}"><span>${label}</span><output id="o_${key}"></output></label>
          <input type="range" id="f_${key}" name="${key}" min="${min}" max="${max}" step="${step}" value="${def}" data-unit="${unit}">`;
        form.appendChild(wrap);
      }
    }
    $$("input[type=range]", form).forEach((el) => {
      const sync = () => ($("#o_" + el.name).textContent = `${fmt(el.value)} ${el.dataset.unit}`);
      el.addEventListener("input", sync);
      el.addEventListener("change", () => schedulePredict());
      sync();
    });
  }
  const fmt = (v) => (Math.abs(v) >= 100 ? Number(v).toFixed(0) : Number(v).toFixed(1).replace(/\.0$/, ""));

  function setForm(values) {
    for (const [k, v] of Object.entries(values)) {
      const el = $("#f_" + k);
      if (!el || v == null) continue;
      el.value = v;
      el.dispatchEvent(new Event("input"));
    }
  }
  function readForm() {
    const out = {};
    $$("#manualForm input[type=range]").forEach((el) => (out[el.name] = Number(el.value)));
    return out;
  }

  // ------------------------------------------------------------ api
  async function api(path, opts = {}) {
    const res = await fetch(`${API}/api/${path}`, {
      headers: { "Content-Type": "application/json" }, ...opts,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }
  function toast(msg, isError = false) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "toast show" + (isError ? " error" : "");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => (t.className = "toast"), 3600);
  }

  // ------------------------------------------------------------ live
  async function loadPlace(place) {
    state.place = place;
    $("#placeLabel").textContent = "Loading live weather…";
    $("#verdictPlace").textContent = placeName(place);
    try {
      const live = await api(`weather?lat=${place.latitude}&lon=${place.longitude}`);
      state.live = live;
      setForm(live.features);
      $("#placeLabel").textContent = `${placeName(place)} · observed ${live.observed_at.replace("T", " ")}`;
      const c = live.current || {};
      $("#nowGrid").innerHTML = [
        ["Temperature", `${c.temperature_2m ?? "—"} °C`], ["Humidity", `${c.relative_humidity_2m ?? "—"}%`],
        ["Pressure", `${c.pressure_msl != null ? Math.round(c.pressure_msl) : "—"} hPa`], ["Cloud cover", `${c.cloud_cover ?? "—"}%`],
        ["Wind", `${c.wind_speed_10m ?? "—"} km/h`], ["Rain now", `${c.precipitation ?? 0} mm`],
      ].map(([k, v]) => `<div class="now"><span>${k}</span><b>${v}</b></div>`).join("");
      render(live.prediction);
      renderHistory(live);
    } catch (e) {
      $("#placeLabel").textContent = "";
      toast(e.message, true);
    }
  }
  const placeName = (p) => [p.name, p.admin1, p.country].filter(Boolean).filter((v, i, a) => a.indexOf(v) === i).join(", ");

  async function search(q) {
    const ul = $("#results");
    ul.innerHTML = "";
    try {
      const { results } = await api(`weather?q=${encodeURIComponent(q)}`);
      if (!results.length) { toast("No places found"); return; }
      for (const r of results) {
        const li = document.createElement("li");
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = placeName(r);
        b.onclick = () => { ul.innerHTML = ""; loadPlace(r); };
        li.appendChild(b);
        ul.appendChild(li);
      }
    } catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------ predict
  let pending;
  function schedulePredict() { clearTimeout(pending); pending = setTimeout(runManual, 250); }
  async function runManual() {
    const body = readForm();
    const clim = state.live && state.live.climatology && state.live.climatology.rain_frequency;
    if (clim != null) body.climatology = clim;
    const btn = $("#predictBtn");
    btn.disabled = true;
    try { render(await api("predict", { method: "POST", body: JSON.stringify(body) })); }
    catch (e) { toast(e.message, true); }
    finally { btn.disabled = false; }
  }

  function render(r) {
    if (!r) return;
    state.last = r;
    const p = r.probability;
    $("#probValue").textContent = Math.round(p * 100);
    $("#gauge").setAttribute("aria-label", `Rain probability ${Math.round(p * 100)} percent`);
    $("#waterGroup").style.transform = `translateY(${166 - p * 150}px)`;
    const type = r.rain_type;
    $("#verdictTitle").textContent = r.will_rain ? `${type.label} expected` : r.condition;
    $("#verdictSub").textContent = r.will_rain
      ? `${r.condition}. About ${fmt(type.effective_mm)} mm likely over the next 24 hours.`
      : `Below the ${Math.round(r.threshold * 100)}% decision threshold tuned by the genetic algorithm.`;
    if (state.place) $("#verdictPlace").textContent = `${placeName(state.place)} · tomorrow`;

    const chips = [
      [r.will_rain ? "Rain: yes" : "Rain: no", r.will_rain ? "warn" : ""],
      [`Intensity: ${type.intensity}`, ""],
      [`If it rains: ~${fmt(r.expected_mm_if_rain)} mm`, ""],
      [`Convective index ${type.convective_index}`, type.thunderstorm ? "danger" : ""],
    ];
    if (r.components.climatology_prior != null) chips.push([`Local history: ${Math.round(r.components.climatology_prior * 100)}% rainy days`, ""]);
    if (state.live && state.live.reference_forecast && state.live.reference_forecast.precip_probability_max != null)
      chips.push([`Open-Meteo forecast: ${state.live.reference_forecast.precip_probability_max}%`, ""]);
    $("#chips").innerHTML = chips.map(([t, c]) => `<span class="chip-pill ${c}">${t}</span>`).join("");

    const c = r.components, w = c.weights;
    bars("#components", [
      [`Gradient boosting · w ${w.gbm}`, c.gradient_boosting],
      [`Logistic regression · w ${w.logistic_regression}`, c.logistic_regression],
      [`Fuzzy logic · w ${w.fuzzy}`, c.fuzzy_logic],
      ["Hybrid result", p],
    ], true);
    const mem = Object.entries(type.memberships).filter(([, v]) => v > 0);
    const topKey = Object.entries(type.memberships).sort((a, b) => b[1] - a[1])[0][0];
    bars("#memberships", (mem.length ? mem : [["No rain", 1]]).map(([k, v]) => [k, v, k === topKey]));
    $("#rules").innerHTML = (r.fuzzy.fired_rules || []).map((rule) =>
      `<li>IF ${Object.entries(rule.if).map(([k, v]) => `${k.replace("_", " ")} is <b>${v}</b>`).join(" AND ")} THEN rain is <b>${rule.then.replace("_", " ")}</b> (${rule.strength})</li>`
    ).join("") || "<li>No rule fired strongly; the system fell back to the base rate.</li>";

    state.rainLevel = p;
    state.storm = type.thunderstorm;
  }

  function bars(sel, rows, pct = false) {
    $(sel).innerHTML = rows.map(([label, v, top]) => `
      <div class="bar ${top ? "top" : ""}"><div class="bar-head"><span>${label}</span><b>${pct ? Math.round(v * 100) + "%" : v.toFixed(2)}</b></div>
      <div class="bar-track"><div class="bar-fill" data-w="${Math.max(2, v * 100)}"></div></div></div>`).join("");
    requestAnimationFrame(() => $$(sel + " .bar-fill").forEach((el) => (el.style.width = el.dataset.w + "%")));
  }

  // ------------------------------------------------------------ charts
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  function chartDefaults() {
    if (!window.Chart) return false;
    Chart.defaults.color = css("--muted");
    Chart.defaults.borderColor = css("--border");
    Chart.defaults.font.family = "Satoshi, system-ui, sans-serif";
    return true;
  }
  function mk(id, cfg) {
    if (state.charts[id]) state.charts[id].destroy();
    state.charts[id] = new Chart(document.getElementById(id), cfg);
  }
  function renderHistory(live) {
    if (!chartDefaults()) { setTimeout(() => renderHistory(live), 300); return; }
    const acc = css("--accent"), w1 = css("--water-1"), warn = css("--warn");
    const base = { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 10 } } } };
    const h = live.history_7d;
    mk("histChart", {
      data: {
        labels: h.map((d) => d.date.slice(5)),
        datasets: [
          { type: "bar", label: "Rain (mm)", data: h.map((d) => d.rain_mm), backgroundColor: w1, borderRadius: 4, yAxisID: "y" },
          { type: "line", label: "Max °C", data: h.map((d) => d.tmax), borderColor: warn, pointRadius: 2, tension: .35, yAxisID: "y1" },
        ],
      },
      options: { ...base, scales: { y: { beginAtZero: true, title: { display: true, text: "mm" } }, y1: { position: "right", grid: { display: false } } } },
    });
    const n = live.next_24h;
    mk("nextChart", {
      type: "line",
      data: { labels: n.map((d) => d.time.slice(11, 16)), datasets: [{ label: "Precip. probability %", data: n.map((d) => d.precip_prob), borderColor: acc, backgroundColor: css("--accent-soft"), fill: true, tension: .35, pointRadius: 0 }] },
      options: { ...base, scales: { y: { min: 0, max: 100 }, x: { ticks: { maxTicksLimit: 8 } } } },
    });
    const m = (live.climatology && live.climatology.monthly_rain_frequency) || [];
    mk("climChart", {
      type: "bar",
      data: { labels: "JFMAMJJASOND".split(""), datasets: [{ label: "% rainy days (≥1 mm)", data: m.map((v) => (v == null ? null : Math.round(v * 100))), backgroundColor: acc, borderRadius: 4 }] },
      options: { ...base, scales: { y: { beginAtZero: true, max: 100 } } },
    });
    const c = live.climatology || {};
    const total7 = h.reduce((s, d) => s + (d.rain_mm || 0), 0);
    $("#kpis").innerHTML = [
      ["Rain, last 7 days", `${total7.toFixed(1)} mm`],
      ["Rainy days, last 7", `${h.filter((d) => d.rain_mm >= 1).length} / ${h.length}`],
      ["Historical rain chance for tomorrow", c.rain_frequency != null ? `${Math.round(c.rain_frequency * 100)}%` : "—"],
      ["Avg. on rainy days (10 yr)", c.mean_rain_mm_on_rainy_days != null ? `${c.mean_rain_mm_on_rainy_days} mm` : "—"],
    ].map(([k, v]) => `<div class="kpi"><span>${k}</span><b>${v}</b></div>`).join("");
  }

  // ------------------------------------------------------------ IoT
  function samplePayload(random = false) {
    const r = (a, b, d = 1) => +(a + Math.random() * (b - a)).toFixed(d);
    const p = random ? r(998, 1018) : 1004.6;
    return {
      device_id: "esp32-roof-01",
      temperature: random ? r(22, 34) : 29.2,
      humidity: random ? r(45, 98, 0) : 88,
      pressure: p,
      pressure_3h_ago: random ? +(p + r(-1, 4)).toFixed(1) : 1007.1,
      rain_analog: random ? r(300, 4095, 0) : 1450,
      gust_kmh: random ? r(5, 60, 0) : 38,
      ...(state.place ? { lat: +state.place.latitude.toFixed(3), lon: +state.place.longitude.toFixed(3) } : {}),
    };
  }
  function setPayload(obj) {
    $("#iotPayload").value = JSON.stringify(obj, null, 2);
    const host = API || location.origin + location.pathname.replace(/\/[^/]*$/, "");
    $("#curlSnippet").textContent = `curl -X POST ${host.startsWith("http") ? host : "https://<your-app>.vercel.app"}/api/iot \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(obj)}'`;
  }
  async function sendIot() {
    let body;
    try { body = JSON.parse($("#iotPayload").value); } catch { toast("Payload is not valid JSON", true); return; }
    const btn = $("#iotSend");
    btn.disabled = true;
    try {
      const r = await api("iot", { method: "POST", body: JSON.stringify(body) });
      $("#iotResult").innerHTML = `<div class="led-row"><span class="led ${r.led}" aria-hidden="true"></span>
        <div><b>${r.rain_probability}% · ${r.rain_type}</b><div class="muted small">${r.condition}. LED ${r.led}${r.alert ? " · buzzer alert" : ""}. Imputed: ${r.imputed.length ? r.imputed.join(", ") : "none"}</div></div></div>`;
      state.rainLevel = r.rain_probability / 100;
      state.storm = r.rain_type === "Thunderstorm";
    } catch (e) { toast(e.message, true); }
    finally { btn.disabled = false; }
  }

  // ------------------------------------------------------------ model card
  async function loadModel() {
    try {
      const h = await api("health");
      const m = h.metrics;
      $("#modelMeta").textContent = `${h.trained_on}. Test period ${m.test_period}. GA threshold ${h.ensemble.threshold}. Rain-amount MAE ${m.rain_amount_mae_mm} mm.`;
      const names = { logistic_regression: "Logistic regression", random_forest: "Random forest", gradient_boosting: "Gradient boosting", neural_network_mlp: "Neural network (MLP)", fuzzy_inference: "Fuzzy inference", hybrid_ga_ensemble: "Hybrid GA ensemble" };
      const rows = Object.entries(m.models).map(([k, v]) => `<tr class="${k === "hybrid_ga_ensemble" ? "best" : ""}"><td>${names[k] || k}</td><td>${v.accuracy.toFixed(3)}</td><td>${v.precision.toFixed(3)}</td><td>${v.recall.toFixed(3)}</td><td>${v.f1.toFixed(3)}</td><td>${v.roc_auc.toFixed(3)}</td></tr>`).join("");
      $("#metrics").innerHTML = `<div class="table-scroll"><table><thead><tr><th>Model</th><th>Acc</th><th>Prec</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    } catch (e) {
      $("#metrics").innerHTML = `<p class="muted">Could not load model metadata: ${e.message}</p>`;
    }
  }

  // ------------------------------------------------------------ rain canvas
  function rain() {
    const cv = $("#rain"), ctx = cv.getContext("2d");
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W, H, drops = [], flash = 0;
    const resize = () => { W = cv.width = innerWidth * devicePixelRatio; H = cv.height = innerHeight * devicePixelRatio; };
    resize(); addEventListener("resize", resize);
    const spawn = () => ({ x: Math.random() * W * 1.2, y: Math.random() * -H, l: 10 + Math.random() * 22, v: 9 + Math.random() * 10, o: .15 + Math.random() * .35 });
    for (let i = 0; i < 500; i++) { const d = spawn(); d.y = Math.random() * H; drops.push(d); }
    const tick = () => {
      const level = state.rainLevel, target = Math.round(30 + level * 470);
      const rgb = css("--rain-color");
      ctx.clearRect(0, 0, W, H);
      if (state.storm && Math.random() < 0.004) flash = 1;
      if (flash > 0) { ctx.fillStyle = `rgba(200,225,255,${flash * 0.18})`; ctx.fillRect(0, 0, W, H); flash -= 0.05; }
      const slant = 1.5 + level * 2.5, dpr = devicePixelRatio;
      ctx.lineWidth = 1 * dpr; ctx.lineCap = "round";
      for (let i = 0; i < target && i < drops.length; i++) {
        const d = drops[i];
        ctx.strokeStyle = `rgba(${rgb},${d.o * (0.5 + level)})`;
        ctx.beginPath(); ctx.moveTo(d.x, d.y); ctx.lineTo(d.x - slant * d.l / 10, d.y + d.l * dpr); ctx.stroke();
        d.y += d.v * dpr * (0.8 + level * 0.6); d.x -= slant * dpr * 0.6;
        if (d.y > H) Object.assign(d, spawn(), { y: -20 });
      }
      if (!reduce) requestAnimationFrame(tick);
    };
    tick();
  }

  // ------------------------------------------------------------ wiring
  function init() {
    buildForm();
    rain();
    $$(".tab").forEach((t) => t.addEventListener("click", () => {
      $$(".tab").forEach((x) => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", x === t); });
      $$(".tabpanel").forEach((p) => (p.hidden = p.dataset.panel !== t.dataset.tab));
    }));
    $("#searchForm").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#city").value.trim(); if (q) search(q); });
    $("#geoBtn").addEventListener("click", () => {
      if (!navigator.geolocation) return toast("Geolocation is not available", true);
      navigator.geolocation.getCurrentPosition(
        (pos) => loadPlace({ name: "My location", latitude: pos.coords.latitude, longitude: pos.coords.longitude }),
        () => toast("Location permission denied", true), { timeout: 8000 });
    });
    $("#predictBtn").addEventListener("click", runManual);
    $("#presetDry").addEventListener("click", () => { setForm({ Humidity9am: 45, Humidity3pm: 28, Pressure9am: 1021, Pressure3pm: 1019, Cloud9am: 1, Cloud3pm: 1, Rainfall: 0, WindGustSpeed: 28, MaxTemp: 33, MinTemp: 18 }); runManual(); });
    $("#presetMonsoon").addEventListener("click", () => { setForm({ Humidity9am: 94, Humidity3pm: 90, Pressure9am: 1003, Pressure3pm: 999.5, Cloud9am: 8, Cloud3pm: 8, Rainfall: 38, WindGustSpeed: 62, MaxTemp: 31, MinTemp: 25, Temp9am: 26, Temp3pm: 29 }); runManual(); });
    $("#iotSend").addEventListener("click", sendIot);
    $("#iotRandom").addEventListener("click", () => setPayload(samplePayload(true)));
    $("#themeToggle").addEventListener("click", () => {
      const root = document.documentElement, next = root.dataset.theme === "storm" ? "overcast" : "storm";
      root.dataset.theme = next;
      $("#themeToggle").setAttribute("aria-label", next === "storm" ? "Switch to light theme" : "Switch to dark theme");
      if (state.live) renderHistory(state.live);
    });
    setPayload(samplePayload());
    loadModel();
    loadPlace({ name: "Mumbai", admin1: "Maharashtra", country: "India", latitude: 19.07283, longitude: 72.88261 });
  }
  document.addEventListener("DOMContentLoaded", init);
})();
