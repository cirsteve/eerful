// eerful demo-ui — vanilla JS SPA
// Connects to /events SSE, renders events into 3-column agent layout +
// gate verdict cards, animates AXL message wire and drift mandate badge.

const $ = (id) => document.getElementById(id);

const state = {
  runId: null,
};

// ---------- helpers ----------
function fmtTime(tsSec) {
  return new Date(tsSec * 1000).toISOString().slice(11, 19);
}

function appendEvent(columnId, evt) {
  const col = $(columnId);
  const row = document.createElement("div");
  row.className = "event-row";

  const ts = document.createElement("span");
  ts.className = "event-ts";
  ts.textContent = fmtTime(evt.ts);
  row.appendChild(ts);

  const kind = document.createElement("span");
  kind.className = "event-kind " + kindClass(evt.kind);
  kind.textContent = displayKind(evt);
  row.appendChild(kind);

  const detail = document.createElement("span");
  detail.className = "event-detail";
  detail.innerHTML = renderDetail(evt);
  row.appendChild(detail);

  col.appendChild(row);
  col.scrollTop = col.scrollHeight;
}

function kindClass(kind) {
  if (kind === "axl_send") return "send";
  if (kind === "axl_recv") return "recv";
  if (kind === "run_started") return "run";
  if (kind === "receipt_minted") return "mint";
  if (kind === "optuna_progress" || kind === "optuna_started" || kind === "optuna_finished") return "optuna";
  return "";
}

function displayKind(evt) {
  if (evt.kind === "axl_send") return "→ " + (evt.payload.envelope_kind || "send");
  if (evt.kind === "axl_recv") return "← " + (evt.payload.envelope_kind || "recv");
  return evt.kind;
}

// Coerce any payload field to a string and escape it. Events arrive
// over the LAN POST endpoint, so even number-shaped fields can be
// adversarial strings; never interpolate raw payload values into the
// innerHTML template.
function safe(v, fallback = "—") {
  if (v == null) return fallback;
  return escapeHtml(String(v));
}
function safeNum(v, digits, fallback = "—") {
  if (v == null || typeof v !== "number" || !Number.isFinite(v)) return fallback;
  return v.toFixed(digits);
}

function renderDetail(evt) {
  const p = evt.payload || {};
  switch (evt.kind) {
    case "run_started":
      return `<span class="key">tool:</span> ${safe(p.tool_response_name, "")}`;
    case "axl_send":
      return `<span class="key">to</span> ${safe(short(p.peer_id_prefix))} · <span class="key">trials</span> <span class="num">${safe(p.n_trials)}</span>`;
    case "axl_recv": {
      // The refiner emits axl_recv for inbound STRATEGY_DRAFTs (no
      // sharpe yet); the explorer emits it for OPTIMIZATION_RESULT
      // (sharpe + best_params). Branch on envelope_kind so the row
      // describes what actually arrived.
      if (p.envelope_kind === "STRATEGY_DRAFT") {
        return `<span class="key">from</span> ${safe(short(p.peer_id_prefix))} · <span class="key">trials</span> <span class="num">${safe(p.n_trials)}</span>`;
      }
      const sharpe = safeNum(p.sharpe, 3);
      return `<span class="key">sharpe</span> <span class="num">${sharpe}</span> · ${formatParams(p.best_params)}`;
    }
    case "working_mandate": {
      const dd = safeNum(p.max_drawdown_pct, 0);
      return `<span class="key">working DD</span> <span class="num">${dd}%</span>${p.drift ? ' <span style="color: var(--bad);">DRIFT</span>' : ""}`;
    }
    case "receipt_minted": {
      const score = (p.score_block && typeof p.score_block.overall === "number")
        ? safeNum(p.score_block.overall, 2) : "—";
      return `<span class="key">${safe(p.bundle)}</span> · <span class="key">overall</span> <span class="num">${score}</span>`;
    }
    case "optuna_progress":
      return `<span class="key">trial</span> <span class="num">${safe(p.trial)}</span> · <span class="key">best sharpe</span> <span class="num">${safeNum(p.best_sharpe, 3)}</span>`;
    case "optuna_started":
      return `<span class="key">trials</span> <span class="num">${safe(p.n_trials)}</span>`;
    case "optuna_finished":
      return `<span class="key">final sharpe</span> <span class="num">${safeNum(p.sharpe, 3)}</span>`;
    default:
      return `<span class="key">${safe(JSON.stringify(p))}</span>`;
  }
}

function formatParams(params) {
  if (!params) return "";
  const entries = Object.entries(params).map(
    ([k, v]) => `<span class="key">${escapeHtml(k)}</span>=<span class="num">${escapeHtml(String(v))}</span>`,
  );
  return entries.join(" ");
}

function short(s) {
  if (!s) return "?";
  return s.length > 16 ? s.slice(0, 12) + "…" : s;
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function setStatus(id, text, working) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("working", !!working);
}

// ---------- wire animation ----------
let wireResetTimer = null;
function animateWire({ direction, label }) {
  const dot = $("wire-dot");
  const lbl = $("wire-label");
  dot.classList.remove("hidden");
  lbl.textContent = label;
  lbl.classList.add("visible");

  // explorer top, refiner bottom: send goes top→bottom, recv bottom→top
  if (direction === "send") {
    dot.classList.remove("recv");
    dot.setAttribute("cy", "20");
    requestAnimationFrame(() => dot.setAttribute("cy", "340"));
  } else {
    dot.classList.add("recv");
    dot.setAttribute("cy", "340");
    requestAnimationFrame(() => dot.setAttribute("cy", "20"));
  }

  if (wireResetTimer) clearTimeout(wireResetTimer);
  wireResetTimer = setTimeout(() => {
    dot.classList.add("hidden");
    lbl.classList.remove("visible");
  }, 1800);
}

// ---------- event handlers ----------
function resetState({ runIdText, runIdLive } = {}) {
  state.runId = null;
  $("explorer-events").innerHTML = "";
  $("refiner-events").innerHTML = "";
  $("verdict-stack").innerHTML = "";
  setStatus("explorer-status", "idle", false);
  setStatus("refiner-status", "idle", false);
  setStatus("gate-status", "awaiting receipts", false);
  $("run-id").textContent = runIdText || "no run yet";
  $("run-id").classList.toggle("live", !!runIdLive);
  $("tool-response").textContent = "";
  $("tool-response").className = "tool-response";
  $("threshold").textContent = "—";
  const m = $("mandate-badge");
  m.classList.remove("drift");
  m.textContent = "mandate: 5% drawdown · 2x leverage · BTC/ETH/SOL";
  // Hide wire dot if it's mid-animation
  const dot = $("wire-dot");
  if (dot) dot.classList.add("hidden");
  const lbl = $("wire-label");
  if (lbl) lbl.classList.remove("visible");
}

function clearForNewRun(evt) {
  const tr = (evt.payload || {}).tool_response_name || "";
  resetState({
    runIdText: "run " + (evt.run_id || "").slice(0, 8),
    runIdLive: true,
  });
  state.runId = evt.run_id;
  setStatus("explorer-status", "running", true);
  setStatus("refiner-status", "awaiting STRATEGY_DRAFT", false);
  $("tool-response").textContent = tr ? `tool_response: ${tr}` : "";
  $("tool-response").className = "tool-response " + (tr.includes("poison") ? "poisoned" : "clean");
}

function handleEvent(evt) {
  // Run boundary
  if (evt.kind === "run_started") {
    clearForNewRun(evt);
    appendEvent("explorer-events", evt);
    return;
  }

  // Source-based routing for column placement
  if (evt.source === "explorer" || evt.source === "agent_multi") {
    appendEvent("explorer-events", evt);
  }
  if (evt.source === "refiner") {
    appendEvent("refiner-events", evt);
  }

  // Wire animation
  if (evt.kind === "axl_send" && evt.source === "explorer") {
    animateWire({ direction: "send", label: evt.payload.envelope_kind || "msg" });
    setStatus("refiner-status", "incoming…", true);
  }
  if (evt.kind === "axl_recv" && evt.source === "explorer") {
    animateWire({ direction: "recv", label: evt.payload.envelope_kind || "msg" });
    setStatus("refiner-status", "done", false);
    setStatus("explorer-status", "rendering artifacts", true);
  }

  // Refiner-side events also drive its status
  if (evt.kind === "axl_recv" && evt.source === "refiner") {
    setStatus("refiner-status", "running Optuna sweep…", true);
  }
  if (evt.kind === "optuna_progress" && evt.source === "refiner") {
    const p = evt.payload || {};
    setStatus("refiner-status", `Optuna trial ${p.trial}…`, true);
  }
  if (evt.kind === "optuna_finished" && evt.source === "refiner") {
    setStatus("refiner-status", "sending result", true);
  }
  if (evt.kind === "axl_send" && evt.source === "refiner") {
    animateWire({ direction: "recv", label: evt.payload.envelope_kind || "result" });
    setStatus("refiner-status", "done", false);
  }

  // Mandate badge
  if (evt.kind === "working_mandate") {
    const p = evt.payload || {};
    const m = $("mandate-badge");
    if (p.drift) {
      m.classList.add("drift");
      m.textContent = `DRIFT — agent's working drawdown: ${Number(p.max_drawdown_pct).toFixed(0)}% (mandate: ${Number(p.principal_mandate_pct).toFixed(0)}%)`;
    } else {
      m.classList.remove("drift");
      m.textContent = `mandate aligned · working DD: ${Number(p.max_drawdown_pct).toFixed(0)}% · 2x leverage · BTC/ETH/SOL`;
    }
  }

  // Receipt minted → flip explorer status
  if (evt.kind === "receipt_minted") {
    setStatus("explorer-status", `receipt minted: ${evt.payload.bundle}`, true);
    setStatus("gate-status", "receipt available — awaiting gate", false);
  }

  // Gate verdict
  if (evt.kind === "gate_verdict") {
    renderVerdict(evt.payload);
    setStatus("gate-status", evt.payload.outcome, false);
    setStatus("explorer-status", "done", false);
    if (evt.payload.threshold != null) {
      $("threshold").textContent = Number(evt.payload.threshold).toFixed(2);
    }
  }
}

function renderVerdict(p) {
  const stack = $("verdict-stack");
  const card = document.createElement("div");
  const isPass = p.outcome === "pass";
  card.className = "verdict " + (isPass ? "pass" : "refuse");

  const bundle = document.createElement("div");
  bundle.className = "verdict-bundle";
  bundle.textContent = p.bundle || "?";
  card.appendChild(bundle);

  const out = document.createElement("div");
  out.className = "verdict-outcome";
  out.textContent = isPass ? "PASS" : "REFUSE — " + (p.outcome || "");
  card.appendChild(out);

  const det = document.createElement("div");
  det.className = "verdict-detail";
  det.textContent = p.detail || "";
  card.appendChild(det);

  if (p.overall != null && p.threshold != null) {
    const score = document.createElement("div");
    score.className = "verdict-score";
    score.innerHTML =
      `<span class="key">overall:</span> <span class="num">${Number(p.overall).toFixed(2)}</span>` +
      ` · <span class="key">threshold:</span> <span class="num">${Number(p.threshold).toFixed(2)}</span>`;
    card.appendChild(score);
  }

  stack.appendChild(card);
  stack.scrollTop = stack.scrollHeight;
}

// ---------- SSE ----------
let es = null;

function connect() {
  if (es) {
    es.close();
    es = null;
  }
  es = new EventSource("/events");
  es.onmessage = (e) => {
    let evt;
    try { evt = JSON.parse(e.data); } catch { return; }
    handleEvent(evt);
  };
  es.onerror = () => {
    setStatus("explorer-status", "stream disconnected — retrying", false);
  };
  es.onopen = () => {
    if (!state.runId) setStatus("explorer-status", "connected — waiting for run", false);
  };
}

// ---------- clear button ----------
async function clearAll() {
  const btn = $("clear-btn");
  btn.disabled = true;
  try {
    await fetch("/admin/clear", { method: "POST" });
  } catch (err) {
    // Best-effort: even if the server clear fails, still reset the
    // client view so the operator gets a fresh canvas.
    console.warn("clear request failed:", err);
  }
  resetState();
  // Reconnect SSE so the client picks up the truncated history (otherwise
  // its file-position cursor would skip past the new EOF and miss the
  // next batch of events).
  connect();
  btn.disabled = false;
}

document.getElementById("clear-btn").addEventListener("click", clearAll);

connect();
