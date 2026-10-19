// Lightyear chat client: one WebSocket per dataset conversation.
const $ = (s) => document.querySelector(s);
const md = (text) => DOMPurify.sanitize(marked.parse(text || ""));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let datasets = [], ds = null, ws = null, busy = false, ready = false;
let cur = null;          // current assistant text bubble {el, text}
const toolEls = {};       // tool_use id -> <details>

const EXAMPLES = {
  _default: [
    "What is in this dataset, and what should I be careful about?",
    "Which cell types change most between the main conditions?",
    "What surrounds the most disease-associated cells?",
  ],
  rrmap2: [
    "What separates chronic peak (CH_PEAK1) from controls in oligodendrocytes?",
    "Is H2-D1 really induced in oligodendrocytes at peak, or is it ambient?",
    "How much of the myeloid change at peak goes with T-cell proximity?",
  ],
  cuprizone: [
    "What separates acute CupRap from controls in microglia?",
    "Does Spp1 in microglia depend on where they sit relative to the lesion?",
    "What does OSM infusion do to the V-SVZ?",
  ],
};

function setStatus(text, isBusy) {
  busy = isBusy;
  $("#status").textContent = text;
  $("#status").classList.toggle("busy", !!isBusy);
  $("#send").hidden = !!isBusy;
  $("#stop").hidden = !isBusy;
  $("#send").disabled = !ready;
}

function scroll() { const c = $("#chat"); c.scrollTop = c.scrollHeight; }

function addMsg(cls, html) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.innerHTML = html;
  $("#chat").appendChild(el);
  scroll();
  return el;
}

function figs(images) {
  if (!images || !images.length) return "";
  return `<div class="figs">${images.map((b) => `<img src="data:image/png;base64,${b}" onclick="window.open(this.src)">`).join("")}</div>`;
}

function connect() {
  if (ws) { ws.onclose = null; ws.close(); }
  $("#chat").querySelectorAll(".msg").forEach((e) => e.remove());
  $("#welcome").hidden = false;
  cur = null;
  ready = false;
  setStatus("starting agent…", false);
  ws = new WebSocket(`ws://${location.host}/ws/${ds}`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { ready = false; setStatus("disconnected - reload or pick a dataset", false); };
}

function handle(m) {
  switch (m.type) {
    case "ready": ready = true; setStatus("ready", false); break;
    case "delta":
      if (!cur) cur = { el: addMsg("assistant md", ""), text: "" };
      cur.text += m.text; cur.el.innerHTML = md(cur.text); scroll(); break;
    case "text":   // final text of a block: replace the streamed version
      if (!cur) cur = { el: addMsg("assistant md", ""), text: "" };
      cur.text = m.text; cur.el.innerHTML = md(m.text); cur = null; scroll(); break;
    case "tool_use": {
      cur = null;
      const el = document.createElement("details");
      el.className = "tool";
      el.innerHTML = `<summary><b>${esc(m.name)}</b> <span class="state">running…</span></summary>
        <div class="body"><pre class="args">${esc(JSON.stringify(m.input, null, 1))}</pre><div class="result md"></div></div>`;
      $("#chat").appendChild(el); toolEls[m.id] = el; scroll(); break;
    }
    case "tool_result": {
      const el = toolEls[m.id];
      if (!el) break;
      el.querySelector(".state").innerHTML = m.is_error ? '<span class="err">error</span>' : "done";
      el.querySelector(".result").innerHTML = md(m.text);
      if (m.images && m.images.length) el.insertAdjacentHTML("afterend", figs(m.images));
      if (m.is_error) el.open = true;
      scroll(); break;
    }
    case "ledger_changed": loadClaims(); loadFindings(); break;
    case "done":
      cur = null;
      addMsg("done", `${m.turns} turns · ${m.duration_s.toFixed(0)} s` + (m.cost_usd ? ` · $${m.cost_usd.toFixed(2)}` : ""));
      setStatus("ready", false); break;
    case "error": addMsg("assistant", `<span class="err">⚠ ${esc(m.text)}</span>`); setStatus("ready", false); break;
  }
}

function ask(text) {
  if (!text.trim() || busy || !ready || !ws || ws.readyState !== 1) return;
  $("#welcome").hidden = true;
  addMsg("user", esc(text));
  ws.send(JSON.stringify({ type: "ask", text }));
  setStatus("thinking…", true);
}

async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(await r.text()); return r.json(); }

async function loadClaims() {
  const rows = await getJSON(`/api/${ds}/claims`);
  const el = $("#tab-claims");
  if (!rows.length) { el.innerHTML = '<p class="empty">No claims tested yet. Claims appear here when the agent runs test_claim.</p>'; return; }
  el.innerHTML = rows.slice().reverse().map((c) => `
    <div class="claim" data-id="${esc(c.id)}">
      <div class="top"><span><b>${esc(c.id)}</b> · ${c.effect >= 0 ? "+" : ""}${c.effect.toFixed(2)} · n ${c.n_a}${c.n_b != null ? "v" + c.n_b : ""}
        ${c.stale ? '· <span class="stale">stale</span>' : ""}</span>
        <span class="badge ${c.verdict}">${c.verdict.replace("_", " ").toUpperCase()}</span></div>
      <div class="stmt">${esc(c.statement)}</div>
    </div>`).join("");
  el.querySelectorAll(".claim").forEach((x) => x.onclick = () => showClaim(x.dataset.id));
}

async function showClaim(id) {
  const c = await getJSON(`/api/${ds}/claims/${id}`);
  $("#claimbody").innerHTML = md(c.markdown) + figs([c.figure]);
  $("#claimdlg").showModal();
}

async function loadFindings() { $("#tab-findings").innerHTML = md((await getJSON(`/api/${ds}/findings`)).markdown); }
async function loadCard() { $("#tab-card").innerHTML = md((await getJSON(`/api/${ds}/card`)).markdown); }

function selectDataset(id) {
  ds = id;
  const d = datasets.find((x) => x.id === id);
  $("#ds-title").textContent = `${d.name} · ${d.n_cells.toLocaleString()} cells · ${d.n_genes.toLocaleString()} genes`;
  $("#ds-desc").textContent = d.description;
  $("#examples").innerHTML = (EXAMPLES[id] || EXAMPLES._default).map((q) => `<button>${esc(q)}</button>`).join("");
  $("#examples").querySelectorAll("button").forEach((b) => b.onclick = () => ask(b.textContent));
  try { localStorage.setItem("lightyear.ds", id); } catch (e) {}
  connect(); loadClaims(); loadFindings(); loadCard();
}

async function init() {
  datasets = await getJSON("/api/datasets");
  $("#dataset").innerHTML = datasets.map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
  let saved = null; try { saved = localStorage.getItem("lightyear.ds"); } catch (e) {}
  const first = datasets.some((d) => d.id === saved) ? saved : datasets[0].id;
  $("#dataset").value = first;
  $("#dataset").onchange = (e) => selectDataset(e.target.value);
  $("#newchat").onclick = () => connect();
  $("#stop").onclick = () => ws && ws.send(JSON.stringify({ type: "interrupt" }));
  $("#ask").onsubmit = (e) => { e.preventDefault(); ask($("#input").value); $("#input").value = ""; };
  $("#input").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask").requestSubmit(); } };
  document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".tabs button, .tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); $(`#tab-${b.dataset.tab}`).classList.add("active");
  });
  selectDataset(first);
}
init();
