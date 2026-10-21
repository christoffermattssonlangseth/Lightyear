// Lightyear chat client: one WebSocket per dataset conversation.
const $ = (s) => document.querySelector(s);
const md = (text) => DOMPurify.sanitize(marked.parse(text || ""));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let datasets = [], ds = null, ws = null, busy = false, ready = false, setup = null;
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
    case "ready": ready = true; setStatus(`ready · ${m.provider}`, false); break;
    case "setup_needed": openSetup(true); break;
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
      addMsg("done", `${m.turns} ${m.cost_usd !== undefined ? "turns" : "tool calls"} · ${m.duration_s.toFixed(0)} s`
        + (m.cost_usd ? ` · $${m.cost_usd.toFixed(2)}` : "") + (m.tokens ? ` · ${(m.tokens / 1000).toFixed(0)}k tokens` : ""));
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
async function sendJSON(url, body, method = "POST") {
  const r = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: body && JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}

// -- setup: provider, sign-in, keys, datasets folder -----------------------------------------
const NAMES = { claude: "Claude", codex: "Codex" };
const radio = (name) => document.querySelector(`input[name="${name}"]:checked`)?.value;

function renderSetup() {
  const st = setup, dlg = $("#setupdlg");
  const prov = radio("provider") || st.provider || "claude";
  dlg.querySelector(`input[name="provider"][value="${prov}"]`).checked = true;
  dlg.querySelectorAll(".pp").forEach((x) => x.hidden = x.dataset.for !== prov);

  const cl = st.claude, cx = st.codex;
  $("#claude-login-opt").hidden = !cl.login_allowed;
  $("#claude-login-detail").textContent = cl.login.detail;
  // first run: prefer an existing sign-in; afterwards, what was saved
  const firstRun = !st.provider;
  const clMode = !cl.login_allowed ? "api_key" : firstRun ? (cl.login.available ? "login" : "api_key") : st.auth.claude;
  const cxMode = firstRun ? (cx.login.available || !cx.installed ? "login" : "api_key") : st.auth.codex;
  if (!radio("claude-auth")) dlg.querySelector(`input[name="claude-auth"][value="${clMode}"]`).checked = true;
  $("#codex-missing").hidden = cx.installed;
  $("#codex-login-detail").textContent = cx.login.detail;
  if (!radio("codex-auth")) dlg.querySelector(`input[name="codex-auth"][value="${cxMode}"]`).checked = true;
  for (const p of ["claude", "codex"]) {
    const row = dlg.querySelector(`.keyrow[data-key="${p}"]`);
    row.hidden = radio(`${p}-auth`) !== "api_key";
    row.querySelector(".current").textContent = st[p].key ? `saved: ${st[p].key}` : "";
    row.querySelector(".delkey").hidden = !st[p].key;
  }
  if (!$("#model").dataset.touched) $("#model").value = (st.model || {})[prov] || "";
  if (!$("#stores-dir").dataset.touched) $("#stores-dir").value = st.stores_dir || "";
  $("#pick-folder").hidden = !st.can_pick_folder;
  $("#setup-close").hidden = !st.ready;
  $("#provider-label").textContent = st.ready ? NAMES[st.provider] : "Setup";
}

async function openSetup(force) {
  setup = await getJSON("/api/setup");
  $("#setup-msg").textContent = force && setup.provider ? "Sign-in for the chosen provider isn't usable right now." : "";
  renderSetup();
  if (!$("#setupdlg").open) $("#setupdlg").showModal();
}

function setupMsg(text, ok) { const el = $("#setup-msg"); el.textContent = text; el.className = "msgline " + (ok ? "ok" : "bad"); }
function showFolder() { $(".advanced").open = true; $("#stores-dir").focus(); }

function wireSetup() {
  const dlg = $("#setupdlg");
  dlg.addEventListener("cancel", (e) => { if (!setup?.ready) e.preventDefault(); });
  $("#setup-close").onclick = () => dlg.close();
  dlg.querySelectorAll('input[type="radio"]').forEach((r) => r.onchange = () => { if (r.name === "provider") $("#model").dataset.touched = ""; renderSetup(); });
  $("#model").oninput = $("#stores-dir").oninput = (e) => e.target.dataset.touched = "1";
  $("#pick-folder").onclick = async () => {
    const r = await sendJSON("/api/pick-folder", {});
    if (r.path) { $("#stores-dir").value = r.path; $("#stores-dir").dataset.touched = "1"; }
  };
  $("#setup-recheck").onclick = async () => { setupMsg("checking…", true); setup = await getJSON("/api/setup"); renderSetup(); setupMsg("", true); };
  dlg.querySelectorAll(".keyrow").forEach((row) => {
    const p = row.dataset.key, inp = row.querySelector("input");
    row.querySelector(".savekey").onclick = async () => {
      if (!inp.value.trim()) return setupMsg("paste a key first", false);
      setupMsg("verifying with " + (p === "claude" ? "Anthropic" : "OpenAI") + "…", true);
      const r = await sendJSON("/api/setup/key", { provider: p, key: inp.value });
      setupMsg(r.detail, r.ok);
      if (r.ok) { inp.value = ""; setup = await getJSON("/api/setup"); renderSetup(); }
    };
    row.querySelector(".delkey").onclick = async () => { setup = await sendJSON(`/api/setup/key/${p}`, null, "DELETE"); renderSetup(); };
  });
  $("#setup-start").onclick = async () => {
    const prov = radio("provider");
    try {
      setup = await sendJSON("/api/setup", {
        provider: prov, auth: { claude: radio("claude-auth"), codex: radio("codex-auth") },
        model: { [prov]: $("#model").value }, stores_dir: $("#stores-dir").value || undefined });
    } catch (e) { if (/folder/i.test(e.message)) showFolder(); return setupMsg(e.message, false); }
    renderSetup();
    if (!setup.ready) return setupMsg(prov === "codex" && !setup.codex.installed ? "Install the Codex CLI first."
      : radio(`${prov}-auth`) === "api_key" ? "Save a working API key first." : "Not signed in yet - see the note above.", false);
    datasets = await getJSON("/api/datasets");
    if (!datasets.length) return setupMsg(`No datasets in ${setup.stores_dir} yet: drop an .h5ad file on this window or press Add dataset…, or under Advanced choose a folder that already holds Lightyear datasets.`, false);
    dlg.close();
    fillDatasets();
  };
}

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

function fillDatasets() {
  $("#dataset").innerHTML = datasets.map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
  let saved = null; try { saved = localStorage.getItem("lightyear.ds"); } catch (e) {}
  const first = datasets.some((d) => d.id === saved) ? saved : datasets[0].id;
  $("#dataset").value = first;
  selectDataset(first);
}

// -- add a dataset: inspect the .h5ad, confirm the columns, run ingest locally --------------------
let ingestInfo = null, ingestTimer = null;
const ingMsg = (text, ok) => { const el = $("#ing-msg"); el.textContent = text; el.className = "msgline " + (ok ? "ok" : "bad"); };

async function openIngest() {
  $("#ing-pick").hidden = !setup?.can_pick_file;
  const st = await getJSON("/api/ingest");
  if (st.state === "running") { showIngest(st); pollIngest(); }
  else { ingestInfo = null; $("#ing-form").hidden = $("#ing-log").hidden = true; $("#ing-start").disabled = true; ingMsg("", true); }
  $("#ingestdlg").showModal();
}

async function readIngest() {
  const path = $("#ing-path").value.trim();
  if (!path) return ingMsg("Choose an .h5ad file first", false);
  ingMsg("reading the file (cell metadata only)…", true);
  $("#ing-start").disabled = true;
  try { ingestInfo = await sendJSON("/api/ingest/inspect", { path }); }
  catch (e) { $("#ing-form").hidden = true; return ingMsg(e.message, false); }
  const i = ingestInfo;
  $("#ing-path").value = i.path;
  $("#ing-summary").textContent = `${i.n_cells.toLocaleString()} cells · ${i.n_genes.toLocaleString()} genes · `
    + (i.spatial ? `coordinates from ${i.spatial}` : "no coordinates");
  const opt = (c) => `<option value="${esc(c.name)}">${esc(c.name)} (${c.n_unique}: ${esc(c.examples.join(", "))}${c.n_unique > c.examples.length ? ", …" : ""})</option>`;
  for (const sel of document.querySelectorAll("#ing-form select[data-role]")) {
    const role = sel.dataset.role, required = role === "sample" || role === "cell_type";
    sel.innerHTML = (required ? '<option value="">choose…</option>' : '<option value="">none</option>') + i.columns.map(opt).join("");
    sel.value = i.roles[role] || "";
  }
  $("#ing-layer").innerHTML = ['<option value="X">main matrix (X)</option>', ...i.layers.map((l) => `<option value="${esc(l)}">layer ${esc(l)}</option>`)].join("");
  $("#ing-id").value = i.id; $("#ing-name").value = i.name;
  const warn = [];
  if (!i.spatial) warn.push("No spatial coordinates found: Lightyear needs obsm['spatial'] (or x/y columns in obs).");
  if (i.x_counts) warn.push("The main matrix looks like raw counts; Lightyear expects log-normalised values"
    + (i.layers.length ? ". If one of the layers is log-normalised, choose it under Expression values." : " (normalise and log1p before adding)."));
  $("#ing-warn").textContent = warn.join(" "); $("#ing-warn").hidden = !warn.length;
  $("#ing-form").hidden = false;
  $("#ing-start").disabled = !i.spatial;
  ingMsg("Check the columns, then add the dataset.", true);
}

function showIngest(st) {
  $("#ing-log").hidden = false;
  $("#ing-log").textContent = st.log.join("\n");
  $("#ing-log").scrollTop = $("#ing-log").scrollHeight;
  $("#ing-cancel").hidden = st.state !== "running";
  $("#ing-start").disabled = st.state === "running";
  if (st.state === "running") ingMsg(`adding ${st.id}… ${st.elapsed_s} s (large datasets take several minutes)`, true);
}

function pollIngest() {
  clearTimeout(ingestTimer);
  ingestTimer = setTimeout(async () => {
    const st = await getJSON("/api/ingest");
    showIngest(st);
    if (st.state === "running") return pollIngest();
    if (st.state === "done") return ingestDone(st.id);
    ingMsg(st.state === "cancelled" ? "Cancelled; nothing was added." : `Failed: ${st.error}`, false);
  }, 1000);
}

async function startIngest() {
  const i = ingestInfo, roles = {};
  document.querySelectorAll("#ing-form select[data-role]").forEach((s) => roles[s.dataset.role] = s.value);
  try {
    showIngest(await sendJSON("/api/ingest", { path: i.path, id: $("#ing-id").value.trim(), name: $("#ing-name").value.trim(),
      roles, layer: $("#ing-layer").value, organism: $("#ing-organism").value, description: $("#ing-desc").value }));
  } catch (e) { return ingMsg(e.message, false); }
  pollIngest();
}

async function ingestDone(id) {
  datasets = await getJSON("/api/datasets");
  try { localStorage.setItem("lightyear.ds", id); } catch (e) {}
  $("#ingestdlg").close();
  if (setup?.ready) { $("#setupdlg").close(); fillDatasets(); }
  else setupMsg(`Dataset ${id} added. Finish choosing the model, then press Start.`, true);
}

// Dropping a file: the app window's native side sends its full path to lightyearDropped. A plain browser page
// never learns where a dropped file lives, so there the drop only explains what to do instead.
window.lightyearDropped = async (path) => {
  $("#dropzone").hidden = true;
  const st = await getJSON("/api/ingest");
  if (!$("#ingestdlg").open) await openIngest();
  if (st.state === "running") return;
  if (!/\.h5ad$/i.test(path)) { $("#ing-path").value = path; return ingMsg("That isn't an .h5ad file. Lightyear reads AnnData .h5ad files.", false); }
  $("#ing-path").value = path;
  readIngest();
};

function wireDrop() {
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  document.addEventListener("dragenter", (e) => { if (hasFiles(e)) { depth++; $("#dropzone").hidden = false; } });
  document.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; $("#dropzone").hidden = true; } });
  document.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  document.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0;
    setTimeout(() => {  // in the app window the native handler follows with the path
      if ($("#dropzone").hidden) return;
      $("#dropzone").hidden = true;
      if (!setup?.can_pick_file) { openIngest(); ingMsg("In a browser, dropped files have no path: paste the file's path above instead.", false); }
    }, 1500);
  });
}

function wireIngest() {
  wireDrop();
  $("#add-ds").onclick = $("#setup-add-ds").onclick = openIngest;
  $("#ingest-close").onclick = () => { clearTimeout(ingestTimer); $("#ingestdlg").close(); };  // a running ingest continues
  $("#ing-read").onclick = readIngest;
  $("#ing-path").onkeydown = (e) => { if (e.key === "Enter") readIngest(); };
  $("#ing-pick").onclick = async () => {
    const r = await sendJSON("/api/pick-file", {});
    if (r.path) { $("#ing-path").value = r.path; readIngest(); }
  };
  $("#ing-start").onclick = startIngest;
  $("#ing-cancel").onclick = async () => showIngest(await sendJSON("/api/ingest", null, "DELETE"));
}

async function init() {
  wireSetup();
  wireIngest();
  $("#dataset").onchange = (e) => selectDataset(e.target.value);
  $("#settings").onclick = () => openSetup(false);
  $("#newchat").onclick = () => ds && connect();
  $("#stop").onclick = () => ws && ws.send(JSON.stringify({ type: "interrupt" }));
  $("#ask").onsubmit = (e) => { e.preventDefault(); ask($("#input").value); $("#input").value = ""; };
  $("#input").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask").requestSubmit(); } };
  document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".tabs button, .tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); $(`#tab-${b.dataset.tab}`).classList.add("active");
  });
  setup = await getJSON("/api/setup");
  datasets = await getJSON("/api/datasets");
  if (!setup.ready || !datasets.length) { setStatus("not set up", false); return openSetup(false); }
  $("#provider-label").textContent = NAMES[setup.provider];
  fillDatasets();
}
init();
