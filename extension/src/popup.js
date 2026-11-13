// The popup: every statistic once, with a dropdown of each source's value, and filters.
const api = globalThis.browser || globalThis.chrome;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const GROUPS = ["quality", "performance", "features", "emotions", "activity"];
const SAVED = "popup_filters";
let REC = null, TAB_ID = null, MODEL = null, OFF = new Set();

function fmt(v, unit) {
  if (v === null || v === undefined) return "—";
  if (typeof v !== "number") return esc(v);
  if (unit === "ratio") return (v * 100).toFixed(1) + "%";
  if (unit === "tok/s") return v.toFixed(1) + " tok/s";
  if (unit === "activation" || unit === "freq") return v.toFixed(3);
  return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
}

// One headline value per statistic over the rows shown: sample-weighted mean of ratios, median of speeds.
function headline(rows) {
  const u = rows[0].unit, vals = rows.filter(r => typeof r.value === "number");
  if (!vals.length) return null;
  if (u === "tok/s") {
    const s = vals.map(r => r.value).sort((a, b) => a - b), m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }
  if (u === "ratio" || u === "activation" || u === "freq") {
    let w = 0, t = 0;
    for (const r of vals) { const k = r.n || 1; w += k; t += k * r.value; }
    return t / w;
  }
  return vals.reduce((a, r) => a + r.value, 0);
}

function shown() {
  if (!REC) return [];
  const q = $("#q").value.trim().toLowerCase(), g = $("#group").value;
  return REC.stats.filter(s => !OFF.has(s.source)
    && (!g || s.group === g)
    && ($("#related").checked || !s.related)
    && ($("#selfrep").checked || !(s.detail && s.detail.self_reported))
    && (!q || (s.label + " " + s.variant + " " + s.source + " " + s.key).toLowerCase().includes(q)));
}

function detailText(d) {
  if (!d) return "";
  return Object.entries(d).filter(([k, v]) => v !== null && v !== undefined && k !== "per_subject" && k !== "self_reported")
    .map(([k, v]) => `${esc(k)}: ${esc(typeof v === "object" ? JSON.stringify(v) : v)}`).join(" · ");
}

function render() {
  const rows = shown();
  const byKey = new Map();
  for (const s of rows) { if (!byKey.has(s.key)) byKey.set(s.key, []); byKey.get(s.key).push(s); }
  let items = [...byKey.values()].map(rs => ({ rows: rs, v: headline(rs), n: rs.reduce((a, r) => a + (r.n || 0), 0) }));
  const sort = $("#sort").value;
  if (sort === "value") items.sort((a, b) => (b.v ?? -Infinity) - (a.v ?? -Infinity));
  else if (sort === "n") items.sort((a, b) => b.n - a.n);
  else if (sort === "label") items.sort((a, b) => a.rows[0].label.localeCompare(b.rows[0].label));
  let html = "", lastGroup = null;
  const grouped = sort === "group";
  if (grouped) items.sort((a, b) => GROUPS.indexOf(a.rows[0].group) - GROUPS.indexOf(b.rows[0].group));
  for (const it of items) {
    const s0 = it.rows[0];
    if (grouped && s0.group !== lastGroup) { html += `<div class="grp">${esc(s0.group)}</div>`; lastGroup = s0.group; }
    const pct = s0.unit === "ratio" && it.v !== null ? Math.max(0, Math.min(100, it.v * 100)) : null;
    const tags = (it.rows.some(r => r.related) ? '<span class="tag">related</span>' : "")
      + (it.rows.some(r => r.detail && r.detail.self_reported) ? '<span class="tag warn" title="reported by the submitter; the board cannot re-check it">self-reported</span>' : "");
    html += `<details class="st"><summary><span class="k" title="${esc(s0.key)}">${esc(s0.label)}${tags}</span>
      <span class="dim">${(k => `${it.rows.length} value${it.rows.length > 1 ? "s" : ""}${k > 1 ? ` · ${k} sources` : ""}`)(new Set(it.rows.map(r => r.source)).size)}${it.n ? " · n " + it.n.toLocaleString() : ""}</span>
      <span class="v">${fmt(it.v, s0.unit)}</span></summary>
      ${pct !== null ? `<div class="bar"><i style="width:${pct}%"></i></div>` : ""}
      <table><tr><th>source</th><th>setup</th><th class="n">value</th><th class="n">n</th></tr>
      ${it.rows.map(r => `<tr><td>${esc(r.source)}${r.via ? `<div class="dim">via ${esc(r.via)}</div>` : ""}</td>
        <td>${esc(r.variant)}${r.related ? '<span class="tag">related</span>' : ""}<div class="dim">${detailText(r.detail)}</div></td>
        <td class="n">${fmt(r.value, r.unit)}</td><td class="n">${r.n ?? ""}</td></tr>`).join("")}</table></details>`;
  }
  $("#stats").innerHTML = html || `<div class="empty">${REC && REC.stats.length ? "Nothing matches these filters."
    : REC && !REC.configured ? 'No sources yet. Add a review board or your Studio under <a href="#" id="go-opts">⚙ settings</a>.'
    : "No source knows this model yet."}</div>`;
  const go = $("#go-opts"); if (go) go.onclick = e => { e.preventDefault(); api.runtime.openOptionsPage(); };
}

function renderSources() {
  const src = (REC && REC.sources) || [];
  $("#srccount").textContent = src.length ? `(${src.filter(s => s.ok).length}/${src.length} answered)` : "";
  $("#sources").innerHTML = src.map(s => `<div class="s"><input type="checkbox" data-id="${esc(s.id)}" ${OFF.has(s.id) ? "" : "checked"}>
    <span>${esc(s.name)}<span class="tag">${esc(s.kind)}</span>${s.via ? `<span class="dim"> via ${esc(s.via)}</span>` : ""}</span>
    <span class="dim">${s.stats ?? (REC.stats.filter(x => x.source === s.id).length)} stats</span>
    ${s.ok ? "" : `<span class="err" title="${esc(s.error)}">error</span>`}</div>`).join("") || '<div class="s dim">none configured</div>';
  document.querySelectorAll("#sources input").forEach(i => i.onchange = () => {
    if (i.checked) OFF.delete(i.dataset.id); else OFF.add(i.dataset.id);
    save(); render();
  });
}

function show(rec) {
  REC = rec;
  $("#model").textContent = MODEL || "No model on this page";
  const on = !!(rec && rec.available);
  $("#glow").src = `icons/${on ? "lit" : "idle"}-32.png`;
  $("#glow").classList.toggle("lit", on);
  $("#sub").textContent = rec ? `${rec.stats.length} statistics from ${rec.sources.filter(s => s.ok).length} sources`
    + (rec.t ? " · " + new Date(rec.t).toLocaleTimeString() : "") : "";
  const groups = new Set((rec ? rec.stats : []).map(s => s.group));
  $("#group").innerHTML = '<option value="">all groups</option>' + GROUPS.filter(g => groups.has(g))
    .map(g => `<option>${g}</option>`).join("");
  load();
  renderSources(); render();
  const acts = [];
  if (rec && rec.open) acts.push(`<a href="${esc(rec.open.hub)}" target="_blank" rel="noopener">Open in OmniCortex (download a file)</a>`,
    `<a href="${esc(rec.open.review)}" target="_blank" rel="noopener">Neuron review</a>`);
  if (MODEL) acts.push(`<a href="https://huggingface.co/${esc(MODEL)}" target="_blank" rel="noopener">Model page</a>`);
  $("#actions").innerHTML = acts.join("");
}

function save() {
  const f = { q: $("#q").value, group: $("#group").value, sort: $("#sort").value, related: $("#related").checked,
    selfrep: $("#selfrep").checked, off: [...OFF] };
  try { api.storage.local.set({ [SAVED]: f }); } catch (e) { /* no storage: filters reset next time */ }
}
let LOADED = null;
function load() {
  const f = LOADED; if (!f) return;
  $("#sort").value = f.sort || "group"; $("#related").checked = f.related !== false; $("#selfrep").checked = f.selfrep !== false;
  if ([...$("#group").options].some(o => o.value === f.group)) $("#group").value = f.group;
  OFF = new Set(f.off || []);
}

async function ask(force) {
  $("#sub").textContent = "asking…";
  let r = TAB_ID !== null ? await api.runtime.sendMessage({ type: "get", tabId: TAB_ID, force }) : { model: null };
  if (!r.model && MODEL) r = { model: MODEL, ...(await api.runtime.sendMessage({ type: "lookup", model: MODEL, force })) };
  if (r.model) MODEL = r.model;
  show(r.record || null);
}

(async () => {
  try { LOADED = (await api.storage.local.get(SAVED))[SAVED] || null; } catch (e) { LOADED = null; }
  const [tab] = await api.tabs.query({ active: true, currentWindow: true }).catch(() => []);
  if (tab) { TAB_ID = tab.id; const hit = tab.url && OC_sites.modelFromUrl(tab.url); if (hit) MODEL = hit.model; }
  ["#q", "#group", "#sort", "#related", "#selfrep"].forEach(s => $(s).addEventListener("input", () => { save(); render(); }));
  $("#refresh").onclick = () => ask(true);
  $("#opts").onclick = () => api.runtime.openOptionsPage();
  if (LOADED) $("#q").value = LOADED.q || "";
  await ask(false);
})();
