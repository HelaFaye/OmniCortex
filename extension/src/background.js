// Background: asks the configured sources about the model on the current page, caches the answer, and lights the
// toolbar icon blue when there is something to show. Chromium runs this as a service worker (importScripts below);
// Firefox loads sites.js and normalize.js before it as background scripts.
if (typeof importScripts === "function" && typeof OC_normalize === "undefined") importScripts("sites.js", "normalize.js");
const api = globalThis.browser || globalThis.chrome;

const DEFAULTS = {
  boards: [],          // ["github:owner/repo", "https://scoreboard.example/"]: read directly, no account needed
  studios: [],         // [{url: "https://box:7870", token: "…"}]: an OmniCortex Studio (adds local data, SAELens)
  ttl_min: 60,
};
const ICONS = state => Object.fromEntries([16, 32, 48, 128].map(s => [s, `icons/${state}-${s}.png`]));

async function settings() {
  const got = await api.storage.sync.get(DEFAULTS).catch(() => DEFAULTS);
  return { ...DEFAULTS, ...got };
}

function boardUrl(spec) {
  spec = spec.trim();
  if (/^https?:\/\//.test(spec)) return spec.endsWith(".json") ? spec : spec.replace(/\/+$/, "") + "/api/board";
  if (spec.startsWith("github:")) spec = spec.slice(7);
  return `https://raw.githubusercontent.com/${spec}/main/board.json`;
}

async function cached(key, ttlMin, fetcher) {
  const now = Date.now();
  try {
    const c = (await api.storage.local.get(key))[key];
    if (c && now - c.t < ttlMin * 60000) return c.v;
  } catch (e) { /* storage unavailable: just fetch */ }
  const v = await fetcher();
  try { await api.storage.local.set({ [key]: { t: now, v } }); } catch (e) { /* too big or no storage: fine */ }
  return v;
}

async function getJson(url, headers = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 20000);
  try {
    const r = await fetch(url, { headers, signal: ctl.signal, credentials: "omit", cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } finally { clearTimeout(timer); }
}

// One record for a model from every source. A board is fetched whole (it never learns which model you looked at);
// a Studio is asked about the model, with its token.
async function lookup(model, { force = false } = {}) {
  const cfg = await settings();
  const ttl = force ? 0 : cfg.ttl_min;
  const records = [];
  for (const spec of cfg.boards) {
    const id = "board:" + spec.replace(/^github:/, "");
    try {
      const board = await cached("board:" + spec, ttl, () => getJson(boardUrl(spec)));
      records.push({ sources: [{ id, kind: "board", name: spec, ok: true, error: null }],
        stats: OC_normalize.boardStats(board, model, id) });
    } catch (e) {
      records.push({ sources: [{ id, kind: "board", name: spec, ok: false, error: String(e.message || e) }], stats: [] });
    }
  }
  for (const st of cfg.studios) {
    const base = String(st.url || "").replace(/\/+$/, "");
    if (!base) continue;
    try {
      const rec = await cached(`studio:${base}:${model}`, Math.min(ttl, 10), () =>
        getJson(`${base}/api/lookup?model=${encodeURIComponent(model)}`,
          st.token ? { Authorization: "Bearer " + st.token } : {}));
      records.push({ ...rec, via: base, open: rec.open ? { hub: base + rec.open.hub, review: base + rec.open.review } : null });
    } catch (e) {
      records.push({ via: base, sources: [{ id: "studio:" + base, kind: "studio", name: base, ok: false,
        error: String(e.message || e) + " (is Studio running, its certificate accepted in this browser, and the token right?)" }],
        stats: [] });
    }
  }
  const out = OC_normalize.merge(model, records);
  out.open = (records.find(r => r.open) || {}).open || null;
  out.configured = cfg.boards.length + cfg.studios.length;
  out.t = Date.now();
  return out;
}

const TAB = new Map();     // tabId -> {model, site, record}

async function light(tabId, rec) {
  const on = !!(rec && rec.available);
  const n = on ? rec.stats.length : 0;
  try {
    await api.action.setIcon({ tabId, path: ICONS(on ? "lit" : "idle") });
    await api.action.setBadgeText({ tabId, text: on ? (n > 99 ? "99+" : String(n)) : "" });
    if (api.action.setBadgeBackgroundColor) await api.action.setBadgeBackgroundColor({ tabId, color: "#1e6bff" });
    await api.action.setTitle({ tabId, title: on ? `OmniCortex: ${n} statistics for ${rec.model}`
      : rec ? `OmniCortex: nothing known about ${rec.model} yet` : "OmniCortex" });
  } catch (e) { /* the tab went away, or this browser has no badge (mobile) */ }
}

async function onPage(tabId, url) {
  const hit = OC_sites.modelFromUrl(url);
  if (!hit) { TAB.delete(tabId); return light(tabId, null); }
  const prev = TAB.get(tabId);
  if (prev && prev.model === hit.model && prev.record) return light(tabId, prev.record);
  TAB.set(tabId, { ...hit, record: null });
  const record = await lookup(hit.model).catch(e => ({ model: hit.model, sources: [], stats: [], available: false, error: String(e) }));
  if ((TAB.get(tabId) || {}).model !== hit.model) return;            // navigated away meanwhile
  TAB.set(tabId, { ...hit, record });
  light(tabId, record);
}

api.runtime.onMessage.addListener((msg, sender, reply) => {
  (async () => {
    if (msg && msg.type === "page" && sender.tab) { await onPage(sender.tab.id, msg.url); return { ok: true }; }
    if (msg && msg.type === "get") {
      const t = TAB.get(msg.tabId);
      if (!t) return { model: null };
      if (msg.force || !t.record) {
        t.record = await lookup(t.model, { force: !!msg.force });
        light(msg.tabId, t.record);
      }
      return { model: t.model, site: t.site, record: t.record };
    }
    if (msg && msg.type === "lookup" && msg.model) return { record: await lookup(msg.model, { force: !!msg.force }) };
    return { error: "unknown message" };
  })().then(reply, e => reply({ error: String(e) }));
  return true;
});

if (api.tabs && api.tabs.onRemoved) api.tabs.onRemoved.addListener(id => TAB.delete(id));
