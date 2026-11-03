const api = globalThis.browser || globalThis.chrome;
const $ = s => document.querySelector(s);

function origin(u) { try { const x = new URL(u); return x.protocol + "//" + x.hostname + "/*"; } catch (e) { return null; } }
function boardOrigin(spec) {
  return /^https?:\/\//.test(spec) ? origin(spec) : null;              // github boards: raw.githubusercontent.com, granted at install
}

(async () => {
  const cfg = await api.storage.sync.get({ boards: [], studios: [], ttl_min: 60 });
  $("#boards").value = cfg.boards.join("\n");
  $("#studios").value = cfg.studios.map(s => (s.url + " " + (s.token || "")).trim()).join("\n");
  $("#ttl").value = cfg.ttl_min;
})();

$("#save").onclick = async () => {
  const boards = $("#boards").value.split("\n").map(s => s.trim()).filter(Boolean);
  const studios = $("#studios").value.split("\n").map(s => s.trim()).filter(Boolean).map(l => {
    const [url, token] = l.split(/\s+/, 2); return { url: url.replace(/\/+$/, ""), token: token || "" };
  });
  const bad = studios.filter(s => !origin(s.url)).map(s => s.url);
  if (bad.length) { $("#msg").innerHTML = `<span class="err">not a URL: ${bad.join(", ")}</span>`; return; }
  // Reaching your own Studio or a scoreboard server needs this browser's permission for that host.
  const origins = [...new Set([...studios.map(s => origin(s.url)), ...boards.map(boardOrigin)].filter(Boolean))];
  let granted = true;
  if (origins.length && api.permissions) {
    try { granted = await api.permissions.request({ origins }); } catch (e) { granted = false; }
  }
  await api.storage.sync.set({ boards, studios, ttl_min: Math.max(1, Math.min(1440, +$("#ttl").value || 60)) });
  $("#msg").innerHTML = granted ? '<span class="ok">saved</span>'
    : '<span class="err">saved, but the browser did not grant access to: ' + origins.join(", ") + "</span>";
};
