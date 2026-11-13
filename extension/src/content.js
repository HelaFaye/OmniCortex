// Runs on model sites only: tells the background which page this is (sites change pages without reloading).
(function () {
  const api = globalThis.browser || globalThis.chrome;
  let last = "";
  function tell() {
    if (location.href === last) return;
    last = location.href;
    try { api.runtime.sendMessage({ type: "page", url: location.href }); } catch (e) { /* extension reloaded */ }
  }
  tell();
  addEventListener("popstate", tell);
  setInterval(tell, 1500);
})();
