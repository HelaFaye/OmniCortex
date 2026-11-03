// Which pages show a model, and which model. Shared by the content script, the background and the tests.
// A model is a Hugging Face-style "org/name" id; sites that mirror Hugging Face repos use the same id.
(function (g) {
  const HF_RESERVED = new Set(["models", "datasets", "spaces", "docs", "blog", "settings", "login", "join", "pricing",
    "organizations", "collections", "papers", "posts", "chat", "api", "tasks", "learn", "enterprise", "new",
    "notifications", "search", "inference-endpoints", "hardware", "changelog", "terms-of-service", "privacy"]);
  const ID = /^[A-Za-z0-9][\w.-]{0,95}$/;

  const SITES = [
    // huggingface.co/org/name[/tree/main/...], also hf.co and the hf-mirror.com mirror
    { name: "Hugging Face", hosts: ["huggingface.co", "hf.co", "hf-mirror.com"],
      model(path) { const [o, n] = path; return o && n && !HF_RESERVED.has(o) ? [o, n] : null; } },
    { name: "ModelScope", hosts: ["modelscope.cn", "www.modelscope.cn"],
      model(path) { return path[0] === "models" && path[1] && path[2] ? [path[1], path[2]] : null; } },
    { name: "LM Studio", hosts: ["lmstudio.ai"],
      model(path) { return path[0] === "models" && path[1] && path[2] ? [path[1], path[2]] : null; } },
  ];

  function modelFromUrl(url) {
    let u;
    try { u = new URL(url); } catch (e) { return null; }
    if (u.protocol !== "https:") return null;
    const site = SITES.find(s => s.hosts.includes(u.hostname));
    if (!site) return null;
    const path = u.pathname.split("/").filter(Boolean).map(decodeURIComponent);
    const m = site.model(path);
    if (!m || !ID.test(m[0]) || !ID.test(m[1])) return null;
    return { model: m[0] + "/" + m[1], site: site.name };
  }

  g.OC_sites = { SITES, modelFromUrl };
})(typeof globalThis !== "undefined" ? globalThis : this);
