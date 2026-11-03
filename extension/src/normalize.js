// The same record scripts/unified_review.py builds (schema omnicortex-lookup/1), for sources the extension
// reads directly (a board's board.json), so the extension works without a Studio. tests/test_unified_review.py
// checks that this file and the Python agree on the same board.
(function (g) {
  const PACKAGING = /[-_.](gguf|ggml|awq|gptq|exl2|mlx|onnx|int4|int8|fp8|bnb[-_]?4bit|\d+bit)$/i;
  const GROUPS = ["quality", "performance", "features", "emotions", "activity"];

  function baseName(repo) {
    let name = String(repo || "").split("/").pop().toLowerCase();
    for (;;) { const n = name.replace(PACKAGING, ""); if (n === name) return name; name = n; }
  }
  function relation(key, repo) {
    if (!repo) return null;
    if (repo.toLowerCase() === key.toLowerCase()) return "same";
    return baseName(repo) === baseName(key) ? "related" : null;
  }
  const stat = (key, group, label, value, source, o = {}) => ({ key, group, label, value, unit: o.unit || "",
    n: o.n === undefined ? null : o.n, source, variant: o.variant || "", related: !!o.related,
    detail: o.detail === undefined ? null : o.detail });
  const num = x => typeof x === "number" && isFinite(x);

  function fromBoard(board, key, source) {
    const rows = Array.isArray(board.rows) ? board.rows : (board.board || []);
    const out = [];
    for (const r of rows) {
      const rel = relation(key, r.hf_repo);
      if (!rel) continue;
      const v = r.variant || {};
      const variant = [rel === "related" ? r.hf_repo : null, r.quant, v.engine, v.lora ? "LoRA" : null,
        num(v.suppression) && v.suppression ? `suppression ${v.suppression}` : null].filter(Boolean).join(" · ");
      const common = { variant, related: rel === "related" };
      const n = r.items === undefined ? null : r.items;
      if (r.accuracy != null) out.push(stat("accuracy", "quality", "Accuracy (TestQA, verified)", r.accuracy, source,
        { ...common, unit: "ratio", n, detail: { submissions: r.submissions === undefined ? null : r.submissions,
          per_subject: r.per_subject === undefined ? null : r.per_subject } }));
      if (r.hallucination_rate != null) out.push(stat("hallucination_rate", "quality", "Confident mistakes",
        r.hallucination_rate, source, { ...common, unit: "ratio", n }));
      for (const s of Object.keys(r.per_subject || {}).sort()) {
        const ps = r.per_subject[s];
        if (ps.accuracy != null) out.push(stat("accuracy:" + s, "quality", "Accuracy: " + s, ps.accuracy, source,
          { ...common, unit: "ratio", n: ps.n === undefined ? null : ps.n }));
      }
      for (const p of r.performance || []) {
        if (!p.gen_tok_s) continue;
        const hw = p.hardware || {};
        const gpus = (hw.gpus || []).map(x => x.name || "?").join(", ") || hw.cpu || "CPU";
        const nz = x => x === undefined ? null : x;
        out.push(stat("gen_tok_s:" + gpus.toLowerCase(), "performance", "Generation speed: " + gpus, p.gen_tok_s, source,
          { ...common, unit: "tok/s", n: nz(p.requests), variant: [variant, gpus].filter(Boolean).join(" · "),
            detail: { hardware: gpus, parallel: nz(p.parallel), ngl: nz(p.ngl), ctx: nz(p.ctx), flash_attn: nz(p.flash_attn) } }));
      }
      const ex = r.extras || {};
      const emo = Object.entries(ex.emotions || {}).sort((a, b) => Math.abs(b[1].mean) - Math.abs(a[1].mean));
      for (const [name, e] of emo) out.push(stat("emotion:" + name, "emotions", name, e.mean, source,
        { ...common, unit: "activation", n: e.n === undefined ? null : e.n, detail: { self_reported: true } }));
      for (const f of ex.features || []) out.push(stat(`feature:${f.sae}:${f.id}`, "features", f.label || "#" + f.id,
        f.freq, source, { ...common, unit: "freq", n: f.n === undefined ? null : f.n,
          detail: { sae: f.sae, id: f.id, self_reported: true } }));
    }
    return out;
  }

  function fromHwReports(board, names, source) {
    const out = [];
    const reps = (board.hw_profiles || {}).reports || {};
    for (const pid of Object.keys(reps).sort()) {
      for (const purpose of Object.keys(reps[pid]).sort()) {
        const a = reps[pid][purpose];
        const hw = new Set((a.hardware || []).map(h => String(h).toLowerCase()));
        if (![...hw].some(h => names.has(h))) continue;
        out.push(stat(`hw:${pid}:${purpose}`, "performance", `Profile ${pid} (${purpose}) worked`,
          a.reports ? Math.round(a.worked / a.reports * 1000) / 1000 : null, source,
          { unit: "ratio", n: a.reports === undefined ? null : a.reports,
            detail: { worked: a.worked === undefined ? null : a.worked, failed: a.failed === undefined ? null : a.failed,
              median_tok_s: a.median_tok_s === undefined ? null : a.median_tok_s } }));
      }
    }
    return out;
  }

  // Board stats plus the hardware reports for the hardware this model was measured on.
  function boardStats(board, key, source) {
    const got = fromBoard(board, key, source);
    const names = new Set();
    for (const s of got) if (s.key.startsWith("gen_tok_s:")) String(s.detail.hardware).toLowerCase().split(", ").forEach(x => names.add(x));
    return got.concat(fromHwReports(board, names, source));
  }

  // Merge several records (from Studios and boards) into one; a source seen twice is kept once.
  function merge(model, records) {
    const seen = new Set(), sources = [], stats = [];
    for (const r of records) {
      for (const s of r.sources || []) {
        if (seen.has(s.id)) continue;
        seen.add(s.id); sources.push({ ...s, via: r.via || null });
        stats.push(...(r.stats || []).filter(x => x.source === s.id).map(x => ({ ...x, via: r.via || null })));
      }
    }
    return { schema: "omnicortex-lookup/1", model, sources, stats, available: stats.length > 0 };
  }

  g.OC_normalize = { GROUPS, baseName, relation, fromBoard, fromHwReports, boardStats, merge };
})(typeof globalThis !== "undefined" ? globalThis : this);
