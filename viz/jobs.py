#!/usr/bin/env python3
"""
Background jobs for Studio: evaluation, retraining and benchmarks from the
browser, without a free-form command line.

Each job kind is a fixed script plus a schema of typed fields. The argv is
built from the schema only: unknown fields are ignored, booleans become bare
flags, numbers must parse, and no value may start with "-" (so a value can
never smuggle in an extra flag). Jobs run as subprocesses with their output in
a log file under ~/.neuronscope/jobs/<id>/, survive the browser closing, and
can be cancelled.

Studio serves the page at /jobs and the API at /api/jobs. Because a job can run
training or execute model-written code (TestQA --allow-exec), Studio allows
jobs only on a loopback bind unless started with --allow-remote-jobs.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"


def _installable() -> list[str]:
    """Packages the install_package job may install: scripts/ns_requirements.py's catalog."""
    sys.path.insert(0, str(S))
    import ns_requirements
    return [p for p in ns_requirements.PACKAGES if p not in ns_requirements.NOT_PIP]


def F(name, flag=None, kind="str", req=False, default=None, help="", choices=None, multi=False, repeat=False):
    """multi: several values after one flag (nargs); repeat: the flag once per value (action=append)."""
    return {"name": name, "flag": flag if flag is not None else "--" + name.replace("_", "-"), "kind": kind,
            "required": req, "default": default, "help": help, "choices": choices, "multi": multi or repeat,
            "repeat": repeat}


# kind -> (title, group, script argv prefix, fields)
SPECS: dict[str, dict] = {
    "testqa": {
        "title": "TestQA run", "group": "Evaluate", "argv": [S / "testqa.py"],
        "help": "Graded bank against an endpoint; publish stats so auto routing sees them.",
        "fields": [
            F("endpoint", req=True, repeat=True, help="label=URL@model, e.g. m=http://127.0.0.1:7870/v1@my-model"),
            F("concurrency", default="auto", help="requests in flight per server: a number, or auto (grows while "
              "throughput rises, up to the server's slots)"),
            F("parallel", repeat=True, help="one endpoint's own concurrency: label=N or label=auto, e.g. slow=1"),
            F("pool", repeat=True, help="another server sharing an endpoint's tasks (same model): label=URL@model; "
              "whichever is free takes the next"),
            F("per_subject", kind="int", help="items per subject (0 = all)"),
            F("subject", multi=True, help="limit to these subjects"),
            F("allow_exec", kind="bool", help="run model-written code"),
            F("sandbox", choices=["none", "docker", "podman"], default="none",
              help="contain that code in a container (recommended)"),
            F("cache", help="cache directory (needed by deficits)"),
            F("out", req=True, help="results JSON"),
            F("publish_stats", help="Studio URL to publish stats to, e.g. http://127.0.0.1:7870"),
        ]},
    "deficits": {
        "title": "Deficit dataset", "group": "Retrain",
        "argv": [S / "deficits.py"], "help": "Failures -> verified SFT/DPO data with replay and a holdout.",
        "fields": [
            F("results", req=True, help="TestQA results JSON"), F("label"), F("cache"),
            F("teacher", help="URL@model of a stronger model"),
            F("expand", kind="int", default=0, help="verified variations per failure"),
            F("student", help="URL@model of the model being retrained: probes which failed facts it knows "
                              "(unknown ones are trained to decline) and answers variations for DPO"),
            F("probe_samples", kind="int", default=4, help="samples per failed fact"),
            F("unknown", choices=["abstain", "skip", "teach"], default="abstain",
              help="facts the model does not know: train a decline, leave out, or teach anyway"),
            F("review_model", help="a model in the neuron review: find the layers to focus training on"),
            F("allow_exec", kind="bool"), F("sandbox", choices=["none", "docker", "podman"], default="none"),
            F("deficit_fraction", kind="float", default=0.25),
            F("general", multi=True, help="extra anchor JSONL files"),
            F("out", req=True)]},
    "finetune": {
        "title": "Fine-tune", "group": "Retrain", "argv": [S / "finetune.py"],
        "help": "QLoRA / LoRA / full SFT, then optional DPO. HF weights, not GGUF.",
        "fields": [
            F("model", req=True, help="HF id or local path"), F("data", req=True, help="deficits output dir"),
            F("out", req=True), F("method", choices=["qlora", "lora", "full"], default="qlora"),
            F("dpo", kind="bool"), F("vision", kind="bool"), F("freeze_projector", kind="bool"),
            F("launch", kind="int", help="GPUs (accelerate)"), F("fsdp", kind="bool"),
            F("layers", default="all", help="all, a list like 10-17, or plan (the focus layers from the dataset)"),
            F("lora_r", kind="int", default=16), F("epochs", kind="float", default=2),
            F("max_steps", kind="int", default=-1), F("max_length", kind="int", default=2048),
            F("gradient_checkpointing", kind="bool")]},
    "engine_optimize": {
        "title": "Find the fastest engine settings", "group": "Setup", "argv": [S / "engine_optimizer.py"],
        "help": "Measure tokens/s over threads, batch, flash attention, KV types... and keep the best (docs/ENGINE.md).",
        "fields": [
            F("model", req=True, help="the GGUF to tune for"), F("out", req=True, help="results folder"),
            F("trials", kind="int", default=24),
            F("objective", choices=["gen", "prompt", "request"], default="gen",
              help="generation speed, prompt speed, or the time of a whole request"),
            F("prompt_tokens", kind="int", default=512), F("gen_tokens", kind="int", default=128),
            F("kv", kind="bool", help="also try quantized KV caches"),
            F("spec", kind="bool", help="also try draft-free speculative decoding"),
            F("placement", kind="bool", help="also try strict CPU placement"),
            F("ngl", kind="int", default=99), F("server", help="llama-server (default from the workspace)"),
            F("apply", help="Studio's settings file, to use the winner (e.g. ~/.neuronscope/studio.json)")]},
    "retrain_cycle": {
        "title": "Retraining cycle", "group": "Retrain", "argv": [S / "retrain_cycle.py"],
        "help": "Measure, curate, train each arm, convert, measure on the holdout, keep or reject. Resumes.",
        "fields": [
            F("out", req=True, help="the cycle's folder (an existing one resumes)"),
            F("base_model", req=True, help="HF id or path of the base weights"),
            F("base_gguf", help="the base model as GGUF, served here for a like-for-like baseline"),
            F("base_endpoint", help="or: URL@model where the base model is already served"),
            F("teacher", help="URL@model of a stronger model"),
            F("arms", default="all", help="layer choices to compare: all, plan, 10-17 (comma list)"),
            F("review_model", help="neuron review model, for the plan arm"),
            F("method", choices=["qlora", "lora", "full"], default="lora"), F("dpo", kind="bool"),
            F("expand", kind="int", default=0), F("per_subject", kind="int", help="TestQA items per subject"),
            F("subject", multi=True), F("allow_exec", kind="bool"),
            F("sandbox", choices=["none", "docker", "podman"], default="docker"),
            F("quant", default="Q4_K_M"), F("llama", help="llama.cpp checkout (default $NS_LLAMA)")]},
    "decode_lab": {
        "title": "Decoding lab (SLED, steering, features, emotions)", "group": "Evaluate", "argv": [S / "decode_lab.py"],
        "help": "Serve one Hugging Face model as several decoding variants on an OpenAI-compatible port, for TestQA "
                "and the retrain gate to compare; it also answers Studio's Features page (reading replies, graphs, "
                "steering). Runs until cancelled (docs/DECODING.md, docs/FEATURES.md).",
        "fields": [
            F("model", req=True, help="HF id or folder"),
            F("variant", repeat=True, help="name={json}, each in single quotes, e.g. 'plain={}' 'sled={\"sled\": {\"alpha\": 2}}'; "
                                           "default: plain"),
            F("port", kind="int", default=8090), F("dtype", choices=["auto", "float32", "bfloat16", "float16"],
                                                   default="auto"),
            F("watch_cett", kind="bool", help="each MLP neuron's CETT for every generated token"),
            F("watch_emotions", help="emotion vectors folder: their activations per token"),
            F("watch_sae", repeat=True, help="feature dictionary folder: the top features per token"),
            F("publish", help="live viewer to stream those to as tokens are generated, e.g. http://127.0.0.1:7890"),
            F("publish_token_file", help="the viewer's access token, if it has one"),
            F("record", help="save each reply as a trace session for the 3D views. Studio fills in its own traces "
                             "folder, which the Lab's Replay and Godot's 3D tab list")]},
    "rp_fit": {
        "title": "Random-projection steering: fit", "group": "Evaluate", "argv": [S / "rp_steer.py", "fit"],
        "help": "Project layers' MLP outputs to a few dozen random dimensions and find the direction that separates "
                "wrong answers from right ones; held-out AUC per layer against the full space. Steer with the "
                "decoding lab's rp variant (docs/DECODING.md).",
        "fields": [F("model", req=True, help="HF id or folder"),
                   F("data", help="JSONL of prompt or messages, reply, label right|wrong"),
                   F("testqa", help="or TestQA results JSON"), F("endpoint", help="which endpoint in those results"),
                   F("layers", multi=True, help="default: half and three quarters of the depth"),
                   F("dim", kind="int", default=48), F("projections", kind="int", default=5),
                   F("dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto"),
                   F("out", req=True, help="e.g. runs/rp.pt")]},
    "rp_export": {
        "title": "Random-projection steering: llama.cpp control vector", "group": "Evaluate",
        "argv": [S / "rp_steer.py", "export"],
        "help": "The fitted directions as a control vector GGUF for llama-server --control-vector-scaled. Fixed at "
                "every step, not gated like the decoding lab's rp variant.",
        "fields": [F("file", req=True, help="rp.pt from the fit"), F("out", req=True, help="e.g. runs/rp-cvec.gguf")]},
    "logit_lens": {
        "title": "Logit lens over a reply", "group": "Evaluate", "argv": [S / "logit_lens.py"],
        "help": "What every layer would say next for each token of a reply, how spread out it is, and where the "
                "answer settles; late settling with a spread-out final layer is marked (docs/CONFIDENCE.md).",
        "fields": [
            F("model", req=True, help="HF id or folder"), F("prompt", help="the question"),
            F("messages", help="or a JSON file with the chat before the reply"),
            F("reply", help="the reply to look into"), F("reply_file", help="or a file with it"),
            F("generate", kind="int", default=0, help="or let the model write one of up to N tokens"),
            F("late", kind="float", default=0.75), F("spread", kind="float", default=0.55),
            F("out", req=True, help="result JSON"), F("html", help="heatmap page to write"),
            F("dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")]},
    "concept_edit": {
        "title": "Concept edit (subspace ablation)", "group": "Retrain", "argv": [S / "concept_edit.py"],
        "help": "Find what separates two prompt sets and project it out of W_O/W_out, norm-preserving, with a KL "
                "damage meter; --search optimizes it (docs/CONCEPT_EDITING.md).",
        "fields": [
            F("model", req=True, help="HF id or folder"), F("concept", req=True, help="prompts with the concept (A)"),
            F("baseline", req=True, help="matched prompts without it (B)"),
            F("neutral", help="ordinary prompts for the KL damage meter"), F("out", req=True),
            F("method", choices=["mean", "pca"], default="mean"), F("k", kind="int", default=1),
            F("search", kind="int", default=0, help="optimization trials (0: one edit with the values below)"),
            F("max_kl", kind="float", default=0.0), F("kl_weight", kind="float", default=1.0),
            F("kernel", help="max_w,pos,min_w,dist"), F("direction_layer", kind="float"),
            F("marker", help="regex over replies: the behaviour to reduce"),
            F("analyze", kind="bool", help="only measure where the concept lives")]},
    "sage_train": {
        "title": "SAGE: adapt with a protected subspace", "group": "Retrain", "argv": [S / "sage_edit.py", "train"],
        "help": "Gated per-head activation edits for a task, kept orthogonal to a protected subspace "
                "(e.g. the 'I don't know' direction).",
        "fields": [
            F("model", req=True), F("data", req=True, help="deficits output folder or JSONL"), F("out", req=True),
            F("protect", help="directions.pt from a concept edit analysis"),
            F("protect_concept", help="or: prompts with the behaviour to protect"),
            F("protect_baseline", help="and matched prompts without it"),
            F("k", kind="int", default=1), F("method", choices=["mean", "pca"], default="mean"),
            F("epochs", kind="float", default=2), F("lr", kind="float", default=0.005),
            F("l0", kind="float", default=0.01), F("layers")]},
    "sage_export": {
        "title": "SAGE: export a control vector", "group": "Retrain", "argv": [S / "sage_edit.py", "export"],
        "help": "The additive edits as a llama.cpp control vector GGUF (approximate; see the docs).",
        "fields": [F("edits", req=True, help="edits.pt"), F("cvector", req=True, help="output .gguf")]},
    "retrain_gate": {
        "title": "Keep or reject a retrain", "group": "Retrain", "argv": [S / "retrain_gate.py"],
        "help": "Before/after on the holdout: fixed vs broken items, hallucination, per subject; a verdict.",
        "fields": [
            F("results", req=True, help="TestQA results JSON with the retrained model"),
            F("after", help="its endpoint label"), F("before", help="the original model's endpoint label"),
            F("before_results", help="a separate results JSON for the original model"),
            F("holdout", help="holdout_ids.json from the deficit dataset"),
            F("alpha", kind="float", default=0.05), F("out", help="comparison JSON")]},
    "merge": {
        "title": "Merge and convert", "group": "Retrain", "argv": [S / "merge_export.py"],
        "help": "Adapter -> merged model -> (quantized) GGUF, or a GGUF LoRA.",
        "fields": [
            F("base", req=True), F("adapter", req=True), F("out", req=True),
            F("gguf", help="quant type, e.g. Q4_K_M (empty: no GGUF)"), F("lora_gguf", kind="bool"),
            F("vision", kind="bool"), F("llama", help="llama.cpp checkout (default $NS_LLAMA)")]},
    "swe_predict": {
        "title": "SWE-bench predict", "group": "Benchmarks", "argv": [S / "benchmarks.py", "swebench", "predict"],
        "help": "Single-shot retrieval patches; score with the Docker harness.",
        "fields": [F("endpoint", req=True, help="URL@model"),
                   F("dataset", default="princeton-nlp/SWE-bench_Lite_bm25_13K"), F("split", default="test"),
                   F("limit", kind="int", default=50), F("out", req=True)]},
    "swe_evaluate": {
        "title": "SWE-bench evaluate", "group": "Benchmarks", "argv": [S / "benchmarks.py", "swebench", "evaluate"],
        "help": "Runs swebench.harness (needs Docker).",
        "fields": [F("predictions", req=True), F("dataset", default="princeton-nlp/SWE-bench_Lite"),
                   F("run_id", req=True), F("max_workers", kind="int", default=2)]},
    "swe_import": {
        "title": "SWE-bench import", "group": "Benchmarks", "argv": [S / "benchmarks.py", "swebench", "import"],
        "help": "Harness report -> stats.",
        "fields": [F("report", req=True), F("model", req=True), F("publish_stats")]},
    "swe_train": {
        "title": "SWE-bench train data", "group": "Retrain", "argv": [S / "benchmarks.py", "swebench", "train-data"],
        "help": "Train split gold patches -> SFT/DPO, prioritised by a test report.",
        "fields": [F("dataset", default="princeton-nlp/SWE-bench_bm25_13K"), F("predictions"), F("report"),
                   F("exclude", multi=True), F("anchors"), F("limit", kind="int", default=0), F("out", req=True)]},
    "livebench": {
        "title": "LiveBench run", "group": "Benchmarks", "argv": [S / "benchmarks.py", "livebench", "run"],
        "help": "Needs a LiveBench checkout.",
        "fields": [F("livebench", req=True), F("endpoint", req=True, help="URL@model"),
                   F("bench", multi=True, default=["live_bench"]), F("publish_stats")]},
    "quant_select": {
        "title": "Choose a quantization", "group": "Benchmarks", "argv": [S / "quant_select.py"],
        "help": "Measure fit, speed and fidelity of a model's quantizations on this computer; keep the fastest faithful one.",
        "fields": [F("model", multi=True, help="GGUF files of one model in different quantizations"),
                   F("from", help="or a high-precision GGUF to make them from"),
                   F("make", help="types to make, e.g. Q8_0,Q6_K,Q4_K_M,TQ2_0"), F("imatrix"),
                   F("objective", choices=["gen", "prompt", "request"], default="gen"),
                   F("tune", kind="int", default=0, help="Bayesian engine-option trials per file"),
                   F("memory_gb", kind="float", help="default: the largest accelerator's free memory, or RAM"),
                   F("fidelity", choices=["auto", "kld", "agreement"], default="auto"),
                   F("max_kld", kind="float", default=0.1), F("min_agreement", kind="float", default=0.6),
                   F("server", help="llama-server"), F("out", req=True)]},
    "clip_bench": {
        "title": "CLIP_benchmark", "group": "Benchmarks", "argv": [S / "clip_bench.py", "eval"],
        "help": "Zero-shot classification at each suppression scale.",
        "fields": [F("model", req=True), F("dataset", req=True, multi=True, help="e.g. imagenetv2 imagenet_sketch"),
                   F("dataset_root", req=True), F("h_neurons", flag="--h_neurons"), F("scales", multi=True, default=["1"]),
                   F("style", choices=["off", "auto", "pixel", "cartoon"], default="off", help="pixel art and cartoons: clean upscaling, padding, style prompts"), F("out", req=True)]},
}

# ---- interpretability: feature dictionaries, circuits and emotion vectors (docs/FEATURES.md)
SITE_CHOICES = ["resid", "mlp_in", "mlp_out", "neurons"]
DTYPES = ["auto", "float32", "bfloat16", "float16"]
SPECS.update({
    "feat_collect": {
        "title": "Features 1: record activations", "group": "Interpret", "argv": [S / "features.py", "collect"],
        "help": "Run a model over text and keep its activations at chosen layers, for training dictionaries. "
                "Several hundred thousand tokens for a first look; millions for a good dictionary.",
        "fields": [F("model", req=True, help="HF id or folder"),
                   F("corpus", req=True, help="folder of .txt/.md, a .txt or .jsonl, or a Hub dataset id "
                                              "(e.g. NeelNanda/pile-10k)"),
                   F("field", help="text field of the rows (default: text, messages, prompt+reply)"),
                   F("split", default="train"), F("config", help="dataset config"),
                   F("site", multi=True, choices=SITE_CHOICES, default=["resid"],
                     help="resid: residual stream after the layer; mlp_in / mlp_out: an MLP's input / output "
                          "(both, for transcoders); neurons: the MLP's hidden neurons (H-Neuron space)"),
                   F("layers", multi=True, help="decoder layers (default: the middle one)"),
                   F("max_tokens", kind="int", default=400000), F("ctx", kind="int", default=256),
                   F("batch", kind="int", default=8), F("dtype", choices=DTYPES, default="auto"),
                   F("device", default="auto", help="auto, cpu, cuda, cuda:1, mps"),
                   F("out", req=True, help="e.g. runs/features/acts-mymodel")]},
    "feat_train": {
        "title": "Features 2: train a dictionary", "group": "Interpret", "argv": [S / "features.py", "train"],
        "help": "A sparse autoencoder on one site and layer, or a transcoder (site mlp_in, target mlp_out) for "
                "attribution graphs. Reports held-out reconstruction and the model's loss with it spliced in.",
        "fields": [F("acts", req=True, help="folder from step 1"), F("site", choices=SITE_CHOICES, default="resid"),
                   F("layer", kind="int", req=True),
                   F("target_site", choices=SITE_CHOICES, help="for a transcoder: what it reconstructs (mlp_out)"),
                   F("arch", choices=["topk", "relu", "jumprelu"], default="topk",
                     help="topk: k features per token (Gao et al.); relu: L1 with decoder norms (Anthropic); "
                          "jumprelu: learned thresholds (Gemma Scope)"),
                   F("k", kind="int", default=32), F("expansion", kind="float", default=8.0,
                                                     help="features per dimension of the layer"),
                   F("features", kind="int", default=0, help="or an exact size"),
                   F("target_l0", kind="float", default=32.0, help="relu / jumprelu: active features per token"),
                   F("lr", kind="float", default=2e-4), F("batch", kind="int", default=4096),
                   F("epochs", kind="float", default=3.0), F("steps", kind="int", default=0),
                   F("device", default="auto"), F("out", req=True, help="e.g. runs/features/mymodel-resid-12")]},
    "feat_import": {
        "title": "Features: import a dictionary", "group": "Interpret", "argv": [S / "features.py", "import"],
        "help": "Use a dictionary trained elsewhere: SAELens folders or any SAELens pretrained release by name, "
                "Gemma Scope params.npz, or safetensors with W_enc / W_dec.",
        "fields": [F("path", help="a folder or file (or use release and sae_id)"),
                   F("release", help="a SAELens pretrained release, e.g. gemma-scope-2b-pt-res: downloaded"),
                   F("sae_id", help="the dictionary in that release, e.g. layer_12/width_16k/average_l0_82"),
                   F("model", help="the model it belongs to"),
                   F("site", choices=SITE_CHOICES), F("layer", kind="int"),
                   F("target_site", choices=SITE_CHOICES), F("acts", help="activations to interpret it on"),
                   F("out", req=True)]},
    "iv_trace": {
        "title": "Interventions: trace a prediction", "group": "Interpret", "argv": [S / "interventions.py", "trace"],
        "help": "The predicted token, and the attention heads and MLP neurons that drove it (direct logit attribution).",
        "fields": [F("model", req=True), F("prompt", req=True), F("top_k", kind="int", default=10),
                   F("target", help="a token to explain instead of the prediction; #ID names a token id"),
                   F("device", default="auto"), F("out", help="JSON output file")]},
    "iv_patch": {
        "title": "Interventions: ablate a component", "group": "Interpret", "argv": [S / "interventions.py", "patch"],
        "help": "Zero- or mean-ablate a layer's residual, attention or MLP output, a head, a neuron or an SAE "
                "feature at chosen positions, and see how the prediction moves.",
        "fields": [F("model", req=True), F("prompt", req=True), F("layer", kind="int", req=True),
                   F("component", choices=["resid_pre", "resid_post", "attn_out", "mlp_out", "mlp_post", "head", "neuron"],
                     default="mlp_out"),
                   F("index", kind="int", help="the head or neuron"), F("mode", choices=["zero", "mean"], default="zero"),
                   F("positions", default="last", help="last, all, 3, 2,5,-1 or 0:4"),
                   F("mean_prompt", repeat=True, help="prompts to average over for mean ablation"),
                   F("sae", help="an SAE folder: set one of its features instead"), F("feature", kind="int"),
                   F("value", kind="float", default=0.0), F("device", default="auto"), F("out")]},
    "iv_act_patch": {
        "title": "Interventions: activation patching (clean vs corrupted)", "group": "Interpret",
        "argv": [S / "interventions.py", "act-patch"],
        "help": "Put clean activations into the corrupted run one at a time, by layer and position or by head, "
                "and see which restore the clean answer.",
        "fields": [F("model", req=True), F("clean", req=True), F("corrupt", req=True),
                   F("target", req=True, help="the clean answer's token"), F("baseline", help="the corrupted answer's token"),
                   F("by", choices=["resid", "attn", "mlp", "head"], default="resid"),
                   F("layers", default="all"), F("positions", default="all"), F("device", default="auto"), F("out")]},
    "iv_path_patch": {
        "title": "Interventions: path patching", "group": "Interpret", "argv": [S / "interventions.py", "path-patch"],
        "help": "A sender head's corrupted output reaches only the receivers you name; everything else stays clean.",
        "fields": [F("model", req=True), F("clean", req=True), F("corrupt", req=True), F("target", req=True),
                   F("baseline"), F("sender", req=True, help="LAYER.HEAD, e.g. 9.6"),
                   F("receiver", req=True, repeat=True, help="LAYER:attn or LAYER:mlp"),
                   F("device", default="auto"), F("out")]},
    "iv_circuit": {
        "title": "Interventions: circuit sketch", "group": "Interpret", "argv": [S / "interventions.py", "circuit"],
        "help": "Candidates by attribution, then each one's ablation effect (and patching effect, with a corrupted "
                "prompt).",
        "fields": [F("model", req=True), F("prompt", req=True), F("top_k", kind="int", default=10),
                   F("corrupt"), F("baseline"), F("device", default="auto"), F("out")]},
    "feat_interpret": {
        "title": "Features 3: describe and label", "group": "Interpret", "argv": [S / "features.py", "interpret"],
        "help": "Top examples and examples across activation levels, logit effects, neighbours and a map for every "
                "feature; optionally labels from a model, scored for specificity.",
        "fields": [F("sae", req=True, help="dictionary folder"), F("acts", help="default: the training activations"),
                   F("labeler", help="URL@model of an OpenAI-compatible endpoint that writes labels"),
                   F("label", kind="int", default=100, help="how many features to label"),
                   F("label_ids", help="or exactly these ids, comma-separated"),
                   F("score", choices=["specificity", "detection", "none"], default="specificity"),
                   F("top", kind="int", default=12), F("buckets", kind="int", default=5),
                   F("keep_codes", kind="int", default=64, help="active features kept per token for browsing"),
                   F("device", default="auto")]},
    "feat_neurons": {
        "title": "Features 4: map onto neurons", "group": "Interpret", "argv": [S / "features.py", "neurons"],
        "help": "Which MLP neurons write and read each feature (weights), which correlate with it (when neurons were "
                "recorded at that layer), and how much of it sits on H-Neurons.",
        "fields": [F("sae", req=True), F("h_neurons", help="h_neurons.json from the classifier"),
                   F("acts"), F("top", kind="int", default=12), F("device", default="auto")]},
    "feat_graph": {
        "title": "Features 5: attribution graph", "group": "Interpret", "argv": [S / "feature_circuits.py", "graph"],
        "help": "Which features the model used for one output and how they connect: transcoders in place of MLPs, "
                "attention and norms frozen, edges pruned by influence (Anthropic's circuit tracing).",
        "fields": [F("transcoders", req=True, multi=True, help="transcoder folders (globs allowed)"),
                   F("prompt", req=True), F("chat", kind="bool"), F("target", repeat=True, help="output tokens"),
                   F("max_features", kind="int", default=400), F("node_threshold", kind="float", default=0.8),
                   F("edge_threshold", kind="float", default=0.98), F("model"), F("device", default="auto"),
                   F("out", req=True, help="e.g. runs/graphs/my-prompt.json")]},
    "feat_attribute": {
        "title": "Features: what drives a completion", "group": "Interpret",
        "argv": [S / "feature_circuits.py", "attribute"],
        "help": "Attribution of one completion (against a baseline) to a dictionary's features, checked by ablating "
                "the top ones.",
        "fields": [F("sae", req=True), F("prompt", req=True), F("token", req=True, help="e.g. ' Austin'"),
                   F("baseline", help="a competing completion"), F("chat", kind="bool"),
                   F("ablate", kind="int", default=10), F("model"), F("device", default="auto"),
                   F("out", req=True)]},
    "feat_intervene": {
        "title": "Features: intervene", "group": "Interpret", "argv": [S / "feature_circuits.py", "intervene"],
        "help": "Change transcoder features and compare the next-token predictions.",
        "fields": [F("transcoders", req=True, multi=True), F("prompt", req=True), F("chat", kind="bool"),
                   F("set", repeat=True, req=True, help="LAYER:FEATURE=VALUE, =xM to multiply, @POS for one position"),
                   F("model"), F("device", default="auto"), F("out")]},
    "emo_stories": {
        "title": "Emotions 1: write stories", "group": "Interpret", "argv": [S / "emotions.py", "stories"],
        "help": "A model writes short stories in which a character feels each emotion without naming it, and neutral "
                "dialogues (the emotions paper's dataset recipe).",
        "fields": [F("writer", help="URL@model of an OpenAI-compatible endpoint"),
                   F("writer_model", help="or a local Hugging Face model"),
                   F("preset", choices=["paper", "core"], default="paper", help="paper: 171 emotions; core: 24"),
                   F("emotions", multi=True, help="or these emotions"),
                   F("topics", kind="int", default=8), F("per_call", kind="int", default=3),
                   F("neutral_topics", kind="int", default=20), F("topics_file"),
                   F("out", req=True, help="e.g. runs/emotions/stories")]},
    "emo_build": {
        "title": "Emotions 2: build vectors", "group": "Interpret", "argv": [S / "emotions.py", "build"],
        "help": "Residual stream means over each emotion's stories, minus the mean over emotions, with neutral "
                "principal components projected out; held-out accuracy per layer.",
        "fields": [F("model", req=True), F("stories", req=True), F("layers", multi=True, help="default: all"),
                   F("start", kind="int", default=50), F("denoise", kind="float", default=0.5),
                   F("judge", help="URL@model that rates valence and arousal, for the map"),
                   F("dtype", choices=DTYPES, default="auto"), F("device", default="auto"),
                   F("out", req=True, help="e.g. runs/emotions/mymodel")]},
    "emo_scan": {
        "title": "Emotions: scan a corpus", "group": "Interpret", "argv": [S / "emotions.py", "scan"],
        "help": "The passages that activate each emotion vector most.",
        "fields": [F("emotions", req=True), F("corpus", req=True), F("field"), F("split", default="train"),
                   F("limit_docs", kind="int", default=500), F("layer", kind="int"), F("top", kind="int", default=5),
                   F("device", default="auto"), F("out", req=True)]},
    "emo_sweep": {
        "title": "Emotions: sweep a quantity", "group": "Interpret", "argv": [S / "emotions.py", "sweep"],
        "help": "Emotion activation just before the reply as one number in the prompt changes.",
        "fields": [F("emotions", req=True), F("template", req=True, help="prompt with {x}"),
                   F("values", multi=True, req=True), F("emotion", multi=True), F("layer", kind="int"),
                   F("device", default="auto"), F("out")]},
    "emo_steer": {
        "title": "Emotions: steer one reply", "group": "Interpret", "argv": [S / "emotions.py", "steer"],
        "help": "A reply with and without an emotion mix added (strength: a fraction of the residual stream's norm; "
                "the paper sweeps -0.1 to 0.1).",
        "fields": [F("emotions", req=True), F("prompt", req=True),
                   F("mix", repeat=True, req=True, help="emotion=strength, e.g. calm=0.05"),
                   F("layers", multi=True), F("max_tokens", kind="int", default=80), F("device", default="auto"),
                   F("out")]},
    "emo_cvec": {
        "title": "Emotions: llama.cpp control vector", "group": "Interpret", "argv": [S / "emotions.py", "cvec"],
        "help": "An emotion mix as a control vector GGUF for llama-server --control-vector.",
        "fields": [F("emotions", req=True), F("mix", repeat=True, req=True), F("layers", multi=True),
                   F("out", req=True, help="e.g. runs/emotions/calm.gguf")]},
})

def U(name, **kw):
    """A field whose flag keeps the underscores (--input_path), as the pipeline scripts spell them."""
    return F(name, flag="--" + name, **kw)


TRUST = U("no_trust_remote_code", kind="bool", help="refuse custom model code from the Hub")
LOCS = ["input", "output", "answer_tokens", "all_except_answer_tokens"]

SPECS.update({
    # ---- setup
    "build_llama": {
        "title": "Build llama.cpp", "group": "Setup", "interp": ["bash"],
        "argv": [S / "build_llama_tools.sh"],
        "help": "Clone llama.cpp, add cett-dump, build llama-server and the tools (~5 min on CPU). "
                "Then set the paths it prints on the Setup page.",
        "fields": [F("backend", choices=["vulkan", "cpu", "cuda", "hip", "metal"], default="vulkan",
                     help="vulkan runs on AMD (incl. APUs), Intel and NVIDIA"),
                   F("dir", help="checkout folder (default ~/llama.cpp)"),
                   F("ref", help="llama.cpp tag or commit (default: tested)"),
                   F("cuda_arch", help="CUDA architectures, e.g. 50-real;61-real (default: this machine's)"),
                   F("portable", kind="bool", help="no -march=native: the binaries run on other CPUs"),
                   F("server_activations", kind="bool", help="also patch llama-server to stream /activations")]},
    "install_package": {
        "title": "Install a Python package", "group": "Setup", "argv": [S / "ns_requirements.py", "install"],
        "help": "pip install one package OmniCortex uses, at the version requirements*.txt asks for, into Studio's "
                "Python. Only packages from the catalog (scripts/ns_requirements.py); torch comes from install.sh.",
        "fields": [F("package", flag="", req=True, choices=_installable())]},
    "env_snapshot": {
        "title": "Snapshot the environment", "group": "Setup", "argv": [S / "ns_requirements.py", "snapshot"],
        "help": "Record installed packages, programs, llama.cpp builds and devices, to compare after an update "
                "(Setup → Requirements → compare).", "fields": []},
    # ---- the text H-Neuron pipeline (docs/PIPELINE.md)
    "preflight": {
        "title": "0 · Preflight", "group": "H-Neurons", "argv": [S / "preflight.py"],
        "help": "Layer count, neuron count and the RAM the classifier will need, without loading weights.",
        "fields": [U("model_path", req=True, help="HF id or local folder, e.g. Qwen/Qwen3-8B"),
                   U("n_pairs", kind="int", default=400)]},
    "collect": {
        "title": "1 · Collect answers", "group": "H-Neurons", "argv": [S / "collect_responses_lmstudio.py"],
        "help": "Ask each TriviaQA question several times; keep consistent right and consistent wrong answers.",
        "fields": [U("base_url", default="http://127.0.0.1:7870/v1", help="any OpenAI-compatible server, e.g. Studio"),
                   U("model", req=True, help="model id on that server"),
                   U("data_path", default="data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet",
                     help="TriviaQA parquet (see data/README.md)"),
                   U("output_path", default="data/consistency_samples.jsonl"),
                   U("sample_num", kind="int", default=5), U("max_questions", kind="int", default=200),
                   U("concurrency", kind="int", default=4)]},
    "tag": {
        "title": "2 · Tag answer tokens", "group": "H-Neurons", "argv": [S / "make_answer_tokens.py"],
        "help": "Find each answer's tokens in the model's tokenization.",
        "fields": [U("input_path", default="data/consistency_samples.jsonl"),
                   U("output_path", default="data/answer_tokens.jsonl"),
                   U("model_path", req=True, help="HF id or folder with the tokenizer"), TRUST]},
    "split": {
        "title": "3 · Balanced split", "group": "H-Neurons", "argv": [S / "sample_balanced_ids.py"],
        "help": "Equal numbers of right and wrong questions. Run once for train, again for test with exclude.",
        "fields": [U("input_path", default="data/answer_tokens.jsonl"),
                   U("output_path", default="data/train_qids.json"), U("num_samples", kind="int", default=400),
                   U("exclude", help="e.g. data/train_qids.json when making the test split"),
                   U("seed", kind="int")]},
    "extract_gguf": {
        "title": "4 · Extract CETT (llama.cpp)", "group": "H-Neurons", "argv": [S / "extract_activations_gguf.py"],
        "help": "Activations from a GGUF with cett-dump: no PyTorch, any GPU llama.cpp supports. Use a Q8_0/F16 file.",
        "fields": [U("gguf", help="model file (default $NS_GGUF)"), U("binary", help="llama-cett-dump (default $NS_CETT)"),
                   U("input_path", default="data/answer_tokens.jsonl"), U("ids_path", default="data/train_qids.json"),
                   U("output_root", default="data/activations"),
                   U("locations", multi=True, choices=LOCS, default=["answer_tokens", "all_except_answer_tokens"]),
                   U("ngl", kind="int", default=99), U("batch", kind="int", default=4096), TRUST]},
    "extract_torch": {
        "title": "4 · Extract CETT (PyTorch)", "group": "H-Neurons", "argv": [S / "extract_activations.py"],
        "help": "Activations from Hugging Face weights; needs PyTorch.",
        "fields": [U("model_path", req=True), U("input_path", default="data/answer_tokens.jsonl"),
                   U("ids_path", default="data/train_qids.json"), U("output_root", default="data/activations"),
                   U("locations", multi=True, choices=LOCS, default=["answer_tokens", "all_except_answer_tokens"]),
                   U("gpu_mem", help="e.g. 14GiB; empty = CPU"), U("cpu_mem", help="e.g. 40GiB"), TRUST]},
    "harvest_train": {
        "title": "Fix This Mistake: micro-tune on the kept pairs", "group": "Retrain",
        "argv": [S / "harvest.py", "train"],
        "help": "A short SFT + DPO run (finetune.py) on the corrections kept with Fix this mistake that no earlier "
                "run used. Writes an adapter; nothing is installed.",
        "fields": [F("model", req=True, help="HF id or path of the base model the chat model came from"),
                   F("out", req=True), F("steps", kind="int", default=40), F("min", kind="int", default=1,
                   help="do nothing unless this many new pairs wait"), F("method", choices=["qlora", "lora"], default="qlora")]},
    "dep_gc": {
        "title": "Clean up build and download leftovers", "group": "Setup", "argv": [S / "dep_gc.py"],
        "help": "Interrupted downloads, Hugging Face cache blobs nothing uses, failed llama.cpp builds, __pycache__, "
                "pip's cache, add-on packages no longer installed. Lists them; deletes only with apply. Models, "
                "workspaces and settings are never touched.",
        "fields": [F("models", repeat=True, help="a model folder to check"), F("llama", repeat=True, help="a llama.cpp checkout"),
                   F("apply", kind="bool", help="delete what is listed")]},
    "dep_snapshot": {
        "title": "Environment snapshot", "group": "Setup", "argv": [S / "dep_snapshot.py", "snapshot"],
        "help": "Every package, the PyTorch build and the llama.cpp tools on this computer, as JSON to compare with "
                "another (dep_snapshot.py diff).", "fields": [F("out", req=True)]},
    "hw_apply": {
        "title": "Hardware: apply a support profile", "group": "Setup", "argv": [S / "hw_profiles.py", "apply"],
        "help": "Build llama.cpp and install PyTorch the way a hardware support profile says, for one purpose "
                "(Setup → Hardware support profiles shows which fit and what each step runs).",
        "fields": [F("id", req=True, help="the profile id"), F("purpose", choices=["general", "inference", "retraining"]),
                   F("llama_dir", help="where llama.cpp is built (default ~/llama.cpp)"),
                   F("python", help="the Python whose environment gets PyTorch (default: OmniCortex's venv)"),
                   F("dry_run", kind="bool", help="only list the commands")]},
    "attrib_saliency": {
        "title": "4 · Saliency instead of CETT (CPU, older GPUs)", "group": "H-Neurons",
        "argv": [S / "neuron_attrib.py", "saliency"],
        "help": "Gradient x activation (or integrated gradients) on the answer: runs in float32 on a CPU or a GPU "
                "without bfloat16. Same output files as extraction; train on one measure only.",
        "fields": [U("model_path", req=True), U("input_path", default="data/answer_tokens.jsonl"),
                   U("ids_path", default="data/train_qids.json"), U("output_root", default="data/saliency"),
                   U("locations", multi=True, choices=LOCS, default=["answer_tokens", "all_except_answer_tokens"]),
                   U("method", choices=["grad_x_act", "integrated"], default="grad_x_act",
                     help="integrated is steps times slower and more faithful"),
                   U("steps", kind="int", default=8), U("dtype", choices=["float32", "float16", "bfloat16"],
                                                         default="float32"),
                   U("device", default="auto", help="cpu, cuda, cuda:1, mps… (auto: a GPU when PyTorch sees one)"),
                   U("max_tokens", kind="int", default=1024), TRUST]},
    "attrib_weights": {
        "title": "4 · Weights-only attribution (no forward pass)", "group": "H-Neurons",
        "argv": [S / "neuron_attrib.py", "weights"],
        "help": "How strongly each neuron writes each answer token, from the GGUF's weights alone (numpy, any "
                "computer). A static measure: what a neuron can promote, not what it did.",
        "fields": [U("gguf", req=True), U("input_path", default="data/answer_tokens.jsonl"),
                   U("ids_path", default="data/train_qids.json"), U("output_root", default="data/dla"),
                   U("abs", kind="bool", help="magnitude only (default keeps the sign)")]},
    "attrib_quant_noise": {
        "title": "Quantization noise per neuron", "group": "H-Neurons",
        "argv": [S / "neuron_attrib.py", "quant-noise"],
        "help": "How far each neuron's weights moved between a reference GGUF (F16/BF16/Q8_0) and a quant, and "
                "whether the H-Neurons were hit harder than the rest.",
        "fields": [U("ref", req=True, help="F16, BF16 or Q8_0 GGUF"), U("quant", req=True, help="the quantized GGUF"),
                   U("neurons", default="models/h_neurons.json"), U("out", default="runs/quant-noise")]},
    "classify": {
        "title": "5 · Find H-Neurons", "group": "H-Neurons", "argv": [S / "classifier.py"],
        "help": "Sparse classifier on the activations; writes models/h_neurons.json and classifier.npz.",
        "fields": [U("acts_root", default="data/activations"), U("train_ids", default="data/train_qids.json"),
                   U("test_ids", default="data/test_qids.json"), U("test_acts_root", default="data/activations_test"),
                   U("train_mode", choices=["1-vs-1", "3-vs-1"], default="3-vs-1"),
                   U("penalty", choices=["l1", "l2"], default="l1"),
                   F("C", flag="--C", kind="float", default=1.0, help="sparsity: lower keeps fewer neurons"),
                   U("out_dir", default="models")]},
    "tune": {
        "title": "7 · Tune the scale", "group": "H-Neurons", "argv": [S / "tune_scale.py"],
        "help": "Measure answers at several suppression scales; --save writes a profile.",
        "fields": [U("model_path", req=True), U("h_neurons", default="models/h_neurons.json"),
                   U("eval_path", default="data/consistency_samples.jsonl"),
                   U("scales", multi=True, default=["1.0", "0.5", "0.25", "0.1", "0.0"]),
                   U("n_eval", kind="int", default=50), U("gpu_mem"), U("config_name", default="trivia"),
                   U("save", kind="bool", default=True), TRUST]},
    "suppress_gguf": {
        "title": "9 · Edit a GGUF", "group": "H-Neurons", "argv": [S / "suppress_gguf.py"],
        "help": "Scale the H-Neurons inside a GGUF; the copy loads anywhere a GGUF does.",
        "fields": [U("gguf", req=True), U("h_neurons", default="models/h_neurons.json"),
                   U("scale", kind="float", req=True, default=0.5), U("out", req=True)]},
    "export_lora": {
        "title": "9 · Export a LoRA", "group": "H-Neurons", "argv": [S / "export_lora.py"],
        "help": "The suppression as an adapter (HF, or GGUF with gguf set).",
        "fields": [U("model_path", req=True), U("profile", req=True), U("output_dir", default="adapters/suppress"),
                   U("scale", kind="float"), U("gguf", help="base GGUF, to write a GGUF adapter"), TRUST]},
    "intervene": {
        "title": "9 · Apply to HF weights", "group": "H-Neurons", "argv": [S / "intervene_model.py"],
        "help": "A suppressed copy of a Hugging Face model.",
        "fields": [U("model_path", req=True), U("profile"), U("h_neurons"), U("scale", kind="float"),
                   U("output_path", default="models/suppressed"), TRUST]},
    "judge": {
        "title": "Check the judge", "group": "H-Neurons", "argv": [S / "judge_agreement.py"],
        "help": "How often the right/wrong labels agree with an LLM judge, before trusting them.",
        "fields": [U("input_path", default="data/consistency_samples.jsonl"),
                   U("base_url", default="http://127.0.0.1:7870/v1"), U("model", req=True),
                   U("n", kind="int", default=100), U("out")]},
    "trace": {
        "title": "Trace one sample", "group": "H-Neurons", "argv": [S / "trace_sample.py"],
        "help": "Per-token CETT for one answer, for the 3D views (Replay).",
        "fields": [U("gguf"), U("binary"), U("input_path", default="data/consistency_samples.jsonl"), U("qid"),
                   U("out", req=True, help="e.g. runs/trace-1"), U("classifier", help="models/classifier.npz"),
                   F("bin_neurons", kind="int", default=512)]},
    # ---- vision (docs/VISION.md)
    **{f"clip_{sub}": {
        "title": f"CLIP · {sub}", "group": "Vision", "argv": [S / "clip_neurons.py", sub],
        "help": {"collect": "Zero-shot label images; keep consistent right and wrong ones.",
                 "extract": "CETT for the image or text tower.",
                 "evaluate": "Error and abstention rates at each scale.",
                 "export": "A suppressed copy of the model."}[sub],
        "fields": fields} for sub, fields in {
        "collect": [U("model", req=True, help="e.g. openai/clip-vit-base-patch32"), U("images", help="image folder"),
                    U("manifest"), U("labels"), U("style", choices=["off", "auto", "pixel", "cartoon"], default="off", help="pixel art and cartoons: clean upscaling, padding, style prompts; extract and evaluate use the same"),
                    U("out", req=True, help="e.g. runs/clip")],
        "extract": [U("model", req=True), U("run", req=True), U("ids", req=True),
                    U("tower", choices=["vision", "text"], default="vision"), U("out", req=True)],
        "evaluate": [U("model", req=True), U("run", req=True), U("ids", req=True), U("h_neurons", req=True),
                     U("tower", choices=["vision", "text"], default="vision"),
                     U("scales", multi=True, default=["1", "0.5", "0"]), U("out")],
        "export": [U("model", req=True), U("h_neurons", req=True), U("tower", choices=["vision", "text"],
                   default="vision"), U("scale", kind="float", req=True), U("out", req=True)]}.items()},
    # ---- analysis
    "compare_models": {
        "title": "Does suppression help?", "group": "Analysis", "argv": [S / "compare_models.py"],
        "help": "Ask the original and the edited model the same fresh questions; count fixes and new errors.",
        "fields": [U("target", req=True, repeat=True, help="label=URL@model, e.g. base=http://127.0.0.1:7870/v1@m"),
                   U("data_path"), U("n", kind="int", default=200), U("out", req=True)]},
    "merge_eval": {
        "title": "Compare models item by item", "group": "Analysis", "argv": [S / "merge_eval.py"],
        "help": "What improved and what regressed between two or more endpoints on the same tasks.",
        "fields": [F("endpoint", req=True, repeat=True, help="label=URL@model"), U("tasks", req=True),
                   U("reference"), U("out")]},
    "cv": {
        "title": "Cross-validate the detector", "group": "Analysis", "argv": [S / "cv.py"],
        "help": "Cross-validated AUROC with permutation tests: is the H-Neuron signal real for this model?",
        "fields": [U("roots", req=True, multi=True, help="activation folders"), U("ids", req=True, multi=True),
                   U("samples"), F("C", flag="--C", multi=True), U("folds", kind="int", default=5), U("out")]},
    "score_tokens": {
        "title": "Score tokens (HTML)", "group": "Analysis", "argv": [S / "score_tokens.py"],
        "help": "H-Neuron activity token by token, written as an HTML page.",
        "fields": [U("model_path", req=True), U("classifier", default="models/classifier.npz"),
                   U("input_path"), U("question"), U("n", kind="int", default=5), U("out"), TRUST]},
    "when": {
        "title": "Where does the score spike?", "group": "Analysis", "argv": [S / "when.py"],
        "help": "Peaks of the hallucination score along one or more traces (pass a wrong and a right one).",
        "fields": [F("traces", flag="", req=True, multi=True, help="trace folders")]},
    "task_neurons": {
        "title": "Neurons per task", "group": "Analysis", "argv": [S / "task_neurons.py"],
        "help": "Which neurons a model recruits for which kind of task.",
        "fields": [U("acts_root", req=True), U("ids", req=True), U("samples", req=True), U("out_dir")]},
    "code_halluc": {
        "title": "Code hallucination vs scale", "group": "Analysis", "argv": [S / "eval_code_hallucination.py"],
        "help": "Invented packages and APIs, checked against a real package index, at each suppression scale.",
        "fields": [U("base_url", req=True, default="http://127.0.0.1:7870/v1"), U("tasks", req=True),
                   U("model"), U("alphas", multi=True, default=["1.0", "0.5", "0.0"]), U("out")]},
    "vram_budget": {
        "title": "Memory budget", "group": "Analysis", "argv": [S / "vram_budget.py"],
        "help": "Model, KV cache and visualizer memory for a GGUF at a context length.",
        "fields": [F("gguf", req=True), F("ctx", kind="int", default=8192), F("vram", help="e.g. 16GiB"),
                   F("ngl", kind="int")]},
    # ---- neuron review (scripts/neuron_review.py; Studio's /review page reads the same store)
    "review_testqa": {
        "title": "Review: ingest a TestQA run", "group": "Review", "argv": [S / "neuron_review.py", "ingest-testqa"],
        "help": "Replay a TestQA run's graded replies through the model that wrote them and record each one's "
                "neuron activity, so the Review page can compare right and wrong answers by subject over time.",
        "fields": [F("results", req=True, help="testqa.py --out file"),
                   F("endpoint", help="which endpoint's replies (default: the first in the file)"),
                   F("gguf", req=True, help="the model file that wrote them"),
                   F("binary", req=True, help="llama-cett-dump"),
                   F("classifier", help="classifier.npz, to record each reply's risk as well"),
                   F("source", default="testqa", help="name shown in the Review filters"),
                   F("ngl", kind="int", default=99), F("limit", kind="int", default=0)]},
    "review_items": {
        "title": "Review: ingest graded replies", "group": "Review", "argv": [S / "neuron_review.py", "ingest-items"],
        "help": "Any other benchmark: JSONL with prompt, response, verdict (right/wrong/abstained) and optionally "
                "subject. Recorded the same way, under its own source name.",
        "fields": [F("items", req=True, help="graded JSONL"), F("source", req=True, help="e.g. livebench, swe-bench"),
                   F("kind", choices=["benchmark", "test", "observed"], default="benchmark"),
                   F("gguf", req=True), F("binary", req=True, help="llama-cett-dump"), F("classifier"),
                   F("ngl", kind="int", default=99), F("limit", kind="int", default=0)]},
    # ---- tuning the scale, scored where the model will run
    "eval_lmstudio": {
        "title": "Score a candidate in LM Studio", "group": "Tuning", "argv": [S / "eval_lmstudio.py"],
        "help": "Wait for a candidate GGUF to appear in LM Studio (local or on another machine), ask it a fixed "
                "TestQA sample, print its score. The adaptive tuner uses this as its evaluator.",
        "fields": [F("model", req=True, help="the candidate GGUF (its file name is matched in LM Studio)"),
                   F("lmstudio", default="http://127.0.0.1:1234", help="LM Studio's server"),
                   F("model_id", help="LM Studio's id, if the file name does not match"),
                   F("metric", choices=["accuracy", "not_hallucinated", "utility"], default="accuracy"),
                   F("per_subject", kind="int", default=10), F("seed", kind="int", default=0),
                   F("tasks", multi=True, help="your own test set instead of the bank"),
                   F("wait", kind="float", default=3600, help="seconds to wait for the transfer")]},
    "tune_adaptive": {
        "title": "Adaptive scale tuning", "group": "Tuning", "argv": [S / "adaptive_tuner.py", "tune"],
        "help": "Sample suppression scales broadly, score each candidate, narrow around the best, repeat. "
                "worker 'local' writes candidates here (put output_dir under the transfer watcher) and scores them "
                "with the evaluator; a URL uses a remote tuning worker. Deletes nothing unless auto_delete.",
        "fields": [F("state", req=True, help="state file, resumable, e.g. runs/tune-model.json"),
                   F("worker", req=True, default="local", help="local, or a tuning worker URL"),
                   F("source", help="local: the GGUF to scale"), F("profile", help="local: h_neurons.json"),
                   F("output_dir", help="local: where candidates go"),
                   F("evaluator", help="local: e.g. python scripts/eval_lmstudio.py --lmstudio "
                                       "http://host:1234 --model {model}"),
                   F("min", kind="float", default=0.1), F("max", kind="float", default=0.9),
                   F("initial_step", kind="float", default=0.1), F("resolution", kind="float", default=0.01),
                   F("batch_size", kind="int", default=2), F("margin_of_error", kind="float", default=0.03),
                   F("baseline_score", kind="float", help="the unedited model's score, from the same evaluator"),
                   F("max_batches", kind="int", default=20),
                   F("auto_delete", kind="bool", help="delete candidates clearly worse than the best (off: "
                                                      "keep everything until you trust the scores)")]},
    # ---- merging and editing
    "merge_selective": {
        "title": "Selective merge", "group": "Merge", "argv": [S / "merge_selective.py"],
        "help": "Take chosen neurons (e.g. the H-Neurons) from a donor model into a base model.",
        "fields": [F("base", req=True), F("donor", req=True), F("h_neurons"), F("layers"),
                   F("alpha", kind="float", default=1.0), U("output_path", req=True), F("dry_run", kind="bool"), TRUST]},
    "sweep_experts": {
        "title": "MoE expert sweep", "group": "Merge", "argv": [S / "sweep_experts.py"],
        "help": "Accuracy and speed of a mixture-of-experts GGUF at each active expert count.",
        "fields": [F("gguf", req=True), F("server", req=True, help="llama-server binary"), F("tasks", req=True),
                   F("experts", req=True, multi=True, default=["2", "4", "8"]), F("ngl", kind="int", default=99),
                   F("out")]},
    "suppress_mmproj": {
        "title": "Edit a vision projector", "group": "Vision", "argv": [S / "suppress_mmproj.py"],
        "help": "Scale CLIP/SigLIP H-Neurons inside a llama.cpp mmproj GGUF.",
        "fields": [U("mmproj", req=True), U("h_neurons", req=True), U("scale", kind="float", req=True),
                   U("out", req=True)]},
    # ---- data
    "export_sft": {
        "title": "Abstention dataset", "group": "Retrain", "argv": [S / "export_sft_dataset.py"],
        "help": "Collected pairs -> an SFT dataset that teaches saying \"I don't know\".",
        "fields": [U("input_path", default="data/consistency_samples.jsonl"), U("output_path", req=True),
                   U("val_fraction", kind="float", default=0.1), U("max_pairs", kind="int")]},
    "vision_synth": {
        "title": "Synthetic vision items", "group": "Vision", "argv": [S / "vision_synth.py"],
        "help": "Images with questions whose answers are true by construction (counting, colours, text).",
        "fields": [F("family", req=True, choices=["arith_minus", "arith_sum", "color_of_shape", "count_color",
                                                  "count_shape", "count_total", "larger_choice", "largest_color",
                                                  "left_right", "left_shape", "letter_count", "ocr_color", "ocr_number",
                                                  "ocr_word", "shape_of_color", "single_shape", "top_bottom"]),
                   F("n", flag="-n", kind="int", default=50), F("out", req=True)]},
    # ---- viewers that open a window on this machine's desktop, or write a report
    "view_timeline": {
        "title": "Timeline (window)", "group": "Views", "argv": [ROOT / "viz" / "timeline.py"],
        "help": "Exact, unbloomed per-token view of a trace, in a desktop window (pygfx).",
        "fields": [F("session", flag="", req=True, help="trace folder, e.g. runs/trace-1 or ~/.neuronscope/traces/<id>"),
                   F("h_neurons"), F("play", kind="bool")]},
    "view_explore": {
        "title": "Explorer (window)", "group": "Views", "argv": [ROOT / "viz" / "explore.py"],
        "help": "Mean and contrast maps, 3D volume, click to inspect, in a desktop window (fastplotlib).",
        "fields": [F("session", flag="", req=True, help="activations session"), F("h_neurons"),
                   F("max_samples", kind="int")]},
    "view_weights": {
        "title": "Weights (window)", "group": "Views", "argv": [ROOT / "viz" / "weights.py"],
        "help": "Weight magnitude, quantization error and H-Neuron enrichment for a GGUF.",
        "fields": [F("gguf", req=True), F("reference", help="higher-precision GGUF to compare against"),
                   F("h_neurons")]},
    "compare": {
        "title": "Compare sessions", "group": "Views", "argv": [ROOT / "viz" / "compare.py"],
        "help": "Depth profiles, concentration and failure overlap between two or more sessions (report).",
        "fields": [F("sessions", flag="", req=True, multi=True), F("json", help="also write the report here"),
                   F("assert_lineage", kind="bool")]},
})

SAFE = re.compile(r"^[^\x00-\x1f]*$")
EXTRA = None      # callable -> more specs (Studio: job kinds from enabled plugins)


HIDDEN = None          # (kind, spec) -> True for jobs of a removed add-on (viz/addons.py); set by Studio


def all_specs() -> dict:
    specs = {**SPECS, **(EXTRA() if EXTRA else {})}
    return {k: v for k, v in specs.items() if not (HIDDEN and HIDDEN(k, v))}


# ---- every option a job's script has, read from its own --help (the form's fields are the common ones)

_OPT_CACHE: dict = {}
_OPT_LINE = re.compile(r"^  (-\S.*?)(?:\s{2,}(\S.*))?$")


def _parse_help(text: str) -> list[dict]:
    """argparse --help -> [{flag, takes_value, multi, choices, metavar, help}] for optional arguments."""
    out, cur, in_opts = [], None, False
    for line in text.splitlines():
        if re.match(r"^\S.*:$", line):                    # a section heading: "options:", "positional arguments:"
            in_opts = not line.lower().startswith("positional")
            cur = None
            continue
        m = _OPT_LINE.match(line) if in_opts else None
        if m:
            inv = m.group(1).strip()
            variants = [v.strip() for v in inv.split(", ")]
            longs = [v for v in variants if v.startswith("--")]
            if not longs:
                cur = None
                continue
            first = longs[0].split(" ", 1)
            flag, rest = first[0], (first[1] if len(first) > 1 else "")
            if flag in ("--help",):
                cur = None
                continue
            neg = any(v.startswith("--no-") for v in variants)
            choices = re.findall(r"\{([^}]*)\}", rest)
            cur = {"flag": flag, "takes_value": bool(rest), "multi": "..." in rest,
                   "choices": choices[0].split(",") if choices else None,
                   "metavar": None if choices else (rest.split(" ")[0].strip("[]") or None), "help": (m.group(2) or "").strip(),
                   "boolean_optional": neg}
            out.append(cur)
        elif cur is not None and line.startswith("    ") and line.strip() and not line.strip().startswith("-"):
            cur["help"] = (cur["help"] + " " + line.strip()).strip()
    return out


def script_options(kind: str) -> list[dict]:
    """The options of a job's script that its form does not show yet (cached until the script changes)."""
    spec = all_specs()[kind]
    script = next((Path(x) for x in spec["argv"] if str(x).endswith(".py")), None)
    stamp = script.stat().st_mtime if script and script.exists() else 0
    hit = _OPT_CACHE.get(kind)
    if hit and hit[0] == stamp:
        return hit[1]
    argv = [*(spec.get("interp") or [sys.executable]), *[str(x) for x in spec["argv"]], "--help"]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=30, env=dict(os.environ, COLUMNS="200"))
        opts = _parse_help(r.stdout)
    except (OSError, subprocess.TimeoutExpired):
        opts = []
    known = {f["flag"] for f in spec["fields"]} | {"--" + f["name"].replace("_", "-") for f in spec["fields"]} \
        | {"--" + f["name"] for f in spec["fields"]}
    opts = [o for o in opts if o["flag"] not in known]
    _OPT_CACHE[kind] = (stamp, opts)
    return opts


def _extra_argv(kind: str, extra: dict) -> list[str]:
    """Values for options from script_options(): only flags the script itself lists."""
    if not extra:
        return []
    if not isinstance(extra, dict):
        raise ValueError("extra options must be an object")
    opts = {o["flag"]: o for o in script_options(kind)}
    out = []
    for flag, v in extra.items():
        o = opts.get(flag)
        if o is None:
            raise ValueError(f"{flag} is not an option of this job's script")
        if not o["takes_value"]:
            if o.get("boolean_optional") and v in (False, "false", "off", "0", 0):
                out.append("--no-" + flag[2:])
            elif v in (True, "true", "on", "1", 1):
                out.append(flag)
            continue
        if v is None or v == "" or v == []:
            continue
        vals = v if isinstance(v, list) else (shlex.split(v) if o["multi"] and isinstance(v, str) else [v])
        clean = []
        for x in vals:
            x = str(x).strip()
            if not x:
                continue
            if (x.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?(e-?\d+)?", x)) or not SAFE.match(x) or len(x) > 4096:
                raise ValueError(f"{flag}: values may not start with '-' (except negative numbers) or contain "
                                 "control characters")
            if o["choices"] and x not in o["choices"]:
                raise ValueError(f"{flag} must be one of {', '.join(o['choices'])}")
            clean.append(x)
        if clean:
            out += [flag] + clean
    return out


def build_argv(kind: str, values: dict) -> list[str]:
    specs = all_specs()
    if kind not in specs:
        raise ValueError(f"unknown job kind {kind!r}")
    spec = specs[kind]
    argv = [*(spec.get("interp") or [sys.executable]), *[str(x) for x in spec["argv"]]]
    for f in spec["fields"]:
        v = values.get(f["name"], f["default"])
        if f["kind"] == "bool":
            if v is True or v in ("true", "on", "1", 1):
                argv.append(f["flag"])
            continue
        if v is None or v == "" or v == []:
            if f["required"]:
                raise ValueError(f"{f['name']} is required")
            continue
        vals = v if isinstance(v, list) else (shlex.split(v) if f["multi"] and isinstance(v, str) else [v])
        out = []
        for x in vals:
            x = str(x).strip()
            if not x:
                continue
            if f["kind"] == "int":
                x = str(int(x))
            elif f["kind"] == "float":
                x = repr(float(x))
            elif x.startswith("-") or not SAFE.match(x) or len(x) > 4096:
                raise ValueError(f"{f['name']}: values may not start with '-' or contain control characters")
            if f["choices"] is not None and x not in f["choices"]:
                raise ValueError(f"{f['name']} must be one of {', '.join(c for c in f['choices'] if c)}")
            out.append(x)
        if out and f["repeat"]:
            for x in out:
                argv += [f["flag"], x]
        elif out:
            argv += ([f["flag"]] if f["flag"] else []) + out       # flag "" = positional
        elif f["required"]:
            raise ValueError(f"{f['name']} is required")
    return argv + _extra_argv(kind, values.get("__extra") or {})


# ---- progress and plain-language hints, read from a job's log

PROG_LINE = re.compile(r"^PROGRESS\s+(\d+)\s*/\s*(\d+)\s*(.*)$")
TQDM = re.compile(r"(?:(?P<label>[^\r\n|]*?):\s*)?(?P<pct>\d{1,3})%\|[^|]*\|\s*(?P<done>\d+)/(?P<total>\d+)"
                  r"\s*\[(?P<el>[^<\]]*)<(?P<eta>[^,\]]*)(?:,\s*(?P<rate>[^\]]*))?\]")
BRACKET = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")
PLAIN = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\b")
# module name -> catalog package (scripts/ns_requirements.py), for an Install button
MODULE_PACKAGE = {"datasets": "datasets", "sklearn": "scikit-learn", "PIL": "pillow", "pandas": "pandas",
                  "pyarrow": "pyarrow", "openai": "openai", "gguf": "gguf", "transformers": "transformers",
                  "safetensors": "safetensors", "accelerate": "accelerate", "peft": "peft", "mcp": "mcp",
                  "requests": "requests", "tqdm": "tqdm", "jinja2": "jinja2", "aiohttp": "aiohttp", "numpy": "numpy"}


def _secs(t: str) -> float | None:
    """tqdm's 01:23 or 1:02:03 -> seconds."""
    try:
        parts = [float(x) for x in t.strip().split(":")]
    except ValueError:
        return None
    s = 0.0
    for x in parts:
        s = s * 60 + x
    return s


def parse_progress(text: str) -> dict | None:
    """The newest progress a log shows: PROGRESS lines, tqdm bars, [n/m] or "n/m" counters.
    -> {done, total, pct, label, eta_s, rate} or None."""
    lines = [l for l in re.split(r"[\r\n]+", text[-65536:]) if l.strip()]
    for line in reversed(lines[-400:]):
        m = PROG_LINE.match(line.strip())
        if m:
            d, t = int(m.group(1)), int(m.group(2))
            return {"done": d, "total": t, "pct": round(100 * d / t, 1) if t else None, "label": m.group(3).strip()}
        m = TQDM.search(line)
        if m:
            d, t = int(m.group("done")), int(m.group("total"))
            return {"done": d, "total": t, "pct": round(100 * d / t, 1) if t else None,
                    "label": (m.group("label") or "").strip()[-80:], "eta_s": _secs(m.group("eta") or ""),
                    "rate": (m.group("rate") or "").strip() or None}
        m = BRACKET.search(line) or PLAIN.match(line)
        if m and 0 < int(m.group(2)) < 10 ** 9 and int(m.group(1)) <= int(m.group(2)):
            d, t = int(m.group(1)), int(m.group(2))
            return {"done": d, "total": t, "pct": round(100 * d / t, 1), "label": ""}
    return None


def last_line(text: str) -> str:
    for l in reversed(re.split(r"[\r\n]+", text[-8192:])):
        l = l.strip()
        if l and not PROG_LINE.match(l):
            return l[-240:]
    return ""


def explain_failure(text: str) -> dict | None:
    """A plain-language reason and, where one exists, a fix for the commonest ways a job fails."""
    t = text[-20000:]
    m = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", t)
    if m:
        mod = m.group(1).split(".")[0]
        pkg = MODULE_PACKAGE.get(mod)
        return {"reason": f"The Python package for '{mod}' is not installed in Studio's Python.",
                "fix": f"Install {pkg}" if pkg else f"pip install {mod}", "install": pkg}
    rules = [
        (r"CUDA out of memory|out of memory|OutOfMemoryError|failed to allocate", "The GPU (or RAM) ran out of memory.",
         "Use a smaller model or quantization, lower the context or batch size, or offload fewer layers."),
        (r"Address already in use|port \d+ .*in use", "A port the job needs is already taken.",
         "Stop what uses it (System page), or choose another port."),
        (r"Connection refused|Failed to establish a new connection|ConnectionRefusedError",
         "The job could not reach a server it needs (an endpoint, Studio, or a host).",
         "Check the endpoint address and that the server is running."),
        (r"FileNotFoundError|No such file or directory", "A file or folder the job was given does not exist.",
         "Check the paths in the form (Browse… picks existing ones)."),
        (r"401|403|Unauthorized|gated repo|Access to model .* is restricted", "Access was refused (token or gated model).",
         "Set HF_TOKEN for gated Hugging Face repos, or check the endpoint's token."),
        (r"CONNECT tunnel failed|Temporary failure in name resolution|Name or service not known|timed out",
         "The network request failed or timed out.", "Check the internet connection, proxy or firewall."),
    ]
    for rx, reason, fix in rules:
        if re.search(rx, t, re.I):
            return {"reason": reason, "fix": fix}
    return None


class JobRunner:
    on_finish = None          # callable(job meta), run when a job ends (Studio: automatic score submission)
    shunt = None              # core_shunt settings: priority and cores for every job

    def __init__(self, root: str | Path, max_running: int = 2, cwd: str | Path | None = None):
        """cwd: where jobs run, so the pipeline's relative data/, models/ and runs/
        land in the workspace's root (default: the repository)."""
        self.root = Path(root).expanduser()
        self.cwd = Path(cwd).expanduser() if cwd else ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_running = max_running
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}

    def _meta_path(self, jid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", jid):
            raise ValueError("bad job id")
        return self.root / jid / "job.json"

    def _write(self, meta: dict) -> None:
        p = self._meta_path(meta["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(p)

    def start(self, kind: str, values: dict) -> dict:
        argv = build_argv(kind, values)
        with self.lock:
            running = sum(p.poll() is None for p in self.procs.values())
            if running >= self.max_running:
                raise RuntimeError(f"{running} jobs already running; wait or cancel one")
            jid = uuid.uuid4().hex[:12]
            d = self.root / jid
            d.mkdir(parents=True)
            log = open(d / "log.txt", "wb")
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            if str(S) not in sys.path:
                sys.path.insert(0, str(S))
            import core_shunt
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=str(self.cwd if self.cwd.is_dir() else ROOT), env=env,
                                    start_new_session=True, **core_shunt.popen_kwargs(self.shunt))
            shunted = core_shunt.apply(proc.pid, self.shunt)    # low priority, efficiency cores (Integrated options)
            log.close()
            self.procs[jid] = proc
            meta = {"id": jid, "kind": kind, "title": all_specs()[kind]["title"], "values": values, "argv": argv[1:],
                    "status": "running", "pid": proc.pid, "started": time.time(), "ended": None, "returncode": None,
                    "shunt": shunted}
            self._write(meta)
        threading.Thread(target=self._wait, args=(jid, proc), daemon=True).start()
        return meta

    def _wait(self, jid: str, proc: subprocess.Popen) -> None:
        rc = proc.wait()
        meta = self.get(jid)
        if meta.get("status") == "running":
            meta["status"] = "done" if rc == 0 else "failed"
        meta.update(returncode=rc, ended=time.time())
        self._write(meta)
        if self.on_finish is not None:
            try:
                self.on_finish(meta)
            except Exception:
                pass

    def get(self, jid: str) -> dict:
        return json.loads(self._meta_path(jid).read_text())

    def list(self, limit: int = 50) -> list[dict]:
        out = []
        for p in self.root.glob("*/job.json"):
            try:
                m = json.loads(p.read_text())
            except Exception:
                continue
            if m.get("status") == "running" and m["id"] not in self.procs:
                m["status"] = "lost"        # Studio restarted while it ran
            out.append(m)
        out.sort(key=lambda m: -m.get("started", 0))
        for m in out[:limit]:
            if m.get("status") in ("running", "failed", "lost"):
                self._annotate(m)
        return out[:limit]

    def _annotate(self, m: dict) -> None:
        """progress, the current step and elapsed time for a running job; a reason and fix for a failed one."""
        try:
            text = self.log(m["id"], 65536)
        except (OSError, ValueError):
            return
        now = m.get("ended") or time.time()
        m["elapsed_s"] = round(now - m.get("started", now))
        if m.get("status") == "running":
            pr = parse_progress(text)
            if pr and pr.get("eta_s") is None and pr.get("done") and pr.get("total") and m["elapsed_s"] > 3:
                pr["eta_s"] = round(m["elapsed_s"] * (pr["total"] - pr["done"]) / pr["done"])
            m["progress"] = pr
            m["step"] = last_line(text)
        else:
            m["hint"] = explain_failure(text)
            m["step"] = last_line(text)

    def log(self, jid: str, tail: int = 20000) -> str:
        meta = self._meta_path(jid)
        if not meta.exists():
            raise FileNotFoundError(f"no job {jid}")     # unknown id, not "no output yet"
        p = meta.parent / "log.txt"
        if not p.exists():
            return ""
        with open(p, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - tail))
            return f.read().decode(errors="replace")

    def cancel(self, jid: str) -> dict:
        meta = self.get(jid)
        proc = self.procs.get(jid)
        if proc is not None and proc.poll() is None:
            meta["status"] = "cancelled"
            self._write(meta)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.terminate()
        return meta


def public_specs() -> dict:
    return {k: {"title": v["title"], "group": v["group"], "help": v["help"],
                "fields": [{x: f[x] for x in ("name", "kind", "required", "default", "help", "choices", "multi")}
                           for f in v["fields"]]} for k, v in all_specs().items()}


PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Jobs</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
a{color:var(--acc)}
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;margin:0}header a{color:var(--acc);font-size:13px}
main{display:grid;grid-template-columns:minmax(260px,380px) 1fr;gap:1rem;padding:1rem;max-width:1400px;margin:0 auto}
@media (max-width:800px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.8rem}
h2{font-size:13px;margin:.2rem 0 .5rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
label{display:block;font-size:12px;color:var(--mut);margin:.45rem 0 .1rem}
input,select{font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
input[type=checkbox]{width:auto;margin-right:.3rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.note{font-size:12px;color:var(--mut)}.err{color:var(--no);font-size:12px;white-space:pre-wrap}
.job{border-bottom:1px solid var(--line);padding:.45rem .3rem;cursor:pointer;display:grid;grid-template-columns:auto 1fr auto;gap:.15rem .6rem;align-items:baseline}
.job.sel{background:var(--code)}.job .jt{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.job .jp{grid-column:1/-1}.job .jstep{grid-column:1/-1;font:11.5px ui-monospace,monospace;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pbar{height:6px;border-radius:3px;background:var(--line);overflow:hidden;position:relative}.pbar>i{display:block;height:100%;background:var(--acc);border-radius:3px;transition:width .4s}
.pbar.ind>i{width:30%!important;animation:ind 1.4s ease-in-out infinite}@keyframes ind{0%{margin-left:-30%}100%{margin-left:100%}}
.pstat{display:flex;gap:.9rem;flex-wrap:wrap;font-size:12px;color:var(--mut);margin-top:.25rem}.pstat b{color:var(--fg);font-weight:600}
.hint{border:1px solid var(--no);border-radius:8px;padding:.55rem .7rem;margin:.5rem 0;background:color-mix(in srgb,var(--no) 8%,transparent)}
.hint b{color:var(--no)}
.job:hover{background:var(--code)}.st{font-size:11px;padding:0 .4rem;border-radius:4px;border:1px solid var(--line)}
.st.running{color:var(--acc)}.st.done{color:var(--ok)}.st.failed,.st.lost{color:var(--no)}
.tip{display:inline-block;width:15px;height:15px;line-height:15px;text-align:center;border-radius:50%;border:1px solid var(--line);font:600 10px ui-sans-serif,system-ui;color:var(--mut);margin-left:.3rem;cursor:help;vertical-align:1px}
.fhelp{font-size:11.5px;color:var(--mut);margin:.1rem 0 0}
pre{background:var(--code);padding:.6rem;border-radius:6px;max-height:60vh;overflow:auto;font:12px ui-monospace,monospace;white-space:pre-wrap;word-break:break-all;margin:.5rem 0 0}
</style>
<header><h1>Studio · Jobs</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/lab">Lab</a><a href="/review">Review</a><a href="/projects">Projects</a><a href="/workers">Workers</a><a href="/jobs">Jobs</a><a href="/connect">Connect</a><span class="note" id="hint"></span></header>
<main>
 <section class="card"><h2>New job</h2>
  <select id="kind"></select><div id="khelp" class="note" style="margin-top:.3rem"></div>
  <div id="profbar" style="margin:.5rem 0 .2rem"></div>
  <form id="form"></form>
  <details id="allopts" style="margin-top:.6rem" hidden><summary class="note" style="cursor:pointer">All options <span id="nopts"></span></summary><div id="extra"></div></details>
  <label for="target" style="margin-top:.7rem">Run on</label>
  <select id="target" title="This computer, or a linked host whose owner allows jobs (Remote page)"><option value="local">This computer</option></select>
  <div class="note" id="tnote"></div>
  <button class="pri" id="run" style="margin-top:.5rem;width:100%">Run</button>
  <div id="ferr" class="err"></div>
 </section>
 <section class="card"><h2>Jobs</h2><div id="jobs" class="note">none yet</div>
  <div id="detail" style="display:none;margin-top:.8rem">
   <div style="display:flex;gap:.5rem;align-items:center"><b id="dtitle"></b><span id="dst" class="st"></span>
    <button id="cancel" style="margin-left:auto">Cancel</button></div>
   <div id="dprog" style="margin-top:.5rem"></div>
   <div id="dhint"></div>
   <details style="margin-top:.3rem"><summary class="note" style="cursor:pointer">Command</summary><div class="note" id="dargv" style="word-break:break-all;margin-top:.3rem"></div></details>
   <pre id="dlog"></pre></div>
 </section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let specs={}, cur=null;
async function api(p,o){const r=await fetch(p,o);const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
// Prefill: values from the link (#kind=testqa&endpoint=...), else what the
// running workspace knows (its GGUF, llama.cpp binaries, Studio's URL).
function suggest(f){ const h=(f.help||'').toLowerCase();
  if(f.name==='gguf'&&!h.includes('quant')) return DEF.gguf;
  if(f.name==='binary'||f.name==='cett') return DEF.cett;
  if(f.name==='server'&&h.includes('llama-server')) return DEF.server;
  if(f.name==='llama') return DEF.llama;
  if(f.name==='record'&&h.includes('trace session')) return DEF.traces;
  return null; }
// Fields that name a file or folder get Browse…; GGUF fields also a list of the models Studio knows.
const PICK={gguf:'file',from:'file',imatrix:'file',testqa:'file',file:'file',server:'file',binary:'file',cett:'file',classifier:'file',predictions:'file',reply_file:'file',h_neurons:'file',
  llama:'dir',cache:'dir',record:'dir',watch_emotions:'dir',out:'any',dataset:'any',data:'any',results:'any',report:'any',traces:'any',state:'any',session:'any',reference:'any',
  model:'any',base:'any',teacher:'any',student:'any',review_model:'any'};
function pickFor(f){ const w=PICK[f.name]; if(!w||f.multi||f.choices) return '';
  return ` data-pick="${w}"`+(f.name==='gguf'?' data-ext="gguf" list="dl-ggufs"':''); }
// Fields with known values get a dropdown of them (still free to type): models, endpoints, results, data files.
const SUGGEST={model:'models',base:'models',teacher:'models',student:'models',review_model:'models',stats_model:'models',
  model_path:'hf_models',results:'results',testqa:'results',neurons:'neuron_sets',h_neurons:'neuron_sets',
  input_path:'data_files',ids_path:'data_files',train_ids:'data_files',test_ids:'data_files',items:'data_files',exclude:'data_files'};
function sugList(f){ if(f.choices||f.name==='gguf') return [];
  if(f.name==='endpoint'||f.name==='pool') return (DEF.models||[]).map(m=>`${m.split('/').pop().replace(/[^\w.-]/g,'-').slice(0,24)}=${DEF.studio||''}@${m}`);
  if(f.name==='model_path') return [...(DEF.hf_models||[]),...(DEF.models||[])];
  return DEF[SUGGEST[f.name]]||[]; }
function sugAttr(f){ return sugList(f).length&&f.name!=='gguf'?` list="dl-f-${f.name}"`:''; }
// Help goes in a tooltip (and under the field while it has focus), so it never overflows the input.
const tip=h=>h?`<span class="tip" title="${esc(h)}" aria-label="${esc(h)}" tabindex="0">i</span>`:'';
const ph=h=>h&&h.length<=38?h:'';
function renderForm(prefill){const k=$('#kind').value, s=specs[k]; prefill=prefill||{};
 $('#khelp').innerHTML=esc(s.help)+(DEF.root?` <span class="note">· runs in ${esc(DEF.root)} (workspace ${esc(DEF.workspace)}, <a href="/settings#workspaces">change</a>)</span>`:'');
 $('#form').innerHTML=s.fields.map(f=>{const id='f_'+f.name, lab=esc(f.name.replace(/_/g,' '))+(f.required?' *':'');
  const sg=suggest(f);
  const d=f.name in prefill?prefill[f.name]:Array.isArray(f.default)?f.default.join(' '):(f.default??sg??'');
  if(f.kind==='bool')return `<label><input type="checkbox" id="${id}" ${[true,'1','true','on'].includes(prefill[f.name])?'checked':''}>${lab}${tip(f.help)}</label>`;
  const inp=f.choices?`<select id="${id}">${f.choices.map(c=>`<option ${c===d?'selected':''}>${esc(c)}</option>`).join('')}</select>`
   :`<input id="${id}" value="${esc(Array.isArray(d)?d.join(' '):d)}" placeholder="${esc(ph(f.help))}" title="${esc(f.help||'')}"${pickFor(f)}${sugAttr(f)}>`
    +(sugList(f).length?`<datalist id="dl-f-${f.name}">${sugList(f).slice(0,200).map(x=>`<option value="${esc(x)}">`).join('')}</datalist>`:'');
  return `<label for="${id}">${lab}${f.multi?' <span class="note">(space-separated)</span>':''}${tip(f.help)}</label>${inp}`}).join('')
  +`<datalist id="dl-ggufs">${(DEF.ggufs||[]).map(p=>`<option value="${esc(p)}">`).join('')}</datalist>`;
 $('#form').querySelectorAll('input[title],select').forEach(el=>{ el.onfocus=()=>{ const h=el.title||''; if(h.length>38){ let n=el.parentNode.querySelector('.fhelp[data-for="'+el.id+'"]');
   if(!n){ n=document.createElement('div'); n.className='fhelp'; n.dataset.for=el.id; (el.closest('span')||el).after(n); } n.textContent=h; } };
   el.onblur=()=>{ const n=el.parentNode.querySelector('.fhelp[data-for="'+el.id+'"]'); if(n) n.remove(); }; });
 loadExtra(k, prefill.__extra||{}); }
// A job kind opened: its form, and its profiles; the saved default fills the form unless a link already did.
function openKind(prefill){ prefill=prefill||{}; const k=$('#kind').value; renderForm(prefill);
  const pr=nsProfiles($('#profbar'),{scope:'job:'+k,get:formValues,set:v=>renderForm(v),system:()=>renderForm({})});
  if(!Object.keys(prefill).length) pr.ready.then(v=>{ if(v&&$('#kind').value===k) renderForm(v); }); }
function formValues(){ const k=$('#kind').value, v={};
  for(const f of specs[k].fields){ const el=$('#f_'+f.name); if(el) v[f.name]=f.kind==='bool'?el.checked:el.value.trim(); }
  const ex={}; document.querySelectorAll('#extra [data-flag]').forEach(el=>{ const val=el.type==='checkbox'?(el.indeterminate?undefined:el.checked):el.value.trim();
    if(val!==undefined&&val!==''&&!(el.type==='checkbox'&&el.dataset.neg!=='1'&&!val)) ex[el.dataset.flag]=val; });
  if(Object.keys(ex).length) v.__extra=ex; return v; }
// Every other option the job's script has, from its own --help.
const OPTS={}; let extraSeq=0;
async function loadExtra(k, pre){ const seq=++extraSeq; let opts=OPTS[k];
  if(!opts){ try{ opts=OPTS[k]=await api('/api/jobs/options?kind='+encodeURIComponent(k)); }catch(e){ opts=[]; } }
  if($('#kind').value!==k||seq!==extraSeq) return;           // a newer form replaced this one meanwhile
  $('#allopts').hidden=!opts.length; $('#nopts').textContent=opts.length?`(${opts.length} more)`:'';
  $('#extra').innerHTML=opts.map((o,i)=>{ const id='x_'+i, nm=esc(o.flag.replace(/^--/,'').replace(/[-_]/g,' ')), v=pre[o.flag];
    if(!o.takes_value) return `<label><input type="checkbox" id="${id}" data-flag="${esc(o.flag)}" data-neg="${o.boolean_optional?1:0}" ${v===true||v==='true'?'checked':''}>${nm}${tip(o.help)}</label>`;
    const ctl=o.choices?`<select id="${id}" data-flag="${esc(o.flag)}"><option value="">(script default)</option>${o.choices.map(c=>`<option ${c===v?'selected':''}>${esc(c)}</option>`).join('')}</select>`
      :`<input id="${id}" data-flag="${esc(o.flag)}" value="${esc(v??'')}" placeholder="${esc(o.metavar?o.metavar.toLowerCase():'')}" title="${esc(o.help)}">`;
    return `<label for="${id}">${nm}${o.multi?' <span class="note">(space-separated)</span>':''}${tip(o.help)}</label>${ctl}`; }).join('');
  if(Object.keys(pre).length) $('#allopts').open=true; }
let DEF={};
async function load(){specs=await api('/api/jobs/specs'); DEF=await api('/api/jobs/defaults').catch(()=>({}));const g={};for(const[k,s]of Object.entries(specs))(g[s.group]??=[]).push([k,s]);
 $('#kind').innerHTML=Object.entries(g).map(([n,xs])=>`<optgroup label="${esc(n)}">${xs.map(([k,s])=>`<option value="${k}">${esc(s.title)}</option>`).join('')}</optgroup>`).join('');
 openFromHash(); refresh();}
function openFromHash(){ const h=decodeURIComponent(location.hash.slice(1)); if(!h) return openKind();
  let kind=h, pre={};
  if(h.includes('=')){ const q=new URLSearchParams(location.hash.slice(1)); kind=q.get('kind')||''; q.forEach((v,k)=>{ if(k!=='kind') pre[k]=v; }); }
  if(specs[kind]){ $('#kind').value=kind; openKind(pre); if(Object.keys(pre).length) $('#ferr').innerHTML='<span class="note">filled in from the link: check the values, then Run</span>'; }
  else openKind(); }
addEventListener('hashchange',()=>{ if(specs) openFromHash(); });
$('#kind').onchange=()=>{openKind();history.replaceState(null,'','#'+$('#kind').value)};
// Where jobs can run: this computer, or linked hosts whose owner allows jobs.
let TARGETS=[], curHost='';
async function loadTargets(){ try{ TARGETS=await api('/api/jobs/targets'); }catch(e){ return; }
  const was=$('#target').value; $('#target').innerHTML=TARGETS.map(t=>`<option value="${esc(t.id)}" ${t.ok?'':'disabled'} title="${esc(t.why||'')}">${esc(t.label)}${t.ok?'':' (not available)'}</option>`).join('');
  if(TARGETS.some(t=>t.id===was&&t.ok)) $('#target').value=was; targetNote(); }
function targetNote(){ const t=TARGETS.find(x=>x.id===$('#target').value)||{}; const off=TARGETS.filter(x=>!x.ok&&x.id!=='local');
  $('#tnote').innerHTML=(t.id&&t.id!=='local'?`Runs on <b>${esc(t.id.slice(5))}</b> with its own OmniCortex; paths are paths on that computer.`:'')
    +(off.length?`<div>${off.map(x=>`${esc(x.label)}: ${esc(x.why||'not available')}`).join('<br>')}</div>`:''); }
$('#target').onchange=targetNote;
$('#run').onclick=async()=>{const k=$('#kind').value, v=formValues(); $('#ferr').textContent='';
 const target=$('#target').value;
 try{const j=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:k,values:v,target})});cur=j.id;curHost=j.host||'';refresh()}
 catch(e){$('#ferr').textContent=e.message}};
$('#cancel').onclick=async()=>{if(cur){await api('/api/jobs/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:cur,host:curHost||undefined})});refresh()}};
loadTargets(); setInterval(loadTargets,30000);
// Progress, time and the current step for running jobs; a reason and a fix for failed ones.
const fmtS=x=>x==null?'':x<60?Math.round(x)+' s':x<3600?Math.floor(x/60)+' min '+Math.round(x%60)+' s':Math.floor(x/3600)+' h '+Math.round(x%3600/60)+' min';
function bar(p){ return p&&p.total?`<div class="pbar" role="progressbar" aria-valuemin="0" aria-valuemax="${p.total}" aria-valuenow="${p.done}"><i style="width:${Math.min(100,p.pct||0)}%"></i></div>`
  :'<div class="pbar ind" role="progressbar" aria-label="working"><i></i></div>'; }
function stats(j){ const p=j.progress, out=[];
  if(p&&p.total) out.push(`<span><b>${p.done.toLocaleString()} / ${p.total.toLocaleString()}</b> (${p.pct}%)</span>`);
  if(j.elapsed_s!=null) out.push(`<span>running ${fmtS(j.elapsed_s)}</span>`);
  if(p&&p.eta_s!=null) out.push(`<span>about ${fmtS(p.eta_s)} left</span>`);
  if(p&&p.rate) out.push(`<span>${esc(p.rate)}</span>`);
  if(p&&p.label) out.push(`<span title="${esc(p.label)}">${esc(p.label.slice(0,60))}</span>`);
  return `<div class="pstat">${out.join('')}</div>`; }
function hintBox(j){ const h=j.hint; if(!h) return j.status==='lost'?'<div class="hint"><b>Lost.</b> Studio restarted while this job ran, so its result is unknown. Run it again.</div>':'';
  return `<div class="hint"><b>Why it failed:</b> ${esc(h.reason)}<div style="margin-top:.3rem">${h.install?`<button data-install="${esc(h.install)}">Install ${esc(h.install)}</button> then run the job again.`:esc(h.fix||'')}</div></div>`; }
async function refresh(){const js=await api('/api/jobs');
 $('#jobs').innerHTML=js.length?js.map(j=>`<div class="job${j.id===cur?' sel':''}" data-id="${j.id}" data-host="${esc(j.host||'')}"><span class="st ${j.status}">${j.status}</span><span class="jt" title="${esc(j.title)}">${esc(j.title)}${j.host?` <span class="note">on ${esc(j.host)}</span>`:''}</span><span class="note">${new Date(j.started*1000).toLocaleTimeString()}</span>
   ${j.status==='running'?`<div class="jp">${bar(j.progress)}</div>${j.step?`<div class="jstep" title="${esc(j.step)}">${esc(j.step)}</div>`:''}`:''}
   ${j.status==='failed'&&j.hint?`<div class="jstep" style="color:var(--no)" title="${esc(j.hint.reason)}">${esc(j.hint.reason)}</div>`:''}</div>`).join(''):'none yet';
 document.querySelectorAll('.job').forEach(e=>e.onclick=()=>{cur=e.dataset.id;curHost=e.dataset.host;refresh()});
 if(cur){const j=js.find(x=>x.id===cur&&(x.host||'')===(curHost||'')); if(j){$('#detail').style.display='';$('#dtitle').textContent=j.title;$('#dst').textContent=j.status;$('#dst').className='st '+j.status;
  $('#dargv').textContent=j.argv.join(' ');$('#cancel').disabled=j.status!=='running';$('#cancel').hidden=j.status!=='running';
  $('#dprog').innerHTML=j.status==='running'?bar(j.progress)+stats(j)+(j.step?`<div class="note" style="font-family:ui-monospace,monospace;margin-top:.2rem;word-break:break-all">now: ${esc(j.step)}</div>`:'')
    :(j.elapsed_s!=null?`<div class="pstat"><span>${j.status} after ${fmtS(j.elapsed_s)}</span></div>`:'');
  $('#dhint').innerHTML=j.status==='running'||j.status==='done'?'':hintBox(j);
  document.querySelectorAll('[data-install]').forEach(b=>b.onclick=async()=>{ b.disabled=true; b.textContent='installing…';
    try{ const r=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'install_package',values:{package:b.dataset.install}})}); cur=r.id; refresh(); }
    catch(e){ b.textContent=e.message; } });
  const l=await api(curHost?`/api/jobs/host/${encodeURIComponent(curHost)}/${cur}/log`:'/api/jobs/'+cur+'/log');const pre=$('#dlog'),atEnd=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-20;
  pre.textContent=(l.log||'(no output yet)').split(/\r?\n/).filter(x=>!/^PROGRESS \d+\/\d+/.test(x)).map(x=>x.split('\r').pop()).join('\n');if(atEnd)pre.scrollTop=pre.scrollHeight}}}
setInterval(()=>refresh().catch(()=>{}),2000);
load().catch(e=>{$('#ferr').textContent=e.message});
</script>"""
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)

