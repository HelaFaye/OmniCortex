# Studio

`viz/studio.py` is a model manager, chat window and OpenAI-compatible server
in front of `llama-server`. It covers the everyday LM Studio workflow and adds
the controls this project exists for.

```bash
python viz/studio.py --models-dir ~/models --server ~/llama.cpp/build/bin/llama-server
python viz/studio.py --models-dir A --models-dir B --idle-ttl 900 --min-graded 20
```

Open http://127.0.0.1:7870. Without `--models-dir` it scans LM Studio's own
model folders (`~/.lmstudio/models`, `~/.cache/lm-studio/models`), so both
apps share one library.

Studio does not implement inference, tokenization or sampling; llama.cpp does
that. Studio handles everything around it.

## Built around what people do most

People who run OmniCortex most often come to do these things, in roughly
this order. Each is a click or two from where they start:

| what people come to do | the short path |
|---|---|
| chat with a local model | the home page; with the model picker on *auto*, sending a message loads the best model for the question |
| get a model that fits this computer | **Discover**: each file says whether it fits in memory, and **Tasks → Distribution** predicts how fast it would run on each device |
| know whether a reply can be trusted | *check replies* under the chat box: token risk from the model's own neurons, with a 3D view of the reply |
| compare models on their own questions | **Compare models on your own questions** on the home page opens a TestQA run with two models already filled in |
| use the models from another app | **Connect**, and the `/v1` address in the header |
| look inside a reply | **Studio → Features**, the 3D replay and the live viewer |
| split larger work across models | **Tasks → Projects**, with **Workers** and **Distribution** to see where the work went |

How the pages get people there:

- **First run.** The home page shows a three-step checklist (engine, a model,
  load it) and a Setup banner until both the engine and a model are present.
  Once all three are done, the checklist shrinks to one line.
- **After setup.** The home page offers the common tasks above. The ones you
  use most move to the front. Visits are counted in this browser only
  (`localStorage`), and nothing is sent anywhere.
- **Search** (Ctrl+K or /) lists your most recent pages first, before typing
  anything. It finds every page, setting, job, help page and control on the
  current page.
- **Fewer things to type.** Fields with known values are dropdowns, and cut
  text has a tooltip ([UI_AND_CLI.md](UI_AND_CLI.md#long-text-and-known-values)).
- **The sidebar** groups pages by purpose (Models, Tasks, Studio, Connect).
  Parts you do not use can be removed under **Add-ons**, which shortens it.

## Feature map

| | LM Studio | Studio |
|---|---|---|
| Model library with metadata (arch, quant, MoE, size, fit) | yes | yes; fit is estimated against this host before loading |
| Hugging Face search and resumable download | yes | yes; multi-part GGUFs offered as one set; `HF_TOKEN` for gated repos |
| Will it run here | yes (green/yellow/red) | Discover and My Models: *Full GPU offload*, *Partial GPU offload*, *Runs on the CPU* or *Likely too large*, for NVIDIA, AMD, Intel and Apple GPUs (integrated ones share RAM); ★ *Recommended* marks the largest file that runs fully on the GPU |
| Per-model load settings (GPU layers, context, batch, threads, flash-attn, KV cache type) | yes | yes; persisted per file |
| Several GPUs (split mode, tensor split, main GPU, which GPUs) | yes | yes; shown when NVIDIA GPUs are present, with a suggested split; fit estimate counts free VRAM on all of them ([HARDWARE.md](HARDWARE.md#nvidia--cuda)) |
| Chat with streaming, reasoning shown separately | yes | yes |
| Saved conversations, export | yes | yes; one JSON file per chat in `~/.omnicortex/chats`, Markdown export |
| Image input for vision models | yes | yes; an `mmproj-*.gguf` beside the model is loaded automatically |
| Presets (system prompt, sampling) | yes | yes; includes JSON-schema and GBNF constrained output |
| OpenAI-compatible local server | yes | yes, at `/v1` on the same port |
| Just-in-time model loading on API requests | yes | yes; a swap waits for in-flight requests to finish |
| Idle auto-unload (TTL) | yes | `--idle-ttl SECONDS` |
| Speculative decoding | yes | draft model, `--draft-max/min` |
| Tokens/s and time to first token per reply | yes | yes |
| **Live suppression α** (LoRA scale, no reload) | no | yes |
| **MoE active-expert override** | no | yes, key read from the model's architecture |
| **`model: "auto"`**: route each prompt to the best-suited local model | no | yes, on each model's measured per-subject accuracy, hallucination and abstention rates |
| **Rolling per-model stats** (graded answers, H-Neuron activations, live abstentions) | no | yes; models without stats are flagged ⚠ |
| Remote access | LM Link (proprietary) | `--host` + token + TLS, device pairing with per-device tokens, and **linked hosts**: another machine's models served from this Studio |
| Chat with your documents (RAG) | yes | yes; BM25 with no model, or hybrid with any embedding GGUF / endpoint; replies cite passages |
| MCP client: chat can call tools from MCP servers | yes | yes; stdio or streamable HTTP, `mcp.json` in the same format; each call asks first unless auto-approved |
| **Jobs page**: TestQA, deficit datasets, fine-tuning, merge/GGUF, SWE-bench, LiveBench, CLIP_benchmark from the browser | no | yes, at `/jobs` |

The transfer system also ships an MCP *server* (see [TRANSFER.md](TRANSFER.md)).

## OpenAI-compatible API

```bash
curl http://127.0.0.1:7870/v1/models
curl http://127.0.0.1:7870/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8b-gguf/qwen3-8b-q6_k", "messages": [{"role": "user", "content": "hi"}]}'
```

- Model ids are `<repo folder>/<file stem>`, lower-cased. A served name set in
  the load panel also works, as do the file name or full path.
- A request that names a model other than the loaded one loads it (JIT).
  `--no-jit` makes such requests fail instead.
- `"model": "auto"` picks a model from measured performance. See below.
  Naming a model is always a manual override, whether or not it has stats.
- Requests with `image_url` content go only to models with a vision projector;
  others get a 400 rather than a confusing failure.
- `stream: true` is passed straight through as SSE.

Point Cline, Continue, Open WebUI or any OpenAI SDK at `http://host:7870/v1`.

## How `auto` chooses

1. The subject classifier turns the last user message into the subjects it
   involves, each weighted (code, math, logic, science, factual, writing,
   vision, graphics, systems, reverse-engineering; several at once when a
   request spans them). When the evidence is thin it says **unknown**, and
   models are ranked on their overall numbers instead. Images restrict the
   candidates to models with a vision projector.
2. **Only models with at least `--min-graded` (default 20) graded results
   compete.** A model with no stats is never auto-selected. You can still load
   it, pick it in the chat's model menu, or name it in an API request.
3. For each candidate and subject, the utility is
   `lower 95% bound of accuracy − cost × upper 95% bound of hallucination rate`.
   Abstaining scores zero, so a model that declines when unsure beats one that
   guesses wrong. `--hallucination-cost` sets the cost (default 1; 0 ranks on
   raw accuracy). Bounds rather than point estimates mean a model with more
   evidence wins over a lucky small sample.
4. A subject a model has fewer than 5 results for borrows its overall rates,
   with confidence bounds as wide as 5 observations. Measured evidence beats
   unverified transfer.
5. The probability-weighted utilities are compared and the best model is
   loaded (JIT). If none qualify, the request fails with 409 and says why.

The chat window's model menu offers **Loaded model**, **Auto (by performance
stats)** and every local model. Models without enough stats show an orange ⚠
there and in the model list; hovering explains why and how to add stats. Each
reply notes which model answered, and for auto picks, why (e.g.
`auto → coder-7b: subject code (97%) -> expected +0.56`).

## Context usage and outputs

The strip above the chat shows the loaded model and how much of its context
window the next request would take. Click it (or Tab to it and press Enter)
for three panes:

- **Environment**: the model, its context window and what it was trained
  for, GPU layers, vision, LoRA, workspace and computer, with a link to the
  model's settings.
- **Context usage**: the system prompt, documents (the passages the last
  reply cited, when a collection is picked), tool definitions (when tools are
  on) and messages, including what you are typing, plus the room kept for the
  reply (the preset's max tokens). Counts come from the running engine's
  tokenizer, or are estimated at about 4 characters a token when no model is
  loaded. The chat template adds a little. Past 75% the count turns amber;
  when the next request would not fit, the strip says so. The last reply's
  measured prompt and reply tokens are shown next to it. `POST /api/context`
  returns the same counts.
- **Compression** (two controls under the usage bar, kept in this browser):
  *Trim document passages* keeps only the sentences of each retrieved chunk
  that share the question's words (BM25-weighted, with the sentence on
  either side), within about a fifth of the window. *When the chat no longer
  fits* can **summarize the oldest messages**: the model writes a summary of
  them once, the summary stands in for them, and it is kept with the chat
  and extended when more messages fall out of the window. Or it can **drop
  the oldest messages**. The newest four messages are always sent word for
  word. Both are lossy: each reply's stats line says what was trimmed, and
  summaries are listed under Outputs (`scripts/context_compress.py`).
- **Outputs**: what this chat produced: 3D views and neuron review links of
  checked replies, tool calls, cited sources, plugin checks, and Export .md.

## Rolling stats

Stats live in `~/.omnicortex/stats/<model id>.jsonl` (append-only) and are
tied to the model *file*: re-downloading or re-quantizing under the same name
starts from zero. Summaries count only the last `--stats-window` (default
500) observations of each kind, and nothing older than 180 days.

| kind | source | used by auto? |
|---|---|---|
| graded: right / hallucinated / abstained, per subject | `testqa.py --publish-stats` or `--record-stats` | **yes** |
| activation: H-Neuron classifier score per reply | Studio, for models with a classifier set in the load panel, when `--cett` (or `$OC_CETT`) is available | no, shown only |
| live: replies and how often the model declined | every chat and `/v1` reply | no, shown only |

Activation scoring runs one extra prefill per reply in the background, after
requests finish, on the CPU by default (`--score-ngl` to change). Use
`--score-every N` to sample. It is shown but not used for ranking, because
the classifier is a weak signal on whole replies (AUROC around 0.7); graded
answers are the evidence `auto` trusts.

**Checking a reply.** With `--cett` and a classifier set for the model, each
assistant reply has a *check* action (and the *check replies* toggle under the
chat runs it after every reply). It runs one extra pass over the reply on the
CPU, scores every token, and shows the result under the reply: how many tokens
crossed the classifier's 0.5 boundary, the peak and mean risk, and the reply
text shaded by risk with the flagged tokens underlined. The check is saved
with the chat and as a trace session in `--traces-dir`
(`~/.omnicortex/traces/<id>`); *open 3D view* shows it at `/viz/<id>/` on
the same Studio, with the same login. `POST /api/trace` with `{model,
messages, text}` does the same from a script; paired devices may call it.
See VISUALIZATION.md, "Which view when", for what to do with a flagged span.

```bash
python scripts/model_stats.py show                               # every model's rolling summary
python scripts/model_stats.py rank "Fix this segfault in my loop"  # what auto would pick, and why
curl http://127.0.0.1:7870/api/stats
```

## Runtime controls

**Suppression α.** Load a model with a OmniCortex LoRA (`export_lora.py`) in
the adapter field, then drag the slider. It sets the adapter scale through
llama-server's `/lora-adapters` with no reload, so you can compare behaviour
at several suppression strengths in one conversation.

**Active experts.** For MoE models the load panel exposes
`<arch>.expert_used_count`. See `scripts/sweep_experts.py` for measuring the
quality and speed curve.

## Documents (RAG)

The **Docs** tab holds collections of files (text, Markdown, code, HTML, JSON,
CSV; PDF with `pypdf`, DOCX with `python-docx`). Pick a collection in the
**docs** selector under the chat and each message retrieves the four best
passages, adds them as a numbered system message, and asks the model to cite
them; the passages appear under the reply, expandable, and are saved with the
chat.

Retrieval is BM25 by default, which needs no model and does well on names,
identifiers and error messages. For meaning-level matches add an embedding
model; search then fuses both rankings (reciprocal-rank fusion):

```bash
python viz/studio.py ... --rag-embed-gguf ~/models/nomic-embed-text-v1.5.Q8_0.gguf   # Studio runs it on demand
python viz/studio.py ... --rag-embed http://127.0.0.1:1234/v1@text-embedding-model  # or any /v1/embeddings
```

The embedding sidecar is a second llama-server (`--embedding`, port
`--backend-port + 1`, CPU unless `--rag-embed-ngl`), so it never displaces the
chat model. A collection remembers which embedding model built it and refuses
to mix in vectors from another. Collections live in `~/.omnicortex/rag`
(`--rag-dir`) and work from the command line too:

```bash
python scripts/rag.py add --collection notes docs/*.md
python scripts/rag.py search --collection notes "how does auto routing pick a model"
```

## Tools (MCP)

Add, edit, test, turn off and remove MCP servers under **Settings →
Connectors** (the Tools tab links there). **+ Add connector** asks for a name
and either the server's URL, with an optional sign-in (a token file, an
environment variable or headers), or the program to start with its
arguments and environment. Saved secrets are never sent back to the page:
they show as dots, and leaving them as dots keeps them. The page writes
`~/.omnicortex/mcp.json` (`--mcp-config`), and the file can be edited by
hand too. It uses the format LM Studio and Claude Desktop use:

```json
{"mcpServers": {
  "files":  {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
  "search": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ..."}},
  "box":    {"url": "https://box.example.ts.net/mcp", "token_file": "~/.config/omnicortex/secrets/box.token"},
  "clock":  {"command": "python", "args": ["clock_server.py"], "autoApprove": ["now"]},
  "old":    {"command": "...", "disabled": true}
}}
```

Keep tokens out of this file:
- `"token_file"` (or `"token_env"`) becomes the server's
  `Authorization: Bearer` header;
- any header or `env` value may say `${file:PATH}` or `${env:NAME}`.

They are read each time the server connects, so after rotating a token,
press **Reconnect / reload config** on the Tools tab; there is nothing to edit here.

Tick **tools** under the chat and the model is offered every connected tool
(as `<server>__<tool>`; llama-server's Jinja templates, on by default, turn
them into the model's native tool format). When the model calls one, the call
and its arguments appear in the reply with **Allow** / **Deny**; nothing runs
until you choose, unless the server's `autoApprove` is `true` or lists that
tool. Results are shown under the call and fed back to the model, for up to 8
rounds per message. A declined call is reported to the model as declined.

The **Tools** tab shows each server's status and tools, and reconnects after
you edit the file. **Test** next to a server asks it for its tools again, and
calls one tool it marks read-only and that needs no arguments (for the
transfer server, `list_targets`). It shows the time taken and the answer. Stdio servers' logs go to `~/.omnicortex/mcp-logs/`.
`python scripts/mcp_client.py tools` checks a config from the command line.

MCP servers run with your permissions. Only configure servers you trust, and
keep `autoApprove` to read-only tools.

## Link: paired devices and other machines' models

The **Link** page (header) does two things, both built on one-time pairing
links instead of sharing the master token.

**Pair a device.** *Create pairing link* gives a link usable once, like
`https://studio.lan:7870/pair#c=K7QX-M2PA-9DTE&fp=…`, claimable for five
minutes by default.
Opened in a browser, it pairs that browser and logs it in; on the command line,
`python scripts/oc_pairing.py claim '<link>' --out laptop.token` writes a token
file for API clients. Each device gets its own token, stored on the host only
as a SHA-256 hash, listed with when it was last seen, and revocable at once.
Paired devices can chat (with documents), use `/v1` and keep saved chats.
Everything that administers the machine needs the master token: jobs (they
run code), load settings and llama-server flags, downloads, MCP tools (they
run with the host owner's permissions), pairing, revoking and links.

**Persistent or temporary access.** Each pairing link grants one of:

- **Persistent**: the device keeps access until you revoke it (a paired
  browser's cookie is renewed for a year at a time; revoking still ends it at
  once).
- **Temporary**: access ends a set time after pairing (1 hour, 8 hours, 1 day,
  7 days, 30 days, or a custom duration). The device's requests are refused
  from that moment, and a paired browser's cookie expires with it.

Choose under **Access** before creating the link; the page the device opens
says which it is getting. The device list shows each device's access and time
left, and **change…** extends or shortens temporary access, renews an expired
device (same token, no re-pairing) or makes it persistent. Expired devices stay
listed for 30 days, then drop off. A linked host with temporary access shows
"access expired" when it runs out, without calling the other machine.

The host's owner sets the policy at startup:

| flag | default | |
|---|---|---|
| `--pair-code-ttl` | `5m` | how long a pairing link can be claimed (up to `1d`) |
| `--pair-durations` | `1h,8h,1d,7d,30d` | the temporary choices offered |
| `--pair-max` | `90d` | the longest temporary access any pairing or extension may grant |
| `--pair-default` | `persistent` | what is preselected (`persistent` or a duration) |
| `--no-persistent-pairing` | off | every device expires; persistent access cannot be granted or restored |

```bash
# a shared machine: guests get a day at most, nothing permanent, 2-minute links
python viz/studio.py ... --no-persistent-pairing --pair-durations 1h,8h,1d --pair-max 1d --pair-code-ttl 2m
```

`POST /api/pair/start` takes `{"persistent": true}` or `{"persistent": false, "ttl": "8h"}`;
`POST /api/devices/update` takes the same plus the device `id`.

**Use another machine's models.** On the GPU box's Studio create a pairing
link, and paste it into this Studio's Connect → Devices page with a name (say `gpu`). Its
models appear in the picker and in `/v1/models` as `gpu:<model id>`; requests
for them, from the chat or from any OpenAI client pointed at this Studio, are
forwarded to that machine with this Studio's device token, streamed back, and
can use this Studio's documents and tools. Stats and `auto` routing stay
per machine.

The link carries the host certificate's SHA-256 fingerprint (`fp=`). The
claiming side checks that the server presents exactly that certificate before
sending the code, and keeps checking it on every later request, so a
self-signed LAN certificate is pinned, not trusted blindly, and a machine in
the middle can neither take the code nor read the traffic. Pairing requires
the host to run with a token; without TLS the link has no fingerprint and
should only be used inside a VPN. Tokens for linked hosts are kept in
`~/.omnicortex/links.json` (0600); devices in `~/.omnicortex/devices.json`.

## Jobs

`/jobs` (**Jobs** in the sidebar) starts evaluation, retraining and
benchmark runs without a terminal: TestQA, deficit datasets, SWE-bench training
data, fine-tuning, merge and GGUF conversion, SWE-bench predict / evaluate /
import, LiveBench and CLIP_benchmark. Each job is a fixed script with a form of
typed fields; there is no free-form command line, and no value may start with
`-`, so a field cannot add flags. Output streams into the page and is kept in
`~/.omnicortex/jobs/<id>/` (`--jobs-dir`); jobs keep running when the browser
closes, and can be cancelled. `--max-jobs` (default 2) limits how many run at
once.

A typical retraining loop on one page: TestQA with a cache → Deficit dataset
→ Fine-tune → Merge and convert → load the new GGUF in Studio → TestQA again
on `holdout_ids.json`. See [RETRAINING.md](RETRAINING.md).

Jobs can train models and run model-written code, so they are available on a
loopback bind only, unless Studio is started with `--allow-remote-jobs`
(still behind the token). `--no-jobs` turns them off entirely.

## Security

Studio loads models, downloads files and runs inference, so binding beyond
loopback is gated:

```bash
python scripts/oc_security.py token --out ~/.config/omnicortex/studio.token
python scripts/oc_security.py selfsigned --host studio.lan --host 192.0.2.10
python viz/studio.py --host 0.0.0.0 --token-file ~/.config/omnicortex/studio.token \
    --tls-cert ns-cert.pem --tls-key ns-key.pem
```

- A non-loopback bind without a token refuses to start.
- A non-loopback bind without TLS needs `--allow-plaintext`, meaning the port
  is only reachable through a VPN or a TLS reverse proxy.
- The browser login sets an HttpOnly, SameSite=Strict cookie (Secure under TLS).
  API clients send `Authorization: Bearer <token>`.
- Failed logins are throttled per address. Request bodies are bounded.
- Jobs are off on a network bind unless `--allow-remote-jobs` is given, and
  only accept JSON requests, so a cross-site form cannot start one.

See [SECURITY.md](SECURITY.md).

## Projects

`/projects` splits a project into tasks by skill, keeps a plan you approve,
starts worker models on whatever accelerators the machine has (AMD first), and
sends every result back for review. See [DIRECTOR.md](DIRECTOR.md).

## Settings

`/settings`: workspaces (where models, llama.cpp, pipeline data and Studio's
state live, by name), the default model per subject for `auto`, and tool
approval rules for MCP tools in chat. See [SETTINGS.md](SETTINGS.md).

## Review

`/review` shows, for every neuron, how its activity relates to wrong answers
across everything recorded for a model: chat checks (mark them right or
wrong under the check), live scoring, project work, and TestQA runs and
benchmarks ingested from Jobs → Review. Filter by source, subject, verdict
and dates; play it over time in 3D. See [REVIEW.md](REVIEW.md).

## Requirements

Setup → Requirements lists what each feature needs and what is missing,
installs missing Python packages, and keeps environment snapshots to
compare. See [REQUIREMENTS.md](REQUIREMENTS.md).

## Connect

`/connect` has ready-made settings for Cline, Claude Desktop and other
OpenAI-compatible apps, and Studio serves MCP at `/mcp` so agents can use
OmniCortex's tools. See [CONNECT.md](CONNECT.md).
