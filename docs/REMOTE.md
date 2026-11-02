# Remote hosts: models on computers that don't run OmniCortex

Use the LLMs on another computer from Studio. That computer doesn't need
OmniCortex: a Windows PC with LM Studio, a server with Ollama, or a box
running llama-server is enough. It runs a small agent, **OmniCortex
Remote** (`remote/`, distributed on its own), and Studio controls it over
MCP.

## Who is who

- **Controller**: this Studio. It decides what runs where and sends the
  instructions. Studio's Remote page says *This computer: CONTROLLER*.
- **Host**: the computer with the models and the agent. The agent's
  console says *This computer: HOST*, shows the pairing code, and lists the
  controllers it obeys. The host's owner can revoke them, and can allow or
  forbid each kind of action.

Any MCP client can be a controller too (Claude Desktop, Cline, Cursor…).
The agent's [README](../remote/README.md) shows how.

Which page to use:

| the other computer runs | use |
|---|---|
| an LLM server (LM Studio, Ollama, llama.cpp…) but not OmniCortex | **Remote** (this page) |
| OmniCortex Studio | **Link** (Studio to Studio pairing, [SECURITY.md](SECURITY.md)) |
| nothing yet; you want to copy GGUFs to it as they are made | the transfer watcher (`scripts/ns_transfer.py`), or Remote's *Send model* once the agent runs there |

Which computer controls and which does the work, for every connection and port: [ROLES.md](ROLES.md).

## Adding a host

Open Studio → **Remote** (*Remote* in the sidebar) → **Add a host**.

1. **Put the agent on the host.** Click **Download OmniCortex Remote**,
   copy the zip to the host, and start it:
   - Windows: double-click `OmniCortex Remote.bat`;
   - macOS and Linux: `./neuronscope-remote.sh`.

   It needs Python 3.9+ and nothing else. Its console opens in the host's
   browser.
2. **Choose how Studio reaches the host.** Pick the same link on both
   sides, in Studio's wizard and in the host's console:

   | link | best when | security | the host listens on |
   |---|---|---|---|
   | **Same network** | both on one home or office network | TLS with the host's own certificate, pinned at pairing; the pairing proof is bound to it | the LAN (port 47871), plus UDP beacons on 47870 while pairing |
   | **VPN**: Tailscale, Headscale, ZeroTier, WireGuard, NetBird, Nebula… | far apart, both in one VPN | the VPN's encryption; token per controller | its VPN address only |
   | **SSH tunnel** | the host has an SSH server, and you have a key for it | SSH; Studio runs `ssh -N -L` itself with your keys and config (it cannot answer password prompts) | itself only (127.0.0.1) |
   | **Public tunnel**: Cloudflare Tunnel, ngrok… | no VPN, no ports you can open | the tunnel's public HTTPS certificate, checked normally; token per controller | itself only; the tunnel forwards |
   | **The host calls this Studio** | the host is behind NAT or a firewall you cannot open | Studio's TLS (pinned through the invite) or a VPN; the host's own token | nothing: it connects out and long-polls Studio |

3. **Pair.**
   - **Same network:** **Search the network** lists hosts whose pairing is
     open.
   - **VPN, SSH, tunnel:** type the address.
   - **Shortcut:** paste the *pairing link* the host's console shows
     (`nsr://…`) into the box at the top of *Add a host*. The link carries
     the address, the code and the certificate.

   Each code:
   - is good for 10 minutes;
   - works once;
   - dies after five wrong tries.

   After pairing, Studio shows the certificate as four plus four
   characters; the host's console shows the same.
4. **For a host that calls in:**
   1. **Make an invite** in Studio. Studio must be reachable from the host:
      on the network with a token and TLS, through a VPN, or behind a
      tunnel. The wizard warns when Studio listens on its own computer
      only.
   2. On the host, paste the invite into the console's **Join** box, or run
      `neuronscope-remote connect nsr1:…`.

   The host shows up in Studio's list as *connected* once it joins, and
   reconnects by itself whenever the agent runs.

## What Studio does with a host

- **Its models are Studio models.** They are listed as `<host>:<model>`
  wherever models are listed, and can be used for:
  - chat;
  - the OpenAI API (`/v1/chat/completions`);
  - TestQA (`--endpoint name=http://studio:7870/v1@gpu-box:qwen3-8b`);
  - the director's worker models.

  For a host that calls in, chat goes as the host's MCP `chat` tool with an
  MCP progress token. The agent reads its backend's reply as it is written
  and posts the new text to Studio a few times a second, so the reply
  streams as it does for any other model. Closing the chat (or pressing
  Stop) tells the host to stop at its next post, which stops generation.
  Agents older than this answer in one piece, which still works.
- **Per host, on Connect → Remote hosts:**
  - *Host info*: OS, RAM, GPUs, LLM servers, folders, what is allowed;
  - *Models*: load and unload;
  - *Try a model*;
  - *Send model*: any of this Studio's GGUFs, optionally installed into
    LM Studio when it arrives;
  - *Remove*.
- **Sending a model:**
  - Studio uploads in 16 MB pieces, resumes where an earlier attempt
    stopped, and the host checks the SHA-256 before keeping the file.
  - A host that calls in downloads the file instead, from a one-time
    Studio link.
  - `install_model` puts it at `<LM Studio models>/neuronscope/<name>/`,
    which is where LM Studio looks, or creates an Ollama model.
- **Everything over MCP.** Each of these is a call to one of the host's
  tools. Other programs reach the same tools through Studio's MCP server:
  - `remote_hosts`: the hosts, how each is reached, and its models;
  - `remote_call`: run a host's tool, for example
    `{"host": "gpu-box", "tool": "load_model", "args": {"model": "qwen3-8b"}}`.

  From a terminal:
  `./neuronscope cli remote-hosts` and
  `./neuronscope cli remote-call --host gpu-box --tool host_info`.

The host's tools are:
- `host_info`
- `list_backends`
- `list_models`
- `chat`
- `load_model` and `unload_model`: LM Studio through its `lms` command,
  Ollama by keep-alive; llama-server serves its one model
- `list_files`
- `receive_file` and `transfers`
- `install_model`
- `delete_file`: off unless the host's owner allows it
- `telemetry`: GPU temperatures (edge, junction, memory), VRAM, load and
  power, and whether the thermal guard has paused the host
- `model_file`: where a model's GGUF is on the host, its size and SHA-256
- `trace_reply`: run a reply through OmniCortex's tracer on the host

## Running Studio's jobs on a host

Any job on the Jobs page (TestQA, benchmarks, SWE-bench, retraining, the
H-Neuron pipeline…) can run on a host instead: pick the host under **Run on**.
That needs two things on the host:

1. **A OmniCortex folder** with its requirements installed. Starting the
   agent from inside one is enough (`remote/` is part of it). Otherwise set
   the folder in the agent's console under *What controllers may do*.
2. **Its owner's permission:** tick *run OmniCortex jobs* in the console.
   It is off by default, because jobs run Python code on that computer.

The host builds the command with its own copy of OmniCortex, so only the
job kinds and fields it knows can run, never an arbitrary command. Paths in
the form are paths on the host. Remote jobs appear in the same list with
their progress, log and failure reason, and Cancel stops them on the host.

The agent can also be restarted from Studio's **System** page (*Remote
agent: name → Restart*). The host's owner can turn that off with *restart
this agent*.

## Thermal guard

The host's console has *GPUs and thermal guard*:
- it shows each GPU's temperatures, VRAM, load and power, read from
  nvidia-smi, rocm-smi, or Linux's own amdgpu sensors (no extra tools
  needed). A value that can't be read shows as ?, and never trips the guard;
- turn the guard on and set a limit (the hottest of junction, edge and
  memory). Above it, the host refuses new chats and traces and unloads what
  LM Studio and Ollama can unload. llama-server and vLLM stay up but get no
  work. It resumes a chosen margin below the limit;
- Studio's Workers page shows the host as *too hot*, and its tasks wait.

## Checking a host model's replies in 3D

*Check* under a reply (or *auto-check*) works for a host's models too, and
the result opens in the same 3D view. The activations have to be computed
where a copy of the model file is, by running the reply through it once
more with `llama-cett-dump`. LM Studio and Ollama cannot report them. Studio
picks the place:

1. **On this computer, when it has the same file.** Studio asks the host
   for the model's GGUF size and SHA-256 and looks for an identical file
   among its own models. A model you sent with *Send model* always
   matches. The local copy's classifier is used, and the host does nothing
   but report the hash.
2. **On the host, otherwise.** Studio sends the reply and the classifier;
   the host runs OmniCortex's tracer and sends back the per-token trace
   (a few MB). The host needs:
   - OmniCortex installed (the agent finds it when it runs from a
     OmniCortex folder; otherwise set `tracer.neuronscope` in the agent's
     `config.json`);
   - `llama-cett-dump` built there (Setup → Build llama.cpp on that machine,
     then set `tracer.cett`, or put it on `PATH`, or set `NS_CETT`);
   - *check replies with OmniCortex's tracer* allowed in the host's
     console (on by default).

   *Host info* says whether the host's tracer is ready.

Which classifier: the one set for the host model (Remote → the host →
*Models* → *Classifier…*), else that of the identical local file, else that
of a local model with the same file name. A classifier belongs to one model
(its layer and neuron counts); a fine-tune of the same architecture needs
its own.

The agent finds a model's file by asking Ollama (`/api/show`) or
llama-server (`/props`), then by name under the models and incoming
folders. When a name is ambiguous, the host's owner can name the file in
`config.json` under `model_files`, e.g.
`"model_files": {"qwen2.5-7b-instruct": "D:/models/qwen2.5-7b-instruct-q4_k_m.gguf"}`.

The reply's text, the question and the trace are stored in Studio, as for
local models. The host keeps nothing.

## Scoring tuning candidates on a host

[LABS.md](LABS.md) explains how the adaptive tuner scores candidates in LM
Studio on another machine with `scripts/eval_lmstudio.py`. With the agent
on that machine, there are two ways:

- point `--lmstudio` at the host's LM Studio over the VPN, as before;
- or send each candidate with *Send model* and let *install* put it where
  LM Studio lists it.

## Security, in short

- **Pairing:**
  - The code never travels.
  - Each side sends an HMAC of it, bound to the TLS certificate it sees.
  - The host answers with its own proof, so a fake host fails too.
- **Tokens:**
  - Each controller gets its own random 256-bit token.
  - The host stores only hashes.
  - Studio keeps the tokens in its links file (0600), next to linked
    Studios.
- **Hosts that call in:**
  - They authenticate to Studio with a token of their own.
  - That token opens only the three routes they use: claim, next, reply.
  - It never opens Studio's owner API.
- **The host's console:**
  - It answers this computer only.
  - It checks the `Host` header.
  - It needs a per-run key from the link the agent prints, so no web page
    can drive it.
- **No shell, no file reading.** The agent has a fixed list of tools. Files
  can be written only into its incoming folder and, by install, into LM
  Studio's folder. Names are reduced to a safe base name.

## Studio API

| route | who | what |
|---|---|---|
| `GET /remote`, `GET /api/remote` | owner | the page; hosts, sends, this Studio's address, local models |
| `GET /remote/download` | owner | the agent as a zip |
| `POST /api/remote/discover` | owner | listen for LAN beacons |
| `POST /api/remote/pair` | owner | pair: `{method, address, code, name}` or `{link}`; SSH: `ssh_target`, `ssh_port` |
| `POST /api/remote/invite` | owner | an invite for a host that calls in |
| `POST /api/remote/call`, `/api/remote/tools` | owner | a host's tool; its tool list |
| `POST /api/remote/send`, `/api/remote/remove` | owner | send one of Studio's models; forget a host |
| `POST /api/remote/claim`; `GET /api/remote/next`; `POST /api/remote/reply` | a host that calls in | join with an invite; wait for messages; answer them |
| `POST /api/remote/progress` | a host that calls in | pieces of an answer still being written; the reply says whether to stop |
| `POST /api/remote/classifier` | owner | the classifier for a host model's 3D checks (`host`, `model`, `classifier` path) |
| `GET /api/remote/file/<ticket>` | a host that calls in | download a file Studio is sending (resumable) |

## When something does not work

- **"no OmniCortex Remote host answered":** the agent is not running, or
  the link differs. Compare the address with *Listening at* in the host's
  console. On Windows, allow Python through the firewall on private
  networks.
- **Search finds nothing:** the beacon does not cross subnets or Wi-Fi
  client isolation. Type the address instead.
- **SSH fails:** Studio cannot answer password prompts. Set up a key with
  `ssh-copy-id`, and check `ssh you@host` works from a terminal.
- **The host that calls in stays "not connected":** check that the host can
  open Studio's address in a browser.
- **"the host does not allow controllers to …":** the host's owner has to
  allow it in the console, under *What controllers may do*.
