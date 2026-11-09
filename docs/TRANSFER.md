# Moving models between machines

Model files are large and tuning often spans machines: a GPU box generates
candidates, an LM Studio host serves them, a laptop drives both. There are
four transfer paths. They solve different problems and can be combined.

| Path | Use when | Encryption | Authentication |
|---|---|---|---|
| WebRTC browser GUI | ad-hoc, two people, no install on the far side | always (DTLS) | 128-bit invite key; verification code compares DTLS fingerprints |
| Resumable CLI / HTTP(S) | a receiving host that should just accept files | TLS when started with a cert, otherwise a VPN | 256-bit bearer token |
| Tuning worker | the remote host can build and evaluate candidates itself | same as CLI | same as CLI |
| MCP control plane | an agent should drive transfers | local stdio | tokens held in the OS keyring / 0600 files |

Read [SECURITY.md](SECURITY.md) before exposing any of them beyond localhost.

```bash
pip install -r requirements-transfer-full.txt     # aiohttp, mcp>=2.3,<3, keyring, cryptography
python scripts/oc_transfer_router.py --help        # one entry point for all modes
```

## 1. WebRTC GUI (peer to peer)

```bash
python scripts/oc_security.py selfsigned --host transfer.lan --host 192.0.2.10   # once
python viz/transfer_lab.py --host 0.0.0.0 --tls-cert ns-cert.pem --tls-key ns-key.pem
```

Open the printed URL on both machines. The first browser creates a room and
shows a short **room code**. **Copy invite link** produces a URL whose `#r=…&k=…` fragment carries the room code and a
128-bit **invitation key**. Fragments are never sent to the server in HTTP
requests. Opening the link joins directly. Typing only the code sends a join
request that someone already in the room must **Admit**, which works for
reading the code aloud.

When peers connect, each side shows a six-digit **verification code**
derived from both DTLS certificate fingerprints. Compare it out of band. If
it matches, nothing (including the signaling server) is in the middle of the
encrypted channel.

Files stream through a DataChannel with backpressure, straight to a folder
you choose (File System Access API; Chromium-based browsers, HTTPS or
localhost). Each file is SHA-256 hashed while streaming. The receiver checks
the hash and discards the file on mismatch.

**Connectivity**:

| mode | flag / selector | trade-off |
|---|---|---|
| Direct | `--ice-mode direct` (default) | fastest; STUN reveals each peer's public IP to the other |
| Relay | `--ice-mode relay --turn-url turns:turn.example:5349 --turn-secret-file turn.secret` | all traffic via your TURN server; peers never learn each other's address |
| LAN / VPN | `--ice-mode lan` or `--no-stun` | host candidates only; for WireGuard / Tailscale / same network |

**TURN credentials** are never in `/ice.json`; a peer receives them only once
it has created a room, joined with the key or been admitted. With
`--turn-secret-file` holding coturn's `static-auth-secret` (coturn configured
with `use-auth-secret`), each peer gets its own credentials in coturn's REST
format, expiring after `--turn-ttl` seconds (default 3600; set it longer than
your longest transfer, since an expired credential cannot refresh the relay
allocation). Static `--turn-user`/`--turn-credential` still work but are
long-lived and shared, and print a warning.

The server relays only signaling (SDP, ICE). It never receives file bytes. It
enforces an Origin allowlist (`--allowed-origin` for reverse-proxy hostnames),
throttles wrong keys and knocks per address, caps rooms at four peers, and
sends CSP / no-referrer / frame-deny headers.

## 2. Resumable CLI / HTTP(S)

Receiver, e.g. on the LM Studio host:

```bash
python scripts/oc_security.py token --out ~/.config/omnicortex/transfer.token
python scripts/oc_transfer.py serve --host 0.0.0.0 --port 8810 \
    --root ~/.lmstudio/models/omnicortex \
    --token-file ~/.config/omnicortex/transfer.token \
    --tls-cert ns-cert.pem --tls-key ns-key.pem
```

Sender:

```bash
export OC_TRANSFER_TOKEN="$(cat transfer.token)"     # or --token-file
python scripts/oc_transfer.py send  --url https://lmstudio-host:8810 --cafile ns-cert.pem --file model.gguf
python scripts/oc_transfer.py batch --url https://lmstudio-host:8810 --cafile ns-cert.pem --dir models/sweep --workers 2
python scripts/oc_transfer.py watch --url https://lmstudio-host:8810 --cafile ns-cert.pem --dir models/sweep
python scripts/oc_transfer.py status --url https://lmstudio-host:8810 --cafile ns-cert.pem
python scripts/oc_transfer.py pull  --url https://lmstudio-host:8810 --cafile ns-cert.pem --id <id> --out back/
```

How it works:
- Chunks of up to 8 MiB each carry their own SHA-256. The whole file is hashed
  again before it is published.
- Interrupted transfers resume from the receiver's offset. A dropped chunk is
  retried with backoff after asking the receiver what it already has.
- Files are published with an atomic rename into `--root[/subdir]`. LM Studio
  never sees a partial file.
- `watch` waits for a file's size and mtime to settle before sending it, and
  remembers what it has already sent across restarts.
- `pull` verifies the downloaded hash.

Without `--tls-cert`, a non-loopback bind refuses to start unless
`--allow-plaintext` is given. Use that flag only when the port is reachable
solely through WireGuard/Tailscale or a TLS reverse proxy.

## 3. Adaptive tuning worker

When the remote host can run OmniCortex itself, ship the source model once
and let it generate and evaluate candidates locally. Only small JSON job
messages cross the network:

```bash
# remote host
python scripts/tuning_worker.py --host 0.0.0.0 --port 8799 \
    --root /models/tuning --source /models/model.Q6_K.gguf --profile /models/h_neurons.json \
    --suppressor scripts/suppress_gguf.py \
    --evaluator 'python my_eval.py --model "{model}" --scale "{scale}"' \
    --token-file worker.token --tls-cert ns-cert.pem --tls-key ns-key.pem

# controller
python scripts/adaptive_tuner.py tune --worker https://remote:8799 --cafile ns-cert.pem \
    --token-file worker.token --state runs/adaptive-state.json \
    --min 0.2 --max 0.8 --batch-size 3 --margin-of-error 0.02 --baseline-score 0.91 --auto-delete
python viz/adaptive_tuning_lab.py          # same thing with a GUI
```

The worker records the SHA-256 of the source model, the profile and every
candidate. Deletion is limited to generated `.gguf` files inside its root.
The evaluator command comes only from the worker's own command line and is
never accepted over the API. Every job record the worker returns is signed
with a key derived from the token and echoes a nonce the controller chose for
that job, so the controller rejects a result that was altered, stripped of its
signature or replayed from another job, even if a proxy terminates TLS in
between. See [LABS.md](LABS.md) for the search strategy.

## 4. MCP control plane

```bash
python scripts/oc_transfer_mcp.py        # stdio MCP server; requires mcp>=2.3,<3
```

Tools: `register_target`, `remove_target`, `list_targets`, `probe_target`,
`send_file`, `send_batch`, `start_watch`, `transfer_status`,
`pull_completed`. They are annotated:
- `list_targets`, `probe_target` and `transfer_status` only read;
- `remove_target` is destructive.

So a client's approval rules can tell them apart (Studio: Settings → Tool
approval).

- `register_target` moves the token into the OS keyring (with `keyring`
  installed) or `~/.config/omnicortex/secrets/<name>.token` (0600). The
  target state file is 0600 and holds no secrets. State from older versions
  with inline tokens is migrated on first load.
- The token reaches the CLI through the child's environment, never argv. It
  is redacted from any output and no tool returns it.
- **The tools act on the computer that runs the server**: its registered
  targets, and file paths on that computer. Each computer has its own
  targets, so a server on another machine starts with none
  (`list_targets` returns `{}`) until you register them there.

### Over the network (`--http`)

By default the server speaks stdio: an MCP client on the same computer
starts it. To control transfers on another computer, serve it over
streamable HTTP there:

```bash
python scripts/oc_security.py token --out ~/.config/omnicortex/secrets/transfer-mcp-http.token
python scripts/oc_transfer_mcp.py --http --token-file ~/.config/omnicortex/secrets/transfer-mcp-http.token
#   -> http://127.0.0.1:8765/mcp, bearer token required
```

- **Token**: every request needs the bearer token, even on 127.0.0.1. It
  must be at least 32 characters. After 20 wrong tokens from one address in
  ten minutes, that address waits.
- **Other websites**: requests from web pages of other sites (an `Origin`
  header) are refused. `--allowed-origin` admits one.
- **Network addresses**: a network address (`--host 0.0.0.0`) needs
  `--tls-cert` and `--tls-key`, or `--allow-plaintext` when only a VPN or a
  TLS proxy can reach the port. The same rule as Studio and the receivers.
- **Health**: `GET /health` answers without the token, for monitors.

**Windows with Tailscale** (a ready HTTPS name, no certificates to manage).
Run these in the OmniCortex folder, the one that contains `scripts\`. The
token goes to `%USERPROFILE%\.config\omnicortex\secrets\transfer-mcp-http.token`,
where `--http` finds it without being told.

PowerShell:

```powershell
cd C:\path\to\omnicortex
py -3 scripts\oc_security.py token --out "$env:USERPROFILE\.config\omnicortex\secrets\transfer-mcp-http.token"
py -3 scripts\oc_transfer_mcp.py --http
tailscale serve --bg 8765        # https://<machine>.<tailnet>.ts.net -> 127.0.0.1:8765
```

Command Prompt: the same, with `"%USERPROFILE%\.config\…"` in place of
`"$env:USERPROFILE\.config\…"`. Skip the token line if you already have a
token file there.

The server stays on 127.0.0.1, and Tailscale terminates TLS and only admits
your tailnet. To start it at log-on, add the second command to Task
Scheduler with the trigger *At log on*.

On the controller, copy the token into a file
(`~/.config/omnicortex/secrets/<name>.token`, mode 0600) and add the
server to Studio's `~/.omnicortex/mcp.json` by reference:

```json
{"mcpServers": {"windows-transfer": {"url": "https://<machine>.<tailnet>.ts.net/mcp",
                                     "token_file": "~/.config/omnicortex/secrets/windows-mcp-http.token"}}}
```

Then press **Reconnect / reload config** and **Test** on Studio's Tools tab. Other MCP clients
take the same URL with an `Authorization: Bearer` header.

## Recommended topology

```text
controller (laptop)
   ├── tuning worker over HTTPS ─────────► GPU host (builds + evaluates candidates)
   ├── oc_transfer watch over HTTPS/VPN ─► LM Studio host (receives finished GGUFs)
   ├── WebRTC GUI ───────────────────────► ad-hoc peer
   └── MCP (stdio, or --http over a VPN) ► the agent driving all of the above
```
