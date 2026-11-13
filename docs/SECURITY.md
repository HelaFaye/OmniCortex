# Security

OmniCortex runs several network services: the Studio UI and API, the WebRTC
signaling server, the resumable transfer receiver, and the tuning worker.
They share one policy, implemented once in `scripts/oc_security.py` and
covered by `tests/test_security.py`.

## Bind policy (all services)

| bind | token | TLS |
|---|---|---|
| `127.0.0.1` / `::1` / `localhost` | optional | optional |
| anything else | **required**, ≥ 32 chars (refuses to start) | **required**, unless `--allow-plaintext` |

Services with a token (Studio, hub, control, live stream, transfer receiver,
tuning worker) follow the table above. Local tools that have **no**
authentication (dashboard, trace viewer, the Quantization / Scale Sweep /
Adaptive Tuning labs, the tracing proxy) refuse any non-loopback bind unless
given `--allow-unauthenticated`. Reach them over an SSH tunnel
(`ssh -L 8796:127.0.0.1:8796 host`) or through the hub instead.

`--allow-plaintext` means "this port is reachable only through WireGuard,
Tailscale or a TLS-terminating reverse proxy". The service still prints a
warning.

```bash
python scripts/oc_security.py token --out ~/.config/omnicortex/transfer.token   # 256-bit, written 0600
python scripts/oc_security.py selfsigned --host myhost.lan --host 192.0.2.10    # P-256 cert for LAN use
```

Pass secrets with `--token-file` or `$OC_TRANSFER_TOKEN` (Studio also reads
`$OC_STUDIO_TOKEN`). `--token` still works but warns, because argv is visible
to every local user. Clients trust a self-signed cert with `--cafile`.

### IPv6 transfer receiver

The resumable transfer receiver selects IPv6 automatically when `--host` is an
IPv6 literal, or explicitly with `--family ipv6` when binding a DNS name that
resolves to IPv6. Bind hosts may be unbracketed (`::`) or bracketed (`[::]`);
URLs still require brackets around IPv6 literals, for example
`https://[2001:db8::10]:8810`.

For an IPv6-only listener on all local IPv6 interfaces, use `--host :: --family
ipv6`. The server sets `IPV6_V6ONLY=1`, so an IPv6 wildcard bind does not
silently expose the same port over IPv4. Non-loopback binds still require a
strong token and TLS unless the operator explicitly selects the VPN-only
`--allow-plaintext` mode. A self-signed certificate must contain the exact
IPv6 address or DNS name clients use, and clients should pass it with
`--cafile`; never use `--insecure` for real transfers.

Example (IPv6-only, TLS):

```bash
python scripts/oc_security.py token --out ~/.config/omnicortex/transfer.token
python scripts/oc_security.py selfsigned --host 2001:db8::10 \
  --cert transfer-cert.pem --key transfer-key.pem
python scripts/oc_transfer.py serve --host :: --family ipv6 --port 8810 \
  --root /srv/omnicortex-incoming \
  --token-file ~/.config/omnicortex/transfer.token \
  --tls-cert transfer-cert.pem --tls-key transfer-key.pem
```

Replace the documentation-only `2001:db8::10` address with an assigned address
on the receiver. Prefer a VPN address (for example a Tailscale address) over
exposing a receiver directly to the public Internet.

## What each path guarantees

| Component | Confidentiality | Authentication | Integrity |
|---|---|---|---|
| WebRTC data channel | DTLS (always on) | 128-bit invitation key or admitted knock; out-of-band verification code over both DTLS fingerprints | DTLS + per-file SHA-256 checked by the receiver |
| WebRTC signaling | TLS/WSS when started with a cert | Origin allowlist; invitation key | — |
| CLI / HTTP transfer | TLS with `--tls-cert`; otherwise the VPN's | 256-bit bearer token, constant-time compare, throttled | per-chunk and per-file SHA-256 (detects corruption; not a signature) |
| Tuning worker | same as CLI | same as CLI | SHA-256 of source, profile and each candidate recorded |
| MCP | local stdio | the local client | process boundary |
| Studio | TLS with `--tls-cert` | token (Bearer or HttpOnly cookie), throttled login | — |

SHA-256 detects accidental corruption. It does not prove who produced the
bytes; that comes from TLS plus the token, or from DTLS plus the verification
code.

## Hardening checklist (status)

**WebRTC**
- [x] HTTPS/WSS (`--tls-cert/--tls-key`); plain HTTP refused off loopback
- [x] 128-bit invitation key separate from the human-readable room code
- [x] room codes lengthened to 10 characters (~49.5 bits); they are labels, not secrets
- [x] throttled joins and knocks per address (8 failures/min → 10 min lockout), identical errors for wrong code and wrong key
- [x] Origin allowlist on the WebSocket upgrade
- [x] verification code from both DTLS fingerprints (detects a MITM, including the signaling server)
- [x] per-file streaming SHA-256, verified by the receiver; mismatches are discarded
- [x] relay-only mode (TURN) for address privacy; LAN/VPN mode with no STUN
- [x] TURN credentials only for peers inside a room, never from `/ice.json`;
  with `--turn-secret-file` (coturn `use-auth-secret`) each peer gets its own,
  expiring after `--turn-ttl`
- [x] `crypto.randomUUID` fallback for non-secure contexts; CSP, no-referrer, frame-deny headers

**HTTP / CLI**
- [x] built-in TLS server and verifying client (`--cafile`)
- [x] token mandatory off loopback; 256-bit generator; ≥ 32-char minimum
- [x] constant-time comparison (`secrets.compare_digest`)
- [x] authentication-throttle state is periodically pruned and bounded to 4,096 tracked peer addresses per process
- [x] tokens from file or environment; argv use warns
- [x] resumable chunks; final SHA-256 before atomic publish
- [x] bounded bodies: JSON 64 KiB, chunk 8 MiB; socket idle timeout 120 s
- [x] failed-auth throttling with periodic expiry and a fixed per-process peer-state cap; cap on concurrent in-progress transfers (`--max-active`)
- [x] Studio uses a bounded HTTP worker pool, idle socket timeouts, and a total request-body deadline (including slow-drip clients)
- [x] 128-bit hex transfer ids; generic 500 responses (no exception text)

**MCP**
- [x] `mcp>=2.3,<3`, version checked at startup
- [x] 0600 state file without secrets
- [x] tokens in the OS keyring (`keyring`) or 0600 files; legacy inline tokens migrated
- [x] tokens passed to subprocesses via environment, never argv
- [x] no tool returns a token; command output is redacted
- [x] stdio only

**Tuning**
- [x] candidate, source-model and profile SHA-256 per job
- [x] atomic publication of candidates
- [x] authenticated worker API (same policy); TLS client in the controller
- [x] job-id/scale validation; deletion restricted to generated GGUFs in the worker root
- [x] persistent per-job state as an audit trail
- [x] signed results: every job record carries an HMAC-SHA256 under a key derived
  from the worker token, bound to a per-job nonce from the controller; the
  controller refuses unsigned, altered or replayed results (protects against a
  TLS-terminating proxy or a plaintext VPN hop; it cannot protect against a
  compromised worker, which holds the key)

**Studio pairing and links**
- [x] one-time pairing codes (~59 bits, single use, throttled claims; claimable for `--pair-code-ttl`, 5 minutes by default)
- [x] persistent or temporary device access; expiry is enforced on every request, the browser cookie
  ends with it, and the host caps it (`--pair-max`) or forbids persistent access (`--no-persistent-pairing`)
- [x] per-device tokens, stored as SHA-256 hashes, revocable; devices are limited to chat, `/v1`, saved
  chats and document search (no jobs, load flags, downloads, MCP tools, pairing or links)
- [x] pairing links carry the TLS certificate fingerprint; claims and every linked-host request pin it
- [x] linked-host tokens in a 0600 file; pairing refused when the host has no token

**Still open / by design**
- Peer identity on WebRTC is proven by comparing the verification code, not by
  accounts. That is deliberate (no accounts), and the code check is manual.
- Rate limits are in-process and per address, and reset on restart. For
  Internet exposure put a reverse proxy in front (Caddy, nginx `limit_req`)
  so limits hold across processes and addresses.
- TestQA without `--sandbox` runs code with rlimits only; with
  `--sandbox docker` (or podman) it is contained (see [TESTQA.md](TESTQA.md)).
  A container shares the host kernel; for hostile code use a VM.

## Recommended deployments

**Private infrastructure**: WireGuard or Tailscale between hosts, services
bound to the tunnel address with `--allow-plaintext` and a token.

**Across the Internet**: TLS on every service (or a TLS reverse proxy such as
Caddy or nginx, with the service on loopback), tokens from files, WebRTC in
relay mode through your own TURN server, and the verification code checked
on every transfer that matters.

## Reporting

Please report vulnerabilities privately to the maintainers rather than in a
public issue.

## Hugging Face custom model code

OmniCortex does not execute custom Python code from a model repository by default. Hugging Face loaders are configured with `trust_remote_code=False`, and the CLI tools that support custom architectures require the explicit `--trust-remote-code` option. Studio exposes the same choice as an unchecked opt-in. The legacy `--no_trust_remote_code` flag remains accepted as an explicit deny for existing command lines.

Enabling this option executes code supplied by the model repository in the OmniCortex process. Use it only for a repository you have reviewed and trust.
