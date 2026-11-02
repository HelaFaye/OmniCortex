# Roles and ports: which computer controls, which does the work

Every connection in OmniCortex has two ends:
- the **controller** decides what to do and sends the request;
- the other end **does the work**: it runs the model or the tool, on its
  own hardware.

In MCP terms the controller is the *client* and the other end is the
*server*. The same program can be a controller in one connection and do the
work in another.

```
 agent app (Cline, Claude Desktop…)     controller
        │  MCP /mcp, or OpenAI /v1
        ▼
 OmniCortex Studio  ───────────────►  does the work for the app,
        │                              and controls what is below
        ├─ this computer's models      worker: started by Studio
        ├─ endpoints (/v1)             worker: a server you started
        └─ OmniCortex Remote hosts    worker: the host runs the model
              │  OpenAI /v1 on the host itself
              ▼
           LM Studio, Ollama, llama-server, vLLM on that host
```

## The connections

| connection | controller | does the work | protocol | where you set it up |
|---|---|---|---|---|
| an agent app uses OmniCortex's tools | the agent app | Studio's computer | MCP at Studio's `/mcp` (or `scripts/ns_mcp.py`, stdio) | Connect page |
| an app chats with your models | the app | Studio's computer (or wherever Studio routes it) | OpenAI API at Studio's `/v1` | Connect page |
| Studio uses another computer's LLMs | Studio | the host | MCP to OmniCortex Remote on the host | Remote page |
| an agent app uses another computer's LLMs | the agent app | the host | MCP: `neuronscope_remote bridge <host>` on the app's computer | CONNECT.md, *MCP, step by step* B |
| Studio sends project tasks to an endpoint | Studio | the endpoint's computer | OpenAI API at the endpoint's `/v1` | Workers page |
| Studio's chat calls tools | Studio | wherever each tool server runs | MCP, from `mcp.json` and plugins | Settings, Plugins |
| two Studios share models | the Studio that links | the linked Studio | Studio's API, with a device token | Connect → Devices |
| an MCP client drives file transfers on another computer | Studio, or any MCP client | the computer running `ns_transfer_mcp.py --http` | MCP at its `/mcp`, bearer token | TRANSFER.md, *Over the network* |

## Ports

| default port | what listens | protocol | is it MCP? |
|---|---|---|---|
| 7870 | OmniCortex Studio: UI, `/v1`, `/mcp` | HTTP(S) | yes, at `/mcp` |
| 47871 | OmniCortex Remote, on a host | HTTPS on a LAN, HTTP on a VPN | yes, at `/mcp` |
| 47872 | OmniCortex Remote's console, on the host itself only | HTTP | no |
| 47870 (UDP) | OmniCortex Remote's beacon while pairing is open | UDP | no |
| 8080 | llama-server (Studio's chat model) | OpenAI API | no |
| 8090 | decode_lab (optional) | OpenAI API | no |
| 8765 | `ns_transfer_mcp.py --http` (optional) | HTTP(S), bearer token | yes, at `/mcp` |
| 8810 | `ns_transfer.py serve`, a file receiver | HTTP(S), bearer token | no |
| 1234 / 11434 / 8000 | LM Studio / Ollama / vLLM | OpenAI API | no |

An LLM server's port is a chat API. Pointing an MCP client at it does
nothing useful. Pointing a chat app at Studio's `/mcp` doesn't work
either. If a plan says "MCP on port 8082", it usually means one of two
things:
- **an endpoint**: an OpenAI-compatible server on 8082, which Studio uses as
  a worker (Workers → *Add an endpoint*);
- **a Remote host**: an agent that listens on its own port (47871 by
  default; `--port` changes it) and in turn talks to the LLM server on the
  same computer.

## Who may do what

- **Studio's tools:** the owner's token or session decides. Paired devices
  may chat but not run jobs (SECURITY.md). Tool approval rules decide what
  an agent may call without asking (Settings → Tool approval).
- **A Remote host:** its owner decides, in the host's console, what
  controllers may do: chat, load, receive files, install, delete, trace. The
  owner can revoke a controller at any time, and turn on the thermal guard.
- **An endpoint:** only its own key, if it has one (`--api-key` on
  llama-server). Keep endpoints on a network you trust.

## Telling them apart in the UI

- Remote page: *This computer: CONTROLLER* → *Hosts*.
- Host console (OmniCortex Remote): *This computer: HOST*, with the paired
  controllers listed.
- Connect page: *Your agent app: CONTROLLER* → *This Studio: does the work*.
- Workers page: each worker's kind (*this computer*, *endpoint*, *remote
  host*) and where it runs.
- MCP: Studio's server instructions and `studio_status` name the computer
  the tools act on. A Remote host's instructions say it is the host.
