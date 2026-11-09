# OmniCortex Remote

Let another computer use the LLMs on this one. You don't need OmniCortex
on this computer. The agent works with LM Studio, Ollama, llama.cpp's
`llama-server`, or any OpenAI-compatible server.

## Who is who

| | **Host** | **Controller** |
|---|---|---|
| what it is | the computer with the models and this agent | OmniCortex Studio, or any MCP client (Claude Desktop, Cline, Cursor…) |
| what it does | runs the models and does what paired controllers ask, within what you allow | decides what to run and sends the instructions |
| what it shows | the console: pairing code, controllers, permissions, activity | its list of hosts and their models |

A controller can do no more than the agent's tools offer, which is a short,
fixed list:
- see the host and its models;
- chat;
- load and unload models;
- receive model files and install them into LM Studio or Ollama;
- delete files (off until you allow it).

It cannot run commands, read your files or reach anything else on the
host. You see every call in the console, and **Revoke** cuts a controller
off at once.

## Install on the host

The host needs Python 3.9 or newer. No other packages are needed.

- **Windows:** unzip the download from Studio (Remote → *Download*) and
  double-click `OmniCortex Remote.bat`. If you don't have Python, get it
  from python.org or the Microsoft Store.
- **macOS, Linux:** run `./omnicortex-remote.sh`.
- **pip:** run `pip install ./omnicortex-remote` (or `pipx install …`), then
  `omnicortex-remote`. Add `[tls]` if the computer has no `openssl` command.
  The *Same network* link needs one of the two to make its certificate.

The agent starts and opens its **console** in your browser. Only this
computer can open the console. If you close the tab, get it back with
`omnicortex-remote console`.

Start the LLM server you use as well. LM Studio: *Developer → Start
server*. Ollama and llama-server: just run them. The agent finds them on
their usual ports, or you can add an address in the console.

## Connect a controller

**1. Choose how the controller reaches this computer** (console, step 1):

| link | when | what the agent does |
|---|---|---|
| **Same network** | home or office | listens on the network with its own certificate, which the controller pins at pairing; announces itself while pairing is open |
| **VPN** (Tailscale, Headscale, ZeroTier, WireGuard, NetBird, Nebula…) | the two computers are far apart but in one VPN | listens on the VPN address only; the VPN encrypts |
| **SSH tunnel** | this computer runs an SSH server | listens on this computer only; the controller runs `ssh -L` (Studio does it for you) |
| **Public tunnel** (Cloudflare Tunnel, ngrok…) | no VPN, no open ports | listens on this computer only; the tunnel gives it a public HTTPS address |
| **This computer calls the controller** | behind a router or firewall you cannot change | listens on nothing; connects out to the controller and waits for instructions |

**2. Pair.** Click **Show a pairing code** in the console. On the
controller, open OmniCortex Studio → **Remote** → **Add a host**, and
either type the code or paste the pairing link. With *Same network*,
Studio lists this host on its own. Each code:
- is valid for 10 minutes;
- works once;
- dies after five wrong tries.

The code itself never crosses the network: each side proves it knows the
code, and the proof is bound to the certificate the controller sees. A
machine in the middle presenting another certificate fails the pairing.

For **This computer calls the controller**, it works the other way round:
1. Studio makes an invite (`nsr1:…`).
2. Paste it into the console's **Join** box, or run
   `omnicortex-remote connect nsr1:…` on this computer.

## Other MCP clients as controllers

On the controller computer, install this same package and pair from the
command line:

```bash
omnicortex-remote add-host 'nsr://192.168.1.20:47871/#c=…'   # the pairing link the host shows
omnicortex-remote add-host 100.101.102.103 --code ABCDE-FGH23
omnicortex-remote hosts
```

`add-host` prints the entry for the MCP client's configuration. The entry
starts `omnicortex-remote bridge NAME`, which carries MCP over
stdin/stdout to the host and pins its certificate:

```json
{"mcpServers": {"host-gpu-box": {"command": "python3", "args": ["/path/to/agent.py", "bridge", "gpu-box"]}}}
```

Clients that speak MCP over HTTP can use `https://HOST:47871/mcp` with the
header `Authorization: Bearer <token>`. This works only if they trust the
host's certificate: either through a VPN or SSH link with plain HTTP, or
a tunnel with a public certificate.

An MCP client on the host itself can run `omnicortex-remote stdio`.

## Commands

```
omnicortex-remote                       start with the saved link and open the console
omnicortex-remote serve --mode lan|vpn|ssh|tunnel|callhome [--bind ADDR] [--public-url URL]
omnicortex-remote pair                  a new pairing code
omnicortex-remote controllers [--revoke ID]
omnicortex-remote connect nsr1:…        call home to a controller
omnicortex-remote console               open the console of the running agent
omnicortex-remote stdio                 MCP over stdin/stdout on this computer
omnicortex-remote add-host | hosts | remove-host | bridge     (controller side)
```

The agent's own port is 47871; its console is on 47872, on this computer
only. On the LAN, the agent sends beacons on UDP 47870 while pairing is
open. Windows asks once whether Python may use the network for the *Same
network* and *VPN* links; allow it on private networks.

## Files

The agent keeps its settings in one folder:
- Windows: `%APPDATA%\OmniCortexRemote`
- macOS: `~/Library/Application Support/OmniCortexRemote`
- elsewhere: `~/.config/omnicortex-remote`
- or the folder in `$OCR_HOME`, if set.

What it holds:
- `config.json`: the link, the LLM servers, and what controllers may do;
- `controllers.json`: who may control this host. Only the hashes of their
  tokens are kept;
- `tls/`: this host's certificate;
- `incoming/`: files controllers sent;
- `hosts.json`: on a controller, the hosts it paired with.

Received models go into `incoming/`. `install_model` then moves them into
LM Studio's models folder as `omnicortex/<name>/<file>.gguf`, or creates an
Ollama model from them.
