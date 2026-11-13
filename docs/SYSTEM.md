# System: what runs, the hardware, and automation

Studio's **System** page (sidebar, bottom) has three parts.

## Running parts

Each part Studio runs on this computer, with its state and a button:
Studio itself (**Restart**), the chat model server (**Reload**, **Stop**),
worker models, the embedding server for documents, MCP connectors
(**Reconnect**), labs, jobs, and each remote host's agent (**Restart**, if
that host's owner allows it). When OmniCortex's files change after Studio
started, every page offers a restart, because pages and fixes added since
then are missing until it restarts.

## Monitor

CPU, memory and every GPU's load, memory, temperature and power on this
computer every 2 s, and each remote host's GPUs every 10 s. It runs while
the page is open or while automation rules are on, and keeps the last hour.
It also shows whether someone is using each computer (below).

## Automation

Rules watch one computer and act on it while their condition holds:

| when | how it is measured |
|---|---|
| its owner uses it | keyboard or mouse input within N seconds, a game controller, or a VR session (SteamVR, Oculus, WMR, Monado, WiVRn, ALVR); Windows, Linux (X11, systemd-logind) and macOS. What a computer cannot tell stays unknown, and a rule never guesses. |
| power over a limit | a Home Assistant sensor: average power (W) or energy used (kWh) per second, minute, hour, day, week, month or year; months and years can start on a billing day |
| switched on | by hand on the page, or through the API |

| then | |
|---|---|
| unload | its models are unloaded |
| busy | it gets no new work: project tasks, worker models, remote jobs |
| requeue | tasks running there go back to the queue and another computer takes them; this does not count as a failed attempt |
| throttle | one task at a time there |

An activity rule stays on for a quiet period after the last input. A power
rule can hold until its period ends (the next billing day, for example).

A remote host can do the first part **on its own**, without Studio: in the
agent's console, *When I use this computer → Give it back to me*. The host
then unloads its models and refuses new work while its owner uses it,
whatever any controller asks. Studio sees the host as busy and moves the
work elsewhere. This is the setting for a friend's computer you are allowed
to use: when they come home to play, the model is out of their way.

### Home Assistant

Enter the address and a long-lived access token (Home Assistant: Profile →
Security). The token is kept in `automation.json` next to Studio's
settings, readable only by you, and never sent back to the page. **Test a
sensor** reads one sensor right away.

### The automation API

Anything that can send an HTTP request can act on a computer, with Studio's
owner token:

```
POST /api/automation/action
{"target": "local" | "host:NAME", "action": "busy" | "unload" | "throttle" | "free",
 "minutes": 0, "reason": "gaming"}
```

`minutes: 0` means until freed. `GET /api/automation` returns the rules,
what is busy and why, and the recent actions. `GET /api/system/monitor`
returns the readings. For example, a Home Assistant `rest_command`:

```yaml
rest_command:
  rig_busy:
    url: "http://studio.local:7870/api/automation/action"
    method: post
    headers:
      Authorization: "Bearer YOUR_STUDIO_TOKEN"
    content_type: "application/json"
    payload: '{"target": "host:rig", "action": "unload", "minutes": 0, "reason": "gaming"}'
```
