# Recorded activations

**On by default.** Studio records the activations of every reply it serves: one profile per reply (how
strongly each layer's neurons fired over it, float16 `[layers, bins]`). They feed:
- Neuron review;
- the Distribution page's activation plots;
- the review system, when you choose to share.

Turn it off, or record one reply in N, under **Settings → Activations**. What was already recorded stays.

**How they are read.** Profiles are read with a method that works for the model. For GGUF models that is
`llama-cett-dump` (Setup → Build llama.cpp). Until it is present, the Activations section says so, and replies
are not profiled. A model with an H-Neuron classifier gets its risk score from the same pass.

## Where they go

| store | for |
|---|---|
| on this computer | the default: the review store (`~/.neuronscope/review`, or `--review-dir`) |
| in another folder | a second disk, a network share, a mounted bucket: a host with little space of its own |
| on another OmniCortex Studio | sent over HTTPS with that Studio's token (`POST /api/review/ingest`) to a server that keeps everyone's activations |
| in memory only | a host with no persistent storage: kept until Studio restarts |

**When the destination can't be reached**, nothing is lost at once. Profiles wait in a bounded queue, which
is also written to a spool file unless you turn that off, and are delivered when the destination answers
again. **Send waiting now** delivers them immediately.

**The size cap** (20 GB by default) removes the oldest observations when the store grows past it, and the
section shows how much space is used and free.

## Locked settings

Settings that can lock you out, expose Studio or lose data start locked, on the Settings and Setup pages:
- where activations are stored, and the size cap;
- the address and port Studio listens on, and the llama-server port;
- the token file, the TLS certificate and its key, and plain HTTP;
- remote jobs, and turning jobs off;
- persistent pairing;
- the MCP configuration.

**🔒 Unlock** shows what the setting does and what changing it can break. The setting becomes editable only
after you tick *I know what I am doing*.

Any page can lock a control the same way: mark it `data-locked="the warning"` (`nsLockAll` in
`viz/ui_links.py`).
