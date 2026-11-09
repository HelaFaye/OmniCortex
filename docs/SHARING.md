# Sharing results with a scoreboard

**Off unless you turn it on.** OmniCortex can send TestQA results to a
scoreboard, so models can be compared across machines and setups. It does
not run a scoreboard of its own: `scripts/scoreboard_server.py` is a
reference server that anyone can host. Nothing is sent until you turn
sharing on and give an address.

Studio: Settings → **Sharing results**. Terminal:
`python scripts/score_cards.py preview|submit|check|history`.

## What is sent

**A card, which may be public:**
- **The model.**
  - The GGUF file's SHA-256, size and name (never its path), and the
    quantization.
  - Its Hugging Face repo, only when the file is verified to be in it.
- **The setup (`variant`)**, so results of different setups are never mixed:
  - the hash of the H-Neuron set and the suppression scale;
  - the LoRA's hash and scale;
  - fine-tune, merge, concept-edit and SAGE lineage recorded next to the
    model;
  - the engine (llama.cpp, vLLM, remote) and its version;
  - the engine options that can change answers: context, KV cache types,
    speculative decoding, flash attention, expert count;
  - sampling.

  Options that only change speed (threads, batch, CPU placement) are left
  out. The `variant_id` is a hash of all of this.
- **Hardware, coarse**: OS, CPU model, cores, RAM and GPU memory in whole
  GB, GPU names.
- **The test**: the TestQA bank's fingerprint (a hash of the items asked),
  and the right / wrong / declined counts per subject.

**Measured speed**, when the run measured it: generation tokens per second,
seconds per request and requests in flight. The card also carries the
settings that decide speed: GPU layers, context, devices, split, KV cache,
flash attention and speculative decoding. These train the speed
predictions ([HARDWARE.md](HARDWARE.md#speed-predictions)).

**A verification log, private.** Per item: the id, the verdict and the
model's reply. A scoreboard uses it only to check the card. The reference
server:
- grades every reply again with the same grader;
- rejects the card if the replies do not grade to its numbers;
- keeps the log in a table no public route reads. Only its operator's
  admin token can fetch a log, to look into a disputed result.

**Never sent:** chats, prompts you typed, file paths, user or computer
names, tokens, anything about other models. Every submission is saved on
your computer exactly as sent (`~/.omnicortex/submissions/`) and listed in
Settings.

## Public or private

| the model | a card for it is |
|---|---|
| found by model search: its file's SHA-256 is in the Hugging Face repo it came from | public |
| unlisted, and you listed it by hand (Settings → Sharing results → *list this unlisted model*) | public, marked "unlisted, published by its submitter" |
| unlisted | private: stored and verified, seen only by the scoreboard's operator |

**Automatic submission:**
- *Submit automatically when a TestQA run finishes* covers models found on
  Hugging Face.
- **Advanced → Also submit results for unlisted models** adds the rest:
  privately, unless listed by hand.
- Manual submission (*Preview exactly what is sent*, then *Submit*) works
  for any model once sharing is on.

## Hosting a scoreboard

```bash
python scripts/scoreboard_server.py --db board.sqlite --port 8765 --admin-token-file admin.token --verify-hf
```

Put it behind TLS (`--tls-cert/--tls-key`, or a reverse proxy) on the
internet. It verifies against its own checkout's TestQA bank: cards from a
different bank version are stored as *unverified* and not shown.

Each public row is one model file in one setup, aggregated over its
submissions:
- accuracy and hallucination rate;
- per-subject counts;
- the hardware the runs came from.

## A scoreboard on GitHub

A board can also live in a GitHub repository, with no server to host:

```bash
python scripts/github_board.py init ./my-board --omnicortex-repo OWNER/omnicortex
```

`init` writes these files:
- `board.pub`, the board's public key;
- `board.key`, its private key, git-ignored;
- a workflow in `.github/workflows/accept.yml`;
- an empty `cards/` folder and `board.json`.

To finish setting it up:
1. Create the repository on GitHub and push these files.
2. Store the contents of `board.key` as the Actions secret `BOARD_KEY`.
3. Delete `board.key`.

**Submitting.** In Studio, set the scoreboard to `github:OWNER/REPO` and save a
GitHub token that can open issues on that repository. A fine-grained token
limited to that one repository is enough. The token is stored on this
computer (mode 0600), never shown again, and removed by saving an empty one.

A submission is an issue:
- **The card** is plain JSON in the issue.
- **The verification log** (each item's reply) is sealed to the board's
  public key: X25519, HKDF-SHA256 and ChaCha20-Poly1305. A log too long for
  one issue continues in comments.

Only the board's workflow can open the log: it holds the private key.

**Accepting.** The workflow runs `github_board.py action` with the
OmniCortex checkout named at `init`, which supplies the TestQA bank and the
graders. It regrades every reply, as the reference server does:
- **A verified card** is committed to
  `cards/<sha[:2]>/<sha>/<variant>/<issue>.json`, and `board.json` is
  rebuilt.
- **A rejected card** (replies that do not grade to its numbers, a different
  bank, a duplicate of an accepted result, an unreadable log) is not
  committed.

Either way, the workflow answers on the issue and closes it. **The log is
never committed.** Once read, it exists only in the issue, sealed.

**Everything accepted is public**, because the repository is. Studio
therefore sends only cards for public models to a GitHub board: models found
on Hugging Face, or ones you listed by hand. A private card is refused before
anything is sent.

**Reading.** `board.json` holds:
- one row per model file and variant;
- the speed measurements cards carried;
- the fitted speed prediction (`perf_model`);
- the measurement weights per subject (`measure_weights`).

Studio reads it from `raw.githubusercontent.com` (cached for an hour) to
train its speed predictions. So does anyone else:
`python scripts/github_board.py fetch OWNER/REPO`.
