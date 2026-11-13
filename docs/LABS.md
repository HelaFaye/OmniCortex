# Labs: quantization, scale sweeps, adaptive tuning

Three GUIs plus CLIs for turning an H-Neuron profile into shipped GGUFs. None of them trains; they quantize, scale and measure.

## Quantization Lab

A **capability-preserving quantization layer** without changing OmniCortex's H-Neuron intervention semantics and without adding an SFT/training system.

### What it does

1. Builds a deterministic calibration corpus from one or more weighted text sets.
2. Computes a llama.cpp importance matrix (`llama-imatrix`).
3. Reads OmniCortex H-Neuron profiles and converts their layer-level concentration into protection scores.
4. Selects the highest-scoring decoder layers under a protection budget.
5. Uses `llama-quantize --imatrix` plus `--tensor-type` to quantize normal tensors at the base level and selected FFN down-projection tensors at a higher precision.
6. Builds in RAM (`/dev/shm`) when enough tmpfs space exists, otherwise falls back to disk scratch unless RAM-only mode was requested.
7. Copies the completed GGUF to the destination atomically and writes a `.omnicortex.json` build manifest containing source/output hashes, profiles, calibration, plan and commands.

The first implementation deliberately protects **whole FFN down tensors per selected layer**, because the normal llama.cpp quantizer exposes tensor-level type overrides rather than arbitrary per-neuron bit allocation. Future work can replace this planner with finer-grained block/channel allocation without changing the GUI or build manifest format.

### CLI

```bash
python scripts/quantization_lab.py inspect /path/to/model-F16.gguf

python scripts/quantization_lab.py plan \
  --model /path/to/model-F16.gguf \
  --profile profiles/<fingerprint>/math.json::1.0 \
  --profile profiles/<fingerprint>/coding.json::1.25 \
  --base-quant Q4_K_M \
  --protected-quant Q6_K \
  --budget-percent 10 \
  --json

python scripts/quantization_lab.py build \
  --model /path/to/model-F16.gguf \
  --output /path/to/model-Q4-preserved.gguf \
  --calibration /data/math.txt::1.0 \
  --calibration /data/general.txt::0.5 \
  --profile profiles/<fingerprint>/math.json::1.0 \
  --profile profiles/<fingerprint>/coding.json::1.25 \
  --base-quant Q4_K_M \
  --protected-quant Q6_K \
  --budget-percent 10 \
  --scratch auto \
  --imatrix-bin ~/llama.cpp/build/bin/llama-imatrix \
  --quantize-bin ~/llama.cpp/build/bin/llama-quantize
```

For production use, start from an F16/BF16/F32 source. The harness refuses to treat a filename that looks already quantized as canonical unless `--allow-requantize-source` is explicitly supplied.

### GUI

```bash
python viz/quant_lab.py \
  --models-dir ~/.models \
  --profiles-dir ./profiles \
  --host 127.0.0.1 \
  --port 8796
```

Then open `http://127.0.0.1:8796/`.

The GUI exposes source/target quants, weighted capability profiles, weighted calibration sources, scratch policy, tool paths, protection budget, a protection-plan preview, and a live build log.

## Scale Sweep Lab

The Scale Sweep Lab builds static GGUF variants from a OmniCortex H-Neuron profile.
It treats the same scale operation as a symmetric control:

- `0 < scale < 1`: suppression
- `scale == 1`: baseline/no-op
- `scale > 1`: amplification

It does not fine-tune or retrain the model.

### GUI

```bash
python viz/scale_sweep_lab.py --host 127.0.0.1 --port 8797
```

Open `http://127.0.0.1:8797/`.

The GUI visualizes the H-Neuron activation/selection footprint by layer and the requested scale grid. It also controls batch size, scratch policy, evaluation tolerance, and automatic deletion of underperforming candidates.

### Evaluation contract

The optional evaluator command receives these environment variables and formatting fields:

- `OC_MODEL` / `{model}` — candidate GGUF path
- `OC_SCALE` / `{scale}` — scale used
- `OC_BASELINE` / `{baseline}` — baseline score if supplied
- `OC_PROFILE` / `{profile}` — H-Neuron profile path

It must print JSON containing either `score` in `[0,1]` or `correct` + `total`.

Examples:

```json
{"correct": 82, "total": 100, "wrong": 12, "abstained": 6}
```

When `correct`/`total` are present, a 95% Wilson interval is calculated. With automatic pruning enabled, a candidate is deleted when its upper confidence bound is below `baseline - margin_of_error`.

### CLI

```bash
python scripts/scale_sweep_lab.py sweep \
  --model /path/to/base.Q6_K.gguf \
  --profile /path/to/h_neurons.json \
  --output-dir /path/to/sweep \
  --min-scale 0.20 \
  --max-scale 0.40 \
  --step 0.05 \
  --batch-size 2 \
  --scratch auto
```

The state file is JSON and can be reused to inspect or resume a run. Existing same-sized candidates are skipped.

### Safety boundaries

The lab delegates GGUF modification to the repository's existing `scripts/suppress_gguf.py` backend. It does not reimplement GGUF internals. Keep the canonical unmodified source GGUF and build every candidate from it; do not chain requantized candidates or repeatedly scale a previously scaled model.

## Adaptive remote tuning

This subsystem separates tuning control from model execution.

The **controller** may run on your workstation. The **worker** runs on the machine that has the LLM/GGUF and evaluator. The worker keeps the source model, H-Neuron profile, generated candidates, and evaluation artifacts local. Only small JSON job/control messages cross the control connection.

Use the Model Transfer service separately for initial source/profile/dataset movement and for retrieving only the final candidates you want to keep.

### Adaptive search behavior

The search is intentionally coarse-to-fine:

```text
wide range
   ↓
batch of N scales
   ↓
evaluate
   ↓
find best result
   ↓
contract interval toward best
   ↓
fresh interior batch
   ↓
repeat
```

A run stops when one of these conditions is met:

- the current interval is at or below the selected autotuning resolution;
- the best result has a 95% Wilson score interval whose width is within the selected margin of error;
- the configured maximum number of batches is reached;
- no new scale can be generated.

The state JSON persists after every job and after every narrowing decision, so reconnecting to the same state file resumes from the saved `next_interval` rather than restarting the broad search.

### Automatic deletion

Enable `--auto-delete` only when a baseline score is supplied. After a batch completes, candidates whose upper confidence bound is below `baseline_score - margin_of_error` are eligible for deletion, except for the best candidate from that batch.

Deletion is constrained to the worker's configured output root.

### Remote worker

Start this on the machine that actually has the LLM/GGUF:

```bash
python scripts/tuning_worker.py \
  --host 0.0.0.0 \
  --port 8799 \
  --root /models/omnicortex-tuning \
  --source /models/model.Q6_K.gguf \
  --profile /models/h_neurons.json \
  --suppressor /opt/omnicortex/scripts/suppress_gguf.py \
  --evaluator 'python /opt/evaluators/my_eval.py --model "{model}" --scale "{scale}"' \
  --workers 1 \
  --token-file ~/.config/omnicortex/worker.token \
  --tls-cert ns-cert.pem --tls-key ns-key.pem
```

A non-loopback bind requires the token and TLS (or `--allow-plaintext` behind a VPN); see [SECURITY.md](SECURITY.md).

### Controller CLI

```bash
python scripts/adaptive_tuner.py tune \
  --state /path/to/adaptive-state.json \
  --worker https://REMOTE-HOST:8799 --cafile ns-cert.pem \
  --min 0.20 \
  --max 0.80 \
  --initial-step 0.20 \
  --resolution 0.01 \
  --batch-size 3 \
  --margin-of-error 0.02 \
  --baseline-score 0.91 \
  --token-file ~/.config/omnicortex/worker.token \
  --auto-delete
```

### Controller GUI

```bash
python viz/adaptive_tuning_lab.py \
  --host 127.0.0.1 \
  --port 8800
```

Open:

```text
http://127.0.0.1:8800/
```

Set:

- Worker URL
- Worker token
- minimum/maximum scale
- initial step
- autotuning resolution
- models per batch
- margin of error
- optional baseline score
- optional automatic deletion

The chart displays the observed score by scale, while the JSON panel exposes the exact persisted state and selected interval.

### Recommended workflow for a remote LLM machine

1. Use **Model Transfer** to move the canonical source GGUF, matching H-Neuron profile, and calibration/evaluator data to the remote machine.
2. Start `tuning_worker.py` on that machine.
3. Run the controller GUI locally.
4. Let the controller narrow the search remotely; intermediate GGUFs stay on the remote disk.
5. Retrieve only retained/final GGUFs with Model Transfer.
6. Keep the controller state JSON as the experiment manifest.

This avoids moving multi-GB candidates after every scale test.

### Scoring candidates in LM Studio on another machine

When the model will run in LM Studio on another computer (say a desktop
over Tailscale), score each candidate there, where it will be used. The
tuner writes candidates locally, the transfer watcher carries each one
over as it appears, and `scripts/eval_lmstudio.py` is the evaluator:

1. Start the watcher on the folder the candidates go to: `oc_transfer.py
   watch`, or the transfer MCP's `start_watch`. Point its destination at a
   folder **inside LM Studio's models folder**, two levels deep
   (`<models>/<publisher>/<name>/`), or LM Studio will not list the files.
2. In LM Studio on the destination: Developer → Server on, **Just-in-time
   model loading** on, and preferably **auto unload unused JIT models**, so
   one candidate is in memory at a time.
3. Score the unedited model once with the same evaluator, for
   `--baseline-score`.
4. Tune (Jobs → Tuning → *Adaptive scale tuning*, or):

```bash
python scripts/adaptive_tuner.py tune --state runs/tune-mymodel.json --worker local \
    --source models/mymodel-Q6_K.gguf --profile models/h_neurons.json \
    --output-dir models/adaptive_candidates \
    --evaluator "python scripts/eval_lmstudio.py --lmstudio http://desktop:1234 --model {model} --metric not_hallucinated" \
    --baseline-score 0.71
```

For each candidate the evaluator:
- waits until LM Studio lists it (up to `--wait`, default an hour, for
  large transfers);
- asks it the same seeded TestQA sample (`--per-subject`, `--seed`, or your
  own `--tasks`) at temperature 0;
- prints `{score, correct, total, …}`, which the tuner reads. Each run's full
  results are kept under `runs/lmstudio-eval/`.

The metrics:
- `accuracy`: right / graded;
- `not_hallucinated`: wrong answers count against it, abstentions do not;
- `utility`: right minus `--cost` × wrong, rescaled to 0–1.

Keep `--auto-delete` off until a few scores look right: nothing is deleted
otherwise. The same evaluator scores a single file:
`python scripts/eval_lmstudio.py --lmstudio http://desktop:1234 --model models/adaptive_candidates/x.gguf`
(Jobs → Tuning → *Score a candidate in LM Studio*).
