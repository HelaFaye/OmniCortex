# The acceptance gate: one verdict for any change

`scripts/accept_gate.py` (Jobs → Retrain → **Acceptance gate**) decides whether a change to a model is kept: a
retrain, a repository adapter, a new quantization. It runs no models. It reads the reports the measuring tools
already write and gives one verdict, with every reason.

| Check | Report it reads | Passes when |
|---|---|---|
| **retrain** | `retrain_gate.py`'s JSON ([RETRAINING.md](RETRAINING.md)) | *keep*: the holdout improved, and neither hallucination nor any subject got significantly worse |
| **fidelity** | `quant_fidelity.py`'s `fidelity.json` ([QUANT_FIDELITY.md](QUANT_FIDELITY.md)) | the quant *keeps*, judged again here with the preset's limits |
| **adapter** | `repo_adapt.py`'s `manifest.json`, with the repository ([REPO_ADAPT.md](REPO_ADAPT.md)) | no core interface changed since training, and the adapter is within its size budget |

**The verdict:**

| Verdict | Meaning | Exit |
|---|---|---|
| **accept** | every check given passed | 0 |
| **review** | none failed, but a person should look (see below) | 2 |
| **reject** | a check failed, or a check named with `--require` wasn't given | 1 |

A check goes to **review** when:
- the retrain gate was inconclusive;
- a quant is past a limit but not twice past it;
- llama-perplexity crashed once and was run again;
- the adapter's manifest has a warning, such as too little replay data;
- the adapter's data came from a working tree with uncommitted changes.

`--require` stops a pipeline from passing by leaving a report out:
`--require retrain --require fidelity` rejects a run that gives only one.

## Presets

| Limit | default | strict |
|---|---|---|
| truth-probe AUROC drop | 0.05 | **0.02** |
| KL divergence | 0.1 | 0.05 |
| same top token | 85% | 90% |
| greedy agreement (without llama-perplexity) | 0.8 | 0.9 |
| TestQA answers lost (95% lower bound) | 3% | 2% |
| H-neuron activity similarity | 0.95 | 0.97 |
| risk flags flipped | 5% | 3% |

- **Use strict** when OmniCortex's own instruments must keep working on the result (the truth probe, the H-neuron
  classifier), and for small models, which lose more to the same quant type.
- **Override one limit** with its own option, such as `--max-auc-drop 0.03`.
- **The same presets work on `quant_fidelity.py`** (`--preset strict`), so its report and the gate agree.

## Example: Qwen2.5-1.5B's quants

From [QUANT_FIDELITY.md](QUANT_FIDELITY.md)'s run:

| Quant | default | strict |
|---|---|---|
| Q8_0 | accept | accept |
| Q4_K_M | accept | **review**: KL divergence 0.056 (limit 0.05), same top token 86.8% (limit 90%) |
| Q2_K | reject | reject |

Q4_K_M kept its truth probe even by the strict limit: its AUROC went from 0.939 to 0.934. What the strict preset
flagged was its next-token drift. Whether that matters depends on the task: chat tolerates it, exact recall
less so. That judgement is why the verdict is *review* rather than *reject*.

## Use

```bash
python scripts/accept_gate.py --preset strict --require retrain --require fidelity \
    --retrain runs/cycle/gate.json \
    --fidelity runs/fidelity/fidelity.json --quant model-Q4_K_M \
    --out runs/accept
```

The report is `accept.json` and `accept.md`. A script can act on the exit status: 0 to promote, 2 to ask a
person, 1 to discard.
