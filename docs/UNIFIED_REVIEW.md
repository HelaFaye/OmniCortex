# Unified review

One record per model from every source OmniCortex knows. Studio's `/api/lookup`, the
[browser extension](BROWSER_EXTENSION.md) and `scripts/unified_review.py` all return the same record
(schema `omnicortex-lookup/1`), so every viewer can show each statistic once, with a dropdown of the
value each source gave and filters by source and group.

```bash
python scripts/unified_review.py lookup Qwen/Qwen2.5-7B-Instruct --board github:owner/omnicortex-board
python scripts/unified_review.py lookup org/name --board-file board.json --saelens --review-dir ~/.neuronscope/review --json
```

## Sources

| Source | What it adds | Checked? |
|---|---|---|
| A review board (`board.json` from a [GitHub board](SHARING.md), or a scoreboard server) | TestQA accuracy, confident-mistake rate and per-subject accuracy; generation speed per hardware; per setup (quant, engine, LoRA, suppression) | Yes: the board re-grades every card from its log |
| The board's hardware-profile reports ([HARDWARE_PROFILES.md](HARDWARE_PROFILES.md)) | How often a profile worked on the hardware this model was measured on | Counted, not re-run |
| Card extras | Emotion-vector means and top SAE features, if the submitter added them | **No**: self-reported, shown with a tag and filterable |
| SAELens's catalogue | Which pretrained dictionaries exist for the model | From SAELens's own list |
| This Studio | Replies observed (activation profiles are collected by default, [ACTIVATIONS.md](ACTIVATIONS.md)), and the share you marked wrong | Your own data |

Values measured on a re-packaging of the model (a `-GGUF`, `-AWQ`, … repo of the same name) are kept
and marked `related`, so a GGUF upload's results show on the original model's page and can be filtered out.

## The record

```json
{"schema": "omnicortex-lookup/1", "model": "org/name", "available": true,
 "sources": [{"id": "board:owner/board", "kind": "board", "ok": true, "error": null, "stats": 12}],
 "stats": [{"key": "accuracy", "group": "quality", "label": "Accuracy (TestQA, verified)", "value": 0.71,
            "unit": "ratio", "n": 400, "source": "board:owner/board", "variant": "Q4_K_M · llama.cpp",
            "related": false, "detail": {"submissions": 2}}]}
```

Groups: `quality`, `performance`, `features`, `emotions`, `activity`. Studio's record also carries
`open.hub` (`/#hub=org/name`: the model's files in Studio, ready to download) and `open.review`.

## Adding emotion and feature summaries to a card

`score_cards.py build|submit --extras extras.json` adds a small, sanitised block:

```json
{"emotions": {"calm": 0.21, "desperate": -0.4},
 "features": [{"sae": "res-l12", "id": "1234", "label": "code comments", "freq": 0.05}]}
```

At most 64 emotions and 32 features, numbers only, labels cut to 80 characters with markup removed.
Boards average them across a row's cards. They cannot be verified from the card's log, so every
viewer marks them self-reported.
