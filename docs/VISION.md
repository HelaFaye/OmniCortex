# H-Neurons for image-text models (CLIP, SigLIP)

The text pipeline finds MLP neurons whose activity predicts a confident wrong
answer. Contrastive image-text models fail the same way: an image is matched,
confidently, to the wrong caption. `scripts/clip_neurons.py` applies the same
method to them and reuses the text pipeline's classifier unchanged.

Supported: any Hugging Face dual-encoder with `vision_model` / `text_model`
towers built from `encoder.layers.N.mlp.fc1 → act → fc2`. That covers OpenAI
CLIP, OpenCLIP checkpoints converted to `transformers`, and SigLIP / SigLIP 2.
The vision towers inside llama.cpp `mmproj` files are covered for applying the
edit (below).

## 1. Collect: which images does the model get confidently wrong?

```bash
python scripts/clip_neurons.py collect --model openai/clip-vit-large-patch14 \
    --images data/my-imagefolder --out runs/clip
```

Input is either an ImageFolder tree (`root/<class name>/*.jpg`; underscores in
folder names become spaces) or `--manifest file.jsonl` with
`{"id", "image", "label", "candidates"?}` rows for per-image candidate sets.

Each image is classified zero-shot under `--views` augmented views (identity,
mirror, crops), with prompt ensembling over `--template`s:

- **t**: every view is right;
- **f**: every view picks the *same* wrong label with probability ≥ `--min_conf`.
  This is the hallucination analogue: a stable, confident mismatch rather than
  crop noise;
- **mixed / uncertain**: dropped, like inconsistent answers in the text pipeline.

It writes `results.jsonl`, a balanced `train_qids.json` / `test_qids.json`
split, and `run.json`. You need at least a few hundred of each class for a
meaningful classifier. Fine-grained label sets (dog breeds, car models, plant
species) produce far more confident mismatches than coarse ones.

## 2. Extract CETT

```bash
python scripts/clip_neurons.py extract --model openai/clip-vit-large-patch14 \
    --run runs/clip --ids runs/clip/train_qids.json --out runs/clip/acts
```

Hooks every `fc2` of the chosen tower (`--tower vision`, the default, or
`text`) and computes, per neuron,

    CETT(layer, token, j) = |a_j| · ‖W_fc2[:, j]‖ / ‖fc2 output‖

averaged over patches (`image/`) and at the CLS token (`cls/`). The text tower
is measured on the caption the model chose. Output uses the same
`act_<id>.npy` + `neuron_index.json` layout as the text pipeline.

Extract in float32 (the default). CETT measures magnitudes, and lower
precision perturbs exactly that quantity.

## 3. Train the classifier

```bash
python scripts/classifier.py --acts_root runs/clip/acts --ans_dir image \
    --train_mode 1-vs-1 --train_ids runs/clip/train_qids.json \
    --test_ids runs/clip/test_qids.json --C 0.05 --out_dir models/clip
```

The resulting `h_neurons.json` records `arch: clip` and the tower, so later
tools address the right modules. As with text models, lower `--C` gives a
sparser set; under 0.1% of neurons is the paper's regime.

## 4. Evaluate a scale sweep

```bash
python scripts/clip_neurons.py evaluate --model openai/clip-vit-large-patch14 \
    --run runs/clip --ids runs/clip/test_qids.json \
    --h_neurons models/clip/h_neurons.json --scales 1 0.75 0.5 0.25 0 \
    --threshold 0.5 --out runs/clip/eval.json
```

The selected neurons are scaled at runtime with forward pre-hooks, which is
mathematically identical to scaling the `fc2` weight columns (a test checks
this). Reported per scale:

| metric | meaning |
|---|---|
| accuracy | top-1 zero-shot accuracy |
| coverage | fraction with top-1 probability ≥ threshold (answered) |
| confident_error_rate | answered **and** wrong: the number suppression should lower |
| selective_accuracy | accuracy among answered |
| conf_auroc | how well confidence separates right from wrong |

A useful edit lowers the confident error rate and raises selective accuracy
or confidence AUROC without a large accuracy loss. As with text models,
suppression trades confident errors for low-confidence ones; it does not add
knowledge. Measure on held-out images, never on the training split.

## Pixel art and cartoons

CLIP learned mostly from photos, and its processor treats every image like
one. Pixel art usually arrives already enlarged into hard square blocks, and
the processor resizes those blocks, whose sharp edges look nothing like a
photo. It also centre-crops, which cuts off the edges of wide or tall art.
`--style` fixes the input side without retraining (`scripts/clip_style.py`):

| style | what happens |
|---|---|
| `off` (default) | the processor's own resize and crop |
| `pixel` | finds the art's own pixel grid (sprites are often shared already scaled by 2, 3 or 4) and shrinks back to it; pads to a square with the background colour; scales up smoothly (bicubic) from that grid to the model's input size |
| `cartoon` | pads to a square instead of cropping |
| `auto` | `pixel` for images with few colours and long runs of identical pixels, `cartoon` for flat-colour images, nothing for photos |

The text side gets style prompts as well: "pixel art of a {}", "a 16-bit
sprite of a {}", "a cartoon of a {}" and similar, averaged with the photo
templates. `collect` records the style in `run.json`, and `extract` and
`evaluate` reuse it unless told otherwise. `clip_bench.py eval --style`
measures whether it helps on your images. Compare `off` and `auto` on the
same set before you rely on it.

What it did on one test (CLIP ViT-B/32, zero-shot, 1,000 CIFAR-10 test
images, and the same images as 16-colour pixel art shipped at 4×):

| | photos (32 px) | pixel art (128 px) |
|---|---|---|
| off | 0.908 | 0.495 |
| style prompts only | 0.907 | 0.519 |
| recovered grid, nearest neighbour | | 0.507 |
| recovered grid, bicubic | 0.908 | 0.727 |
| `auto` (grid, bicubic and prompts) | 0.907 | **0.750** |

Recovering the grid matters, and so does enlarging it *smoothly*. Hard
edges from nearest neighbour gained nothing. These are pixelated photos
rather than drawn sprites, so check on your own art.

```bash
python scripts/clip_style.py inspect sprite.png --out sprite-224.png   # what auto sees, and the result
python scripts/clip_neurons.py collect --model openai/clip-vit-base-patch32 --images data/sprites --style auto --out runs/clip-sprites
```

## Benchmark it

For standard numbers (ImageNetV2, ImageNet-Sketch, VTAB and the rest of CLIP_benchmark) at each suppression scale, see [BENCHMARKS.md](BENCHMARKS.md#clip_benchmark-imagenetv2-imagenet-sketch-vtab-).

## 5. Ship it

**Hugging Face checkpoint** (for `transformers`, OpenCLIP-style pipelines, or
conversion):

```bash
python scripts/clip_neurons.py export --model openai/clip-vit-large-patch14 \
    --h_neurons models/clip/h_neurons.json --scale 0.5 --out models/clip-supp050
```

**llama.cpp / LM Studio vision projector.** Vision-language models in GGUF
ship their image encoder as `mmproj-*.gguf`. If that encoder is the CLIP or
SigLIP you profiled, edit it in place:

```bash
python scripts/suppress_mmproj.py --mmproj mmproj-model-f16.gguf \
    --h_neurons models/clip/h_neurons.json --scale 0.5 \
    --out mmproj-model-f16-supp050.gguf
```

The down projection is located by shape, not name, because older converters
stored fc1 as `ffn_down` and fc2 as `ffn_up`. F16 and F32 projectors are
edited exactly. Quantized projectors are refused, because a column edit would
cross quantization blocks. The script verifies one edited and one untouched
column and writes a `.suppression.json` sidecar.

Profile the *same* encoder weights you edit. A VLM's projector is often
fine-tuned from the public CLIP, so neuron indices from the public checkpoint
may not line up. Export the VLM's own vision tower to HF format and profile
that when in doubt.

Studio pairs `mmproj` files with their model automatically, so dropping the
edited projector next to the model and reloading is enough.
