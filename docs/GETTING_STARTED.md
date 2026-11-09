# Getting started

New to the vocabulary? [GLOSSARY.md](GLOSSARY.md) explains GGUF, CETT,
H-Neurons, abstention, MCP and the rest, and why each matters here.

OmniCortex is a set of Python tools plus two small llama.cpp additions. Most
of it runs anywhere Python 3.10+ runs; the parts that touch model internals
want either PyTorch (any backend) or a llama.cpp build.

| Component | Linux | macOS | Windows | Needs |
|---|---|---|---|---|
| Studio (model manager, chat, `/v1` API) | yes | yes | yes | Python stdlib, a `llama-server` binary |
| Transfer (WebRTC GUI, resumable CLI, MCP) | yes | yes | yes | stdlib; `aiohttp` for the GUI |
| TestQA, subject classifier | yes | yes | yes¹ | `requests` |
| CLIP / SigLIP H-Neurons | yes | yes | yes | `torch`, `transformers`, `pillow` |
| Text H-Neurons, PyTorch path | yes | yes | yes | `torch`, `transformers` |
| Text H-Neurons, GGUF path (`cett-dump`) | yes | yes | build with CMake | llama.cpp checkout |

¹ The code interpreter's CPU/memory limits use POSIX `resource`; on Windows
only the wall-clock timeout applies.

## The short way

Get OmniCortex (clone it, or *Download ZIP* and unpack it), then
double-click the installer for your system:

| system | double-click | if it will not start |
|---|---|---|
| Windows | `Install OmniCortex.bat` | it offers to install Python with winget, or opens python.org |
| macOS | `Install OmniCortex.command` | right-click → Open the first time (it is a downloaded script); needs Python 3.10+ from python.org or Homebrew, since the built-in one is 3.9 |
| Linux | `install-gui.sh` | it prints the package to install if Python's venv module is missing |

The installer opens in your browser:
1. **This computer**: the system, Python, GPUs and free space, and anything
   peculiar about them. It checks for:
   - the Microsoft Store's Python;
   - long Windows paths;
   - an Intel Mac, which has no current PyTorch;
   - an AMD GPU on Windows, which has no ROCm PyTorch;
   - membership of the `video` and `render` groups for AMD on Linux;
   - an RDNA4 card, which needs ROCm 7 and no `HSA_OVERRIDE_GFX_VERSION`;
   - older NVIDIA cards, which need the CUDA 12.6 wheels.
2. **Choices**: the PyTorch build recommended for this computer, or NVIDIA,
   AMD, CPU only, or none (Studio, TestQA, Projects and the llama.cpp path
   don't need it); the extras; a shortcut (Start menu, Applications folder,
   app menu); and where to install.
3. **Installing**: each step with its output. A failed step says what to
   try, and *Install* again resumes quickly.
4. **Done**: *Open OmniCortex* starts Studio on its Setup page, which
   builds llama.cpp (or uses one you have) and finds your models.

The first time you open Studio it offers a two-minute tour, which is
optional. It points at each part of Studio in turn. The **?** button at the
bottom right of every page brings it back, jumps to the current page's
stop, or opens these help pages inside Studio.

Terminal instead:

```bash
python3 scripts/installer.py --yes            # the same, with the recommended choices
python3 scripts/installer.py --check          # what it found and would do; installs nothing
python3 scripts/installer.py --yes --torch none
./install.sh                                  # the original Linux/macOS script, still supported
```

Then start it from the shortcut, `OmniCortex.bat` or `./omnicortex`.
Everything after that is in the browser; see [UI_AND_CLI.md](UI_AND_CLI.md)
for where each feature lives, and the steps below for doing the same from
a terminal.

## 1. Install

```bash
git clone <this repo> omnicortex && cd omnicortex
./install.sh              # creates venv/, picks a PyTorch build for your GPU, smoke-tests it
source venv/bin/activate
```

`install.sh` detects AMD (ROCm), NVIDIA (CUDA) or neither (CPU) and installs a
matching PyTorch wheel. PyTorch is only needed for the PyTorch extraction path,
CLIP/SigLIP and fine-tuning; for Studio, TestQA, Projects, transfer and the
llama.cpp path, `./install.sh --no-torch` is a much smaller install. On NVIDIA it checks every GPU's architecture: older
cards (Maxwell, Pascal, Volta, e.g. a Tesla M10) need the CUDA 12.6 wheels and
`torch<2.15`, which `python scripts/cuda_info.py` reports and the installer
picks. On macOS or Windows, or if you prefer to manage it yourself:

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install torch                                    # pick the right build from pytorch.org
pip install -r requirements.txt
pip install -r requirements-transfer-full.txt        # optional: WebRTC GUI, MCP, keyring
```

Then check what you have:

```bash
python scripts/doctor.py
```

It checks, in order, Python and packages, your GPUs and the devices Studio's
worker models can use, your llama.cpp build, the model in `OC_GGUF`, and which
pipeline stages already have output. Each problem comes with what it blocks
and the fix, and it ends with what to do next: one line for using models in
Studio, one for the H-Neuron pipeline.

## 2. Build llama.cpp (for serving and GGUF extraction)

```bash
scripts/build_llama_tools.sh --backend vulkan     # or cpu, cuda, metal, hip; --dir to choose the checkout
```

It clones llama.cpp, adds the `cett-dump` extractor, and builds
`llama-cett-dump`, `llama-server`, `llama-quantize` and `llama-eval-callback`
(about 5 minutes on a laptop CPU), then prints the lines to put in
`env.local.sh`. A warning that llama-server's built-in web UI could not be
downloaded is harmless: Studio is the UI.
Vulkan works on AMD, Intel and NVIDIA and is the most portable choice. With
`--backend cuda` the CUDA architectures come from your GPUs (`--cuda-arch` to
build for others), and pre-Turing cards need a CUDA 12.x toolkit, or
`docker/llama-cuda.Dockerfile`; see [HARDWARE.md](HARDWARE.md#nvidia--cuda).

`cett-dump` is tested against current llama.cpp: on a model converted with
llama.cpp's own converter its CETT values match the PyTorch hook path on every
layer (`tests/test_llamacpp_integration.py`, run with `OC_LLAMA` set).

## 3. Get a model and point OmniCortex at it

No GGUF yet? Start Studio (step 4) and use its **Models** tab to search Hugging
Face and download one; a 7-8B model at Q4_K_M (about 5 GB) is a good first
choice. Studio, TestQA and Projects need nothing more.

For the H-Neuron pipeline, tell the scripts which model to study:

```bash
export OC_GGUF=~/models/Qwen3-8B-Q6_K.gguf
export OC_LLAMA=~/llama.cpp
source env.sh            # derives OC_CETT, OC_LLAMA_SERVER, OC_LAYERS; checks the file
```

Put those exports in `env.local.sh` (gitignored) to make them stick. Use a
Q8_0 or F16 file here if you can: quantization rounds the very activation
sizes the pipeline measures (see [GLOSSARY.md](GLOSSARY.md#models-and-files)).

## 4. Pick a starting point

**Run models like LM Studio, with the OmniCortex dials:**

```bash
python viz/studio.py --models-dir ~/models --server "$OC_LLAMA_SERVER"
```

Open http://127.0.0.1:7870. Any OpenAI client can use http://127.0.0.1:7870/v1.
Studio prints the ids of the models it found; that id is what goes in an API
request's `"model"` and in `MODEL_ID` below. See [STUDIO.md](STUDIO.md).

**Benchmark one or more endpoints:**

```bash
MODEL_ID=acme/model-q4_k_m        # one of the ids Studio printed
python scripts/testqa.py --endpoint "mine=http://127.0.0.1:7870/v1@$MODEL_ID" --per-subject 5   # a quick first pass
python scripts/testqa.py --endpoint "mine=http://127.0.0.1:7870/v1@$MODEL_ID" --allow-exec --sandbox docker
```

`--allow-exec` grades code questions by running the model's code; with
`--sandbox docker` (or `podman`) that happens in a locked-down container,
without it only under local CPU and memory limits. See [TESTQA.md](TESTQA.md).

**Find and suppress H-Neurons in a text model:** [PIPELINE.md](PIPELINE.md).

**Do the same for CLIP or SigLIP:** [VISION.md](VISION.md).

**Move models between machines:** [TRANSFER.md](TRANSFER.md), and read
[SECURITY.md](SECURITY.md) before exposing anything beyond localhost.

## The first real test for the GGUF path

```bash
printf 'The capital of France is' > /tmp/seq.txt
"$OC_CETT" -m "$OC_GGUF" -ngl 99 -b 4096 --n-layers "$OC_LAYERS" \
    --prompt-file /tmp/seq.txt --out /tmp/d.bin
```

It should end with `cett-dump: finished, 1 ok, 0 failed`. To see what it
captured:

```bash
python -c "import sys; sys.path.insert(0, 'scripts')
from extract_activations_gguf import read_aggregate
spans, agg = read_aggregate('/tmp/d.bin')[:2]
print('spans x layers x neurons:', agg.shape)"
```

The middle number must equal `$OC_LAYERS`: one averaged row of neuron
activations per decoder layer. If the run fails with "capture failed (0
records)", the architecture names its FFN output node differently. List the
node names:

```bash
"$OC_LLAMA"/build/bin/llama-eval-callback -m "$OC_GGUF" -p "hi" -n 1 2>&1 \
    | grep -oE '^[a-z_]+-[0-9]+' | sort -u | head -30
```

then add the prefix to `FFN_DOWN_PREFIXES` in
`llama-tools/cett-dump/cett-dump.cpp` and rebuild.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The suite needs no model and no GPU. It covers the security policy, both
transfer paths (including TLS), the signaling server, the CLIP pipeline on a
tiny random CLIP, TestQA against fake endpoints, and Studio's API against a
fake `llama-server`. The C++ tools have their own build notes in
`llama-tools/*/`.
