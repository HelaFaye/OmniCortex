# Building cett-dump with Vulkan

Vulkan is the point of this tool: it removes ROCm and PyTorch from the
extraction path entirely, and a Q6_K 9B plus the eval callback fits in 16GB
where bf16 in PyTorch does not.

```bash
git clone https://github.com/ggml-org/llama.cpp
cp -r /path/to/omnicortex/llama-tools/cett-dump llama.cpp/tools/
echo 'add_subdirectory(cett-dump)' >> llama.cpp/tools/CMakeLists.txt

cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target llama-cett-dump
```

On Linux you need `vulkan-headers`, `shaderc` and your driver's Vulkan
ICD (`vulkan-radeon` for AMD). Confirm the GPU is visible with `vulkaninfo
--summary` before building.

Smoke test against any small GGUF:

```bash
printf 'The capital of France is' > /tmp/seq.txt
./build/bin/llama-cett-dump -m tiny.gguf -ngl 99 -b 4096 \
    --prompt-file /tmp/seq.txt --out /tmp/dump.bin
```

Or just run `scripts/build_llama_tools.sh --backend vulkan`.

Every token is flagged as an output so llama.cpp computes the final layer for
all of them; that costs an n_vocab x n_tokens logits buffer (about 1 GB for
2k tokens at a 150k vocabulary). `--last-layer-outputs-only` trades the last
layer's activations for that memory.

Expect one record per decoder layer. If you get zero, the node naming differs
in your llama.cpp version -- run `llama-eval-callback` on the same model and
grep its output for the ffn_down node names, then adjust
`parse_ffn_down_layer`.

## Build notes, verified against llama.cpp 0.4.0-dev

Three things drifted between writing this and building it, all fixed in the
files here:

- `common_params_parse` moved to `arg.h`
- `common_init_from_params` returns `common_init_result_ptr` (a `unique_ptr`),
  and `model`/`context` are accessor methods, not fields
- the common library target is `llama-common`, not `common`; CMakeLists now
  detects which exists

`<cmath>` was also missing, which only worked elsewhere by transitive include.
The same drifts will apply to `llama-tools/server-activations/` when you build
that.

## Known limits

- **MoE is supported** via the `ffn_moe_down-<il>` node. `ggml_mul_mat_id`
  carries the expert ids in `src[2]`, so each (token, slot) knows which expert
  produced it and the index becomes (layer, expert, neuron). Means are divided
  by each expert's own routed-token count, not the span length.
- **One batch.** The sequence must fit in `-b`, so every ffn_down node fires
  exactly once and each layer yields one record. Raise `-b` past your longest
  sequence rather than letting it split.
- **The API moves.** This is written against a recent llama.cpp and modelled on
  `examples/eval-callback` to keep divergence small, but names change. If it
  fails to compile, diff against that example first.
