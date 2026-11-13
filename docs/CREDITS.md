# Credits

OmniCortex is built on other people's research and software. This page
lists every paper, method, dataset and project it uses, and where each one
is used. When a feature adapts someone's work, it gets a line here and a
mention in the code or doc that uses it.

Something missing or credited wrongly? Please open an issue.

## Where it started

| Work | Used for |
|---|---|
| **H-Neurons**: Gao et al. (2025), *H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs*, [arXiv:2512.01797](https://arxiv.org/abs/2512.01797); code at [thunlp/H-Neurons](https://github.com/thunlp/H-Neurons) | The method OmniCortex grew from: CETT, sparse classifiers over MLP neurons, and suppressing the neurons tied to over-confident answers. What changed from upstream is listed in [RESEARCH_NOTES.md](RESEARCH_NOTES.md). |

## Reading what a model computes

| Work | Used for |
|---|---|
| Rudrendu Paul & Sourav Nandy, [RudrenduPaul/NeuronScope](https://github.com/RudrenduPaul/NeuronScope) (MIT) | The `trace`, `activations`, `patch` and `circuit` operations and their versioned JSON in `interventions.py`; and the reason this project is now called OmniCortex |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [nnsight](https://nnsight.net/) | The hook-point vocabulary (`resid_pre`, `attn_out`, `mlp_out`, `mlp_post`) and the activation- and path-patching workflows `interventions.py` follows |
| Wang et al. (2022), *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*, [arXiv:2211.00593](https://arxiv.org/abs/2211.00593) | Path patching and the clean/corrupted prompt design of `act-patch` and `path-patch` |
| Joseph Bloom et al., [SAELens](https://github.com/decoderesearch/SAELens) (MIT) | Its dictionary format, and its catalogue of pretrained releases that `features.py import --release` downloads from |
| [Neuronpedia](https://www.neuronpedia.org) | Feature descriptions (`interventions.py neuronpedia`) |
| nostalgebraist (2020), [*interpreting GPT: the logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | `logit_lens.py`; the per-layer logits SLED compares |
| Belrose et al. (2023), *Eliciting Latent Predictions from Transformers with the Tuned Lens*, [arXiv:2303.08112](https://arxiv.org/abs/2303.08112) | Named in `logit_lens.py` and [CONFIDENCE.md](CONFIDENCE.md) as the more accurate lens for early layers |
| Xiao et al. (2024), *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM), ICLR, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | `when.py` ignores the first position (the attention sink); the engine's "tokens to keep" with context shift |
| Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*, [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html) | Direct logit attribution from weights alone (`neuron_attrib.py weights`) |
| Sundararajan, Taly & Yan (2017), *Axiomatic Attribution for Deep Networks*, ICML, [arXiv:1703.01365](https://arxiv.org/abs/1703.01365) | Integrated gradients over neuron activations (`neuron_attrib.py saliency --method integrated`) |
| Nanda (2023), *Attribution Patching: Activation Patching At Industrial Scale*, [neelnanda.io](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) | Gradient × activation as a first-order estimate of patching each neuron (`neuron_attrib.py saliency`) |

## Features, circuits and emotion concepts

What [FEATURES.md](FEATURES.md) implements, and where it departs from these works, is described there.

| Work | Used for |
|---|---|
| Elhage et al. (2022), *Toy Models of Superposition*, [arXiv:2209.10652](https://arxiv.org/abs/2209.10652) | Why neurons mix concepts and dictionaries are needed |
| Bricken et al. (2023), [*Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*](https://transformer-circuits.pub/2023/monosemantic-features/index.html), Transformer Circuits Thread; Cunningham et al. (2023), *Sparse Autoencoders Find Highly Interpretable Features in Language Models*, [arXiv:2309.08600](https://arxiv.org/abs/2309.08600) | Dictionary learning with sparse autoencoders: `sae.py`, `features.py` |
| Templeton, Conerly et al. (2024), [*Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), Transformer Circuits Thread | Residual-stream dictionaries; examples across activation buckets; the 0-3 specificity rubric for labels; logit effects; neighbours by decoder cosine; the feature-neuron correlation comparison; clamping features (error term kept, units of the maximum activation); attribution and ablation of a completion (`feature_circuits.py attribute`) |
| Conerly, Templeton, Bricken, Marcus & Henighan (2024), [*Update on how we train SAEs*](https://transformer-circuits.pub/2024/april-update/index.html), Circuits Updates, April 2024 | The `relu` architecture's training: decoder-norm-weighted L1, √d input scaling, initialisation, schedules, folding decoder norms afterwards |
| Gao et al. (2024), *Scaling and evaluating sparse autoencoders*, [arXiv:2406.04093](https://arxiv.org/abs/2406.04093) | The `topk` architecture and its auxiliary loss for dead features |
| Rajamanoharan et al. (2024), *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*, [arXiv:2407.14435](https://arxiv.org/abs/2407.14435); Lieberum et al. (2024), *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*, [arXiv:2408.05147](https://arxiv.org/abs/2408.05147) | The `jumprelu` architecture; importing Gemma Scope's `params.npz` |
| [SAELens](https://github.com/jbloomAus/SAELens) (Bloom et al.) | The folder format `features.py import` reads |
| Dunefsky, Chlenski & Nanda (2024), *Transcoders Find Interpretable LLM Feature Circuits*, [arXiv:2406.11944](https://arxiv.org/abs/2406.11944) | Transcoders (`--site mlp_in --target-site mlp_out`) |
| Ameisen et al. (2025), [*Circuit Tracing: Revealing Computational Graphs in Language Models*](https://transformer-circuits.pub/2025/attribution-graphs/methods.html); Lindsey et al. (2025), [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), Transformer Circuits Thread | Attribution graphs: the local replacement model with error nodes, frozen attention patterns and norm denominators, edge weights, logit targets, influence-based pruning, replacement and completeness scores, interventions (`feature_circuits.py graph`, `intervene`) |
| [circuit-tracer](https://github.com/safety-research/circuit-tracer) (Hanna, Piotrowski, Lindsey & Ameisen, 2025) | The open-source implementation of those graphs; read to check ours against (its pruning and influence computation) |
| Marks et al. (2024), *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models*, [arXiv:2403.19647](https://arxiv.org/abs/2403.19647); Nanda (2023), *Attribution Patching: Activation Patching At Industrial Scale* | Linear attribution with dictionary error terms; Scaling Monosemanticity's attribution is attribution patching with a zero baseline |
| Bills et al. (2023), *Language models can explain neurons in language models*, OpenAI; Paulo et al. (2024), *Automatically Interpreting Millions of Features in Large Language Models*, [arXiv:2410.13928](https://arxiv.org/abs/2410.13928) | Labels written by a model, and the detection score (`interpret --score detection`) |
| Sofroniew, Kauvar, Saunders, Chen et al. (2026), [*Emotion Concepts and their Function in a Large Language Model*](https://transformer-circuits.pub/2026/emotions/index.html), Transformer Circuits Thread | `emotions.py`: the 171 emotion words, the story and neutral-dialogue recipe, vectors from story means from the 50th token minus the mean over emotions, denoising by the neutral components explaining 50% of their variance, the two-thirds-depth layer, z-scored traces, corpus scans, the dose sweep, the valence-arousal map, steering in fractions of the residual norm |
| Zou et al. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency*, [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) | Earlier linear emotion representations and steering, cited by the emotions paper |
| Gao et al. (2020), *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*, [arXiv:2101.00027](https://arxiv.org/abs/2101.00027), through [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) | The example corpus in the docs and the Features page |

## Editing and steering

| Work | Used for |
|---|---|
| Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) | `concept_edit.py`: a difference-of-means direction, removed from the weights (abliteration) |
| Wollschläger et al. (2025), *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence*, ICML, [arXiv:2502.17420](https://arxiv.org/abs/2502.17420) | The idea of a multi-dimensional concept cone. `concept_edit.py --method pca` builds a simpler cone, from PCA |
| Jim Lai (grimjim) (2025), *projected abliteration* and *norm-preserving biprojected abliteration*, Hugging Face community blog | Biprojection and norm preservation in `concept_edit.py` |
| [Heretic](https://github.com/p-e-w/heretic) (Philipp Emanuel Weidmann) | `concept_edit.py --search` follows its parameterization of ablation strength per layer |
| Turner et al. (2023), *Steering Language Models With Activation Engineering* (ActAdd), [arXiv:2308.10248](https://arxiv.org/abs/2308.10248); Panickssery (Rimsky) et al. (2024), *Steering Llama 2 via Contrastive Activation Addition*, ACL, [arXiv:2312.06681](https://arxiv.org/abs/2312.06681) | Adding a direction to the residual stream: `decode_lab.py`'s `steer` variant, control-vector exports |
| Nonta & Rambhatla (2026), *Refusal-Orthogonal Gated Editing for Safer Localized LLM Adaptation* (SAGE), ICLR 2026 TTU workshop | `sage_edit.py` |
| Louizos, Welling & Kingma (2018), *Learning Sparse Neural Networks through L0 Regularization*, ICLR, [arXiv:1712.01312](https://arxiv.org/abs/1712.01312) | The Hard Concrete gates in `sage_edit.py` |
| Johnson & Lindenstrauss (1984), *Extensions of Lipschitz mappings into a Hilbert space*, Contemporary Mathematics 26 | Why a random low-dimensional projection keeps a spread-out signal: `rp_steer.py` |
| Friedman (1989), *Regularized Discriminant Analysis*, JASA 84(405) | The shrinkage LDA in `rp_steer.py` |

## Decoding and serving

| Work | Used for |
|---|---|
| Zhang et al. (2024), *SLED: Self Logits Evolution Decoding for Improving Factuality in Large Language Models*, NeurIPS, [arXiv:2411.02433](https://arxiv.org/abs/2411.02433) | `decode_lab.py`'s `sled` variant ([DECODING.md](DECODING.md)) |
| Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding*, ICML, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | Engine options for speculative decoding with a draft model |
| Li et al. (2025), *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) | The `draft-eagle3` engine option, where llama.cpp supports it |
| Saxena (2023), [prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) | The n-gram speculative decoding options |
| Ma et al. (2024), *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (BitNet b1.58), [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) | The ternary TQ1_0 / TQ2_0 types `quant_select.py` can try |

## Training, merging and data

| Work | Used for |
|---|---|
| Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | LoRA exports of an edit; `finetune.py` |
| Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | `finetune.py --method qlora` |
| Rafailov et al. (2023), *Direct Preference Optimization*, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) | `deficits.py` writes `dpo.jsonl`; `finetune.py --dpo` |
| Gekhman et al. (2024), *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*, EMNLP, [arXiv:2405.05904](https://arxiv.org/abs/2405.05904) | Why `deficits.py` teaches a decline, not an answer, for facts the model does not hold ([RETRAINING.md](RETRAINING.md)) |
| Goddard et al. (2024), *Arcee's MergeKit*, [arXiv:2403.13257](https://arxiv.org/abs/2403.13257); [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) | `mergekit_omnicortex.py` registers a merge method in it |
| Ilharco et al. (2023), *Editing Models with Task Arithmetic*, [arXiv:2212.04089](https://arxiv.org/abs/2212.04089); Yadav et al. (2023), *TIES-Merging*, [arXiv:2306.01708](https://arxiv.org/abs/2306.01708); Yu et al. (2024), *Language Models are Super Mario* (DARE), [arXiv:2311.03099](https://arxiv.org/abs/2311.03099) | The merge methods OmniCortex's neuron-selected merge is compared with ([MERGING.md](MERGING.md)); mergekit implements them |

## Retrieval, search and statistics

| Work | Used for |
|---|---|
| Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond* | `rag.py`, `context_compress.py` |
| Cormack, Clarke & Büttcher (2009), *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR | `rag.py` combines BM25 and embeddings |
| Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization* (TPE), NeurIPS; Akiba et al. (2019), *Optuna*, KDD, [arXiv:1907.10902](https://arxiv.org/abs/1907.10902) | `engine_optimizer.py`'s search |
| Wilson (1927), *Probable inference, the law of succession, and statistical inference*, JASA 22(158) | Confidence intervals on right / wrong / declined rates (TestQA, model stats) |
| McNemar (1947), *Note on the sampling error of the difference between correlated proportions or percentages*, Psychometrika 12(2) | Paired comparisons: the retrain gate, model comparisons, merges, expert sweeps |
| Efron (1979), *Bootstrap Methods: Another Look at the Jackknife*, Annals of Statistics 7(1) | Bootstrap intervals in `cv.py` |

## Vision

| Work | Used for |
|---|---|
| Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | `clip_neurons.py` |
| Zhai et al. (2023), *Sigmoid Loss for Language Image Pre-Training* (SigLIP), [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) | `clip_neurons.py` |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | `clip_bench.py` presents a transformers model through OpenCLIP's interface, which CLIP_benchmark expects |
| [LAION-AI/CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) | Zero-shot image benchmarks ([VISION.md](VISION.md)) |

## Pose

| Work | Used for |
|---|---|
| Bazarevsky et al. (2020), *BlazePose: On-device Real-time Body Pose tracking*, [arXiv:2006.10204](https://arxiv.org/abs/2006.10204), and Google's [MediaPipe pose landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) | `plugins/pose/extract.py --video`: the 33-landmark model and its world coordinates |
| Sandler et al. (2018), *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, [arXiv:1801.04381](https://arxiv.org/abs/1801.04381) | The backbone family behind the on-device pose models |
| Cao et al., [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (BODY_25) | `extract.py --openpose`: its JSON output imported |
| Shin et al. (2024), *WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion*, and Xu et al. (2020), *GHUM & GHUML* | Pose sources whose full rotations can be passed through (`--frames-jsonl`) |
| [toyxyz](https://github.com/toyxyz)'s RGB mocap workflows | The idea of driving engines from video-derived pose data |
| The [VMC protocol](https://protocol.vmc.info/) | `stream.py --vmc`: how engines and avatar tools receive the poses |

## Looped models and NVIDIA's training stack

| Work | Used for |
|---|---|
| NVIDIA, [Nemotron-Labs Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) (OpenMDW-1.1), and its llama.cpp support (ggml-org/llama.cpp#30306, the fork its card pins) | Looped-model serving, `loop_sweep.py`, `loop_lens.py` ([DORY.md](DORY.md)) |
| Azaria & Mitchell (2023), Marks & Tegmark (2023), Burns et al. (2022), Zou et al. (2023): truth directions and probes in model activations | `truth_probe.py` ([TRUTH_PROBE.md](TRUTH_PROBE.md)) |
| evilJazz, the MI50 32 GB VBIOS comparison (a GitHub gist, with the gfx906 Discord community and TechPowerUp's VBIOS collection) | the VBIOS table in `amd_vbios.py` ([PROXMOX.md](PROXMOX.md)) |
| The mi50-arm and m10-arm guides (MI50 and Tesla M10 on RK3588) | the container route, coherency and BAR checks, and the ARM hardware profiles ([PROXMOX.md](PROXMOX.md)) |
| [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) (Apache-2.0) | The resources-server interface `integrations/nemo_gym/omnicortex_testqa` implements |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) (Apache-2.0) | GRPO training on that environment |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Background: how Dory was pre-trained |
| [orailnoor/cross-platform-llm-client](https://github.com/orailnoor/cross-platform-llm-client) (MIT) | The starting point for the phone client plan ([MOBILE_CLIENTS.md](MOBILE_CLIENTS.md)) |

## Datasets and benchmarks

| Work | Used for |
|---|---|
| Joshi et al. (2017), *TriviaQA*, ACL, [arXiv:1705.03551](https://arxiv.org/abs/1705.03551) | The default labels for text H-Neurons |
| Jimenez et al. (2024), *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, ICLR, [arXiv:2310.06770](https://arxiv.org/abs/2310.06770); [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) | `benchmarks.py swebench`, SWE-bench retraining data |
| White et al. (2025), *LiveBench*, ICLR, [arXiv:2406.19314](https://arxiv.org/abs/2406.19314); [LiveBench/LiveBench](https://github.com/LiveBench/LiveBench) | `benchmarks.py livebench` |
| Deng et al. (2009), *ImageNet*, CVPR; Recht et al. (2019), *Do ImageNet Classifiers Generalize to ImageNet?* ([ImageNetV2](https://github.com/modestyachts/ImageNetV2)), ICML; Wang et al. (2019), *Learning Robust Global Representations by Penalizing Local Predictive Power* ([ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)), NeurIPS; Hendrycks et al. (2021), *Natural Adversarial Examples* (ImageNet-A), CVPR, and *The Many Faces of Robustness* (ImageNet-R), ICCV | Vision benchmarks through CLIP_benchmark |
| Zhai et al. (2019), *A Large-scale Study of Representation Learning with the Visual Task Adaptation Benchmark* (VTAB), [google-research/task_adaptation](https://github.com/google-research/task_adaptation) | Vision benchmarks through CLIP_benchmark |
| Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10/100) | Vision benchmarks through CLIP_benchmark |
| [BenchLM](https://benchlm.ai/benchmarks) and other published leaderboards | `benchmarks.py leaderboard import`. Imported scores keep their benchmark's name (HumanEval, MMLU, GSM8K, SimpleQA…) only to sort them by subject; OmniCortex does not use those datasets itself |

## Workflows and graphs

[GRAPHS.md](GRAPHS.md) takes these ideas; none of their code is included.

| Project | What OmniCortex took |
|---|---|
| [PySpur](https://github.com/PySpur-Dev/PySpur) (PySpur-Dev) | Human-in-the-loop breakpoints that pause a workflow until a person approves; node-level debugging; test cases and evals over a workflow; execution traces; loops; JSON-schema structured outputs; adding a node as one file |
| [comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party) (heshengtao) | An MCP tool node; the reasoning of thinking models split from the answer; workflows saved as files and reused as steps |
| [ComfyUI-LLMs-Toolkit](https://github.com/ComfyUI-Kelin/ComfyUI-LLMs-Toolkit) (ComfyUI-Kelin) | Nodes that report an error instead of breaking the workflow; JSON build, extract and combine nodes; a translator node |
| [ComfyUI-LLMs](https://github.com/leoleelxh/ComfyUI-LLMs) (leoleelxh) | One model node for any OpenAI-compatible endpoint, with the model switchable per node |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | The node-graph way of wiring models together that all of these build on |

## Software it builds on

| Project | Used for |
|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | Serving (`llama-server`), quantizing (`llama-quantize`, `llama-imatrix`), control vectors, GGUF and `gguf-py`, converters. `llama-cett-dump` is OmniCortex's addition built on it ([ENGINE.md](ENGINE.md)) |
| [PyTorch](https://pytorch.org) | Hooks, extraction, training, decoding variants |
| Hugging Face [transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets), [huggingface_hub](https://github.com/huggingface/huggingface_hub), [safetensors](https://github.com/huggingface/safetensors), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [accelerate](https://github.com/huggingface/accelerate) | Loading models and data, LoRA, SFT and DPO training; the Hub's parquet export when `datasets` is missing |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | QLoRA |
| [scikit-learn](https://scikit-learn.org), [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org), [Apache Arrow](https://arrow.apache.org) | Classifiers, arrays, tables, parquet |
| [Optuna](https://optuna.org) | Engine optimization |
| [three.js](https://threejs.org) and [Godot Engine](https://godotengine.org) | The 3D views and the Godot client |
| [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm) | Serving stacks OmniCortex connects to |
| [Model Context Protocol](https://modelcontextprotocol.io) and [FastMCP](https://github.com/jlowin/fastmcp); [Cline](https://github.com/cline/cline) | `oc_mcp.py` and the MCP client; the Cline configuration |
| [Tailscale](https://tailscale.com), [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) | Reaching hosts and Studio from elsewhere ([REMOTE.md](REMOTE.md), [SECURITY.md](SECURITY.md)) |
| WebRTC | Peer-to-peer model transfer ([TRANSFER.md](TRANSFER.md)) |
| [Home Assistant](https://www.home-assistant.io) REST API | Power rules on the System page ([SYSTEM.md](SYSTEM.md)) |
| [psutil](https://github.com/giampaolo/psutil) | System monitoring, where installed |

Each project keeps its own licence. Using its models, datasets or outputs
is subject to that licence, and the same goes for the model weights you
load into OmniCortex.
