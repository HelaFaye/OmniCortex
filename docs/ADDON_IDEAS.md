# Add-on ideas not built yet

This lists every idea from the add-on brainstorms that is not built yet. Each entry says what OmniCortex
already has to build on, the realistic version, and what stands in the way. What *is* built is in
[INTEGRATED_OPTIONS.md](INTEGRATED_OPTIONS.md) and [ADDONS.md](ADDONS.md). The pose pipeline is in
[POSE.md](POSE.md).

Effort: **S** is a few days, **M** about two weeks, **L** a month or more, **R** a research problem with no
known working method.

## Steering and creative control

**Style and genre knobs (S–M).**
- **Builds on:** SAE dictionaries, SAELens releases, `features.py`, emotion-vector steering, and the decoding
  lab's steering.
- **Realistic version:** a page of sliders, one per labelled SAE feature or steering vector. Each slider adds
  scale × direction to the residual stream at generation time (PyTorch path), or exports a llama.cpp control
  vector (GGUF path).
- **In the way:** "Cyberpunk noir" is only a slider if some dictionary has a clean feature for it. Most style
  features are entangled, so expect to curate and test them per model.

**Tension and emotion valve: the jewel as an input (M).**
- **Builds on:** the emotion jewel (web and Godot), emotion vectors, and steering.
- **Realistic version:** drag a vertex of the jewel. Each vertex is an emotion vector, and its distance from
  the centre becomes that vector's steering scale for the next tokens.
- **In the way:** a llama-server control vector is fixed at load time (`--control-vector-scaled`), so live
  dragging needs the PyTorch path or a server patch that rescales per request.

**Deterministic seed rigging for structured output (S).**
- **Builds on:** grammars and JSON schemas in presets, and temperature 0.
- **Realistic version:** lock the structure with a grammar generated from a template (indentation, keys,
  call shapes). Sampling then only varies the open slots.
- **In the way:** pinning "spatial-maths neurons" is not needed for this. Grammar-constrained decoding is
  the reliable route.

**Synthetic critic: amplify, then filter (M).**
- **Builds on:** the LoRA scale slider (negative scales amplify), graphs, TestQA's code sandbox, and a judge
  model.
- **Realistic version:** a graph that samples at an amplified H-neuron scale and runs code in the sandbox. A
  judge keeps what passes; the novel-but-working outputs go to a review list.
- **In the way:** amplification mostly produces confident nonsense, so the yield of useful novelty needs
  measuring before it is worth a button.

**Lore-consistency monitor (M).**
- **Builds on:** SAE features, Neural Reset's stream hook (`_proxy`), the claim spotter, and RAG.
- **Realistic version:** map each lore rule to SAE features plus keywords. While a reply streams, check the
  features (PyTorch path) or the text (any engine). On a breach, cut and resume the way Neural Reset does,
  with the rule restated.
- **In the way:** the text check works on any engine; the feature check needs a model whose activations
  Studio can read per token (`--server-activations`).

**Neural debate node for ComfyUI (M).**
- **Builds on:** the ComfyUI add-on, graphs, and two models.
- **Realistic version:** two models draft and critique a diffusion prompt in a few rounds, and a third merges
  their drafts. Agreement is measured on the text.
- **In the way:** "suppressing conflicting tokens via H-neurons" across two different models has no defined
  meaning, so it is a text-level debate.

**Sync-pulse: temporal consistency for frame prompts (S).**
- **Builds on:** graphs and presets.
- **Realistic version:** keep the fixed parts (characters, setting, lighting) in a structured state. Each
  frame's prompt is that state plus the action, enforced by a template or grammar, and drift is checked by
  diffing entities across frames.
- **In the way:** clamping neuron coordinates across steps is not needed; the state object does the job.

## Merging and model structure

**Neural chimera, per-sentence routing between two fine-tunes (M).**
- **Builds on:** `merge_export.py`, neuron-granular merges (MERGING.md), morphing's router, and per-subject
  routing.
- **Realistic version:** route each request, or each paragraph, to whichever fine-tune scores better on
  its kind of content, with the router learned from TestQA results per subject.
- **In the way:** switching sub-layer paths sentence by sentence inside one model needs both models loaded,
  plus a custom forward pass.

**3D latent sculpting (R).**
- **Builds on:** the Godot client, the neuron review's 3D view, and concept translation.
- **In the way:** a dragged terrain does not define a merge. A sculpt would have to be read as a set of
  feature-level blend weights, and testing that idea comes before any interface.

## Visual and spatial interfaces (Godot / VR)

**VR neuron painting (L).**
- **Builds on:** the Godot client's 3D feature and neuron views, and feature packs (the output format).
- **Realistic version:** OpenXR in Godot. Point at feature clusters, group and label them, and strike
  through clusters to write a suppression list.
- **In the way:** VR input and comfort work; thousands of features need level-of-detail rendering.

**Activation wind tunnel (M).**
- **Builds on:** per-token traces, the risk timeline, and the 3D replay.
- **Realistic version:** a time-scrubbable 3D stream of layers × tokens, with attention arcs for the
  strongest heads.
- **In the way:** attention weights need the PyTorch path. A "baffle" that damps a layer is an intervention
  (`model_lab.py hijack` / ablation), not a live serving control.

**Attribution wiring boards (M).**
- **Builds on:** attribution graphs (`feat_graph`), circuits (`interventions.py circuit`), and path patching.
- **Realistic version:** a 3D or 2D board of the attribution graph. Snipping a wire runs the ablation and
  shows the changed prediction.
- **In the way:** making a snip permanent means exporting an edit, which works for neurons and features but
  not for arbitrary paths.

**Feature terrariums for agents (M).**
- **Builds on:** Projects (agents and tasks), permissions, and the system monitor.
- **Realistic version:** a room per agent task, showing its permission boundaries, its recent tool calls,
  its risk timeline, and a pause or deny dial wired to the existing approval rules.
- **In the way:** nothing fundamental; it is UI work on data Studio already has.

**Multimodal attribution stencils (L).**
- **Builds on:** `clip_neurons.py`, the vision lab, and the mmproj tools.
- **Realistic version:** gradient or occlusion saliency over image patches for a vision model's answer,
  drawn as an overlay. Clicking a region re-asks with that region masked.
- **In the way:** editing projector weights per image is not useful; masking the input is the honest
  control.

## Provenance, security and trust

**Activation watermarking (R).**
- **Builds on:** steering and sampling hooks.
- **Realistic version:** sampling-level watermarks (green lists, as in Kirchenbauer et al.) are known to
  work, and could be a sampler option on the PyTorch path or a llama.cpp patch.
- **In the way:** a watermark carried by activation oscillations, detectable in plain text, has no known
  method. Sampling watermarks survive light editing only.

**MCP digital-twin guard (M).**
- **Builds on:** the MCP tool approval rules, permissions, per-reply risk scoring, and the Godot dashboard.
- **Realistic version:** an approval rule that denies, or asks about, a tool call when the reply that made
  it scored above a risk threshold.
- **In the way:** risk is scored after the reply. For tool calls, scoring the call's own tokens first means
  holding the call until the score arrives, which is a short delay.

**Clean-slate license screen (M).**
- **Builds on:** graph screens, permissions' regex engine, and the code security node.
- **Realistic version:** a graph node that fingerprints generated code (normalised token shingles) against a
  local index of code you must not copy. On a match it flags, or asks for a rewrite.
- **In the way:** the index has to be built from the corpora you care about. Near-copies with renamed
  identifiers need the normalisation step.

**Proof-of-alignment mining pool (R).**
- **Builds on:** the review board, feature packs, neuro-fuzzing, remote workers, and signed worker results.
- **Realistic version:** a shared deficit queue. Workers run fuzzing and TestQA on a model and submit
  feature packs with their before and after TestQA cards, and the board re-checks the cards.
- **In the way:** "proof" only goes as far as the board can re-verify (cards, yes; neuron edits generally,
  no). Rewards and an open network invite gaming.

## Distributed and multi-model

**Hive-mind consensus routing (M).**
- **Builds on:** linked hosts, workers, per-reply risk, and morphing's router.
- **Realistic version:** send a prompt to several hosts' models. Each scores its own risk (its classifier),
  and the lowest-risk confident answer is used, or a vote decides.
- **In the way:** moving "middle and late layers" to a cleaner machine mid-generation is layer streaming
  with a router, which llama.cpp RPC does not support per request.

**P2P asset forge for game engines (M).**
- **Builds on:** Connect's transfer plane, the remote agent, graphs, and the Godot client as a model.
- **Realistic version:** an engine plugin (Godot first) that sends the selected scene context (bounding
  boxes, tags) to Studio and gets generated scripts or assets back into the project.
- **In the way:** Unreal and Unity plugins are separate C++ and C# projects.

**Sensory substitution matrix (R).**
- **Builds on:** concept translation (linear maps between representation spaces) and the vision path.
- **Realistic version:** render the sensor data as images a vision model can read (spectrograms, range
  maps), which works today without training. Mapping raw sensor data straight into a vision encoder's
  feature space needs paired data and a learned projector.
- **In the way:** a projector needs training, so it is not "without training from scratch".

**Neuro-feedback biometric steering (M, plus hardware).**
- **Builds on:** the presence monitor (input idle, gamepad, VR), steering, and Neural Reset's stream hook.
- **Realistic version:** a biometric stream (heart rate, eye tracker) becomes an event. When it fires
  mid-reply, Studio cuts the reply and resumes it more plainly, or steers it by a "clarity" vector on the
  PyTorch path.
- **In the way:** consumer EEG is too noisy for this. Heart rate and gaze are usable, but they need consent,
  calibration, and care with the data.

## Pose

The pose pipeline is built ([POSE.md](POSE.md)). The remaining step there is a native UE5 C++ plugin: the
vertex-factory shader inside Unreal. VMC4UE already receives the VMC stream.
