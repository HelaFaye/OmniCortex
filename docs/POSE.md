# Pose add-on

`plugins/pose` turns video into tracked 3D poses and catches the frames where the tracking is physically
impossible. It turns each one into a training pair (as tracked = rejected, corrected = chosen), lets you review
them, and refines the pose model with DPO once enough are approved. It also streams the poses, with per-joint
stress, to game engines and DCC tools.

The add-on is off until you turn it on (**Settings → Plugins**). Its jobs then appear under **Jobs → Plugins:
pose**, and its review page is at `/plugins/pose/`.

## The pipeline

1. **Extract** (`extract.py`) produces frame payloads, one JSON object per line. Sources:
   - a video through MediaPipe's pose landmarker (`pip install mediapipe opencv-python` plus the
     `.task` model file), up to `--people` performers per frame, in world metres;
   - OpenPose's BODY_25 JSON output. 2D OpenPose is scaled to pseudo-metres at z = 0; its 3D output is used
     as is;
   - an existing frames file from WHAM, an SMPL-X fit or anything else, re-tracked.

   Each performer keeps a TrackID: hips are matched to the nearest track, closest first, and a track ends
   after it goes unseen. `--images` writes the video's frames for review and vision DPO.
2. **Sieve** (`sieve.py`) checks every performer, frame by frame:
   - knees bent past straight the wrong way;
   - elbows or knees that flip inside out;
   - joints moving faster than a body can;
   - bones stretching away from their median length;
   - feet below the floor;
   - feet sliding while planted.

   Each flagged frame is corrected: the failing joint and those below it are interpolated between the
   nearest clean frames, keeping the limb attached; with no clean neighbours, the joint is clamped. The pair
   is staged under `pending/`. A "correction" that changes nothing (a hand landing back where it belongs) is
   dropped.
3. **Review** (`/plugins/pose/`) shows each item over its video frame:
   - the skeleton as tracked, in green;
   - the failing joints, in red;
   - the correction, dashed blue, projected through a camera fitted to the healthy joints.

   Keys: **a** approve, **d** discard, **j**/**k** next and previous.
4. **Orchestrate** (`orchestrate.py`) runs a refinement cycle once `--threshold` new pairs are approved:
   1. pauses the stream;
   2. exports the dataset;
   3. sweeps `finetune.py --dpo` (QLoRA) over learning rates × LoRA alphas;
   4. reads each run's training log, falling back to TensorBoard event files;
   5. picks the run with the largest stable reward margin;
   6. writes the winner into your settings file;
   7. optionally merges it into a GGUF, applies vision-side H-neuron suppression, and swaps it in.

   The swap keeps a timestamped backup and uses an atomic rename. Without `--model`, a cycle only exports,
   so the loop can be tried without a GPU.
5. **Stream** (`stream.py`) sends the poses at their own timing, or follows a file as it grows:
   - **VMC** (OSC over UDP): Unity humanoid bone names with local rotations, so VSeeFace, VMC4B (Blender),
     EVMC4U (Unity) and VMC4UE (Unreal) receive them;
   - **JSON over UDP**: the payload plus per-joint stress, for the clients below.

   UDP never blocks the stream. While the orchestrator holds `PAUSE`, the stream waits.

## Engine clients

Download them from the add-on (`/plugins/pose/download/...`).

**Godot 4** (`omnicortex-pose-godot.zip`):
- `pose_receiver.gd` draws each performer as a stick figure glowing green to yellow to red with joint
  stress, with an x-ray pass that shows through walls.
- `stress_skin.gdshader` colours a rigged character by its skin weights (`BONE_INDICES` × `BONE_WEIGHTS` ×
  per-bone stress).
- `xray_outline.gdshader` is the see-through pass for a rigged character.

**Blender 4** (`omnicortex_pose.zip`, installed as an add-on):
- **Listen** moves one empty per joint, coloured by stress.
- **Bake stress** writes the clip's worst moment per vertex into a colour attribute `omnicortex_stress`,
  from the mesh's bone weights. Each bone also gets an animated custom property `oc_stress` over the frames.
- **Export FBX** includes the vertex colours and custom properties. FBX carries one set of vertex colours,
  so the per-frame log travels as the bones' animated property.

**Unreal and Unity** receive the VMC stream through VMC4UE and EVMC4U. A native UE5 C++ plugin with the
vertex-factory shader is not included.

## What it does not do

- **Trace which neurons drove a mistake on a video frame.** `llama-cett-dump` reads text. On the vision
  side, `clip_neurons.py` finds the encoder's overconfident neurons from images a model gets wrong (approved
  frames are such images), and `orchestrate.py --suppress` applies them to the merged projector.
- **Refine MediaPipe, OpenPose or WHAM themselves.** The refinement loop trains a vision-language model
  that outputs pose JSON. For the classic estimators, the reviewed pairs are a clean dataset for their own
  training code.
- **Recover twist.** Joint rotations from positions are swing only: twist about a bone cannot be recovered
  from where its ends are. Pass full rotations through from estimators that give them.
- **Run in an engine in CI.** The engine scripts have not been run in an engine here. Their maths (the
  stress ramp, vertex blending, rig-name matching) lives in `pose_stress.py` and `pose_core.py`, which are
  tested.
