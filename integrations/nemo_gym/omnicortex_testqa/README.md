# omnicortex_testqa: a NeMo Gym environment

OmniCortex's TestQA grading as an RL reward that does not pay for guessing:
- a correct answer earns 1;
- a decline earns `abstain_reward` (0.25);
- a confident mistake earns `wrong_reward` (-1).

1. Copy this folder to `Gym/resources_servers/omnicortex_testqa` and set `OMNICORTEX_ROOT` to an OmniCortex
   checkout. The server imports `scripts/gym_bridge.py` and TestQA's grader from it.
2. Make the training data:
   `python scripts/gym_bridge.py export --bank qa/bank --fuzz qa/fuzz/round1 --harvest --out .../data/train.jsonl`.
   `data/example.jsonl` holds five rows made the same way.
3. Run it like any Gym environment (`gym env start --resources-server omnicortex_testqa ...`), and train on it with
   NeMo RL's GRPO through Gym.

With `studio_url` and `risk_weight` set, and a classifier for the policy model in that Studio, a correct answer
given at high H-neuron risk earns `risk_weight` × risk less. That pushes the policy toward answers it knows.

This has not been run against NeMo Gym in OmniCortex's CI. `app.py` follows the interfaces of Gym's own example
servers (`SimpleResourcesServer`, `verify` returning `reward`). The grading and the reward live in `gym_bridge.py`,
which is tested.
