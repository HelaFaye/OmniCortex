#!/usr/bin/env python3
"""
Merge method reference and config generator, ordered by how likely each is to
work on a first attempt.

Every method below is verified present in mergekit's registry. The ordering is
practical, not chronological: it starts with what to reach for and ends with
what to reach for only when you know why.

    python scripts/merge_methods.py                 # the ranked list
    python scripts/merge_methods.py --method ties --base A --models B C
    python scripts/merge_methods.py --verify        # check against installed mergekit

The ranking assumes the common case: two or three models sharing a base,
same architecture, and you want measured improvement. It changes if you have
ten checkpoints of one fine-tune (model_stock rises) or want to remove a
behaviour rather than add one (task_arithmetic rises).

Whatever you pick, put the result through merge_eval.py. Merging is cheap and
plausible-looking; a merged model produces fluent text whether or not it
retained anything, so the only evidence is a paired comparison against the
sources on tasks you did not tune on.
"""

import argparse
import json

# tier: 1 = reach for first, 4 = only with a reason
METHODS = [
    dict(name="slerp", tier=1, models=(2, 2), base="optional",
         summary="Spherical interpolation between exactly two models.",
         why="The default first attempt. Averaging flattens weight magnitudes; "
             "interpolating along the arc preserves them, which matters because "
             "direction and magnitude are what the network computes with. "
             "Two models only.",
         params={"t": 0.5},
         note="t is the blend point; t: 0.5 is even. A gradient list applies "
              "different blends to different layers."),

    dict(name="ties", tier=1, models=(2, None), base="required",
         summary="Trim task vectors, elect a sign per parameter, then merge.",
         why="The workhorse for three or more models. Addresses the two ways "
             "naive averaging destroys information: a small weight dragging "
             "down a large one of the same sign, and opposing signs cancelling "
             "to near zero. Needs a base to compute task vectors from.",
         params={"weight": 1.0, "density": 0.5},
         note="density is the fraction of each task vector kept. 0.5 is a "
              "reasonable start; lower it if the merge feels muddy."),

    dict(name="dare_ties", tier=1, models=(2, None), base="required",
         summary="Randomly drop most of each task vector, rescale, then TIES.",
         why="Usually a small improvement on TIES for the same cost. The DARE "
             "finding is that ~90% of fine-tuning deltas can be dropped and "
             "rescaled with little loss, which leaves less to interfere.",
         params={"weight": 1.0, "density": 0.5},
         note="Stochastic, so set a seed if you need reproducibility."),

    dict(name="task_arithmetic", tier=2, models=(1, None), base="required",
         summary="Add or subtract fine-tuning deltas as vectors.",
         why="The idea the others build on, and still the right tool when you "
             "want to REMOVE something: a negative weight subtracts a "
             "behaviour. Simple and predictable, with no interference handling.",
         params={"weight": 1.0},
         note="weight: -1.0 subtracts. That is the mechanism behind "
              "'un-fine-tuning' a behaviour."),

    dict(name="linear", tier=2, models=(2, None), base="optional",
         summary="Weighted average of weights. Model soup.",
         why="Works when the models are already very close -- checkpoints of "
             "the same run, or light fine-tunes of a shared base. Flattens "
             "magnitudes, so it degrades as the models diverge.",
         params={"weight": 0.5},
         note="If linear works, prefer it: fewest knobs, most predictable."),

    dict(name="model_stock", tier=2, models=(3, None), base="required",
         summary="Geometric centre estimated from three or more fine-tunes.",
         why="Approximates the centre a large model soup would converge to, "
             "from far fewer models. Only worth it when you genuinely have "
             "three or more fine-tunes of one base.",
         params={},
         note="Needs at least three models plus the base. Fewer and it has "
              "nothing to estimate from."),

    dict(name="dare_linear", tier=3, models=(2, None), base="required",
         summary="DARE dropping without the sign election.",
         why="DARE minus TIES. Cheaper, and occasionally better when sign "
             "conflicts are not the problem. Try it if dare_ties disappoints.",
         params={"weight": 1.0, "density": 0.5}),

    dict(name="della", tier=3, models=(2, None), base="required",
         summary="Magnitude-weighted stochastic dropping, then TIES.",
         why="DARE drops uniformly at random; DELLA drops with probability "
             "inversely related to magnitude, so large deltas usually survive. "
             "More principled, more knobs to get wrong.",
         params={"weight": 1.0, "density": 0.5, "epsilon": 0.1, "lambda": 1.0},
         note="epsilon is the spread of drop probability, lambda the final "
              "scaling. Copy a published config before tuning blind."),

    dict(name="della_linear", tier=3, models=(2, None), base="required",
         summary="DELLA dropping without the sign election.",
         why="As dare_linear is to dare_ties.",
         params={"weight": 1.0, "density": 0.5, "epsilon": 0.1, "lambda": 1.0}),

    dict(name="breadcrumbs", tier=3, models=(2, None), base="required",
         summary="Mask both the largest and smallest deltas, keep the middle.",
         why="Discards outliers at both ends on the theory that extremes are "
             "noise or overfit. Narrower use than TIES; worth a try when TIES "
             "produces something erratic.",
         params={"weight": 1.0, "density": 0.9, "gamma": 0.01}),

    dict(name="breadcrumbs_ties", tier=3, models=(2, None), base="required",
         summary="Breadcrumbs masking plus sign election.",
         why="The combination, for the same reasons as dare_ties over "
             "dare_linear.",
         params={"weight": 1.0, "density": 0.9, "gamma": 0.01}),

    dict(name="nuslerp", tier=3, models=(2, 2), base="optional",
         summary="SLERP with per-model weights and optional task-vector mode.",
         why="SLERP when you want an uneven blend, or want to interpolate task "
             "vectors rather than raw weights.",
         params={"weight": 1.0}),

    dict(name="multislerp", tier=3, models=(2, None), base="optional",
         summary="Spherical interpolation generalised past two models.",
         why="SLERP's answer to the three-model case, as an alternative to "
             "TIES."),

    dict(name="karcher", tier=4, models=(2, None), base="optional",
         summary="Karcher mean -- the intrinsic average on the weight manifold.",
         why="The mathematically principled version of a soup. Iterative and "
             "slower; interesting rather than routine.",
         params={}),

    dict(name="sce", tier=4, models=(2, None), base="required",
         summary="Select, Calculate, Erase.",
         why="Variance-based selection of which parameters to merge. Newer and "
             "less exercised; benchmark against TIES before adopting."),

    dict(name="nearswap", tier=4, models=(2, 2), base="required",
         summary="Swap in donor weights where they are close to the base.",
         why="Very conservative: touches only what barely differs. Rarely what "
             "you want, occasionally exactly what you want.",
         params={"t": 0.5}),

    dict(name="arcee_fusion", tier=4, models=(2, 2), base="required",
         summary="Dynamic-threshold selective fusion.",
         why="Arcee's in-house method. Reasonable, but no more likely to work "
             "than TIES on a first attempt.",
         params={}),

    dict(name="passthrough", tier=4, models=(1, None), base="no",
         summary="Stack layer ranges from different models. Frankenmerge.",
         why="Not a merge -- it concatenates layers and CHANGES THE PARAMETER "
             "COUNT. This is how Goliath-120B was built from two 70Bs. "
             "Unpredictable, expensive to evaluate, occasionally spectacular.",
         params={},
         note="Uses slices/layer_range, not models/weight. Nothing else in "
              "this toolkit assumes a changed parameter count -- your existing "
              "profiles will not apply to the result."),

    dict(name="omnicortex_select", tier=4, models=(1, None), base="required",
         summary="Merge only at classifier-identified H-Neurons.",
         why="Ours, from scripts/mergekit_omnicortex.py. Every method above "
             "selects by magnitude or variance; this selects by measured "
             "behaviour. Requires a trained classifier, so it is last by "
             "prerequisite, not by merit.",
         params={"weight": 1.0},
         note="Needs OMNICORTEX_NEURONS set and the plugin imported."),
]

TIER_LABEL = {
    1: "REACH FOR THESE FIRST",
    2: "SOLID, NARROWER USE",
    3: "WORTH TRYING WHEN THE ABOVE DISAPPOINT",
    4: "ONLY WITH A SPECIFIC REASON",
}


def show_list(verbose=False):
    for tier in (1, 2, 3, 4):
        ms = [m for m in METHODS if m["tier"] == tier]
        print(f"\n=== {TIER_LABEL[tier]} ===")
        for m in ms:
            lo, hi = m["models"]
            count = f"{lo}" if hi == lo else (f"{lo}+" if hi is None else f"{lo}-{hi}")
            print(f"\n  {m['name']:<18} {count} models, base {m['base']}")
            print(f"    {m['summary']}")
            if verbose:
                for line in _wrap(m["why"], 68):
                    print(f"      {line}")
                if m.get("note"):
                    for line in _wrap("NOTE: " + m["note"], 68):
                        print(f"      {line}")


def _wrap(text, width):
    out, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width:
            out.append(cur); cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def gen_config(name, base, models):
    m = next((x for x in METHODS if x["name"] == name), None)
    if m is None:
        raise SystemExit(f"unknown method {name!r}; run without --method to list")

    lo, hi = m["models"]
    if len(models) < lo:
        raise SystemExit(f"{name} needs at least {lo} model(s), got {len(models)}")
    if hi and len(models) > hi:
        raise SystemExit(f"{name} takes at most {hi} model(s), got {len(models)}")
    if m["base"] == "required" and not base:
        raise SystemExit(f"{name} requires --base (it merges task vectors)")

    if name == "passthrough":
        print(f"""# {m['summary']}
# CHANGES THE PARAMETER COUNT. Nothing else in this toolkit assumes that.
slices:
  - sources:
      - model: {models[0]}
        layer_range: [0, 24]
  - sources:
      - model: {models[1] if len(models) > 1 else models[0]}
        layer_range: [8, 32]
merge_method: passthrough
dtype: bfloat16""")
        return

    lines = [f"# {m['summary']}", f"merge_method: {name}"]
    if base:
        lines.append(f"base_model: {base}")
    if name == "slerp":
        lines += [f"models:", f"  - model: {models[0]}", f"  - model: {models[1]}",
                  "parameters:",
                  f"  t: {m['params'].get('t', 0.5)}   # 0 = first model, 1 = second"]
    else:
        lines.append("models:")
        for mm in models:
            lines.append(f"  - model: {mm}")
            if m["params"]:
                lines.append("    parameters:")
                for k, v in m["params"].items():
                    lines.append(f"      {k}: {v}")
    lines.append("dtype: bfloat16")
    print("\n".join(lines))
    if m.get("note"):
        print(f"\n# NOTE: {m['note']}")
    print(f"\n# mergekit-yaml merge.yaml ./out")
    print(f"# then: python scripts/merge_eval.py --endpoint base=... "
          f"--endpoint merged=... --tasks data/eval_tasks.jsonl")


def verify():
    try:
        from mergekit.merge_methods import REGISTERED_MERGE_METHODS
        have = set(REGISTERED_MERGE_METHODS)
    except Exception:
        try:
            from mergekit.merge_methods.registry import STATIC_MERGE_METHODS
            have = {m.name() for m in STATIC_MERGE_METHODS}
        except Exception as e:
            raise SystemExit(f"could not inspect mergekit: {e}\n"
                             "  pip install -e /path/to/mergekit")
    ours = {m["name"] for m in METHODS} - {"omnicortex_select"}
    missing = sorted(ours - have)
    extra = sorted(have - ours)
    print(f"{len(have)} methods registered in the installed mergekit")
    if missing:
        print(f"\nlisted here but NOT installed: {missing}")
        print("  your mergekit is older than this reference")
    if extra:
        print(f"\ninstalled but not listed here: {extra}")
    if not missing and not extra:
        print("reference matches the installed mergekit exactly")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method")
    p.add_argument("--base")
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--short", action="store_true", help="names and summaries only")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    if a.verify:
        return verify()
    if a.json:
        return print(json.dumps(METHODS, indent=2))
    if a.method:
        return gen_config(a.method, a.base, a.models)

    print(__doc__.strip().split("\n\n")[0])
    show_list(verbose=not a.short)
    print("\n\nStart with slerp for two models, ties for three or more. "
          "Everything else\nneeds a reason. And evaluate the result -- a merge "
          "that retained nothing still\nproduces fluent text.")


if __name__ == "__main__":
    main()
