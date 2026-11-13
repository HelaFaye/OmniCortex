#!/usr/bin/env python3
"""
Load a model with a suppression profile applied, for interactive use or as a
library. No weights are modified; the profile is applied as forward pre-hooks
and the scale can change between generations.

    # list profiles matching a model
    python scripts/apply_profile.py --model_path <m> --list

    # chat with suppression on, comparing scales side by side
    python scripts/apply_profile.py --model_path <m> \
        --config_name trivia-q8 --compare 1.0 0.25 \
        --prompt "Who wrote the novel Stoner?"

As a library:

    from profiles import Profile, SuppressionHandle
    with SuppressionHandle(model, Profile.load(path)) as h:
        ...                 # suppressed
        h.set_scale(1.0)    # off, no reload
"""
import argparse, json, os, sys
import torch
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_common import load_model
from profiles import Profile, SuppressionHandle, fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--profile", help="Explicit path; otherwise looked up by fingerprint")
    p.add_argument("--profiles_root", default="profiles")
    p.add_argument("--config_name")
    p.add_argument("--list", action="store_true", help="Show matching profiles and exit")
    p.add_argument("--prompt")
    p.add_argument("--compare", nargs="+", type=float,
                   help="Generate once per scale for comparison")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--gpu_mem")
    p.add_argument("--cpu_mem", default="40GiB")
    try:
        from oc_cli import add_trust_remote_code_argument
    except ImportError:
        from scripts.oc_cli import add_trust_remote_code_argument
    add_trust_remote_code_argument(p)
    a = p.parse_args()
    trc = a.trust_remote_code

    if a.list:
        cfg = AutoConfig.from_pretrained(a.model_path, trust_remote_code=trc)
        fp, geom = fingerprint(cfg)
        print(f"fingerprint: {fp}\ngeometry   : {geom}\n")
        found = Profile.find(cfg, a.profiles_root, a.config_name)
        if not found:
            print(f"no profiles under {a.profiles_root}/{fp}/")
            return
        for path in found:
            d = json.load(open(path))
            ev = d.get("evaluations") or [{}]
            print(f"  {os.path.basename(path):<24} scale={d['scale']:<5} "
                  f"neurons={d['total_neurons']:<5} created={d['created']}")
            sel = ev[-1].get("selected")
            if sel is not None:
                print(f"      tuned on {ev[-1].get('n_eval')} questions, selected {sel}")
        return

    path = a.profile
    if not path:
        cfg = AutoConfig.from_pretrained(a.model_path, trust_remote_code=trc)
        found = Profile.find(cfg, a.profiles_root, a.config_name)
        if not found:
            raise SystemExit(f"no matching profile; run with --list to check")
        if len(found) > 1 and not a.config_name:
            raise SystemExit("several profiles match; pass --config_name:\n  " +
                             "\n  ".join(os.path.basename(f) for f in found))
        path = found[0]
    print(f"profile: {path}")

    model, tokenizer = load_model(a.model_path, a.gpu_mem, a.cpu_mem, trc)
    device = next(model.parameters()).device
    profile = Profile.load(path)

    with SuppressionHandle(model, profile) as h:
        print(h.summary())
        if not a.prompt:
            print("no --prompt given; profile loads cleanly against this model")
            return
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": a.prompt}],
            add_generation_prompt=True, return_tensors="pt").to(device)
        for scale in (a.compare or [profile.data["scale"]]):
            h.set_scale(scale)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=a.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            label = "off" if scale == 1.0 else f"scale {scale}"
            print(f"\n--- {label} ---\n{text.strip()}")


if __name__ == "__main__":
    main()
