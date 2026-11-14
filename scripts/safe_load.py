"""
Loading PyTorch files without running code from them.

A .pt file is a pickle. Unpickling an untrusted one can run arbitrary code, and some of the files OmniCortex loads
can come from elsewhere: feature dictionaries (sae.py), steering and emotion vectors (decode_lab.py, rp_steer.py,
emotions.py). OmniCortex's own files hold only tensors, numbers, strings, lists and dicts, which PyTorch's
weights_only loader reads safely, so that is the only loader used.

A file that needs more is refused with an explanation. For a file you made yourself and trust, OC_TRUST_PICKLE=1
allows the full loader.
"""
from __future__ import annotations

import os


def torch_load(path, map_location="cpu"):
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:                    # UnpicklingError, or RuntimeError wrapping it
        if "weights_only" not in str(e) and "Unpickl" not in type(e).__name__ and "unpickl" not in str(e).lower():
            raise
        if os.environ.get("OC_TRUST_PICKLE") == "1":
            return torch.load(path, map_location=map_location, weights_only=False)
        raise ValueError(f"{path} holds Python objects beyond tensors, numbers and text, and loading it could run "
                         "code from the file. OmniCortex's own files never need that. If you made this file "
                         "yourself and trust it, set OC_TRUST_PICKLE=1.") from e
