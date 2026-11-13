#!/usr/bin/env python3
"""Check that this PyTorch install really works on this computer's GPUs: every GPU has
kernels in the wheel, and an fp32 matmul on it matches the CPU. Exit status 1 when not.
The installers (installer.py, install.sh) run it after installing PyTorch."""
import sys

import torch

print("torch     :", torch.__version__)
print("hip       :", getattr(torch.version, "hip", None))
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    a = torch.randn(256, 256)
    err = float(((a.to("mps") @ a.to("mps")).cpu() - a @ a).abs().max() / (a @ a).abs().max())
    print(f"device    : Apple MPS (fp32 matches CPU, rel err {err:.1e})")
    if err > 1e-3:
        sys.exit("MPS result differs from the CPU")
if torch.cuda.is_available():
    archs = torch.cuda.get_arch_list()
    a = torch.randn(512, 512, dtype=torch.float32)
    ref = a @ a
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cc = f"sm_{p.major}{p.minor}"
        free, total = torch.cuda.mem_get_info(i)
        print(f"device {i}  : {p.name}  {cc}  {total / 2**30:.1f} GiB ({free / 2**30:.1f} free)")
        if not getattr(torch.version, "hip", None) and cc not in archs and \
                not any(x.startswith("compute_") for x in archs):
            sys.exit(f"this torch wheel has no kernels for {cc} (has {', '.join(archs)}); "
                     "scripts/cuda_info.py shows which wheel this GPU needs")
        got = (a.to(f"cuda:{i}") @ a.to(f"cuda:{i}")).cpu()
        err = float((got - ref).abs().max() / ref.abs().max())
        if err > 1e-3:
            sys.exit(f"device {i}: fp32 matmul differs from CPU by {err:.2e}")
        print(f"            fp32 matches CPU (rel err {err:.1e})")
elif not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
    print("no GPU visible to torch: work that uses PyTorch runs on the CPU")
