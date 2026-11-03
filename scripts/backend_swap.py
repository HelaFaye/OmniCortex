#!/usr/bin/env python3
"""
Backend hot-swap (docs/INTEGRATED_OPTIONS.md): when llama-server cannot load a model, or dies while serving, try
again on a setup that is more likely to fit instead of failing the chat or the job.

The fallback ladder, tried in order until one starts:
    1. the same backend with fewer layers on the GPU (half, then a quarter, then none)
    2. each alternate llama-server you list (Settings -> Integrated options), e.g. a Vulkan build when the CUDA one
       runs out of memory or its driver misbehaves, at the original GPU layer count and then with half
    3. CPU only (no GPU layers, no GPU split)

Only failures that look like resources or the driver start the ladder (out of memory, failed allocations, device
lost, a crash); a bad setting (unknown flag, missing file) fails as before, because no fallback would fix it.

Not done: moving a running generation between backends mid-token. A server that dies mid-reply is restarted on
the ladder and the request is sent again once; the client sees one slow reply, or an error if that fails too.
"""
from __future__ import annotations

import re

RESOURCE = re.compile(
    r"out of memory|failed to allocate|cudaMalloc|ErrorOutOfDeviceMemory|OutOfDeviceMemory|hipErrorOutOfMemory|"
    r"device lost|DeviceLost|CUDA error|ggml_cuda|vk::|VK_ERROR|unable to allocate|insufficient memory|"
    r"exited|died|killed|segmentation fault|SIGSEGV|core dumped|not healthy|timed out", re.I)
SETTING = re.compile(r"unknown argument|invalid argument|error: unknown|no such file|failed to open|"
                     r"must be|not found: |is not a valid", re.I)


def is_resource_failure(message: str) -> bool:
    m = str(message or "")
    return bool(RESOURCE.search(m)) and not SETTING.search(m)


def ladder(settings: dict, servers: list[str] | None = None, current_server: str | None = None) -> list[tuple[dict, str | None, str]]:
    """-> [(settings, server binary or None for the same one, why)] to try after the first failure."""
    out = []
    ngl = settings.get("ngl")
    try:
        ngl = int(ngl) if ngl not in (None, "") else 999
    except (TypeError, ValueError):
        ngl = 999
    full = ngl if ngl < 999 else 99
    for n in (full // 2, full // 4):
        if 0 < n < full:
            out.append(({**settings, "ngl": n}, None, f"same backend, {n} layers on the GPU"))
    for srv in servers or []:
        if srv and srv != current_server:
            out.append(({**settings}, srv, f"{srv}, {ngl if ngl < 999 else 'all'} GPU layers"))
            if full // 2 > 0:
                out.append(({**settings, "ngl": full // 2}, srv, f"{srv}, {full // 2} layers on the GPU"))
    cpu = {**settings, "ngl": 0, "split_mode": "", "tensor_split": "", "main_gpu": -1, "gpus": ""}
    out.append((cpu, None, "CPU only"))
    seen, uniq = set(), []
    for s, srv, why in out:
        k = (s.get("ngl"), srv, s.get("split_mode"))
        if k not in seen:
            seen.add(k)
            uniq.append((s, srv, why))
    return uniq
