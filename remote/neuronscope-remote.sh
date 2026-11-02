#!/bin/sh
# OmniCortex Remote: this computer becomes a HOST that a controller (OmniCortex Studio,
# or an MCP client) can use after you pair it.
cd "$(dirname "$0")" || exit 1
exec python3 -m neuronscope_remote "$@"
