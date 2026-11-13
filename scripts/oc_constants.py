"""Lightweight shared constants/helpers for OmniCortex.

This module intentionally has no PyTorch/Transformers dependency.  The GGUF
activation path needs only token-string normalization and the decoder-layer
matcher, so importing those helpers must not pull in the heavyweight training
stack.
"""

import re


# Decoder layers: model.layers.N or language_model.model.layers.N
TEXT_LAYER_RE = re.compile(r"(?:^|\.)(?:language_model\.)?model\.layers\.(\d+)\.")
THINK_CLOSE = "</think>"


def normalise_piece(tok):
    """Normalize common tokenizer word-boundary markers to spaces."""
    return tok.replace("\u2581", " ").replace("\u0120", " ")
