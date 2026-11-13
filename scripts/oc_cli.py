"""Shared CLI option for explicitly opting into Hugging Face custom code."""
from __future__ import annotations

import argparse


def add_trust_remote_code_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add an opt-in remote-code flag, retaining the old negative flag safely.

    Remote repository Python code is never trusted by default. The legacy
    ``--no_trust_remote_code`` spelling remains accepted as an explicit deny.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--trust-remote-code", "--trust_remote_code",
        dest="trust_remote_code", action="store_true", default=False,
        help="allow execution of custom Python code from the model repository",
    )
    group.add_argument(
        "--no-trust-remote-code", "--no_trust_remote_code",
        dest="trust_remote_code", action="store_false", default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    return parser
