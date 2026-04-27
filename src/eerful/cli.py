"""Placeholder CLI entrypoint.

Wired into pyproject.toml's `[project.scripts]` so `uv run eerful` resolves
on a freshly cloned main without ImportError. The real subcommands
(`verify`, `publish-evaluator`, `evaluate`) land alongside §7.1 Steps 4-7
and the 0G storage / TeeML adapters.
"""

from __future__ import annotations


def main() -> int:
    print("eerful CLI not yet implemented; see docs/spec.md for the EER v0.4 protocol.")
    return 0
