"""EvaluatorBundle — the public criteria a producer was evaluated under.

Spec §6.5. Bundles are immutable after publication; a new version of an
evaluator publishes a new bundle with a new evaluator_id (the content hash
of the bundle's canonical JSON).

This module is Day 1 scaffolding: field types only. `evaluator_id()`,
`canonical_bytes()`, and schema validation land Day 2-3.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EvaluatorBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    """Human-readable version, e.g. 'trading-critic@1.2.0'."""

    model_identifier: str
    """An identifier the 0G Compute Network recognizes for verifiable services.
    The TeeML attestation does NOT bind the loaded model to this identifier
    (spec §8)."""

    system_prompt: str
    """The criteria, in natural language."""

    output_schema: dict | None = None
    """JSON Schema that `output_score_block` MUST validate against when both
    are present (spec §6.5)."""

    inference_params: dict | None = None
    """Producer-recommended inference parameters (temperature, max_tokens, etc.).
    Not enforced at the protocol level — the TEE doesn't attest to them."""

    metadata: dict | None = None
    """Publisher-defined; informational."""
