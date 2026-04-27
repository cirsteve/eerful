"""EvaluatorBundle — the public criteria a producer was evaluated under.

Spec §6.5. Bundles are immutable after publication; a new version of an
evaluator publishes a new bundle with a new evaluator_id (the content hash
of the bundle's canonical JSON).
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from eerful.canonical import Bytes32Hex, canonical_json_bytes, is_bytes32_hex


class EvaluatorBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    """Human-readable version, e.g. 'trading-critic@1.2.0'."""

    model_identifier: str
    """An identifier the 0G Compute Network recognizes for verifiable services.
    The TeeML attestation does NOT bind the loaded model weights to this
    identifier (spec §8)."""

    system_prompt: str
    """The criteria, in natural language."""

    output_schema: dict[str, Any] | None = None
    """JSON Schema that `output_score_block` MUST validate against when both
    are present (spec §6.5)."""

    inference_params: dict[str, Any] | None = None
    """Producer-recommended inference parameters (temperature, max_tokens, etc.).
    Not enforced at the protocol level — the TEE doesn't attest to them."""

    accepted_compose_hashes: list[Bytes32Hex] | None = None
    """Allowlist of attested compose-hashes (spec §6.5, §8.3). When set,
    verification Step 5 fails if the report's compose-hash is not in this
    list. The strongest defense against the §8 model-binding gap that the
    receipt format alone can offer.

    Canonicalization: `None` means "no allowlist, no gating"; a populated
    list means "gate Step 5 on these hashes". An empty list is rejected at
    construction so there is exactly one canonical form for "no gating"
    (mirrors §10.1's `extensions={}` → `null` policy). Without this rule,
    `None` and `[]` would canonical-JSON-encode differently (`null` vs `[]`)
    and produce diverging `evaluator_id`s for what publishers intend as the
    same bundle."""

    metadata: dict[str, Any] | None = None
    """Publisher-defined; informational."""

    @model_validator(mode="after")
    def _validate_accepted_compose_hashes(self) -> EvaluatorBundle:
        """Enforce the §6.5 / §6.4 invariants the type alias can't express:

        - Each entry is a syntactically valid `Bytes32Hex` (the
          `BeforeValidator` lowercases but does not bound length).
        - The list is non-empty when present (canonical-form rule above).
        """
        items = self.accepted_compose_hashes
        if items is None:
            return self
        if len(items) == 0:
            raise ValueError(
                "accepted_compose_hashes must be omitted (None) rather than empty; "
                "an empty list has no canonical form (see §6.5 docstring)."
            )
        for h in items:
            if not is_bytes32_hex(h):
                raise ValueError(
                    f"accepted_compose_hashes entry is not a valid Bytes32Hex: {h!r}"
                )
        return self

    def canonical_bytes(self) -> bytes:
        """Canonical JSON encoding of the bundle (spec §6.4).

        `mode="json"` defends against future fields whose Python-native form
        isn't JSON-serializable (datetime, UUID, Decimal, etc.) — keeps
        canonicalization total even as the bundle schema grows.
        """
        return canonical_json_bytes(self.model_dump(mode="json"))

    def evaluator_id(self) -> Bytes32Hex:
        """Content hash of the bundle. The receipt's `evaluator_id` field is
        a verifier-fetchable handle: store the bundle's `canonical_bytes()`
        keyed by this id, and a verifier rehashes after fetch (spec §7.1
        Step 2)."""
        return "0x" + hashlib.sha256(self.canonical_bytes()).hexdigest()
