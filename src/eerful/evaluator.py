"""EvaluatorBundle — the public criteria a producer was evaluated under.

Spec §6.5. Bundles are immutable after publication; a new version of an
evaluator publishes a new bundle with a new evaluator_id (the content hash
of the bundle's canonical JSON).
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from eerful.canonical import Address, Bytes32Hex, canonical_json_bytes, is_address, is_bytes32_hex
from eerful.zg.attestation import ComposeCategory


class ComposeHashEntry(BaseModel):
    """Publisher-declared entry in a bundle's `accepted_compose_hashes`
    allowlist (spec §6.5).

    Each entry asserts that a known compose-hash falls in a known §8.2
    category and was published by a known provider. The publisher
    classifies the category at bundle-publication time; verifiers MUST
    treat the publisher's declaration as the authority. The diagnostic
    returned by `categorize_compose` is a sanity check, not a substitute.

    Cryptographically committed via the bundle's `evaluator_id` (sha256 of
    canonical bundle JSON) — restructuring the entry shape changes every
    receipt's `evaluator_id` derivation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hash: Bytes32Hex
    """Lowercase 0x-prefixed sha256(app_compose) — the attested
    compose-hash that Step 5 looks up for gating."""

    category: ComposeCategory
    """Publisher's §8.2 classification. The PrincipalPolicy's
    `required_categories` filters on this field at gate time."""

    provider_address: Address
    """0G compute provider hosting this compose. Informational on the
    bundle side; load-bearing on the executor's diversity rules
    (`distinct_compose_hashes`)."""

    notes: str | None = None
    """Publisher-supplied human-readable annotation (e.g. 'vLLM --model
    zai-org/GLM-5-FP8'). Not parsed."""

    @model_validator(mode="after")
    def _validate_invariants(self) -> ComposeHashEntry:
        """Enforce length invariants the BeforeValidator can't express.

        `Bytes32Hex` and `Address` lowercase but don't bound length; without
        this check, a 31-byte hash or 19-byte address would propagate into
        Step 5's lookup and the executor's diversity comparisons.
        """
        if not is_bytes32_hex(self.hash):
            raise ValueError(
                f"hash is not a valid 32-byte hex string: {self.hash!r}"
            )
        if not is_address(self.provider_address):
            raise ValueError(
                f"provider_address is not a valid 20-byte hex EVM address: {self.provider_address!r}"
            )
        return self


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

    accepted_compose_hashes: list[ComposeHashEntry] | None = None
    """Allowlist of attested composes (spec §6.5, §8.3). When set,
    verification Step 5 fails if the report's compose-hash is not in any
    entry. Each entry carries the publisher's §8.2 category declaration
    and the provider address it was published from — the executor's
    `required_categories` and `distinct_compose_hashes` rules read
    those fields at gate time.

    Canonicalization: `None` means "no allowlist, no gating"; a populated
    list means "gate Step 5 on these entries". An empty list is rejected
    at construction so there is exactly one canonical form for "no
    gating" (mirrors §10.1's `extensions={}` → `null` policy). Without
    this rule, `None` and `[]` would canonical-JSON-encode differently
    (`null` vs `[]`) and produce diverging `evaluator_id`s for what
    publishers intend as the same bundle."""

    metadata: dict[str, Any] | None = None
    """Publisher-defined; informational."""

    @model_validator(mode="after")
    def _validate_accepted_compose_hashes(self) -> EvaluatorBundle:
        """Enforce the §6.5 list-level invariant the type alias can't express:
        the list is non-empty when present (canonical-form rule above).
        Per-entry hash and address shape is enforced by `ComposeHashEntry`
        itself.
        """
        items = self.accepted_compose_hashes
        if items is None:
            return self
        if len(items) == 0:
            raise ValueError(
                "accepted_compose_hashes must be omitted (None) rather than empty; "
                "an empty list has no canonical form (see §6.5 docstring)."
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
