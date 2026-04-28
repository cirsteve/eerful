"""EnhancedReceipt — the EER artifact (spec §6).

A self-contained, third-party-verifiable record of one TEE-attested LLM
evaluation. Binds together:

- the public evaluator definition (by content hash, `evaluator_id`)
- an optional private input commitment (`input_commitment`)
- the attested model execution (`enclave_pubkey`, `enclave_signature`,
  `attestation_report_hash`)
- the optional structured score (`output_score_block`)
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

from eerful.canonical import Address, Bytes32Hex, BytesHex, canonical_json_bytes, to_lower_hex
from eerful.errors import VerificationError


def _to_rfc3339_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("created_at must be timezone-aware UTC")
    offset = dt.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("created_at must be UTC (offset 00:00)")
    s = dt.isoformat()
    if s.endswith("+00:00"):
        s = s[:-6] + "Z"
    return s


def derive_receipt_id(payload: dict[str, Any]) -> Bytes32Hex:
    return "0x" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


SPEC_VERSION: str = "0.5"
"""Current EER spec version (§10.2). Receipts produced under this
version carry `spec_version == SPEC_VERSION` — verifiers refuse
older receipts (asymmetric break documented in §10.2). v0.5 adds
`spec_version`, `evaluator_storage_root`, and `attestation_storage_root`
as required fields; the storage_root pair lets any verifier fetch any
receipt's bytes from any storage instance (Tier 2 cross-instance fix)."""


class EnhancedReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt_id: Bytes32Hex

    # Spec version pin (in canonical signing payload, spec §10.2).
    # Receipts authored under v0.5 must carry "0.5" — verifiers refuse
    # mismatching versions rather than guess at semantics.
    spec_version: str

    # Producer claims (in canonical signing payload)
    created_at: datetime
    evaluator_id: Bytes32Hex
    evaluator_version: str
    # 0G Merkle root (or backend-equivalent retrieval locator) of the
    # evaluator bundle bytes. Paired with `evaluator_id` (sha256) as a
    # (integrity_hash, retrieval_locator) tuple per spec §6.1; required
    # so any verifier with any storage instance can fetch the bundle
    # without depending on producer-side sha256→root index.
    evaluator_storage_root: Bytes32Hex
    input_commitment: Bytes32Hex | None = None
    previous_receipt_id: Bytes32Hex | None = None

    # Compute provider attribution (in canonical signing payload)
    provider_address: Address
    chat_id: str
    response_content: str

    # Structured evaluation (in canonical signing payload)
    output_score_block: dict[str, Any] | None = None

    # Attestation report identity (in canonical signing payload, spec §6.3).
    # Binding the report hash into receipt_id forecloses a same-key /
    # different-report swap post-construction.
    attestation_report_hash: Bytes32Hex
    # Symmetric to evaluator_storage_root: backend retrieval locator
    # for the attestation report bytes.
    attestation_storage_root: Bytes32Hex

    # Attestation signature material (NOT in canonical signing payload, spec §6.3).
    # Cached for offline verification; integrity is established by the Step 5
    # report-binds-pubkey check and the Step 6 signature verification, not
    # by receipt_id (a signature cannot be over itself).
    enclave_pubkey: BytesHex
    enclave_signature: BytesHex

    # Extensions (in canonical signing payload, spec §10.1)
    extensions: dict[str, Any] | None = None

    SIGNING_PAYLOAD_FIELDS: ClassVar[tuple[str, ...]] = (
        "attestation_report_hash",
        "attestation_storage_root",
        "chat_id",
        "created_at",
        "evaluator_id",
        "evaluator_storage_root",
        "evaluator_version",
        "extensions",
        "input_commitment",
        "output_score_block",
        "previous_receipt_id",
        "provider_address",
        "response_content",
        "spec_version",
    )

    SIGNING_PAYLOAD_HEX_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "attestation_report_hash",
            "attestation_storage_root",
            "evaluator_id",
            "evaluator_storage_root",
            "input_commitment",
            "previous_receipt_id",
            "provider_address",
        }
    )

    @classmethod
    def _payload_from_source(cls, source: dict[str, Any]) -> dict[str, Any]:
        """Project `source` to the canonical signing payload shape (spec §6.3).

        `source` must contain every name in `SIGNING_PAYLOAD_FIELDS`. The
        single source of truth is that tuple; adding a field there
        automatically pulls it into `signing_payload()`, `build()`, and
        `_verify_receipt_id` together. Hex fields are canonicalized to
        lowercase here (idempotent on already-canonical input) so `build()`
        and `signing_payload()` agree whether the projection runs before or
        after pydantic's BeforeValidator (spec §6.4).
        """
        payload: dict[str, Any] = {}
        for name in cls.SIGNING_PAYLOAD_FIELDS:
            value = source[name]
            if name == "created_at":
                value = _to_rfc3339_z(value)
            elif name in cls.SIGNING_PAYLOAD_HEX_FIELDS and value is not None:
                value = to_lower_hex(value)
            payload[name] = value
        return payload

    def signing_payload(self) -> dict[str, Any]:
        """Fields included in receipt_id derivation (spec §6.3)."""
        return self._payload_from_source(
            {name: getattr(self, name) for name in self.SIGNING_PAYLOAD_FIELDS}
        )

    def signing_payload_bytes(self) -> bytes:
        """Canonical JSON of the signing payload (spec §6.4)."""
        return canonical_json_bytes(self.signing_payload())

    @classmethod
    def build(
        cls,
        *,
        created_at: datetime,
        evaluator_id: Bytes32Hex,
        evaluator_storage_root: Bytes32Hex,
        evaluator_version: str,
        provider_address: Address,
        chat_id: str,
        response_content: str,
        attestation_report_hash: Bytes32Hex,
        attestation_storage_root: Bytes32Hex,
        enclave_pubkey: BytesHex,
        enclave_signature: BytesHex,
        input_commitment: Bytes32Hex | None = None,
        previous_receipt_id: Bytes32Hex | None = None,
        output_score_block: dict[str, Any] | None = None,
        extensions: dict[str, Any] | None = None,
        spec_version: str = SPEC_VERSION,
    ) -> EnhancedReceipt:
        """Construct a receipt with receipt_id derived from the canonical
        signing payload."""
        all_fields: dict[str, Any] = {
            "chat_id": chat_id,
            "created_at": created_at,
            "evaluator_id": evaluator_id,
            "evaluator_storage_root": evaluator_storage_root,
            "evaluator_version": evaluator_version,
            "input_commitment": input_commitment,
            "previous_receipt_id": previous_receipt_id,
            "provider_address": provider_address,
            "response_content": response_content,
            "output_score_block": output_score_block,
            "attestation_report_hash": attestation_report_hash,
            "attestation_storage_root": attestation_storage_root,
            "enclave_pubkey": enclave_pubkey,
            "enclave_signature": enclave_signature,
            "extensions": extensions,
            "spec_version": spec_version,
        }
        payload = cls._payload_from_source(all_fields)
        return cls(receipt_id=derive_receipt_id(payload), **all_fields)

    @model_validator(mode="after")
    def _verify_spec_version(self) -> EnhancedReceipt:
        # Strict equality (not >=): a future v0.6 implementation would
        # add this validator with `SPEC_VERSION = "0.6"` and refuse v0.5
        # receipts. Asymmetric break documented in §10.2.
        if self.spec_version != SPEC_VERSION:
            raise VerificationError(
                step=1,
                reason=(
                    f"spec_version mismatch: this implementation produces "
                    f"and verifies {SPEC_VERSION!r}, receipt declares "
                    f"{self.spec_version!r}"
                ),
            )
        return self

    @model_validator(mode="after")
    def _verify_receipt_id(self) -> EnhancedReceipt:
        expected = derive_receipt_id(self.signing_payload())
        if expected != self.receipt_id:
            raise VerificationError(
                step=1,
                reason=f"receipt_id mismatch: expected {expected}, got {self.receipt_id}",
            )
        return self
