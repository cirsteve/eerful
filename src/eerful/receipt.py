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

from eerful.canonical import Address, Bytes32Hex, BytesHex, canonical_json_bytes
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


def _signing_payload_dict(
    *,
    chat_id: str,
    created_at: datetime,
    evaluator_id: Bytes32Hex,
    evaluator_version: str,
    extensions: dict[str, Any] | None,
    input_commitment: Bytes32Hex | None,
    output_score_block: dict[str, Any] | None,
    previous_receipt_id: Bytes32Hex | None,
    provider_address: Address,
    response_content: str,
) -> dict[str, Any]:
    return {
        "chat_id": chat_id,
        "created_at": _to_rfc3339_z(created_at),
        "evaluator_id": evaluator_id,
        "evaluator_version": evaluator_version,
        "extensions": extensions,
        "input_commitment": input_commitment,
        "output_score_block": output_score_block,
        "previous_receipt_id": previous_receipt_id,
        "provider_address": provider_address,
        "response_content": response_content,
    }


def derive_receipt_id(payload: dict[str, Any]) -> Bytes32Hex:
    return "0x" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class EnhancedReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt_id: Bytes32Hex

    # Producer claims (in canonical signing payload)
    created_at: datetime
    evaluator_id: Bytes32Hex
    evaluator_version: str
    input_commitment: Bytes32Hex | None = None
    previous_receipt_id: Bytes32Hex | None = None

    # Compute provider attribution (in canonical signing payload)
    provider_address: Address
    chat_id: str
    response_content: str

    # Structured evaluation (in canonical signing payload)
    output_score_block: dict[str, Any] | None = None

    # Attestation block (NOT in canonical signing payload; spec §6.3).
    # Cached for offline verification; integrity is established by the Step 5
    # report-binds-pubkey check and the Step 6 signature verification, not
    # by receipt_id.
    attestation_report_hash: Bytes32Hex
    enclave_pubkey: BytesHex
    enclave_signature: BytesHex

    # Extensions (in canonical signing payload, spec §10.1)
    extensions: dict[str, Any] | None = None

    SIGNING_PAYLOAD_FIELDS: ClassVar[tuple[str, ...]] = (
        "chat_id",
        "created_at",
        "evaluator_id",
        "evaluator_version",
        "extensions",
        "input_commitment",
        "output_score_block",
        "previous_receipt_id",
        "provider_address",
        "response_content",
    )

    def signing_payload(self) -> dict[str, Any]:
        """Fields included in receipt_id derivation (spec §6.3)."""
        return _signing_payload_dict(
            chat_id=self.chat_id,
            created_at=self.created_at,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            extensions=self.extensions,
            input_commitment=self.input_commitment,
            output_score_block=self.output_score_block,
            previous_receipt_id=self.previous_receipt_id,
            provider_address=self.provider_address,
            response_content=self.response_content,
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
        evaluator_version: str,
        provider_address: Address,
        chat_id: str,
        response_content: str,
        attestation_report_hash: Bytes32Hex,
        enclave_pubkey: BytesHex,
        enclave_signature: BytesHex,
        input_commitment: Bytes32Hex | None = None,
        previous_receipt_id: Bytes32Hex | None = None,
        output_score_block: dict[str, Any] | None = None,
        extensions: dict[str, Any] | None = None,
    ) -> EnhancedReceipt:
        """Construct a receipt with receipt_id derived from the canonical
        signing payload."""
        payload = _signing_payload_dict(
            chat_id=chat_id,
            created_at=created_at,
            evaluator_id=evaluator_id,
            evaluator_version=evaluator_version,
            extensions=extensions,
            input_commitment=input_commitment,
            output_score_block=output_score_block,
            previous_receipt_id=previous_receipt_id,
            provider_address=provider_address,
            response_content=response_content,
        )
        return cls(
            receipt_id=derive_receipt_id(payload),
            created_at=created_at,
            evaluator_id=evaluator_id,
            evaluator_version=evaluator_version,
            input_commitment=input_commitment,
            previous_receipt_id=previous_receipt_id,
            provider_address=provider_address,
            chat_id=chat_id,
            response_content=response_content,
            output_score_block=output_score_block,
            attestation_report_hash=attestation_report_hash,
            enclave_pubkey=enclave_pubkey,
            enclave_signature=enclave_signature,
            extensions=extensions,
        )

    @model_validator(mode="after")
    def _verify_receipt_id(self) -> EnhancedReceipt:
        expected = derive_receipt_id(self.signing_payload())
        if expected != self.receipt_id:
            raise VerificationError(
                step=1,
                reason=f"receipt_id mismatch: expected {expected}, got {self.receipt_id}",
            )
        return self
