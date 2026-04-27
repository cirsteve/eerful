"""EnhancedReceipt — the EER artifact (spec §6).

A self-contained, third-party-verifiable record of one TEE-attested LLM
evaluation. Binds together:

- the public evaluator definition (by content hash, `evaluator_id`)
- an optional private input commitment (`input_commitment`)
- the attested model execution (`enclave_pubkey`, `enclave_signature`,
  `attestation_report_hash`)
- the optional structured score (`output_score_block`)

This module is Day 1 scaffolding: field types only. `signing_payload_for_fields`,
`derive_receipt_id`, `EnhancedReceipt.build`, and the `_verify_receipt_id`
model_validator land Day 2 once the canonical encoding pipeline is end-to-end.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from eerful.canonical import Address, Bytes32Hex, BytesHex


class EnhancedReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Identity (derived from the canonical signing payload; not in it)
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
    output_score_block: dict | None = None

    # Attestation report identity (in canonical signing payload, see spec §6.3)
    attestation_report_hash: Bytes32Hex

    # Attestation block (NOT in canonical signing payload — the signature
    # cannot be over itself)
    enclave_pubkey: BytesHex
    enclave_signature: BytesHex

    # Extensions (in canonical signing payload, see spec §10.1)
    extensions: dict | None = None
