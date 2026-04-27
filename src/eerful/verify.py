"""§7.1 verification algorithm — Steps 1–3.

Steps 4–7 require 0G storage access and TEE attestation chain validation;
they land alongside Track B. The functions here are I/O-free: callers fetch
the bundle bytes (and, later, the attestation report bytes) and pass them
in. Each step is a separately testable function per the plan; an
orchestrator runs Steps 1–3 in spec order.

Failures raise `VerificationError(step=N, reason=...)`.
"""

from __future__ import annotations

import hashlib

import jsonschema
from pydantic import ValidationError

from eerful.errors import VerificationError
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt, derive_receipt_id


def verify_step_1_receipt_integrity(receipt: EnhancedReceipt) -> None:
    """Step 1: receipt_id matches sha256(canonical_json(signing_payload))."""
    expected = derive_receipt_id(receipt.signing_payload())
    if expected != receipt.receipt_id:
        raise VerificationError(
            step=1,
            reason=f"receipt_id mismatch: expected {expected}, got {receipt.receipt_id}",
        )


def verify_step_2_evaluator_bundle(
    receipt: EnhancedReceipt,
    bundle_bytes: bytes,
) -> EvaluatorBundle:
    """Step 2: hash of fetched bundle equals receipt.evaluator_id; bundle parses."""
    expected = "0x" + hashlib.sha256(bundle_bytes).hexdigest()
    if expected != receipt.evaluator_id:
        raise VerificationError(
            step=2,
            reason=(
                f"evaluator_id mismatch: storage returned a bundle hashing to "
                f"{expected}, receipt claims {receipt.evaluator_id}"
            ),
        )
    try:
        bundle = EvaluatorBundle.model_validate_json(bundle_bytes)
    except ValidationError as e:
        raise VerificationError(step=2, reason=f"bundle deserialization failed: {e}") from e
    return bundle


def verify_step_3_output_schema(
    receipt: EnhancedReceipt,
    bundle: EvaluatorBundle,
) -> None:
    """Step 3: output_score_block validates against bundle.output_schema, when both present.

    Per spec §6.5: schema validation runs only when both the bundle declares
    a schema AND the receipt carries a score block. A receipt without a score
    block is structurally valid (the response_content is the result, no
    machine-readable score is asserted).
    """
    if bundle.output_schema is None or receipt.output_score_block is None:
        return
    try:
        jsonschema.validate(receipt.output_score_block, bundle.output_schema)
    except jsonschema.ValidationError as e:
        raise VerificationError(
            step=3,
            reason=f"output_score_block schema validation failed: {e.message}",
        ) from e


def verify_through_step_3(
    receipt: EnhancedReceipt,
    bundle_bytes: bytes,
) -> EvaluatorBundle:
    """Run Steps 1–3 in spec order; return the verified bundle for Step 4+."""
    verify_step_1_receipt_integrity(receipt)
    bundle = verify_step_2_evaluator_bundle(receipt, bundle_bytes)
    verify_step_3_output_schema(receipt, bundle)
    return bundle
