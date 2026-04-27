"""§7.1 verification algorithm.

The functions here are I/O-free: callers fetch the bundle bytes and (for
Step 5) the attestation report bytes and pass them in. Each step is a
separately testable function; an orchestrator runs them in spec order.

Coverage as of v0.4 reference impl:

- Steps 1–3 land in v0.4 (offline integrity + bundle binding + schema).
- Step 5's compose-hash subset (§6.5 allowlist + §8.2 category diagnostic)
  is implemented here. The TDX chain, NVIDIA GPU attestation, and pubkey
  binding subsets of Step 5 are deferred to the dstack-verifier
  integration in Track B follow-up; they are NOT yet enforced.
- Step 6 (enclave signature) is implemented: recovers the secp256k1
  pubkey from `enclave_signature` over `response_content` via EIP-191
  personal_sign and confirms it equals `enclave_pubkey`.
- Step 4 (fetch the attestation report from Storage) is the caller's
  responsibility for now; on Day 3 it will be wired to a 0G Storage
  client and added here.

Failures raise `VerificationError(step=N, reason=...)`.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, ValidationError

from eerful.canonical import Bytes32Hex
from eerful.errors import VerificationError
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt, derive_receipt_id
from eerful.zg.attestation import (
    ComposeCategory,
    ParsedAttestationReport,
    categorize_compose,
    parse_attestation_report,
)
from eerful.zg.compute import recover_pubkey_from_personal_sign

ComposeHashGating = Literal["enforced", "skipped"]
"""Whether Step 5's compose-hash gate ran. `enforced` means the bundle
declared `accepted_compose_hashes` and the attested compose-hash was in
the list; `skipped` means the bundle did not declare an allowlist (per
§6.5, no gating is performed in that case)."""


class Step5Result(BaseModel):
    """What Step 5's compose-hash subset establishes.

    `gating` reflects the §6.5 binary: `enforced` only when the bundle
    populated `accepted_compose_hashes` AND the attested hash was in it;
    `skipped` otherwise. `category` is the §8.2 diagnostic — informational,
    not a gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    compose_hash: Bytes32Hex
    gating: ComposeHashGating
    category: ComposeCategory


class VerificationResult(BaseModel):
    """Aggregated outcome of a successful (Steps 1–3 + 5-compose) run.

    The CLI uses this to render the verdict; downstream consumers (jig,
    higher-layer reputation systems) take it as the verified handle on the
    receipt. Construction implies all enforced steps passed; failure paths
    raise `VerificationError` instead of returning a result with a flag."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle: EvaluatorBundle
    step5: Step5Result | None = None
    """`None` when the caller didn't pass an attestation report (e.g. running
    Steps 1–3 only). Present once Step 5's compose-hash subset has run."""


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
    except jsonschema.SchemaError as e:
        raise VerificationError(
            step=3,
            reason=f"invalid output_schema in evaluator bundle: {e.message}",
        ) from e
    except jsonschema.ValidationError as e:
        raise VerificationError(
            step=3,
            reason=f"output_score_block schema validation failed: {e.message}",
        ) from e


def verify_step_5_compose_hash_gating(
    bundle: EvaluatorBundle,
    report_bytes: bytes,
) -> Step5Result:
    """Step 5 (compose-hash subset): enforce §6.5's `accepted_compose_hashes`.

    Parses the attestation report, extracts the compose-hash that RTMR3
    binds (cross-checked against the dstack event log), and:

    - If the bundle declares `accepted_compose_hashes`, fails when the
      attested hash is not in the list (`gating="enforced"` on success).
    - If the bundle does not declare it, skips gating per §6.5 ("no
      compose-hash gating is performed") and returns `gating="skipped"`.

    The §8.2 category is computed either way as a diagnostic.

    This function does NOT yet perform the rest of Step 5 (TDX quote chain
    against Intel roots, NVIDIA GPU attestation, pubkey-to-receipt
    binding); those are deferred to the dstack-verifier integration. A
    receipt that passes the compose-hash gate has not been fully Step-5
    verified — see this module's docstring.
    """
    parsed: ParsedAttestationReport = parse_attestation_report(report_bytes)
    category = categorize_compose(
        parsed,
        expected_model_identifier=bundle.model_identifier,
    )

    allowlist = bundle.accepted_compose_hashes
    if allowlist is None:
        return Step5Result(
            compose_hash=parsed.compose_hash,
            gating="skipped",
            category=category,
        )
    if parsed.compose_hash not in allowlist:
        raise VerificationError(
            step=5,
            reason=(
                f"attested compose-hash {parsed.compose_hash} is not in the "
                f"evaluator bundle's accepted_compose_hashes (size {len(allowlist)})"
            ),
        )
    return Step5Result(
        compose_hash=parsed.compose_hash,
        gating="enforced",
        category=category,
    )


def verify_step_6_enclave_signature(receipt: EnhancedReceipt) -> None:
    """Step 6: enclave_signature is a valid EIP-191 personal_sign over
    response_content under enclave_pubkey.

    The 0G TeeML provider signs response bodies via EIP-191
    `personal_sign` (`keccak256("\\x19Ethereum Signed Message:\\n" + len + msg)`).
    Step 6 recovers the secp256k1 pubkey from `(response_content, signature)`
    and confirms it matches `receipt.enclave_pubkey` byte-for-byte after
    canonicalization. Equivalent to "the signature is valid" because
    secp256k1 recovery is unambiguous for a well-formed (r, s, v) triple.
    """
    try:
        recovered_pubkey, _recovered_address = recover_pubkey_from_personal_sign(
            receipt.response_content,
            receipt.enclave_signature,
        )
    except (ValueError, TypeError) as e:
        raise VerificationError(
            step=6,
            reason=f"enclave_signature could not be recovered: {e}",
        ) from e
    if recovered_pubkey != receipt.enclave_pubkey:
        raise VerificationError(
            step=6,
            reason=(
                f"recovered pubkey {recovered_pubkey} does not match "
                f"enclave_pubkey {receipt.enclave_pubkey}"
            ),
        )


def verify_through_step_3(
    receipt: EnhancedReceipt,
    bundle_bytes: bytes,
) -> EvaluatorBundle:
    """Run Steps 1–3 in spec order; return the verified bundle for Step 4+."""
    verify_step_1_receipt_integrity(receipt)
    bundle = verify_step_2_evaluator_bundle(receipt, bundle_bytes)
    verify_step_3_output_schema(receipt, bundle)
    return bundle


def verify_receipt(
    receipt: EnhancedReceipt,
    bundle_bytes: bytes,
    report_bytes: bytes | None = None,
) -> VerificationResult:
    """Run all currently-implemented verification steps in spec order.

    Steps 1–3 always run, then Step 6 (enclave signature). Step 5's
    compose-hash subset runs when `report_bytes` is provided (Step 4 —
    fetching the report — is the caller's responsibility until 0G Storage
    integration lands).

    Returns the aggregated `VerificationResult` on success. On failure,
    raises `VerificationError(step=N, ...)` from the first step that
    fails; later steps don't run.
    """
    bundle = verify_through_step_3(receipt, bundle_bytes)
    verify_step_6_enclave_signature(receipt)
    if report_bytes is None:
        return VerificationResult(bundle=bundle)
    step5 = verify_step_5_compose_hash_gating(bundle, report_bytes)
    return VerificationResult(bundle=bundle, step5=step5)
