"""§7.1 Steps 1-3 verification."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pytest

from eerful.errors import VerificationError
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.verify import (
    verify_step_1_receipt_integrity,
    verify_step_2_evaluator_bundle,
    verify_step_3_output_schema,
    verify_through_step_3,
)

CREATED = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["score"],
    "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}},
}


def _bundle(**overrides: Any) -> EvaluatorBundle:
    fields: dict[str, Any] = dict(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
    )
    fields.update(overrides)
    return EvaluatorBundle(**fields)


def _receipt(bundle: EvaluatorBundle, **overrides: Any) -> EnhancedReceipt:
    fields: dict[str, Any] = dict(
        created_at=CREATED,
        evaluator_id=bundle.evaluator_id(),
        evaluator_version=bundle.version,
        provider_address="0x" + "b" * 40,
        chat_id="chat-123",
        response_content="hello",
        attestation_report_hash="0x" + "e" * 64,
        enclave_pubkey="0x" + "c" * 64,
        enclave_signature="0x" + "d" * 128,
    )
    fields.update(overrides)
    return EnhancedReceipt.build(**fields)


# ---------------- Step 1 ----------------


def test_step_1_passes_for_valid_receipt():
    verify_step_1_receipt_integrity(_receipt(_bundle()))


def test_step_1_fails_when_receipt_id_tampered():
    r = _receipt(_bundle())
    bad = EnhancedReceipt.model_construct(**{**r.model_dump(), "receipt_id": "0x" + "0" * 64})
    with pytest.raises(VerificationError) as exc:
        verify_step_1_receipt_integrity(bad)
    assert exc.value.step == 1


# ---------------- Step 2 ----------------


def test_step_2_passes_for_matching_bundle():
    b = _bundle()
    r = _receipt(b)
    out = verify_step_2_evaluator_bundle(r, b.canonical_bytes())
    assert out.evaluator_id() == b.evaluator_id()


def test_step_2_fails_when_hash_mismatch():
    b = _bundle()
    r = _receipt(b)
    other = _bundle(system_prompt="different criteria")
    with pytest.raises(VerificationError) as exc:
        verify_step_2_evaluator_bundle(r, other.canonical_bytes())
    assert exc.value.step == 2
    assert "evaluator_id mismatch" in exc.value.reason


def test_step_2_fails_when_bundle_malformed():
    """Receipt's evaluator_id matches the bytes' hash, so the hash check
    passes and we exercise the JSON parse / pydantic-validation branch
    rather than re-testing the hash mismatch path."""
    malformed = b"not valid json"
    digest = "0x" + hashlib.sha256(malformed).hexdigest()
    r = _receipt(_bundle(), evaluator_id=digest)
    with pytest.raises(VerificationError) as exc:
        verify_step_2_evaluator_bundle(r, malformed)
    assert exc.value.step == 2
    assert "deserialization" in exc.value.reason


# ---------------- Step 3 ----------------


def test_step_3_passes_when_no_schema_no_score():
    b = _bundle()
    verify_step_3_output_schema(_receipt(b), b)


def test_step_3_passes_when_schema_but_no_score():
    b = _bundle(output_schema=SCHEMA)
    verify_step_3_output_schema(_receipt(b), b)


def test_step_3_passes_when_score_but_no_schema():
    b = _bundle()
    verify_step_3_output_schema(_receipt(b, output_score_block={"score": 0.7}), b)


def test_step_3_passes_when_score_validates():
    b = _bundle(output_schema=SCHEMA)
    verify_step_3_output_schema(_receipt(b, output_score_block={"score": 0.7}), b)


def test_step_3_fails_when_score_out_of_range():
    b = _bundle(output_schema=SCHEMA)
    with pytest.raises(VerificationError) as exc:
        verify_step_3_output_schema(_receipt(b, output_score_block={"score": 1.7}), b)
    assert exc.value.step == 3


def test_step_3_fails_when_score_missing_required():
    b = _bundle(output_schema=SCHEMA)
    with pytest.raises(VerificationError) as exc:
        verify_step_3_output_schema(_receipt(b, output_score_block={"unrelated": 1}), b)
    assert exc.value.step == 3


def test_step_3_wraps_schema_error_when_output_schema_invalid():
    """A malformed `output_schema` raises jsonschema.SchemaError; Step 3 must
    wrap it as VerificationError(step=3) per the module's contract that all
    failures surface through VerificationError, not the underlying library."""
    bad_schema: dict[str, Any] = {"type": "object", "required": "score"}  # required must be a list
    b = _bundle(output_schema=bad_schema)
    with pytest.raises(VerificationError) as exc:
        verify_step_3_output_schema(_receipt(b, output_score_block={"score": 0.5}), b)
    assert exc.value.step == 3
    assert "invalid output_schema" in exc.value.reason


# ---------------- Orchestration ----------------


def test_through_step_3_passes_full_chain():
    b = _bundle(output_schema=SCHEMA)
    r = _receipt(b, output_score_block={"score": 0.5})
    out = verify_through_step_3(r, b.canonical_bytes())
    assert out.evaluator_id() == b.evaluator_id()


def test_through_step_3_short_circuits_at_step_2():
    b = _bundle(output_schema=SCHEMA)
    r = _receipt(b, output_score_block={"score": 1.7})
    other = _bundle(system_prompt="different")
    with pytest.raises(VerificationError) as exc:
        verify_through_step_3(r, other.canonical_bytes())
    assert exc.value.step == 2
