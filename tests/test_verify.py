"""§7.1 Steps 1-3 verification."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from eerful.errors import VerificationError
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.verify import (
    verify_receipt,
    verify_step_1_receipt_integrity,
    verify_step_2_evaluator_bundle,
    verify_step_3_output_schema,
    verify_step_5_compose_hash_gating,
    verify_through_step_3,
)


def _build_report(
    *,
    app_compose: dict[str, Any] | None = None,
    compose_hash_override: str | None = None,
) -> tuple[bytes, str]:
    """Synthesize an attestation report; return `(bytes, compose_hash_lowercase)`.

    Mirrors `tests/test_attestation.py::_build_report` but exposes the
    attested compose-hash so step-5 tests can populate
    `accepted_compose_hashes` for hit/miss cases.
    """
    if app_compose is None:
        app_compose = {
            "docker_compose_file": (
                "services:\n  vllm:\n    image: vllm/vllm-openai:nightly\n"
                "    command: --model zai-org/GLM-5-FP8\n"
            ),
        }
    raw = json.dumps(app_compose, sort_keys=True)
    real_hash = hashlib.sha256(raw.encode()).hexdigest()
    declared = compose_hash_override or real_hash

    event_log: list[dict[str, Any]] = [
        {
            "imr": 3,
            "event_type": 134217729,
            "digest": "00" * 48,
            "event": "compose-hash",
            "event_payload": declared,
        }
    ]
    tcb = {
        "compose_hash": declared,
        "event_log": event_log,
        "app_compose": raw,
    }
    envelope = {
        "quote": "00",
        "event_log": json.dumps(event_log),
        "report_data": "",
        "vm_config": "{}",
        "tcb_info": json.dumps(tcb),
        "nvidia_payload": {},
    }
    return json.dumps(envelope).encode(), "0x" + declared.lower()

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


# ---------------- Step 5 (compose-hash subset) ----------------


def test_step_5_passes_when_allowlist_matches():
    """Bundle declares allowlist + report's compose-hash is in it →
    `gating='enforced'`, no error."""
    report_bytes, hash_hex = _build_report()
    b = _bundle(accepted_compose_hashes=[hash_hex])
    result = verify_step_5_compose_hash_gating(b, report_bytes)
    assert result.gating == "enforced"
    assert result.compose_hash == hash_hex
    assert result.category == "A"


def test_step_5_fails_when_allowlist_misses():
    """Bundle declares allowlist + report's compose-hash is NOT in it →
    Step 5 fails. This is the load-bearing §6.5 invariant: a publisher
    who lists known-good composes must reject everything else."""
    report_bytes, hash_hex = _build_report()
    other = "0x" + "f" * 64
    b = _bundle(accepted_compose_hashes=[other])
    with pytest.raises(VerificationError) as exc:
        verify_step_5_compose_hash_gating(b, report_bytes)
    assert exc.value.step == 5
    assert hash_hex in exc.value.reason
    assert "not in" in exc.value.reason


def test_step_5_skips_when_no_allowlist():
    """Bundle without allowlist → Step 5 reports `gating='skipped'` and does
    not raise. Spec §6.5: 'When absent, no compose-hash gating is performed.'
    A receipt is still verifiable; the §8 caveat applies."""
    report_bytes, hash_hex = _build_report()
    b = _bundle()  # accepted_compose_hashes defaults to None
    assert b.accepted_compose_hashes is None
    result = verify_step_5_compose_hash_gating(b, report_bytes)
    assert result.gating == "skipped"
    assert result.compose_hash == hash_hex
    assert result.category == "A"


def test_step_5_uppercase_allowlist_matches_lowercase_report():
    """If a bundle's allowlist is uppercase (only possible if construction
    bypasses the BeforeValidator — defensive test), step 5 still matches.
    The bundle validator forces lowercase, so this is belt-and-suspenders."""
    report_bytes, hash_hex = _build_report()
    # Bundle stores lowercase regardless of input case (tested in test_evaluator).
    b = _bundle(accepted_compose_hashes=[hash_hex.upper()])
    result = verify_step_5_compose_hash_gating(b, report_bytes)
    assert result.gating == "enforced"


def test_step_5_propagates_parse_errors():
    """Malformed report is a Step 5 failure regardless of allowlist status;
    the parser raises VerificationError(step=5) and Step 5 must not swallow
    it. Forces the RTMR3 mismatch path: keeps `tcb_info.compose_hash`
    aligned with `sha256(app_compose)` so the app_compose anchor passes,
    then diverges only the event log payload."""
    report_bytes, _ = _build_report()
    envelope = json.loads(report_bytes)
    tcb = json.loads(envelope["tcb_info"])
    for e in tcb["event_log"]:
        if e.get("event") == "compose-hash":
            e["event_payload"] = "cd" * 32  # diverge from tcb_info.compose_hash
    envelope["tcb_info"] = json.dumps(tcb)
    b = _bundle()
    with pytest.raises(VerificationError) as exc:
        verify_step_5_compose_hash_gating(b, json.dumps(envelope).encode())
    assert exc.value.step == 5
    assert "compose-hash mismatch" in exc.value.reason


# ---------------- end-to-end orchestrator ----------------


def test_verify_receipt_runs_full_chain_when_report_provided():
    report_bytes, hash_hex = _build_report()
    b = _bundle(accepted_compose_hashes=[hash_hex])
    r = _receipt(b)
    result = verify_receipt(r, b.canonical_bytes(), report_bytes)
    assert result.bundle.evaluator_id() == b.evaluator_id()
    assert result.step5 is not None
    assert result.step5.gating == "enforced"


def test_verify_receipt_skips_step5_when_report_omitted():
    b = _bundle()
    r = _receipt(b)
    result = verify_receipt(r, b.canonical_bytes(), None)
    assert result.step5 is None


def test_verify_receipt_short_circuits_at_step_2_before_step_5():
    """A bundle-bytes mismatch at Step 2 must surface as Step 2, not Step 5,
    even when a report is supplied — spec §7.1 ordering is normative."""
    report_bytes, hash_hex = _build_report()
    b = _bundle(accepted_compose_hashes=[hash_hex])
    r = _receipt(b)
    other = _bundle(system_prompt="something else")
    with pytest.raises(VerificationError) as exc:
        verify_receipt(r, other.canonical_bytes(), report_bytes)
    assert exc.value.step == 2
