"""§7.1 verification — Steps 1, 2, 3, 4, 5, 6 reference tests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pytest
from eth_keys import keys
from eth_utils import keccak

from eerful.errors import StorageError, TrustViolation, VerificationError
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.verify import (
    fetch_evaluator_bundle_bytes,
    verify_receipt,
    verify_receipt_with_storage,
    verify_step_1_receipt_integrity,
    verify_step_2_evaluator_bundle,
    verify_step_3_output_schema,
    verify_step_4_attestation_report,
    verify_step_5_compose_hash_gating,
    verify_step_6_enclave_signature,
    verify_through_step_3,
)
from eerful.zg.storage import MockStorageClient


_TEST_PRIVKEY = b"\x42" * 32


def _sign_personal(text: str, privkey_bytes: bytes = _TEST_PRIVKEY) -> tuple[str, str]:
    """EIP-191 personal_sign over `text`. Returns (pubkey_hex, signature_hex).

    Mirrors what the 0G TeeML provider's signature endpoint produces,
    minus the 27/28 v-shift (eth-keys emits v in {0, 1}; both forms are
    accepted by `recover_pubkey_from_personal_sign`)."""
    text_bytes = text.encode("utf-8")
    msg_hash = keccak(
        b"\x19Ethereum Signed Message:\n" + str(len(text_bytes)).encode() + text_bytes
    )
    pk = keys.PrivateKey(privkey_bytes)
    sig = pk.sign_msg_hash(msg_hash)
    return "0x" + pk.public_key.to_bytes().hex(), "0x" + sig.to_bytes().hex()


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
    """Build a receipt whose enclave signature is genuinely valid for
    `response_content` (so Step 6 passes by default). Tests exercising
    Step 6 failure modes override `enclave_signature` or `enclave_pubkey`
    explicitly."""
    response_content = overrides.get("response_content", "hello")
    pubkey, sig = _sign_personal(response_content)
    fields: dict[str, Any] = dict(
        created_at=CREATED,
        evaluator_id=bundle.evaluator_id(),
        evaluator_version=bundle.version,
        provider_address="0x" + "b" * 40,
        chat_id="chat-123",
        response_content=response_content,
        attestation_report_hash="0x" + "e" * 64,
        enclave_pubkey=pubkey,
        enclave_signature=sig,
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


# ---------------- Step 6 ----------------


def test_step_6_passes_for_valid_signature():
    verify_step_6_enclave_signature(_receipt(_bundle()))


def test_step_6_fails_when_signature_over_wrong_message():
    """Construct a receipt where the signature was computed over a
    different message than response_content — recovered pubkey will
    differ from enclave_pubkey."""
    pubkey, sig = _sign_personal("a different message")
    r = _receipt(
        _bundle(),
        response_content="this is the response that the receipt asserts",
        enclave_pubkey=pubkey,
        enclave_signature=sig,
    )
    with pytest.raises(VerificationError) as exc:
        verify_step_6_enclave_signature(r)
    assert exc.value.step == 6
    assert "does not match" in exc.value.reason


def test_step_6_fails_when_pubkey_belongs_to_different_key():
    """response_content was signed by privkey A, but receipt declares
    pubkey of privkey B."""
    _, sig_a = _sign_personal("hello", b"\x01" * 32)
    pubkey_b, _ = _sign_personal("hello", b"\x02" * 32)
    r = _receipt(
        _bundle(),
        response_content="hello",
        enclave_pubkey=pubkey_b,
        enclave_signature=sig_a,
    )
    with pytest.raises(VerificationError) as exc:
        verify_step_6_enclave_signature(r)
    assert exc.value.step == 6


def test_step_6_fails_when_signature_tampered():
    """Flip a byte in the signature; recovery still produces *some*
    pubkey, just not the right one."""
    pubkey, sig = _sign_personal("hello")
    # Flip one nibble in the middle (avoid touching the v byte at the end).
    flipped = sig[:-30] + ("e" if sig[-30] != "e" else "f") + sig[-29:]
    r = _receipt(
        _bundle(),
        response_content="hello",
        enclave_pubkey=pubkey,
        enclave_signature=flipped,
    )
    with pytest.raises(VerificationError) as exc:
        verify_step_6_enclave_signature(r)
    assert exc.value.step == 6


def test_step_6_wraps_malformed_signature_as_step_6_error():
    """A signature that's the wrong byte-length raises ValueError inside
    the recovery helper; the verifier must wrap it as a Step 6 failure
    so the spec-step attribution stays consistent."""
    r = _receipt(
        _bundle(),
        response_content="hello",
        enclave_signature="0x" + "ab" * 30,  # 30 bytes, not 65
    )
    with pytest.raises(VerificationError) as exc:
        verify_step_6_enclave_signature(r)
    assert exc.value.step == 6
    assert "could not be recovered" in exc.value.reason


def test_step_6_wraps_eth_keys_bad_signature_as_step_6_error():
    """A 65-byte signature that's structurally invalid (e.g. all-zero r/s)
    raises eth_keys.exceptions.BadSignature, not ValueError. Step 6 must
    still attribute the failure to itself, never let the underlying
    library exception escape."""
    # 64 bytes of zeros + v=0 — passes length check, fails secp256k1
    # recovery (s=0 is rejected; an all-zero signature doesn't lift to a
    # valid pubkey).
    r = _receipt(
        _bundle(),
        response_content="hello",
        enclave_signature="0x" + "00" * 65,
    )
    with pytest.raises(VerificationError) as exc:
        verify_step_6_enclave_signature(r)
    assert exc.value.step == 6
    assert "could not be recovered" in exc.value.reason


# ---------------- verify_receipt orchestrator ----------------


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


# ---------------- Step 4 + storage-aware orchestrator ----------------


def test_step_4_returns_report_bytes_when_storage_has_them():
    """Step 4: storage holds bytes whose sha256 equals
    receipt.attestation_report_hash → fetch returns those bytes."""
    storage = MockStorageClient()
    report_bytes, _ = _build_report()
    rh = storage.upload_blob(report_bytes)
    r = _receipt(_bundle(), attestation_report_hash=rh)
    out = verify_step_4_attestation_report(r, storage)
    assert out == report_bytes


def test_step_4_fails_when_storage_has_no_entry():
    """Step 4: nothing at the receipt's attestation_report_hash → storage
    raises StorageError, Step 4 wraps it as VerificationError(step=4)."""
    storage = MockStorageClient()  # empty
    r = _receipt(_bundle(), attestation_report_hash="0x" + "e" * 64)
    with pytest.raises(VerificationError) as exc:
        verify_step_4_attestation_report(r, storage)
    assert exc.value.step == 4
    assert "not retrievable" in exc.value.reason


def test_step_4_locally_rehashes_storage_bytes_defense_in_depth():
    """Defense in depth: if a `StorageClient` impl forgoes its own
    content check (legal under the Protocol — it's not a normative
    obligation), Step 4 still catches a mismatch by re-hashing locally.
    Spec §7.1 puts the hash check on the verifier."""

    class _UncheckedStorage:
        """A StorageClient that returns bytes hashing to a different
        value than the one requested. Our adapters check this, but the
        Protocol doesn't enforce it."""

        def upload_blob(self, data: bytes) -> str:
            raise NotImplementedError

        def download_blob(self, content_hash: str) -> bytes:
            return b"this hashes to something else entirely"

    r = _receipt(_bundle(), attestation_report_hash="0x" + "e" * 64)
    with pytest.raises(VerificationError) as exc:
        verify_step_4_attestation_report(r, _UncheckedStorage())
    assert exc.value.step == 4
    assert "content hash mismatch" in exc.value.reason
    assert "hashing to" in exc.value.reason


def test_step_4_fails_when_storage_returns_wrong_bytes_as_trust_violation():
    """Step 4: storage tampers with bytes (key no longer hashes to the
    requested hash) → adapter raises TrustViolation, Step 4 attributes it."""

    class _LyingStorage:
        def upload_blob(self, data: bytes) -> str:
            raise NotImplementedError

        def download_blob(self, content_hash: str) -> bytes:
            raise TrustViolation(
                f"download-blob hash mismatch: requested {content_hash}, "
                "received bytes hash to 0x" + "0" * 64
            )

    r = _receipt(_bundle(), attestation_report_hash="0x" + "e" * 64)
    with pytest.raises(VerificationError) as exc:
        verify_step_4_attestation_report(r, _LyingStorage())
    assert exc.value.step == 4
    assert "content hash mismatch" in exc.value.reason


def test_fetch_evaluator_bundle_bytes_returns_bytes_when_present():
    storage = MockStorageClient()
    b = _bundle()
    storage.upload_blob(b.canonical_bytes())
    r = _receipt(b)
    assert fetch_evaluator_bundle_bytes(r, storage) == b.canonical_bytes()


def test_fetch_evaluator_bundle_bytes_attributes_storage_miss_to_step_2():
    storage = MockStorageClient()  # empty
    r = _receipt(_bundle())
    with pytest.raises(VerificationError) as exc:
        fetch_evaluator_bundle_bytes(r, storage)
    assert exc.value.step == 2
    assert "not retrievable" in exc.value.reason


def test_fetch_evaluator_bundle_bytes_attributes_trust_violation_to_step_2():
    class _LyingStorage:
        def upload_blob(self, data: bytes) -> str:
            raise NotImplementedError

        def download_blob(self, content_hash: str) -> bytes:
            raise TrustViolation("byzantine")

    r = _receipt(_bundle())
    with pytest.raises(VerificationError) as exc:
        fetch_evaluator_bundle_bytes(r, _LyingStorage())
    assert exc.value.step == 2
    assert "content hash mismatch" in exc.value.reason


def test_verify_receipt_with_storage_passes_full_chain():
    """End-to-end: bundle + report fetched from storage, all spec steps
    pass, VerificationResult shows enforced gating."""
    storage = MockStorageClient()
    report_bytes, hash_hex = _build_report()
    b = _bundle(accepted_compose_hashes=[hash_hex])
    storage.upload_blob(b.canonical_bytes())
    rh = storage.upload_blob(report_bytes)
    r = _receipt(b, attestation_report_hash=rh)

    result = verify_receipt_with_storage(r, storage)
    assert result.bundle.evaluator_id() == b.evaluator_id()
    assert result.step5 is not None
    assert result.step5.gating == "enforced"


def test_verify_receipt_with_storage_skips_step_5_when_fetch_report_false():
    """Explicit offline-Step-5 mode: no report fetch, no Step 5, no error
    even if the report wouldn't be in storage."""
    storage = MockStorageClient()
    b = _bundle()
    storage.upload_blob(b.canonical_bytes())
    # NOTE: report is intentionally NOT uploaded.
    r = _receipt(b)
    result = verify_receipt_with_storage(r, storage, fetch_report=False)
    assert result.step5 is None


def test_verify_receipt_with_storage_short_circuits_at_step_2_before_step_4():
    """Spec §7.1 ordering is normative: Step 2 fails before Step 4 even
    if Step 4 would also fail. Bundle missing from storage → Step 2."""
    storage = MockStorageClient()  # bundle and report both missing
    b = _bundle()
    r = _receipt(b)
    with pytest.raises(VerificationError) as exc:
        verify_receipt_with_storage(r, storage)
    assert exc.value.step == 2


def test_verify_receipt_with_storage_step_4_runs_before_step_5():
    """Bundle present + report missing → Step 4 fails (not Step 5).
    Catches a regression where Step 4 attribution leaks to Step 5
    because Step 5 is what touches the report bytes."""
    storage = MockStorageClient()
    b = _bundle(accepted_compose_hashes=["0x" + "f" * 64])
    storage.upload_blob(b.canonical_bytes())
    # Report intentionally absent — Step 4 fetch will fail.
    r = _receipt(b)
    with pytest.raises(VerificationError) as exc:
        verify_receipt_with_storage(r, storage)
    assert exc.value.step == 4


def test_verify_receipt_with_storage_propagates_step_5_failure():
    """Storage holds the right bytes but the bundle's allowlist excludes
    the attested compose-hash → Step 5 fails through the storage path."""
    storage = MockStorageClient()
    report_bytes, _ = _build_report()
    b = _bundle(accepted_compose_hashes=["0x" + "f" * 64])  # not the attested hash
    storage.upload_blob(b.canonical_bytes())
    rh = storage.upload_blob(report_bytes)
    r = _receipt(b, attestation_report_hash=rh)
    with pytest.raises(VerificationError) as exc:
        verify_receipt_with_storage(r, storage)
    assert exc.value.step == 5


def test_verify_receipt_with_storage_step_4_fetches_attestation_report_hash_not_evaluator_id():
    """Defensive: Step 4 must look up by `attestation_report_hash`, not
    by `evaluator_id`. Catches a swapped-fetch-key regression that
    would silently pass when the bundle and report happened to live in
    the same store."""
    storage = MockStorageClient()
    b = _bundle()
    storage.upload_blob(b.canonical_bytes())
    # Report has a *different* sha256 than the bundle, so a lookup by
    # evaluator_id would return the bundle bytes, fail to parse as a
    # report at Step 5, and be misattributed.
    report_bytes, _ = _build_report()
    rh = storage.upload_blob(report_bytes)
    assert rh != b.evaluator_id()
    r = _receipt(b, attestation_report_hash=rh)
    out = verify_step_4_attestation_report(r, storage)
    assert out == report_bytes


def test_verify_receipt_with_storage_storage_error_at_step_4_uses_storage_subtype():
    """A non-mock storage that raises a real StorageError (not a
    TrustViolation) still attributes to Step 4. Covers the
    `StorageError` branch separately from the `TrustViolation` branch
    above."""

    class _FlakyStorage:
        def upload_blob(self, data: bytes) -> str:
            raise NotImplementedError

        def download_blob(self, content_hash: str) -> bytes:
            raise StorageError("bridge offline")

    r = _receipt(_bundle(), attestation_report_hash="0x" + "e" * 64)
    with pytest.raises(VerificationError) as exc:
        verify_step_4_attestation_report(r, _FlakyStorage())
    assert exc.value.step == 4
    assert "bridge offline" in exc.value.reason


def test_verify_receipt_runs_step_5_before_step_6():
    """When both Step 5 and Step 6 would fail, §7.1 ordering attributes
    the failure to Step 5. A receipt with a bogus signature AND a
    compose-hash that's not in the bundle's allowlist must surface as
    Step 5, not Step 6."""
    report_bytes, hash_hex = _build_report()
    # Allowlist contains some other hash, so Step 5 would fail.
    different_hash = "0x" + "f" * 64
    b = _bundle(accepted_compose_hashes=[different_hash])
    # Fresh receipt with default valid signature, then swap the
    # signature for a structurally-invalid one. Step 5 should fire
    # first; Step 6 never runs.
    r = _receipt(
        b,
        enclave_signature="0x" + "ab" * 30,  # would fail Step 6 if reached
    )
    with pytest.raises(VerificationError) as exc:
        verify_receipt(r, b.canonical_bytes(), report_bytes)
    assert exc.value.step == 5
    assert hash_hex in exc.value.reason
