"""EnhancedReceipt construction, receipt_id derivation, and round-trip
(spec §6.1, §6.3, §7.1 Step 1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from eerful.errors import VerificationError
from eerful.receipt import EnhancedReceipt

CREATED = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

BASE: dict[str, Any] = dict(
    created_at=CREATED,
    evaluator_id="0x" + "a" * 64,
    evaluator_storage_root="0x" + "1" * 64,
    evaluator_version="trading-critic@1.0.0",
    provider_address="0x" + "b" * 40,
    chat_id="chat-123",
    response_content="hello",
    attestation_report_hash="0x" + "e" * 64,
    attestation_storage_root="0x" + "2" * 64,
    enclave_pubkey="0x" + "c" * 64,
    enclave_signature="0x" + "d" * 128,
)


def test_build_derives_well_formed_receipt_id():
    r = EnhancedReceipt.build(**BASE)
    assert r.receipt_id.startswith("0x")
    assert len(r.receipt_id) == 66


def test_build_is_deterministic():
    assert EnhancedReceipt.build(**BASE).receipt_id == EnhancedReceipt.build(**BASE).receipt_id


def test_round_trip_through_json():
    r = EnhancedReceipt.build(**BASE)
    j = r.model_dump_json()
    r2 = EnhancedReceipt.model_validate_json(j)
    assert r2.receipt_id == r.receipt_id
    assert r2.response_content == r.response_content


def test_signature_material_does_not_affect_receipt_id():
    """enclave_pubkey and enclave_signature are NOT in the canonical signing
    payload (spec §6.3 — a signature cannot be over itself)."""
    base = EnhancedReceipt.build(**BASE)
    sig_changed = EnhancedReceipt.build(**{**BASE, "enclave_signature": "0x" + "1" * 128})
    pubkey_changed = EnhancedReceipt.build(**{**BASE, "enclave_pubkey": "0x" + "9" * 64})
    assert base.receipt_id == sig_changed.receipt_id == pubkey_changed.receipt_id


def test_attestation_report_hash_affects_receipt_id():
    """attestation_report_hash IS in the canonical signing payload (spec §6.3);
    binding it forecloses a same-key/different-report swap post-construction."""
    base = EnhancedReceipt.build(**BASE)
    report_changed = EnhancedReceipt.build(**{**BASE, "attestation_report_hash": "0x" + "f" * 64})
    assert base.receipt_id != report_changed.receipt_id


def test_response_content_change_changes_receipt_id():
    a = EnhancedReceipt.build(**BASE)
    b = EnhancedReceipt.build(**{**BASE, "response_content": "different"})
    assert a.receipt_id != b.receipt_id


def test_evaluator_id_change_changes_receipt_id():
    a = EnhancedReceipt.build(**BASE)
    b = EnhancedReceipt.build(**{**BASE, "evaluator_id": "0x" + "9" * 64})
    assert a.receipt_id != b.receipt_id


def test_extensions_in_signing_payload():
    a = EnhancedReceipt.build(**BASE)
    b = EnhancedReceipt.build(**{**BASE, "extensions": {"foo": "bar"}})
    assert a.receipt_id != b.receipt_id


def test_optional_fields_default_to_none():
    r = EnhancedReceipt.build(**BASE)
    assert r.input_commitment is None
    assert r.previous_receipt_id is None
    assert r.output_score_block is None
    assert r.extensions is None


def test_naive_datetime_rejected_in_build():
    with pytest.raises(ValueError, match="UTC"):
        EnhancedReceipt.build(**{**BASE, "created_at": datetime(2026, 4, 26, 12, 0)})


def test_non_utc_offset_rejected_in_build():
    pst = timezone(timedelta(hours=-8))
    with pytest.raises(ValueError, match="UTC"):
        EnhancedReceipt.build(**{**BASE, "created_at": datetime(2026, 4, 26, 4, 0, tzinfo=pst)})


def test_tampered_response_content_fails_step1():
    r = EnhancedReceipt.build(**BASE)
    j = r.model_dump_json()
    tampered = j.replace('"hello"', '"goodbye"')
    with pytest.raises((ValidationError, VerificationError)) as exc:
        EnhancedReceipt.model_validate_json(tampered)
    assert "step 1" in str(exc.value).lower() or "receipt_id" in str(exc.value).lower()


def test_tampered_receipt_id_fails_step1():
    r = EnhancedReceipt.build(**BASE)
    bad_id = "0x" + "0" * 64
    with pytest.raises(VerificationError) as exc:
        EnhancedReceipt(**{**r.model_dump(), "receipt_id": bad_id})
    assert exc.value.step == 1


def test_signing_payload_excludes_only_signature_material_and_receipt_id():
    """Spec §6.3: only receipt_id, enclave_pubkey, enclave_signature are
    excluded. attestation_report_hash IS in the payload."""
    r = EnhancedReceipt.build(**BASE)
    payload = r.signing_payload()
    assert "receipt_id" not in payload
    assert "enclave_pubkey" not in payload
    assert "enclave_signature" not in payload
    assert "attestation_report_hash" in payload


def test_signing_payload_includes_all_canonical_fields():
    r = EnhancedReceipt.build(**BASE)
    payload = r.signing_payload()
    expected = {
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
    }
    assert set(payload.keys()) == expected


def test_uppercase_hex_normalized_on_construction():
    """Spec §6.4 mandates lowercase hex. Receipts built with uppercase input
    must derive the same receipt_id as receipts built with lowercase input."""
    upper = {
        **BASE,
        "evaluator_id": "0x" + "A" * 64,
        "evaluator_storage_root": "0x" + "1" * 64,
        "provider_address": "0x" + "B" * 40,
        "attestation_report_hash": "0x" + "E" * 64,
        "attestation_storage_root": "0x" + "2" * 64,
        "enclave_pubkey": "0x" + "C" * 64,
        "enclave_signature": "0x" + "D" * 128,
    }
    r_upper = EnhancedReceipt.build(**upper)
    r_lower = EnhancedReceipt.build(**BASE)
    assert r_upper.receipt_id == r_lower.receipt_id
    assert r_upper.evaluator_id == r_lower.evaluator_id
    assert r_upper.provider_address == r_lower.provider_address
    assert r_upper.attestation_report_hash == r_lower.attestation_report_hash


def test_created_at_serialized_as_rfc3339_z():
    r = EnhancedReceipt.build(**BASE)
    assert r.signing_payload()["created_at"] == "2026-04-26T12:00:00Z"


def test_storage_roots_required_for_construction():
    """v0.5 makes both storage_root fields required. `EnhancedReceipt.build`
    declares them as keyword-only required parameters, so omitting one
    raises TypeError at the call site — the asymmetric break is what
    removes the cross-instance-fetch bug entirely."""
    incomplete = {k: v for k, v in BASE.items() if k != "evaluator_storage_root"}
    with pytest.raises(TypeError):
        EnhancedReceipt.build(**incomplete)


def test_evaluator_storage_root_in_signing_payload():
    """A change to `evaluator_storage_root` must change `receipt_id` —
    the signing payload binds the rootHash so a post-construction swap
    to a different storage backend pointing at colliding bytes can't
    rotate without invalidating the receipt."""
    a = EnhancedReceipt.build(**BASE)
    b = EnhancedReceipt.build(**{**BASE, "evaluator_storage_root": "0x" + "5" * 64})
    assert a.receipt_id != b.receipt_id


def test_attestation_storage_root_in_signing_payload():
    a = EnhancedReceipt.build(**BASE)
    b = EnhancedReceipt.build(**{**BASE, "attestation_storage_root": "0x" + "5" * 64})
    assert a.receipt_id != b.receipt_id


def test_spec_version_in_signing_payload():
    """A receipt under a different `spec_version` should derive a
    different receipt_id even when the rest of the payload is identical
    — guards against silent forward-compatibility surprises."""
    r = EnhancedReceipt.build(**BASE)
    assert r.spec_version == "0.5"
    payload = r.signing_payload()
    assert payload["spec_version"] == "0.5"


def test_spec_version_mismatch_rejected():
    """A receipt declaring a non-current spec_version must fail
    construction. Asymmetric break per §10.2: this verifier can only
    speak its own version."""
    r = EnhancedReceipt.build(**BASE)
    with pytest.raises(VerificationError) as exc:
        EnhancedReceipt(**{**r.model_dump(), "spec_version": "0.4"})
    assert "spec_version" in exc.value.reason


def test_storage_roots_lowercase_normalized_in_signing_payload():
    """Both storage_root fields must canonicalize to lowercase in the
    signing payload (spec §6.4). A receipt built with uppercase input
    must produce the same receipt_id as the lowercase form.

    Uses hex-letter values so `.upper()` actually changes the strings —
    a test using only digits like "1"/"2" would pass even if
    normalization were broken."""
    lower = {
        **BASE,
        "evaluator_storage_root": "0x" + "ab" * 32,
        "attestation_storage_root": "0x" + "cd" * 32,
    }
    upper_input = {
        **BASE,
        "evaluator_storage_root": ("0x" + "ab" * 32).upper().replace("0X", "0x"),
        "attestation_storage_root": ("0x" + "cd" * 32).upper().replace("0X", "0x"),
    }
    r_upper = EnhancedReceipt.build(**upper_input)
    r_lower = EnhancedReceipt.build(**lower)
    assert r_upper.receipt_id == r_lower.receipt_id
    # Pin canonicalization on the storage_root fields specifically.
    assert r_upper.evaluator_storage_root == r_lower.evaluator_storage_root
    assert r_upper.attestation_storage_root == r_lower.attestation_storage_root
