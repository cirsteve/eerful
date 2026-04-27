"""EnhancedReceipt construction, receipt_id derivation, and round-trip
(spec §6.1, §6.3, §7.1 Step 1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eerful.errors import VerificationError
from eerful.receipt import EnhancedReceipt

CREATED = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

BASE = dict(
    created_at=CREATED,
    evaluator_id="0x" + "a" * 64,
    evaluator_version="trading-critic@1.0.0",
    provider_address="0x" + "b" * 40,
    chat_id="chat-123",
    response_content="hello",
    attestation_report_hash="0x" + "e" * 64,
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


def test_attestation_block_does_not_affect_receipt_id():
    """attestation_report_hash, enclave_pubkey, enclave_signature are NOT
    part of the canonical signing payload (spec §6.3)."""
    a = EnhancedReceipt.build(**{**BASE, "enclave_signature": "0x" + "1" * 128})
    b = EnhancedReceipt.build(**{**BASE, "enclave_signature": "0x" + "2" * 128})
    c = EnhancedReceipt.build(**{**BASE, "attestation_report_hash": "0x" + "f" * 64})
    d = EnhancedReceipt.build(**{**BASE, "enclave_pubkey": "0x" + "9" * 64})
    assert a.receipt_id == b.receipt_id == c.receipt_id == d.receipt_id


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
    with pytest.raises(Exception) as exc:
        EnhancedReceipt.model_validate_json(tampered)
    assert "step 1" in str(exc.value).lower() or "receipt_id" in str(exc.value).lower()


def test_tampered_receipt_id_fails_step1():
    r = EnhancedReceipt.build(**BASE)
    bad_id = "0x" + "0" * 64
    with pytest.raises(VerificationError) as exc:
        EnhancedReceipt(**{**r.model_dump(), "receipt_id": bad_id})
    assert exc.value.step == 1


def test_signing_payload_excludes_attestation_block():
    r = EnhancedReceipt.build(**BASE)
    payload = r.signing_payload()
    assert "attestation_report_hash" not in payload
    assert "enclave_pubkey" not in payload
    assert "enclave_signature" not in payload
    assert "receipt_id" not in payload


def test_signing_payload_includes_all_producer_fields():
    r = EnhancedReceipt.build(**BASE)
    payload = r.signing_payload()
    expected = {
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
    }
    assert set(payload.keys()) == expected


def test_created_at_serialized_as_rfc3339_z():
    r = EnhancedReceipt.build(**BASE)
    assert r.signing_payload()["created_at"] == "2026-04-26T12:00:00Z"
