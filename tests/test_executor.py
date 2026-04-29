"""Executor — `evaluate_gate` six-check sequence + helpers.

Tests cover:

- PASS path
- All 6 REFUSE outcomes
- `PolicyError` on missing tier / missing bundle_name (programming bug,
  not a refusal)
- `canonical_set_hash` order-independence + N=1 vs N=2 distinctness
- `tee_signer_address_from_pubkey` derivation correctness

Receipt-shape helpers (`_sign_personal`, `_build_report`) are inlined
rather than imported from `test_verify.py` — extracting to `tests/_fakes.py`
is the parked "consumer #3" follow-on (next_steps.md).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pytest
from eth_keys import keys
from eth_utils import keccak

from eerful.errors import PolicyError
from eerful.evaluator import ComposeHashEntry, EvaluatorBundle
from eerful.executor import (
    GateOutcome,
    canonical_set_hash,
    evaluate_gate,
    tee_signer_address_from_pubkey,
)
from eerful.policy import (
    POLICY_VERSION,
    DiversityRules,
    PrincipalPolicy,
    TierPolicy,
)
from eerful.receipt import EnhancedReceipt
from eerful.zg.storage import MockStorageClient


# ---------------- shared fixtures / helpers ----------------


_PROVIDER = "0x" + "b" * 40
_OTHER_PROVIDER = "0x" + "c" * 40
_ENTRY_PROVIDER = "0x" + "d" * 40

_PRIVKEY_A = b"\x42" * 32
_PRIVKEY_B = b"\x77" * 32

CREATED = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)


def _sign_personal(text: str, privkey_bytes: bytes = _PRIVKEY_A) -> tuple[str, str]:
    """EIP-191 personal_sign over `text` — same shape as the 0G TeeML
    provider's response signature. Returns (pubkey_hex, signature_hex)."""
    text_bytes = text.encode("utf-8")
    msg_hash = keccak(
        b"\x19Ethereum Signed Message:\n" + str(len(text_bytes)).encode() + text_bytes
    )
    pk = keys.PrivateKey(privkey_bytes)
    sig = pk.sign_msg_hash(msg_hash)
    return "0x" + pk.public_key.to_bytes().hex(), "0x" + sig.to_bytes().hex()


def _build_report(
    *,
    model_arg: str = "zai-org/GLM-5-FP8",
) -> tuple[bytes, str]:
    """Synthesize a parseable attestation report whose compose names a
    vLLM model; returns (bytes, compose_hash_lowercase). The
    heuristic categorizer will see this as Category A."""
    app_compose = {
        "docker_compose_file": (
            f"services:\n  vllm:\n    image: vllm/vllm-openai:nightly\n"
            f"    command: --model {model_arg}\n"
        ),
    }
    raw = json.dumps(app_compose, sort_keys=True)
    real_hash = hashlib.sha256(raw.encode()).hexdigest()
    event_log = [
        {
            "imr": 3,
            "event_type": 134217729,
            "digest": "00" * 48,
            "event": "compose-hash",
            "event_payload": real_hash,
        }
    ]
    tcb = {
        "compose_hash": real_hash,
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
    return json.dumps(envelope).encode(), "0x" + real_hash.lower()


_DEFAULT_SCORE: dict[str, Any] = {"overall": 0.8}
_UNSET: object = object()


def _make_receipt_and_storage(
    *,
    accepted_compose_hashes: list[ComposeHashEntry] | None = None,
    output_score_block: Any = _UNSET,
    privkey: bytes = _PRIVKEY_A,
    response_content: str = "ok",
    model_arg: str = "zai-org/GLM-5-FP8",
    provider_address: str = _PROVIDER,
) -> tuple[EnhancedReceipt, EvaluatorBundle, MockStorageClient]:
    """Build a single verifying receipt with bundle + report uploaded.

    `output_score_block` defaults to `_DEFAULT_SCORE` (`overall=0.8`) for
    PASS-path tests. A sentinel distinguishes "use default" from "explicitly
    None" — passing `None` produces a receipt with no score block (the
    REFUSE_SCORE-on-missing path needs this)."""
    bundle = EvaluatorBundle(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
        accepted_compose_hashes=accepted_compose_hashes,
    )
    storage = MockStorageClient()
    bundle_upload = storage.upload_blob(bundle.canonical_bytes())
    report_bytes, _ = _build_report(model_arg=model_arg)
    report_upload = storage.upload_blob(report_bytes)

    pubkey, sig = _sign_personal(response_content, privkey)
    score = _DEFAULT_SCORE if output_score_block is _UNSET else output_score_block

    receipt = EnhancedReceipt.build(
        created_at=CREATED,
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle_upload.storage_root,
        evaluator_version=bundle.version,
        provider_address=provider_address,
        chat_id="chat-123",
        response_content=response_content,
        attestation_report_hash=report_upload.content_hash,
        attestation_storage_root=report_upload.storage_root,
        enclave_pubkey=pubkey,
        enclave_signature=sig,
        output_score_block=score,
    )
    return receipt, bundle, storage


def _entry_for(hash_hex: str, *, category: str = "A") -> ComposeHashEntry:
    return ComposeHashEntry(
        hash=hash_hex,
        category=category,  # type: ignore[arg-type]
        provider_address=_ENTRY_PROVIDER,
    )


def _policy(
    *,
    bundle_id: str,
    n_attestations: int = 1,
    score_threshold: float = 0.6,
    required_categories: list[str] | None = None,
    diversity: DiversityRules | None = None,
) -> PrincipalPolicy:
    tier = TierPolicy(
        n_attestations=n_attestations,
        score_threshold=score_threshold,
        required_categories=required_categories,  # type: ignore[arg-type]
        diversity=diversity if diversity is not None else DiversityRules(),
    )
    return PrincipalPolicy(
        policy_version=POLICY_VERSION,
        principal_id="test",
        bundles={"proposal_grade": bundle_id},
        tiers={"low_consequence": tier},
    )


# ---------------- canonical_set_hash ----------------


def test_canonical_set_hash_is_deterministic_and_hex():
    r, _, _ = _make_receipt_and_storage()
    h = canonical_set_hash([r])
    assert h.startswith("0x")
    assert len(h) == 66
    assert h == canonical_set_hash([r])


def _make_distinct_receipts() -> tuple[EnhancedReceipt, EnhancedReceipt]:
    """Two receipts with distinct receipt_ids — distinct chat_ids feed
    different signing payloads, so the derived receipt_ids differ.
    Different signing keys alone wouldn't suffice: enclave_pubkey and
    enclave_signature aren't in `SIGNING_PAYLOAD_FIELDS`, so two
    receipts identical except for signature share the same receipt_id."""
    bundle = EvaluatorBundle(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
    )
    storage = MockStorageClient()
    bundle_upload = storage.upload_blob(bundle.canonical_bytes())
    report_bytes, _ = _build_report()
    report_upload = storage.upload_blob(report_bytes)
    pubkey, sig = _sign_personal("ok")
    common: dict[str, Any] = dict(
        created_at=CREATED,
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle_upload.storage_root,
        evaluator_version=bundle.version,
        provider_address=_PROVIDER,
        response_content="ok",
        attestation_report_hash=report_upload.content_hash,
        attestation_storage_root=report_upload.storage_root,
        enclave_pubkey=pubkey,
        enclave_signature=sig,
        output_score_block={"overall": 0.8},
    )
    r1 = EnhancedReceipt.build(chat_id="chat-1", **common)
    r2 = EnhancedReceipt.build(chat_id="chat-2", **common)
    return r1, r2


def test_canonical_set_hash_order_independent():
    """Set semantics: same receipts in any order yield the same hash.
    Without sort, two callers passing the same set in different orders
    would chain differently downstream."""
    r1, r2 = _make_distinct_receipts()
    assert canonical_set_hash([r1, r2]) == canonical_set_hash([r2, r1])


def test_canonical_set_hash_n1_differs_from_receipt_id():
    """N=1 set hash hashes the receipt_id once more — it's not equal to
    the receipt's own id. Locks down what the formula actually does, so
    a future change that special-cases N=1 surfaces as a test break."""
    r, _, _ = _make_receipt_and_storage()
    assert canonical_set_hash([r]) != r.receipt_id


def test_canonical_set_hash_changes_when_set_changes():
    r1, r2 = _make_distinct_receipts()
    assert canonical_set_hash([r1]) != canonical_set_hash([r1, r2])


def test_canonical_set_hash_dedupes_duplicates():
    """Set semantics: passing the same receipt twice must yield the
    same hash as passing it once. Without dedup, an accidental duplicate
    in the input list would shift the downstream chain anchor for what
    is semantically the same set."""
    r, _, _ = _make_receipt_and_storage()
    assert canonical_set_hash([r, r]) == canonical_set_hash([r])
    assert canonical_set_hash([r, r, r]) == canonical_set_hash([r])


# ---------------- tee_signer_address_from_pubkey ----------------


def test_tee_signer_address_matches_eth_keys_canonical_address():
    """The derived address must equal what eth_keys reports as the
    canonical address — without this property, the diversity rule
    compares ad-hoc strings instead of EVM addresses."""
    pk = keys.PrivateKey(_PRIVKEY_A)
    pubkey_hex = "0x" + pk.public_key.to_bytes().hex()
    expected = "0x" + pk.public_key.to_canonical_address().hex()
    assert tee_signer_address_from_pubkey(pubkey_hex) == expected


def test_tee_signer_address_distinguishes_distinct_keys():
    pk_a = keys.PrivateKey(_PRIVKEY_A)
    pk_b = keys.PrivateKey(_PRIVKEY_B)
    addr_a = tee_signer_address_from_pubkey("0x" + pk_a.public_key.to_bytes().hex())
    addr_b = tee_signer_address_from_pubkey("0x" + pk_b.public_key.to_bytes().hex())
    assert addr_a != addr_b


def test_tee_signer_address_rejects_wrong_length():
    """Pubkey is X||Y form, 64 bytes. A SEC1-prefixed (65-byte) or
    truncated input is malformed and must surface, not silently truncate."""
    with pytest.raises(ValueError):
        tee_signer_address_from_pubkey("0x" + "ab" * 32)  # 32 bytes, half


# ---------------- evaluate_gate: PolicyError ----------------


def test_evaluate_gate_raises_on_unknown_tier():
    r, bundle, storage = _make_receipt_and_storage()
    p = _policy(bundle_id=bundle.evaluator_id())
    with pytest.raises(PolicyError, match="tier 'high_consequence'"):
        evaluate_gate(
            policy=p,
            tier="high_consequence",
            bundle_name="proposal_grade",
            receipts=[r],
            storage=storage,
        )


def test_evaluate_gate_raises_on_unknown_bundle_name():
    r, bundle, storage = _make_receipt_and_storage()
    p = _policy(bundle_id=bundle.evaluator_id())
    with pytest.raises(PolicyError, match="bundle_name 'unknown'"):
        evaluate_gate(
            policy=p,
            tier="low_consequence",
            bundle_name="unknown",
            receipts=[r],
            storage=storage,
        )


# ---------------- evaluate_gate: PASS ----------------


def test_evaluate_gate_passes_minimum_viable_receipt():
    """Single receipt, no allowlist on bundle, no required_categories,
    score above threshold → PASS with canonical_set_hash set."""
    r, bundle, storage = _make_receipt_and_storage()
    p = _policy(bundle_id=bundle.evaluator_id())
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.PASS
    assert result.canonical_set_hash is not None
    assert result.canonical_set_hash.startswith("0x")
    assert result.receipts_supplied == 1
    assert result.receipts_required == 1


def test_evaluate_gate_passes_with_enforced_allowlist():
    """Bundle declares allowlist, receipt's compose hits it →
    Step 5 enforced and PASS."""
    _, bundle, storage = _make_receipt_and_storage()  # discard receipt; need allowlist version
    report_bytes, hash_hex = _build_report()
    bundle = EvaluatorBundle(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
        accepted_compose_hashes=[_entry_for(hash_hex, category="A")],
    )
    storage = MockStorageClient()
    bundle_upload = storage.upload_blob(bundle.canonical_bytes())
    report_upload = storage.upload_blob(report_bytes)
    pubkey, sig = _sign_personal("ok")
    receipt = EnhancedReceipt.build(
        created_at=CREATED,
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle_upload.storage_root,
        evaluator_version=bundle.version,
        provider_address=_PROVIDER,
        chat_id="chat-1",
        response_content="ok",
        attestation_report_hash=report_upload.content_hash,
        attestation_storage_root=report_upload.storage_root,
        enclave_pubkey=pubkey,
        enclave_signature=sig,
        output_score_block={"overall": 0.8},
    )
    p = _policy(bundle_id=bundle.evaluator_id(), required_categories=["A"])
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[receipt],
        storage=storage,
    )
    assert result.outcome == GateOutcome.PASS


# ---------------- evaluate_gate: REFUSE_INSUFFICIENT_RECEIPTS ----------------


def test_evaluate_gate_refuses_when_too_few_receipts():
    r, bundle, storage = _make_receipt_and_storage()
    p = _policy(bundle_id=bundle.evaluator_id(), n_attestations=4)
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_INSUFFICIENT_RECEIPTS
    assert result.canonical_set_hash is None
    assert result.receipts_supplied == 1
    assert result.receipts_required == 4
    assert "1 receipt" in result.detail
    assert "requires 4" in result.detail


def test_evaluate_gate_refuses_zero_receipts():
    """N=0 supplied is the trivial REFUSE_INSUFFICIENT case — still
    bumps against the n_attestations >= 1 floor."""
    _, bundle, storage = _make_receipt_and_storage()
    p = _policy(bundle_id=bundle.evaluator_id(), n_attestations=1)
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_INSUFFICIENT_RECEIPTS


def test_evaluate_gate_refuses_when_duplicate_receipts_pad_count():
    """Passing the same receipt N times must NOT satisfy an N=2
    attestation tier when diversity rules are off — the count check
    measures distinct receipt_ids, not raw input length. This is the
    receipt-level distinctness floor that sits below the diversity
    rules' provider-level distinctness."""
    r, bundle, storage = _make_receipt_and_storage()
    p = _policy(bundle_id=bundle.evaluator_id(), n_attestations=2)
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r, r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_INSUFFICIENT_RECEIPTS
    assert "1 distinct" in result.detail
    assert "2 receipt" in result.detail


# ---------------- evaluate_gate: REFUSE_BUNDLE_MISMATCH ----------------


def test_evaluate_gate_refuses_when_evaluator_id_does_not_match_policy():
    """Receipt names a different evaluator than the principal committed
    to → REFUSE_BUNDLE_MISMATCH. The principal's pre-commitment to bundle
    hashes is the load-bearing trust anchor for the gate."""
    r, _, storage = _make_receipt_and_storage()
    other_bundle_id = "0x" + "f" * 64
    p = _policy(bundle_id=other_bundle_id)
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_BUNDLE_MISMATCH
    assert other_bundle_id in result.detail
    assert r.evaluator_id in result.detail


# ---------------- evaluate_gate: REFUSE_INVALID_RECEIPT ----------------


def test_evaluate_gate_refuses_when_receipt_fails_step_4_storage_miss():
    """Storage missing the report → Step 4 fails → REFUSE_INVALID_RECEIPT.
    The detail surfaces the §7.1 step number for diagnostic purposes."""
    r, bundle, storage = _make_receipt_and_storage()
    # Drop the report from storage by building a fresh (empty) one and
    # uploading only the bundle.
    fresh_storage = MockStorageClient()
    fresh_storage.upload_blob(bundle.canonical_bytes())
    p = _policy(bundle_id=bundle.evaluator_id())
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=fresh_storage,
    )
    assert result.outcome == GateOutcome.REFUSE_INVALID_RECEIPT
    assert "Step 4" in result.detail


def test_evaluate_gate_refuses_when_storage_raises_value_error():
    """`BridgeStorageClient.download_blob` raises ValueError when a
    receipt's storage_root is malformed hex (Pydantic's BeforeValidator
    only lowercases — length isn't enforced at the field level). The
    gate must translate ValueError to REFUSE_INVALID_RECEIPT, not
    crash, so the rails fail closed even on malformed receipt input."""
    r, bundle, _ = _make_receipt_and_storage()

    class _ValidatingStorage:
        """Mimics BridgeStorageClient's hex validation — raises
        ValueError instead of returning bytes when given a malformed
        content_hash. Mock doesn't do this; bridge does."""

        def upload_blob(self, data: bytes) -> Any:
            raise NotImplementedError

        def download_blob(self, content_hash: str, storage_root: str) -> bytes:
            raise ValueError(
                f"content_hash must be 0x-prefixed 64-char lowercase hex, got {content_hash!r}"
            )

    p = _policy(bundle_id=bundle.evaluator_id())
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=_ValidatingStorage(),
    )
    assert result.outcome == GateOutcome.REFUSE_INVALID_RECEIPT
    assert "malformed" in result.detail
    assert r.receipt_id in result.detail


def test_evaluate_gate_refuses_when_receipt_fails_step_5_allowlist():
    """Bundle declares allowlist not matching the report's compose →
    Step 5 fails → REFUSE_INVALID_RECEIPT."""
    report_bytes, _ = _build_report()
    bundle = EvaluatorBundle(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
        accepted_compose_hashes=[_entry_for("0x" + "9" * 64)],
    )
    storage = MockStorageClient()
    bundle_upload = storage.upload_blob(bundle.canonical_bytes())
    report_upload = storage.upload_blob(report_bytes)
    pubkey, sig = _sign_personal("ok")
    receipt = EnhancedReceipt.build(
        created_at=CREATED,
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle_upload.storage_root,
        evaluator_version=bundle.version,
        provider_address=_PROVIDER,
        chat_id="chat-1",
        response_content="ok",
        attestation_report_hash=report_upload.content_hash,
        attestation_storage_root=report_upload.storage_root,
        enclave_pubkey=pubkey,
        enclave_signature=sig,
        output_score_block={"overall": 0.8},
    )
    p = _policy(bundle_id=bundle.evaluator_id())
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[receipt],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_INVALID_RECEIPT
    assert "Step 5" in result.detail


# ---------------- evaluate_gate: REFUSE_CATEGORY ----------------


def test_evaluate_gate_refuses_when_declared_category_not_in_required():
    """Bundle declares the entry as Category C; tier requires ['A'] →
    REFUSE_CATEGORY. This is the load-bearing path for the §6 supply-gap
    documentation: a publisher allowlisting a Cat C compose can still
    have the gate refuse if the principal demanded Cat A."""
    report_bytes, hash_hex = _build_report()
    bundle = EvaluatorBundle(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
        accepted_compose_hashes=[_entry_for(hash_hex, category="C")],
    )
    storage = MockStorageClient()
    bundle_upload = storage.upload_blob(bundle.canonical_bytes())
    report_upload = storage.upload_blob(report_bytes)
    pubkey, sig = _sign_personal("ok")
    receipt = EnhancedReceipt.build(
        created_at=CREATED,
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle_upload.storage_root,
        evaluator_version=bundle.version,
        provider_address=_PROVIDER,
        chat_id="chat-1",
        response_content="ok",
        attestation_report_hash=report_upload.content_hash,
        attestation_storage_root=report_upload.storage_root,
        enclave_pubkey=pubkey,
        enclave_signature=sig,
        output_score_block={"overall": 0.8},
    )
    p = _policy(bundle_id=bundle.evaluator_id(), required_categories=["A"])
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[receipt],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_CATEGORY
    assert "'C'" in result.detail
    assert "'A'" in result.detail


def test_evaluate_gate_refuses_when_tier_requires_categories_but_bundle_has_no_allowlist():
    """Tier demands category enforcement; bundle declared no allowlist →
    no declared category to check → REFUSE_CATEGORY. Covers the
    tier-vs-bundle mismatch case where the policy's expectations
    outrun what the bundle can prove."""
    r, bundle, storage = _make_receipt_and_storage()  # default: no allowlist
    p = _policy(bundle_id=bundle.evaluator_id(), required_categories=["A"])
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_CATEGORY
    assert "no declared compose-category" in result.detail


# ---------------- evaluate_gate: REFUSE_DIVERSITY ----------------


def _make_two_receipts_same_signer() -> tuple[
    EnhancedReceipt, EnhancedReceipt, EvaluatorBundle, MockStorageClient
]:
    """Two receipts signed by the SAME enclave key — same
    tee_signer_address. The diversity rule's negative test."""
    bundle = EvaluatorBundle(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
    )
    storage = MockStorageClient()
    bundle_upload = storage.upload_blob(bundle.canonical_bytes())
    report_bytes, _ = _build_report()
    report_upload = storage.upload_blob(report_bytes)

    pubkey, sig = _sign_personal("ok")
    common_kwargs: dict[str, Any] = dict(
        created_at=CREATED,
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle_upload.storage_root,
        evaluator_version=bundle.version,
        provider_address=_PROVIDER,
        response_content="ok",
        attestation_report_hash=report_upload.content_hash,
        attestation_storage_root=report_upload.storage_root,
        enclave_pubkey=pubkey,
        enclave_signature=sig,
        output_score_block={"overall": 0.8},
    )
    r1 = EnhancedReceipt.build(chat_id="chat-1", **common_kwargs)
    r2 = EnhancedReceipt.build(chat_id="chat-2", **common_kwargs)
    return r1, r2, bundle, storage


def test_evaluate_gate_refuses_when_distinct_signers_required_but_same_signer():
    """Two receipts from the same enclave key, distinct_signers=True →
    REFUSE_DIVERSITY. This is the rule that gives N>1 actual security."""
    r1, r2, bundle, storage = _make_two_receipts_same_signer()
    p = _policy(
        bundle_id=bundle.evaluator_id(),
        n_attestations=2,
        diversity=DiversityRules(distinct_signers=True),
    )
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r1, r2],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_DIVERSITY
    assert "distinct_signers" in result.detail


def test_evaluate_gate_passes_n2_when_signers_distinct():
    """Two receipts from different enclave keys, distinct_signers=True →
    PASS. Locks down the positive path of the diversity rule."""
    bundle = EvaluatorBundle(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
    )
    storage = MockStorageClient()
    bundle_upload = storage.upload_blob(bundle.canonical_bytes())
    report_bytes, _ = _build_report()
    report_upload = storage.upload_blob(report_bytes)

    response_content = "ok"
    pubkey_a, sig_a = _sign_personal(response_content, _PRIVKEY_A)
    pubkey_b, sig_b = _sign_personal(response_content, _PRIVKEY_B)

    common: dict[str, Any] = dict(
        created_at=CREATED,
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle_upload.storage_root,
        evaluator_version=bundle.version,
        provider_address=_PROVIDER,
        response_content=response_content,
        attestation_report_hash=report_upload.content_hash,
        attestation_storage_root=report_upload.storage_root,
        output_score_block={"overall": 0.8},
    )
    r1 = EnhancedReceipt.build(
        chat_id="chat-1", enclave_pubkey=pubkey_a, enclave_signature=sig_a, **common
    )
    r2 = EnhancedReceipt.build(
        chat_id="chat-2", enclave_pubkey=pubkey_b, enclave_signature=sig_b, **common
    )
    p = _policy(
        bundle_id=bundle.evaluator_id(),
        n_attestations=2,
        diversity=DiversityRules(distinct_signers=True),
    )
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r1, r2],
        storage=storage,
    )
    assert result.outcome == GateOutcome.PASS


def test_evaluate_gate_refuses_distinct_compose_when_compose_hashes_collide():
    """Two receipts both attesting the same compose-hash + tier requires
    distinct_compose_hashes → REFUSE_DIVERSITY on duplicate.

    The bundle has no allowlist here, but Step 5 still runs (with
    `gating="skipped"`) and `Step5Result.compose_hash` is populated —
    so the executor can compare compose-hashes across the receipt set
    even without an allowlist, and catches the collision."""
    r1, r2, bundle, storage = _make_two_receipts_same_signer()
    p = _policy(
        bundle_id=bundle.evaluator_id(),
        n_attestations=2,
        diversity=DiversityRules(distinct_compose_hashes=True),
    )
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r1, r2],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_DIVERSITY
    assert "distinct_compose_hashes" in result.detail
    assert "duplicate" in result.detail


# ---------------- evaluate_gate: REFUSE_SCORE ----------------


def test_evaluate_gate_refuses_when_overall_below_threshold():
    r, bundle, storage = _make_receipt_and_storage(
        output_score_block={"overall": 0.4}
    )
    p = _policy(bundle_id=bundle.evaluator_id(), score_threshold=0.6)
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_SCORE
    assert "0.4" in result.detail
    assert "0.6" in result.detail


def test_evaluate_gate_refuses_when_no_score_block():
    """all_must_pass requires a score block on every receipt; missing
    one is a refusal, not a pass-by-default."""
    r, bundle, storage = _make_receipt_and_storage(
        output_score_block=None
    )
    # _make_receipt_and_storage's default is overall=0.8; passing None
    # overrides to None. But Pydantic on EnhancedReceipt allows None
    # output_score_block, so the receipt builds.
    # Sanity: assert the receipt actually has no score block.
    assert r.output_score_block is None
    p = _policy(bundle_id=bundle.evaluator_id(), score_threshold=0.6)
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_SCORE
    assert "no output_score_block" in result.detail


def test_evaluate_gate_refuses_when_overall_is_non_numeric():
    r, bundle, storage = _make_receipt_and_storage(
        output_score_block={"overall": "high"},  # str, not numeric
    )
    p = _policy(bundle_id=bundle.evaluator_id(), score_threshold=0.6)
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_SCORE
    assert "not a numeric value" in result.detail


def test_evaluate_gate_refuses_when_overall_is_bool():
    """Python `True` is `int`; without explicit bool rejection,
    `overall=True` would pass the `isinstance(overall, (int, float))`
    check and compare True > 0.6 = True, silently passing."""
    r, bundle, storage = _make_receipt_and_storage(
        output_score_block={"overall": True},
    )
    p = _policy(bundle_id=bundle.evaluator_id(), score_threshold=0.6)
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.REFUSE_SCORE
    assert "not a numeric value" in result.detail


def test_evaluate_gate_passes_when_overall_equals_threshold():
    """Threshold inclusive: overall == threshold passes. Locks the
    boundary semantics."""
    r, bundle, storage = _make_receipt_and_storage(
        output_score_block={"overall": 0.6}
    )
    p = _policy(bundle_id=bundle.evaluator_id(), score_threshold=0.6)
    result = evaluate_gate(
        policy=p,
        tier="low_consequence",
        bundle_name="proposal_grade",
        receipts=[r],
        storage=storage,
    )
    assert result.outcome == GateOutcome.PASS
