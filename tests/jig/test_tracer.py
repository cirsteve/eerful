"""`attach_receipt_to_span` — span metadata decoration helper."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jig.core.types import Span, SpanKind

from eerful.jig import attach_receipt_to_span
from eerful.receipt import EnhancedReceipt


def _span(metadata: dict[str, Any] | None = None) -> Span:
    return Span(
        id="span-1",
        trace_id="trace-1",
        kind=SpanKind.LLM_CALL,
        name="test",
        started_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        metadata=metadata,
    )


def _make_receipt() -> EnhancedReceipt:
    return EnhancedReceipt.build(
        created_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        evaluator_id="0x" + "ab" * 32,
        evaluator_version="v1",
        provider_address="0x" + "b" * 40,
        chat_id="chat-1",
        response_content="hi",
        attestation_report_hash="0x" + "cd" * 32,
        # Pubkey/sig don't need to be valid here; we're not running Step 6
        # — just exercising metadata write-through.
        enclave_pubkey="0x" + "11" * 64,
        enclave_signature="0x" + "22" * 65,
    )


def test_attach_receipt_to_span_adds_three_namespaced_keys() -> None:
    span = _span()
    receipt = _make_receipt()
    attach_receipt_to_span(span, receipt)
    assert span.metadata == {
        "eerful.receipt_id": receipt.receipt_id,
        "eerful.evaluator_id": receipt.evaluator_id,
        "eerful.attestation_report_hash": receipt.attestation_report_hash,
    }


def test_attach_receipt_to_span_preserves_existing_metadata() -> None:
    """Pre-existing metadata keys must survive the merge — instrumentation
    from other layers (request_id, user_id, etc.) shouldn't get clobbered."""
    span = _span(metadata={"request_id": "req-42", "user_id": "u-7"})
    receipt = _make_receipt()
    attach_receipt_to_span(span, receipt)
    assert span.metadata is not None
    assert span.metadata["request_id"] == "req-42"
    assert span.metadata["user_id"] == "u-7"
    assert span.metadata["eerful.receipt_id"] == receipt.receipt_id


def test_attach_receipt_to_span_handles_none_metadata() -> None:
    """A fresh span has metadata=None (default). The helper must
    initialize it, not raise on the dict merge."""
    span = _span(metadata=None)
    assert span.metadata is None
    attach_receipt_to_span(span, _make_receipt())
    assert span.metadata is not None
    assert "eerful.receipt_id" in span.metadata


def test_attach_receipt_to_span_overwrites_existing_eerful_keys() -> None:
    """If a span was previously decorated with stale eerful.* metadata
    (re-use, replay), the new receipt's values win — the spec semantic
    is "this span's receipt is X," not "any receipt this span has ever
    seen." Last write wins for our namespace."""
    span = _span(metadata={"eerful.receipt_id": "0x" + "00" * 32})
    receipt = _make_receipt()
    attach_receipt_to_span(span, receipt)
    assert span.metadata is not None
    assert span.metadata["eerful.receipt_id"] == receipt.receipt_id
    assert span.metadata["eerful.receipt_id"] != "0x" + "00" * 32
