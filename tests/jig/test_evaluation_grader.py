"""`EvaluationGrader` — produces scores + persists receipts via FeedbackLoop.

Tests cover:
- Score extraction: one Score per top-level numeric field; strings,
  bools, dicts, lists ignored.
- `score_dimensions` filter scoping.
- Feedback persistence carries receipt identifiers in metadata.
- Tracer integration: LLM_CALL span opened with receipt metadata when
  `_tracer` + `_span_id` are in context; no-op without them.
- Empty / None score block surfaces as empty `[Score]` rather than raising.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from jig.core.types import ScoreSource, SpanKind

from eerful.evaluator import EvaluatorBundle
from eerful.jig import EvaluationClient, EvaluationGrader
from eerful.zg.storage import MockStorageClient

from tests.jig.conftest import FakeComputeClient, RecordingFeedback, RecordingTracer


# ---------------- helper: client constructor ----------------


def _make_client(
    bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> EvaluationClient:
    return EvaluationClient(
        compute=fake_compute,
        storage=MockStorageClient(),
        bundle=bundle,
        evaluator_id=bundle.evaluator_id(),
        provider_address="0x" + "b" * 40,
    )


# ---------------- score extraction ----------------


@pytest.mark.asyncio
async def test_grade_returns_score_per_numeric_field(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Bundle's response yields {risk: 0.4, novelty: 0.7, commentary: "..."}.
    Grader returns 2 scores (risk, novelty); commentary is a string."""
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client)
    scores = await grader.grade(input="strategy v1", output="some output")
    dims = {s.dimension for s in scores}
    assert dims == {"risk", "novelty"}
    for s in scores:
        assert s.source == ScoreSource.LLM_JUDGE


@pytest.mark.asyncio
async def test_grade_filters_by_score_dimensions(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Caller restricts to specific dimensions — others are dropped."""
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client, score_dimensions=["risk"])
    scores = await grader.grade(input="x", output="y")
    assert [s.dimension for s in scores] == ["risk"]


@pytest.mark.asyncio
async def test_grade_ignores_bool_values(
    trading_critic_bundle: EvaluatorBundle,
) -> None:
    """`bool` is an `int` subclass; accepting `{"valid": True}` as a 1.0
    score is almost never what the bundle author wanted. Drop them."""
    fake = FakeComputeClient(
        response_content=json.dumps({"score": 0.5, "valid": True})
    )
    client = _make_client(trading_critic_bundle, fake)
    grader = EvaluationGrader(client=client)
    scores = await grader.grade(input="x", output="y")
    assert {s.dimension for s in scores} == {"score"}


@pytest.mark.asyncio
async def test_grade_returns_empty_when_score_block_is_none(
    trading_critic_bundle: EvaluatorBundle,
) -> None:
    """Malformed (non-JSON) response → receipt.output_score_block is
    None → grader returns []. The receipt still got built; we just
    have nothing to score against."""
    fake = FakeComputeClient(response_content="this is not json")
    client = _make_client(trading_critic_bundle, fake)
    grader = EvaluationGrader(client=client)
    scores = await grader.grade(input="x", output="y")
    assert scores == []


@pytest.mark.asyncio
async def test_grade_handles_nested_dicts_and_lists(
    trading_critic_bundle: EvaluatorBundle,
) -> None:
    """Top-level scoring only — nested objects don't recurse. Avoids
    silently giving meaning to producer-internal structure."""
    fake = FakeComputeClient(
        response_content=json.dumps(
            {
                "risk": 0.4,
                "details": {"sub": 0.9},
                "tags": [1, 2, 3],
            }
        )
    )
    client = _make_client(trading_critic_bundle, fake)
    grader = EvaluationGrader(client=client)
    scores = await grader.grade(input="x", output="y")
    assert [s.dimension for s in scores] == ["risk"]


# ---------------- feedback persistence ----------------


@pytest.mark.asyncio
async def test_grade_persists_receipt_metadata_to_feedback(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """When a feedback loop is configured, the grader writes a stored
    result whose metadata carries the receipt's three identifiers and
    a tags list including the receipt_id."""
    feedback = RecordingFeedback()
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client, feedback=feedback)
    scores = await grader.grade(input="strategy v1", output="raw output")

    assert len(feedback.stored) == 1
    stored = feedback.stored[0]
    md = stored["metadata"]
    assert md is not None
    assert "eer_receipt_id" in md
    assert "eer_evaluator_id" in md
    assert "eer_attestation_report_hash" in md
    assert md["eer_receipt_id"] in md["tags"]
    assert "eer" in md["tags"]
    # Scores got written under the same result_id.
    assert len(feedback.scored) == 1
    assert feedback.scored[0][0] == stored["result_id"]
    assert feedback.scored[0][1] == scores


@pytest.mark.asyncio
async def test_grade_skips_feedback_when_scores_empty(
    trading_critic_bundle: EvaluatorBundle,
) -> None:
    """Empty scores → no `store_result`, no `score`. Skipping prevents
    junk rows for malformed-response cases."""
    feedback = RecordingFeedback()
    fake = FakeComputeClient(response_content="not json")
    client = _make_client(trading_critic_bundle, fake)
    grader = EvaluationGrader(client=client, feedback=feedback)
    scores = await grader.grade(input="x", output="y")
    assert scores == []
    assert feedback.stored == []
    assert feedback.scored == []


@pytest.mark.asyncio
async def test_grade_no_feedback_returns_scores_unchanged(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """No FeedbackLoop wired → scores still flow back. The feedback
    integration is opt-in, not required for the grader to function."""
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client)
    scores = await grader.grade(input="x", output="y")
    assert len(scores) == 2


# ---------------- tracer integration ----------------


@pytest.mark.asyncio
async def test_grade_attaches_receipt_metadata_to_llm_call_span(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """With _tracer + _span_id in context, grader opens an LLM_CALL
    child span carrying the receipt's identifiers."""
    tracer = RecordingTracer()
    parent = tracer.start_trace("test", kind=SpanKind.PIPELINE_RUN)

    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client)
    await grader.grade(
        input="x",
        output="y",
        context={"_tracer": tracer, "_span_id": parent.id},
    )

    llm_spans = [s for s in tracer.spans.values() if s.kind == SpanKind.LLM_CALL]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.parent_id == parent.id
    assert span.metadata is not None
    assert "eerful.receipt_id" in span.metadata
    assert "eerful.evaluator_id" in span.metadata
    assert "eerful.attestation_report_hash" in span.metadata
    # end_span propagated the score block as output and the usage struct.
    assert span.output is not None
    assert span.usage is not None
    assert span.usage.input_tokens == 42


@pytest.mark.asyncio
async def test_grade_no_tracer_in_context_does_not_raise(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Grader called without a tracer (e.g. direct .grade() outside a
    pipeline) skips span integration silently."""
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client)
    scores = await grader.grade(input="x", output="y", context=None)
    assert len(scores) == 2


@pytest.mark.asyncio
async def test_grade_partial_context_still_skips_tracer(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Half-set context (tracer but no span id, or vice-versa) means
    we don't have what we need to open a child span — skip silently."""
    tracer = RecordingTracer()
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client)
    # Only _tracer, no _span_id.
    await grader.grade(input="x", output="y", context={"_tracer": tracer})
    assert tracer.spans == {}


# ---------------- span lifecycle ----------------


class _RaisingCompute:
    """A compute fake that raises on `infer_full`. Used to assert the
    LLM_CALL span gets closed with an error when the call fails,
    rather than dangling open or being absent entirely."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def infer_full(self, **kwargs: Any) -> Any:
        raise self._exc


@pytest.mark.asyncio
async def test_grade_closes_llm_call_span_with_error_when_complete_raises(
    trading_critic_bundle: EvaluatorBundle,
) -> None:
    """If the underlying compute call raises, the LLM_CALL span exists
    in the trace and is closed with `error=...`. Without this, a
    failed call would either leave a dangling open span (debugging
    nightmare) or no span at all (looks like the call never happened)."""
    fake_raising = _RaisingCompute(RuntimeError("upstream-bridge-down"))
    client = EvaluationClient(
        compute=fake_raising,
        storage=MockStorageClient(),
        bundle=trading_critic_bundle,
        evaluator_id=trading_critic_bundle.evaluator_id(),
        provider_address="0x" + "b" * 40,
    )
    grader = EvaluationGrader(client=client)
    tracer = RecordingTracer()
    parent = tracer.start_trace("test", kind=SpanKind.PIPELINE_RUN)

    with pytest.raises(RuntimeError, match="upstream-bridge-down"):
        await grader.grade(
            input="x",
            output="y",
            context={"_tracer": tracer, "_span_id": parent.id},
        )

    llm_spans = [s for s in tracer.spans.values() if s.kind == SpanKind.LLM_CALL]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.error is not None
    assert "upstream-bridge-down" in span.error
    # No metadata: we never got a receipt to attach.
    assert span.metadata is None or "eerful.receipt_id" not in span.metadata


@pytest.mark.asyncio
async def test_grade_does_not_mask_complete_error_with_tracer_failure(
    trading_critic_bundle: EvaluatorBundle,
) -> None:
    """If `tracer.end_span` itself raises during error cleanup, the
    grader must still surface the ORIGINAL `complete()` exception, not
    the tracer's. Cleanup-must-not-hide-the-real-error: a broken
    tracer (disk full, sqlite lock, etc.) shouldn't mask the actual
    call failure that the user needs to debug."""

    class _FailingTracer(RecordingTracer):
        def end_span(
            self,
            span_id: str,
            output: Any = None,
            error: str | None = None,
            usage: Any = None,
        ) -> None:
            raise OSError("tracer-disk-full")

    fake_raising = _RaisingCompute(RuntimeError("call-failed"))
    client = EvaluationClient(
        compute=fake_raising,
        storage=MockStorageClient(),
        bundle=trading_critic_bundle,
        evaluator_id=trading_critic_bundle.evaluator_id(),
        provider_address="0x" + "b" * 40,
    )
    grader = EvaluationGrader(client=client)
    tracer = _FailingTracer()
    parent = tracer.start_trace("test", kind=SpanKind.PIPELINE_RUN)

    with pytest.raises(RuntimeError, match="call-failed") as exc_info:
        await grader.grade(
            input="x",
            output="y",
            context={"_tracer": tracer, "_span_id": parent.id},
        )
    # Make sure it's the call's exception, not the tracer's.
    assert "tracer-disk-full" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_grade_success_path_does_not_raise_when_tracer_end_span_fails(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """A failure in success-path `tracer.end_span` (disk full, sqlite
    lock, schema mismatch) must NOT convert a successful evaluation
    into a grading failure. compute + storage already succeeded; the
    receipt is valid; the caller deserves their scores. Mirrors the
    error-path's tracer-cleanup suppression for symmetry."""

    class _SuccessFailingTracer(RecordingTracer):
        def end_span(
            self,
            span_id: str,
            output: Any = None,
            error: str | None = None,
            usage: Any = None,
        ) -> None:
            raise OSError("tracer-disk-full-on-success")

    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client)
    tracer = _SuccessFailingTracer()
    parent = tracer.start_trace("test", kind=SpanKind.PIPELINE_RUN)

    # Should NOT raise — tracer.end_span throwing must be swallowed.
    scores = await grader.grade(
        input="x",
        output="y",
        context={"_tracer": tracer, "_span_id": parent.id},
    )
    # And we still get the scores, even though tracing didn't complete.
    assert len(scores) == 2


@pytest.mark.asyncio
async def test_grade_success_path_tracer_failure_does_not_block_feedback(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Receipt persistence to FeedbackLoop runs AFTER the tracer block.
    If tracer.end_span raises and we don't suppress, feedback never
    runs and the receipt is lost. With suppression, feedback proceeds
    normally."""

    class _SuccessFailingTracer(RecordingTracer):
        def end_span(
            self,
            span_id: str,
            output: Any = None,
            error: str | None = None,
            usage: Any = None,
        ) -> None:
            raise OSError("tracer-disk-full")

    feedback = RecordingFeedback()
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client, feedback=feedback)
    tracer = _SuccessFailingTracer()
    parent = tracer.start_trace("test", kind=SpanKind.PIPELINE_RUN)

    await grader.grade(
        input="x",
        output="y",
        context={"_tracer": tracer, "_span_id": parent.id},
    )
    # Feedback got the receipt despite tracer failure.
    assert len(feedback.stored) == 1


@pytest.mark.asyncio
async def test_grade_llm_call_span_input_recorded_at_open_time(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """The LLM_CALL span's `input` field is set when we open the span
    (before complete runs), so it's available even if the call later
    fails. Asserts the round-1 fix: span open is BEFORE the await,
    not after (the previous code opened it after, missing the input
    in the dangling-failure case)."""
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client)
    tracer = RecordingTracer()
    parent = tracer.start_trace("test", kind=SpanKind.PIPELINE_RUN)
    await grader.grade(
        input="strategy-input",
        output="some-output",
        context={"_tracer": tracer, "_span_id": parent.id},
    )
    llm_span = next(
        s for s in tracer.spans.values() if s.kind == SpanKind.LLM_CALL
    )
    assert llm_span.input == "strategy-input"


# ---------------- input formatting ----------------


@pytest.mark.asyncio
async def test_grade_serializes_dict_output_as_json(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Structured (non-string) outputs flow into the user message as
    JSON so the bundle's critic can read them."""
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client)
    await grader.grade(
        input="strategy v1",
        output={"trades": 3, "pnl": 0.07},
    )
    sent = fake_compute.calls[-1]["messages"]
    user_msg = next(m for m in sent if m["role"] == "user")
    assert "OUTPUT:" in user_msg["content"]
    assert '"pnl"' in user_msg["content"]
    assert "INPUT:" in user_msg["content"]
    assert "strategy v1" in user_msg["content"]
