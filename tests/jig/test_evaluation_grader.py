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
from datetime import datetime, timezone
from typing import Any

import pytest
from jig.core.types import (
    EvalCase,
    FeedbackLoop,
    FeedbackQuery,
    Score,
    ScoreSource,
    ScoredResult,
    Span,
    SpanKind,
    TracingLogger,
    Usage,
)

from eerful.evaluator import EvaluatorBundle
from eerful.jig import EvaluationClient, EvaluationGrader
from eerful.zg.storage import MockStorageClient

from tests.jig.conftest import FakeComputeClient


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


class _RecordingFeedback(FeedbackLoop):
    """Captures `store_result` + `score` calls so tests can assert
    what got persisted. Mirrors the jig test fake pattern."""

    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []
        self.scored: list[tuple[str, list[Score]]] = []
        self._counter = 0

    async def store_result(
        self,
        content: str,
        input_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self._counter += 1
        result_id = f"result-{self._counter:03d}"
        self.stored.append(
            {
                "result_id": result_id,
                "content": content,
                "input_text": input_text,
                "metadata": metadata,
            }
        )
        return result_id

    async def score(self, result_id: str, scores: list[Score]) -> None:
        self.scored.append((result_id, scores))

    async def get_signals(
        self,
        query: str,
        limit: int = 3,
        min_score: float | None = None,
        source: ScoreSource | None = None,
    ) -> list[ScoredResult]:
        return []

    async def query(self, q: FeedbackQuery) -> list[ScoredResult]:
        return []

    async def export_eval_set(
        self,
        since: datetime | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        return []


@pytest.mark.asyncio
async def test_grade_persists_receipt_metadata_to_feedback(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """When a feedback loop is configured, the grader writes a stored
    result whose metadata carries the receipt's three identifiers and
    a tags list including the receipt_id."""
    feedback = _RecordingFeedback()
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
    feedback = _RecordingFeedback()
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


class _RecordingTracer(TracingLogger):
    """Minimal in-memory tracer for assertions on span structure +
    metadata. Records every span by id so tests can pull them back."""

    def __init__(self) -> None:
        self.spans: dict[str, Span] = {}
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"span-{self._counter:03d}"

    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span:
        span = Span(
            id=self._new_id(),
            trace_id="trace-test",
            kind=kind,
            name=name,
            started_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
            metadata=metadata,
        )
        self.spans[span.id] = span
        return span

    def start_span(
        self, parent_id: str, kind: SpanKind, name: str, input: Any = None
    ) -> Span:
        span = Span(
            id=self._new_id(),
            trace_id="trace-test",
            kind=kind,
            name=name,
            started_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
            parent_id=parent_id,
            input=input,
        )
        self.spans[span.id] = span
        return span

    def end_span(
        self,
        span_id: str,
        output: Any = None,
        error: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        span = self.spans[span_id]
        span.output = output
        span.error = error
        span.usage = usage

    async def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans.values() if s.trace_id == trace_id]

    async def list_traces(
        self,
        since: datetime | None = None,
        limit: int = 50,
        name: str | None = None,
    ) -> list[Span]:
        return list(self.spans.values())


@pytest.mark.asyncio
async def test_grade_attaches_receipt_metadata_to_llm_call_span(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """With _tracer + _span_id in context, grader opens an LLM_CALL
    child span carrying the receipt's identifiers."""
    tracer = _RecordingTracer()
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
    tracer = _RecordingTracer()
    client = _make_client(trading_critic_bundle, fake_compute)
    grader = EvaluationGrader(client=client)
    # Only _tracer, no _span_id.
    await grader.grade(input="x", output="y", context={"_tracer": tracer})
    assert tracer.spans == {}


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
