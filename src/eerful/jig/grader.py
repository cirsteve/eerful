"""`EvaluationGrader` — jig `Grader[Any]` that produces an EER per call.

Wraps an `EvaluationClient`. Each `.grade()`:

1. Builds a `CompletionParams` from `(input, output)` — the bundle's
   system_prompt is the criteria; the user message is "score this
   output for this input."
2. Calls the client. The receipt comes back attached to the response.
3. Extracts one `Score` per top-level numeric field in the receipt's
   `output_score_block`. Filters by `score_dimensions` if set.
4. Decorates the GRADING-sibling LLM_CALL span (if a tracer is in
   context) with the receipt's identifiers so traces stay queryable
   by receipt_id.
5. Persists the receipt via `FeedbackLoop` (if one was supplied) so
   downstream consumers can pull receipts back by their tags.

The `Grader[Any]` parameterization is intentional: jig's `Grader[T]`
binds `T` to the agent's `output_schema` type. We don't constrain
that — any `output` whose `(input, output)` pair the bundle's system
prompt knows how to score is valid. Producers that want stricter
typing can subclass and re-parameterize.
"""

from __future__ import annotations

import json
from typing import Any

from jig.core.types import (
    CompletionParams,
    FeedbackLoop,
    Grader,
    Message,
    Role,
    Score,
    ScoreSource,
    SpanKind,
    TracingLogger,
)

from eerful.jig.client import EvaluationClient
from eerful.jig.tracer import attach_receipt_to_span


class EvaluationGrader(Grader[Any]):
    """jig `Grader` backed by an `EvaluationClient`.

    The grader's job is to ask the bundle's critic LLM "score this
    output for this input." The bundle's `system_prompt` is the
    criteria; the user message we send is the input + output pair.
    Producers control the message format by writing their bundle's
    system prompt to expect a specific input/output framing — for
    D.1's trading critic, the convention is a `INPUT:` / `OUTPUT:`
    block.

    `score_dimensions` optionally filters which numeric fields of the
    receipt's `output_score_block` become `Score` objects. Default
    `None` returns one score per top-level numeric field. String /
    list / dict fields are always ignored — `Score.value: float`
    won't accept anything else.

    `feedback`, when set, receives the receipt as a stored result
    plus its scores. Storage shape: `metadata` carries the receipt's
    three identifiers (receipt_id, evaluator_id,
    attestation_report_hash) plus a `tags` list including the
    receipt_id so `FeedbackQuery(tags=[<receipt_id>])` finds it.
    """

    def __init__(
        self,
        *,
        client: EvaluationClient,
        score_dimensions: list[str] | None = None,
        feedback: FeedbackLoop | None = None,
    ) -> None:
        self._client = client
        self._score_dimensions = score_dimensions
        self._feedback = feedback

    async def grade(
        self,
        input: Any,
        output: Any,
        context: dict[str, Any] | None = None,
    ) -> list[Score]:
        params = self._build_params(input, output)
        response = await self._client.complete(params)
        receipt = response.eer
        scores = self._extract_scores(receipt.output_score_block)

        # Tracer integration: if jig's pipeline runner put a tracer +
        # parent span_id in context, open an LLM_CALL child and attach
        # the receipt's identifiers. The LLM_CALL span is a sibling of
        # the runner-managed GRADING span (jig's pipeline doesn't
        # update `_span_id` mid-flight, so children open under the
        # PIPELINE_RUN root). Acceptable for v1; both spans are
        # visible in the trace so downstream queries can find either.
        ctx = context or {}
        tracer = ctx.get("_tracer")
        span_id = ctx.get("_span_id")
        if isinstance(tracer, TracingLogger) and isinstance(span_id, str):
            llm_span = tracer.start_span(
                span_id, SpanKind.LLM_CALL, "eerful.evaluate", input=input
            )
            attach_receipt_to_span(llm_span, receipt)
            tracer.end_span(
                llm_span.id,
                output=receipt.output_score_block,
                usage=response.usage,
            )

        # Feedback persistence: write the receipt as a stored result
        # so downstream consumers can pull it back by tag. Skipped
        # silently when scores is empty (FeedbackLoop.score on an
        # empty list would be a no-op anyway, and store_result without
        # scores leaves an unscored row that's noise).
        if self._feedback is not None and scores:
            result_id = await self._feedback.store_result(
                content=receipt.response_content,
                input_text=str(input),
                metadata={
                    "eer_receipt_id": receipt.receipt_id,
                    "eer_evaluator_id": receipt.evaluator_id,
                    "eer_attestation_report_hash": receipt.attestation_report_hash,
                    # tags include the receipt_id so a later
                    # FeedbackQuery(tags=[receipt_id]) finds this row.
                    "tags": ["eer", receipt.receipt_id],
                },
            )
            await self._feedback.score(result_id, scores)

        return scores

    # ---------------- internals ----------------

    def _build_params(self, input: Any, output: Any) -> CompletionParams:
        """Format `(input, output)` as the user message the bundle's
        critic prompt expects.

        Convention: `INPUT:\\n{input}\\n\\nOUTPUT:\\n{output}`. Bundles
        whose system prompts expect a different framing can subclass
        the grader and override this method. JSON-stringify outputs
        that aren't strings so structured agent results (dicts /
        pydantic models) flow through legibly.
        """
        if not isinstance(output, str):
            try:
                output_text = json.dumps(output, default=str, sort_keys=True)
            except (TypeError, ValueError):
                output_text = repr(output)
        else:
            output_text = output
        user_msg = f"INPUT:\n{input}\n\nOUTPUT:\n{output_text}"
        return CompletionParams(
            messages=[Message(role=Role.USER, content=user_msg)]
        )

    def _extract_scores(self, score_block: dict[str, Any] | None) -> list[Score]:
        """Walk the receipt's score block; emit one Score per top-level
        numeric field that passes the optional `score_dimensions` filter.

        bool is filtered out because Python's `bool` is an `int` subclass
        — accepting it would let `{"valid": True}` become a score of
        value 1.0, which is almost never what the bundle author meant.
        """
        if score_block is None:
            return []
        out: list[Score] = []
        for dim, value in score_block.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue
            if self._score_dimensions is not None and dim not in self._score_dimensions:
                continue
            out.append(
                Score(dimension=dim, value=float(value), source=ScoreSource.LLM_JUDGE)
            )
        return out
