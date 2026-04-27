"""Span-decoration helper for attaching EER receipt identifiers to a jig span.

Not its own `TracingLogger` — that would force users to choose between
their existing tracer (SQLiteTracer / FakeTracer / external observability
backend) and seeing receipt metadata. Instead, this is a one-function
helper any tracer can call.

Caller pattern (used by `EvaluationGrader.grade`):

    span = tracer.start_span(parent_id, SpanKind.LLM_CALL, "eerful.evaluate", input=...)
    response = await client.complete(params)
    attach_receipt_to_span(span, response.eer)
    tracer.end_span(span.id, output=...)

The helper mutates `span.metadata` in place. jig's in-tree tracers
(`FakeTracer`, `SQLiteTracer`) hold the `Span` object and serialize it
at flush/end time, so the metadata update is visible in the persisted
trace. A future tracer that snapshots metadata at `start_span` time
rather than at end would silently drop these updates — flagged in the
docstring so it's not a hidden trap.
"""

from __future__ import annotations

from jig.core.types import Span

from eerful.receipt import EnhancedReceipt


def attach_receipt_to_span(span: Span, receipt: EnhancedReceipt) -> None:
    """Record the EER receipt's identifiers on `span.metadata`.

    Adds three keys, namespaced under `eerful.` to avoid collision with
    other instrumentation:

    - `eerful.receipt_id` — the receipt's content identity
    - `eerful.evaluator_id` — the bundle the receipt attests
    - `eerful.attestation_report_hash` — the report Step 4 fetches

    Existing `metadata` keys are preserved (dict merge, not replace).
    Callers downstream (replay, debugging, audit) can pull the receipt
    by ID from storage using `eerful.receipt_id` alone.

    Mutates `span.metadata` in place — the helper assumes the caller
    holds the same `Span` object the tracer's internal store has a
    reference to (which is the contract for jig's `start_span` /
    `end_span` flow). Tracers that snapshot metadata at `start_span`
    time rather than at flush would silently drop these updates;
    none of jig's in-tree tracers do that as of this writing.
    """
    extras = {
        "eerful.receipt_id": receipt.receipt_id,
        "eerful.evaluator_id": receipt.evaluator_id,
        "eerful.attestation_report_hash": receipt.attestation_report_hash,
    }
    span.metadata = {**(span.metadata or {}), **extras}
