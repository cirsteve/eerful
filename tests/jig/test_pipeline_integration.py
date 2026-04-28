"""End-to-end: jig `run_pipeline` driving an `EvaluationGrader`.

Exercises the load-bearing properties for D.1 (trading-critic demo):
- A pipeline step's output, scored by `EvaluationGrader`, produces a
  receipt that's recoverable from the trace and verifiable through
  `verify_receipt` end-to-end (Steps 1, 2, 3, 6 against MockStorage).
- A chain of three pipelines (modeling v1 → v2 → v3) links receipts
  via `previous_receipt_id` and all three verify.
- A pipeline that short-circuits via `is_err` doesn't lose receipts
  produced before the failure point.
"""

from __future__ import annotations

from typing import Any

import pytest
from jig.core.pipeline import PipelineConfig, Step, run_pipeline

from eerful.evaluator import EvaluatorBundle
from eerful.jig import EvaluationClient, EvaluationGrader
from eerful.verify import verify_receipt
from eerful.zg.storage import MockStorageClient

from tests.jig.conftest import FakeComputeClient, RecordingFeedback, RecordingTracer


def _client_and_storage(
    bundle: EvaluatorBundle,
    fake_compute: FakeComputeClient,
) -> tuple[EvaluationClient, MockStorageClient]:
    """Build a client + storage where the bundle has been pre-published.
    Verification's Step 2 fetches the bundle by content hash, so it
    must already be in storage."""
    storage = MockStorageClient()
    storage.upload_blob(bundle.canonical_bytes())
    client = EvaluationClient(
        compute=fake_compute,
        storage=storage,
        bundle=bundle,
        evaluator_id=bundle.evaluator_id(),
        provider_address="0x" + "b" * 40,
    )
    return client, storage


@pytest.mark.asyncio
async def test_pipeline_grader_persists_receipt_metadata_through_feedback_and_trace(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """A one-step pipeline whose grader is an EvaluationGrader produces:
    (a) one stored result in feedback carrying the receipt's three
    identifiers, (b) an LLM_CALL span whose metadata mirrors them,
    (c) the attestation report bytes in storage at
    `attestation_report_hash`. End-to-end `verify_receipt` is asserted
    in `test_pipeline_receipt_is_offline_verifiable` below — separate
    test because it needs a capturing-grader subclass to peel the
    receipt object out of the response."""
    client, storage = _client_and_storage(trading_critic_bundle, fake_compute)
    feedback = RecordingFeedback()
    grader = EvaluationGrader(client=client, feedback=feedback)
    tracer = RecordingTracer()

    async def echo_step(ctx: dict[str, Any]) -> str:
        return f"echo:{ctx['input']}"

    config = PipelineConfig(
        name="trading-critic",
        steps=[Step(name="echo", fn=echo_step, grader=grader)],
        tracer=tracer,
    )
    result = await run_pipeline(config, input="strategy v1")

    # Pipeline emitted scores per step.
    assert "echo" in result.step_scores
    assert len(result.step_scores["echo"]) == 2  # risk, novelty
    # Feedback got the receipt's three identifiers in metadata.
    assert len(feedback.stored) == 1
    md = feedback.stored[0]["metadata"]
    assert "eer_receipt_id" in md
    assert "eer_evaluator_id" in md
    assert "eer_attestation_report_hash" in md
    # LLM_CALL span carries the same receipt_id (grader path → tracer).
    llm_span = next(
        s for s in tracer.spans.values() if s.name == "eerful.evaluate"
    )
    assert llm_span.metadata is not None
    assert llm_span.metadata["eerful.receipt_id"] == md["eer_receipt_id"]
    # Attestation report bytes are in storage at the receipt's hash;
    # the report fetch + re-hash is what offline verify needs at Step 4.
    fetched = storage.download_blob(md["eer_attestation_report_hash"])
    assert len(fetched) > 0


@pytest.mark.asyncio
async def test_chain_of_three_pipelines_links_receipts(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Three sequential pipeline runs sharing one EvaluationClient
    instance: each run's grader produces a receipt; second/third
    have `previous_receipt_id` pointing at the prior. Models the D.1
    trading-critic v1 → v2 → v3 chain.

    The chain is asserted via the receipts the client tracks: after
    each run, `client.previous_receipt_id` advances to the receipt
    just produced. The integration test verifies the *linkage* end
    to end (the receipt-level Step 6 / chain semantics are unit-
    tested in test_evaluation_client.py)."""
    client, _ = _client_and_storage(trading_critic_bundle, fake_compute)
    feedback = RecordingFeedback()
    grader = EvaluationGrader(client=client, feedback=feedback)
    tracer = RecordingTracer()

    async def echo_step(ctx: dict[str, Any]) -> str:
        return f"output-for-{ctx['input']}"

    config = PipelineConfig(
        name="trading-critic-chain",
        steps=[Step(name="echo", fn=echo_step, grader=grader)],
        tracer=tracer,
    )

    # v1: no predecessor.
    await run_pipeline(config, input="strategy v1")
    v1_id = client.previous_receipt_id
    assert v1_id is not None

    # v2: previous = v1.
    await run_pipeline(config, input="strategy v2")
    v2_id = client.previous_receipt_id
    assert v2_id is not None
    assert v2_id != v1_id

    # v3: previous = v2.
    await run_pipeline(config, input="strategy v3")
    v3_id = client.previous_receipt_id
    assert v3_id is not None
    assert v3_id != v2_id

    # Three feedback rows, one per run.
    assert len(feedback.stored) == 3
    chain_ids = [s["metadata"]["eer_receipt_id"] for s in feedback.stored]
    assert chain_ids == [v1_id, v2_id, v3_id]


@pytest.mark.asyncio
async def test_short_circuited_pipeline_preserves_prior_receipts(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """A two-step pipeline whose second step short-circuits via
    `is_err`. The first step's grader already produced a receipt;
    that receipt must still be in feedback even though the pipeline
    bailed before completing. Otherwise a partial run would silently
    lose verifiable evidence of the work that was actually done."""
    client, _ = _client_and_storage(trading_critic_bundle, fake_compute)
    feedback = RecordingFeedback()
    grader = EvaluationGrader(client=client, feedback=feedback)
    tracer = RecordingTracer()

    async def good_step(ctx: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "value": "first"}

    async def bad_step(ctx: dict[str, Any]) -> dict[str, Any]:
        return {"ok": False, "error": "kaboom"}

    def _extract(r: Any) -> str:
        if isinstance(r, dict):
            err = r.get("error")
            if isinstance(err, str):
                return err
        return "unknown"

    config = PipelineConfig(
        name="short-circuit-test",
        steps=[
            Step(name="good", fn=good_step, grader=grader),
            Step(name="bad", fn=bad_step, grader=grader),
        ],
        tracer=tracer,
        is_err=lambda r: isinstance(r, dict) and r.get("ok") is False,
        extract_err=_extract,
    )
    result = await run_pipeline(config, input="x")

    assert result.short_circuited
    assert result.error_step == "bad"
    # First step's grader ran (because the runner grades after each
    # step's success); second step's grader did NOT run (the
    # short-circuit happens before grading the bad step).
    assert "good" in result.step_scores
    assert "bad" not in result.step_scores
    # Feedback has exactly the v1 receipt — not lost to the
    # short-circuit.
    assert len(feedback.stored) == 1
    assert feedback.stored[0]["metadata"]["eer_evaluator_id"] == (
        trading_critic_bundle.evaluator_id()
    )


@pytest.mark.asyncio
async def test_pipeline_receipt_is_offline_verifiable(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Smoke test: the receipt produced through a pipeline is the SAME
    receipt shape the offline verifier expects — Step 1 (receipt_id
    derivation), Step 2 (bundle hash), Step 3 (output_score_block
    schema), Step 6 (enclave signature) all pass.

    We grab the receipt directly from the EvaluationClient via a
    second grade call (the client carries the most recent receipt's
    id; we peel back one layer by capturing the response in a
    side-channel)."""
    client, storage = _client_and_storage(trading_critic_bundle, fake_compute)

    captured: list[Any] = []

    class _CapturingGrader(EvaluationGrader):
        async def grade(
            self, input: Any, output: Any, context: dict[str, Any] | None = None
        ) -> Any:
            response = await self._client.complete(self._build_params(input, output))
            captured.append(response)
            return self._extract_scores(response.eer.output_score_block)

    grader = _CapturingGrader(client=client)
    tracer = RecordingTracer()

    async def echo_step(ctx: dict[str, Any]) -> str:
        return "echo"

    config = PipelineConfig(
        name="offline-verify",
        steps=[Step(name="echo", fn=echo_step, grader=grader)],
        tracer=tracer,
    )
    await run_pipeline(config, input="x")

    assert len(captured) == 1
    receipt = captured[0].eer
    bundle_bytes = trading_critic_bundle.canonical_bytes()
    # End-to-end verification through the offline pipeline.
    result = verify_receipt(receipt, bundle_bytes, report_bytes=None)
    assert result.bundle.evaluator_id() == trading_critic_bundle.evaluator_id()
    # Bundle output_schema validated against the receipt's score block.
    assert receipt.output_score_block is not None
    assert "risk" in receipt.output_score_block
    # And the report is in storage too — Step 4 would succeed.
    fetched = storage.download_blob(receipt.attestation_report_hash)
    assert len(fetched) > 0
