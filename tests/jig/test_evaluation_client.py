"""`EvaluationClient` — receipt-producing jig LLM client.

Tests cover the load-bearing properties of the EER produce path:
- Returned response is a real `LLMResponse` subclass (Q5 canary).
- Receipt round-trips through `verify_receipt` Steps 1, 2, 3, 6.
- Bundle's system_prompt and inference_params override caller-side
  conflicts (the receipt has to attest the bundle's criteria).
- Tools are rejected (TeeML doesn't speak them).
- Chain pattern works: previous_receipt_id stamped automatically and
  per-call overridable via provider_params.
- input_commitment populated only when commit_inputs is True; salt
  persisted to SaltStore when one is supplied.
- Score block parsed best-effort: malformed → None receipt field, not
  a raised exception (the protocol records what happened).
- Attestation report is uploaded to Storage and the storage-side hash
  matches the compute-side hash.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
from jig.core.types import (
    CompletionParams,
    LLMResponse,
    Message,
    Role,
    ToolDefinition,
)

from eerful.commitment import SaltStore, compute_input_commitment
from eerful.errors import EvaluationClientError
from eerful.evaluator import EvaluatorBundle
from eerful.jig import EerfulLLMResponse, EvaluationClient
from eerful.verify import verify_receipt
from eerful.zg.storage import MockStorageClient

from tests.jig.conftest import FakeComputeClient


# ---------------- Q5 canary ----------------


def test_eerful_llm_response_is_a_llm_response_subclass() -> None:
    """jig's runner does `isinstance(resp, LLMResponse)`. If a future
    upstream change breaks dataclass subclassing (e.g. adds a defaulted
    field to LLMResponse, forcing our `eer` to also default), this test
    is the canary."""
    assert issubclass(EerfulLLMResponse, LLMResponse)


# ---------------- happy path ----------------


def _client(
    *,
    bundle: EvaluatorBundle,
    fake_compute: FakeComputeClient,
    storage: MockStorageClient | None = None,
    **kwargs: Any,
) -> tuple[EvaluationClient, MockStorageClient]:
    storage = storage or MockStorageClient()
    return (
        EvaluationClient(
            compute=fake_compute,
            storage=storage,
            bundle=bundle,
            evaluator_id=bundle.evaluator_id(),
            provider_address="0x" + "b" * 40,
            **kwargs,
        ),
        storage,
    )


def _user_only_params(text: str) -> CompletionParams:
    return CompletionParams(messages=[Message(role=Role.USER, content=text)])


@pytest.mark.asyncio
async def test_complete_returns_eerful_llm_response_with_verifying_receipt(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """End-to-end: a single .complete() returns an EerfulLLMResponse
    whose receipt verifies through the offline pipeline (Steps 1, 2, 3, 6)."""
    client, storage = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)

    response = await client.complete(_user_only_params("rate this strategy"))

    assert isinstance(response, EerfulLLMResponse)
    assert isinstance(response, LLMResponse)
    receipt = response.eer

    # Verify Steps 1, 2, 3, 6 against the storage we uploaded to.
    bundle_bytes = trading_critic_bundle.canonical_bytes()
    storage.upload_blob(bundle_bytes)  # caller would do this via publish-evaluator
    result = verify_receipt(receipt, bundle_bytes, report_bytes=None)
    assert result.bundle.evaluator_id() == trading_critic_bundle.evaluator_id()


@pytest.mark.asyncio
async def test_complete_uploads_attestation_report_to_storage(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """The producer side of Step 4: report bytes land in storage at
    receipt.attestation_report_hash."""
    client, storage = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    response = await client.complete(_user_only_params("hi"))
    fetched = storage.download_blob(response.eer.attestation_report_hash)
    # Round-trip: storage content matches what the receipt commits to.
    assert hashlib.sha256(fetched).hexdigest() == response.eer.attestation_report_hash[2:]


@pytest.mark.asyncio
async def test_complete_uses_bundle_inference_params_over_call_params(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Bundle pins `temperature=0.0`, `max_tokens=500`. Caller passes
    NEITHER; the bundle's values flow through to ComputeClient.infer_full."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    await client.complete(_user_only_params("x"))
    call = fake_compute.calls[-1]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 500


# ---------------- validation failures ----------------


@pytest.mark.asyncio
async def test_complete_rejects_tools(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """TeeML doesn't sign tool-call results. Silently dropping tools
    would produce a receipt whose response did NOT use the tools the
    caller specified — wrong default."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        tools=[ToolDefinition(name="t", description="d", parameters={})],
    )
    with pytest.raises(EvaluationClientError, match="tool calls"):
        await client.complete(params)


@pytest.mark.asyncio
async def test_complete_rejects_conflicting_system_prompt(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Caller-supplied system prompt that differs from the bundle's
    would let the receipt attest evaluation under different criteria
    than evaluator_id claims. Refuse."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        system="be lenient instead",
    )
    with pytest.raises(EvaluationClientError, match="system_prompt"):
        await client.complete(params)


@pytest.mark.asyncio
async def test_complete_accepts_matching_system_prompt(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Caller passing the same system prompt is harmless — defensive
    behavior (`if equal, no-op`) over strict (refuse anything non-None)."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        system=trading_critic_bundle.system_prompt,
    )
    response = await client.complete(params)
    assert isinstance(response, EerfulLLMResponse)


@pytest.mark.asyncio
async def test_complete_rejects_conflicting_temperature(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Bundle pins temperature=0.0 for cross-receipt comparability;
    caller passing 0.7 would silently drift the inference and the
    receipt would attest a different decoding policy than declared."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        temperature=0.7,
    )
    with pytest.raises(EvaluationClientError, match="temperature"):
        await client.complete(params)


# ---------------- provider_params strict validation ----------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_value",
    ["true", "false", "0", "1", 1, 0, [], {}, "yes"],
)
async def test_commit_inputs_non_bool_value_raises(
    trading_critic_bundle: EvaluatorBundle,
    fake_compute: FakeComputeClient,
    bad_value: Any,
) -> None:
    """`provider_params['eerful.commit_inputs']` must be a real bool.
    Truthy-coercing strings (`'false'`, `'0'`) or ints would silently
    flip a commitment on or off based on YAML/JSON ambiguity. Refuse
    loudly so the producer knows their config didn't take effect."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        provider_params={"eerful.commit_inputs": bad_value},
    )
    with pytest.raises(EvaluationClientError, match="commit_inputs"):
        await client.complete(params)


@pytest.mark.asyncio
async def test_commit_inputs_real_bool_accepted(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Sanity: actual `True` / `False` flow through unchanged."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        provider_params={"eerful.commit_inputs": True},
    )
    response = await client.complete(params)
    assert response.eer.input_commitment is not None


@pytest.mark.asyncio
async def test_previous_receipt_id_invalid_hex_raises(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """A non-hex string value would otherwise reach
    `EnhancedReceipt.build` and fail with a confusing pydantic error
    deep in the receipt module. Catching it at the call boundary
    surfaces a clear `EvaluationClientError` mentioning
    `previous_receipt_id`."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        provider_params={"eerful.previous_receipt_id": "definitely-not-hex"},
    )
    with pytest.raises(EvaluationClientError, match="previous_receipt_id"):
        await client.complete(params)


@pytest.mark.asyncio
async def test_previous_receipt_id_wrong_length_raises(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """A valid hex string of the wrong length is also rejected."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        # Valid hex but too short for Bytes32Hex.
        provider_params={"eerful.previous_receipt_id": "0xab"},
    )
    with pytest.raises(EvaluationClientError, match="previous_receipt_id"):
        await client.complete(params)


@pytest.mark.asyncio
async def test_previous_receipt_id_non_string_raises(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        provider_params={"eerful.previous_receipt_id": 12345},
    )
    with pytest.raises(EvaluationClientError, match="previous_receipt_id"):
        await client.complete(params)


@pytest.mark.asyncio
async def test_salt_non_bytes_raises(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Silently regenerating on a wrong-type salt would defeat the
    chain pattern's stable input commitment without any error. Refuse."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        provider_params={
            "eerful.commit_inputs": True,
            "eerful.salt": "0xab" * 16,  # caller probably meant to pass bytes
        },
    )
    with pytest.raises(EvaluationClientError, match="salt"):
        await client.complete(params)


@pytest.mark.asyncio
async def test_salt_bytes_accepted_and_used(
    trading_critic_bundle: EvaluatorBundle,
    fake_compute: FakeComputeClient,
    tmp_path: Path,
) -> None:
    """Sanity: passing actual bytes for `eerful.salt` pins it across
    the call. The persisted salt in SaltStore equals what we passed."""
    salt_store = SaltStore(tmp_path / "salts.json")
    client, _ = _client(
        bundle=trading_critic_bundle,
        fake_compute=fake_compute,
        salt_store=salt_store,
    )
    pinned_salt = b"\xaa" * 32
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        provider_params={
            "eerful.commit_inputs": True,
            "eerful.salt": pinned_salt,
        },
    )
    response = await client.complete(params)
    salt_back, _ = salt_store.get(response.eer.receipt_id)
    assert salt_back == pinned_salt


# ---------------- chain pattern ----------------


@pytest.mark.asyncio
async def test_chain_pattern_carries_previous_receipt_id_automatically(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Two consecutive .complete() calls: second receipt references
    first via previous_receipt_id without the caller plumbing it."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    r1 = await client.complete(_user_only_params("strategy v1"))
    r2 = await client.complete(_user_only_params("strategy v2"))
    assert r1.eer.previous_receipt_id is None
    assert r2.eer.previous_receipt_id == r1.eer.receipt_id


@pytest.mark.asyncio
async def test_chain_pattern_per_call_override_via_provider_params(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Caller can pin `previous_receipt_id` per call. Useful for the
    D.1 chain when reconstructing across processes (e.g. v3's writer
    is a different process than v1's)."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    pinned = "0x" + "ab" * 32
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        provider_params={"eerful.previous_receipt_id": pinned},
    )
    r = await client.complete(params)
    assert r.eer.previous_receipt_id == pinned


# ---------------- input_commitment ----------------


@pytest.mark.asyncio
async def test_commit_inputs_true_populates_input_commitment_and_persists_salt(
    trading_critic_bundle: EvaluatorBundle,
    fake_compute: FakeComputeClient,
    tmp_path: Path,
) -> None:
    """commit_inputs=True + a SaltStore: receipt has non-None
    input_commitment, the salt is persisted under receipt_id, and
    `compute_input_commitment(input_bytes, evaluator_id, salt)`
    reconstructs to the same value."""
    salt_store = SaltStore(tmp_path / "salts.json")
    client, _ = _client(
        bundle=trading_critic_bundle,
        fake_compute=fake_compute,
        salt_store=salt_store,
        commit_inputs=True,
    )
    user_text = "strategy: market-neutral-v0"
    response = await client.complete(_user_only_params(user_text))
    receipt = response.eer
    assert receipt.input_commitment is not None
    salt, _ = salt_store.get(receipt.receipt_id)
    expected = compute_input_commitment(
        user_text.encode("utf-8"), trading_critic_bundle.evaluator_id(), salt
    )
    assert receipt.input_commitment == expected


@pytest.mark.asyncio
async def test_commit_inputs_false_leaves_input_commitment_none(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    client, _ = _client(
        bundle=trading_critic_bundle, fake_compute=fake_compute, commit_inputs=False
    )
    response = await client.complete(_user_only_params("x"))
    assert response.eer.input_commitment is None


# ---------------- score block parsing ----------------


@pytest.mark.asyncio
async def test_malformed_score_block_results_in_none_not_raise(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Best-effort parse: non-JSON response_content → output_score_block
    is None, receipt is still produced. The protocol records what
    happened; a producer with garbled output should still have a
    verifiable record of the failure."""
    fake_compute.set_next_response("this is not json")
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    response = await client.complete(_user_only_params("x"))
    assert response.eer.output_score_block is None


@pytest.mark.asyncio
async def test_response_content_is_signed_text_not_caller_input(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Receipt's response_content equals what the enclave signed —
    Step 6 verification depends on this. The fake's response_content
    flows through as the receipt's response_content."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    response = await client.complete(_user_only_params("user prompt"))
    assert response.eer.response_content == fake_compute._response_content
    # And jig's LLMResponse.content matches it too.
    assert response.content == response.eer.response_content


# ---------------- usage flow-through ----------------


@pytest.mark.asyncio
async def test_usage_tokens_flow_through_to_jig_usage(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """OQ-4 Phase 1: ComputeResult's input_tokens / output_tokens land
    on jig's Usage struct so BudgetTracker can see EER calls."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    response = await client.complete(_user_only_params("x"))
    assert response.usage.input_tokens == 42
    assert response.usage.output_tokens == 21
    # Phase 2 not yet wired; cost stays None.
    assert response.usage.cost is None


@pytest.mark.asyncio
async def test_usage_tokens_default_to_zero_when_provider_omitted(
    trading_critic_bundle: EvaluatorBundle,
) -> None:
    """A ComputeResult without input_tokens/output_tokens (provider
    didn't report) lands as Usage(0, 0) — jig.Usage uses ints, not
    Optional. Documented that "0" here means "unknown" not "definitely
    zero." Producer's BudgetTracker sees no cost ceiling impact."""
    fake_no_usage = FakeComputeClient(response_content='{"score": 0.5}')
    # Override to drop the token counts.
    original_infer = fake_no_usage.infer_full

    def infer_full_no_usage(**kwargs: Any) -> Any:
        result = original_infer(**kwargs)
        return result.model_copy(update={"input_tokens": None, "output_tokens": None})

    fake_no_usage.infer_full = infer_full_no_usage  # type: ignore[method-assign]
    client = EvaluationClient(
        compute=fake_no_usage,
        storage=MockStorageClient(),
        bundle=trading_critic_bundle,
        evaluator_id=trading_critic_bundle.evaluator_id(),
        provider_address="0x" + "b" * 40,
    )
    response = await client.complete(_user_only_params("x"))
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0


# ---------------- attestation hash mismatch ----------------


@pytest.mark.asyncio
async def test_attestation_hash_mismatch_after_upload_raises(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """If the storage adapter returns a different content hash than
    what compute reported, that's byzantine — refuse to build the
    receipt rather than silently produce one that won't verify at
    Step 4."""

    class _LyingStorage:
        def upload_blob(self, data: bytes) -> str:
            return "0x" + "0" * 64  # always lies

        def download_blob(self, content_hash: str) -> bytes:
            raise NotImplementedError

    client = EvaluationClient(
        compute=fake_compute,
        storage=_LyingStorage(),
        bundle=trading_critic_bundle,
        evaluator_id=trading_critic_bundle.evaluator_id(),
        provider_address="0x" + "b" * 40,
    )
    with pytest.raises(EvaluationClientError, match="attestation report hash mismatch"):
        await client.complete(_user_only_params("x"))


# ---------------- system message handling ----------------


@pytest.mark.asyncio
async def test_complete_strips_caller_system_messages_and_prepends_bundle_prompt(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """A caller-supplied system message in `params.messages` is
    silently dropped (the bundle owns the system slot). The wire
    request's first message is always the bundle's system_prompt."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[
            Message(role=Role.SYSTEM, content="ignored caller system"),
            Message(role=Role.USER, content="hello"),
        ]
    )
    await client.complete(params)
    sent = fake_compute.calls[-1]["messages"]
    assert sent[0] == {"role": "system", "content": trading_critic_bundle.system_prompt}
    # Caller's "ignored caller system" must not appear anywhere.
    assert all(m["content"] != "ignored caller system" for m in sent)
    # And the user message survives.
    assert {"role": "user", "content": "hello"} in sent


# ---------------- evaluator_id consistency check ----------------


def test_constructor_rejects_mismatched_evaluator_id(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """A stale or wrong `evaluator_id` (e.g., from a previous bundle
    version) would silently produce receipts that fail verification
    later. The constructor refuses to build a client whose
    `evaluator_id` doesn't match the bundle's content hash, so the
    error surfaces at construction time with a clear message."""
    wrong_id = "0x" + "00" * 32
    with pytest.raises(EvaluationClientError, match="does not match"):
        EvaluationClient(
            compute=fake_compute,
            storage=MockStorageClient(),
            bundle=trading_critic_bundle,
            evaluator_id=wrong_id,
            provider_address="0x" + "b" * 40,
        )


def test_constructor_accepts_matching_evaluator_id(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Sanity: passing the bundle's actual evaluator_id constructs cleanly."""
    EvaluationClient(
        compute=fake_compute,
        storage=MockStorageClient(),
        bundle=trading_critic_bundle,
        evaluator_id=trading_critic_bundle.evaluator_id(),
        provider_address="0x" + "b" * 40,
    )


# ---------------- caller inference params ----------------


@pytest.mark.asyncio
async def test_caller_temperature_forwarded_when_bundle_does_not_pin(
    fake_compute: FakeComputeClient,
) -> None:
    """Bundle with no inference_params: caller-supplied `temperature`
    flows through to ComputeClient.infer_full. Spec §6.5 frames
    inference_params as informational, so a bundle that omits
    `temperature` is signalling 'use whatever the caller wants.'"""
    bundle = EvaluatorBundle(
        version="bundle-without-params@1.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
        # inference_params intentionally omitted
    )
    client = EvaluationClient(
        compute=fake_compute,
        storage=MockStorageClient(),
        bundle=bundle,
        evaluator_id=bundle.evaluator_id(),
        provider_address="0x" + "b" * 40,
    )
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        temperature=0.7,
        max_tokens=2048,
    )
    await client.complete(params)
    call = fake_compute.calls[-1]
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_caller_inference_params_silently_dropped_when_bundle_pins(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Bundle pins temperature=0.0, max_tokens=500. Caller passing
    nothing → bundle values flow through. (Caller passing matching
    values is also fine, exercised in
    `test_complete_uses_bundle_inference_params_over_call_params`.)"""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        temperature=None,
        max_tokens=None,
    )
    await client.complete(params)
    call = fake_compute.calls[-1]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 500


# ---------------- tools=[] compat ----------------


@pytest.mark.asyncio
async def test_complete_accepts_empty_tools_list(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Some serializers emit `tools: []` for 'no tools' — that's
    semantically the same as `None` and we shouldn't break callers
    on it. Only a non-empty list is the unsupported case."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        tools=[],
    )
    response = await client.complete(params)
    assert isinstance(response, EerfulLLMResponse)


# ---------------- empty salt rejected ----------------


@pytest.mark.asyncio
async def test_empty_salt_rejected_explicitly(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """An empty-bytes salt (`b\"\"`) provides zero entropy and defeats
    §6.7's brute-force-protection purpose. We reject it loudly rather
    than silently regenerating (would be a footgun: producer thinks
    they pinned a salt, gets a random one) or accepting it (would
    silently produce a known-broken commitment)."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    params = CompletionParams(
        messages=[Message(role=Role.USER, content="x")],
        provider_params={
            "eerful.commit_inputs": True,
            "eerful.salt": b"",
        },
    )
    with pytest.raises(EvaluationClientError, match="empty"):
        await client.complete(params)


# ---------------- concurrent complete() chain safety ----------------


@pytest.mark.asyncio
async def test_concurrent_complete_calls_serialize_chain_state(
    trading_critic_bundle: EvaluatorBundle, fake_compute: FakeComputeClient
) -> None:
    """Two `complete()` calls dispatched via `asyncio.gather` on the
    same client must produce a properly chained pair: one's
    `previous_receipt_id` must equal the other's `receipt_id`. Without
    the per-instance lock, both calls could read the same predecessor
    (None) and produce a branched chain instead of a linked one.

    The receipts arrive in arbitrary order (gather doesn't guarantee
    completion order matches submission order under to_thread), so
    we check the linkage symmetrically: one of the two receipts must
    be the predecessor of the other."""
    client, _ = _client(bundle=trading_critic_bundle, fake_compute=fake_compute)
    r1, r2 = await asyncio.gather(
        client.complete(_user_only_params("a")),
        client.complete(_user_only_params("b")),
    )
    chain = sorted(
        [r1.eer, r2.eer],
        key=lambda r: r.previous_receipt_id is None,
        reverse=True,  # the one with None predecessor first
    )
    first, second = chain
    assert first.previous_receipt_id is None
    assert second.previous_receipt_id == first.receipt_id


