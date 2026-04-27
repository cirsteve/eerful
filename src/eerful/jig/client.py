"""`EvaluationClient` — a jig `LLMClient` that produces an EER per call.

Each `.complete()`:

1. Validates the caller's `CompletionParams` against the bound
   evaluator bundle. The bundle's `system_prompt` and `inference_params`
   are authoritative — the receipt has to attest the bundle's criteria,
   not whatever the caller passes. Conflicts raise `EvaluationClientError`.
2. Runs TeeML inference via `ComputeClient.infer_full` (sync, wrapped
   with `asyncio.to_thread` to satisfy `LLMClient.complete`'s async
   contract).
3. Uploads the attestation report to Storage and confirms the bridge's
   content hash matches what the broker reported.
4. Builds an `EnhancedReceipt` per spec §6 — including an optional
   `input_commitment` when `commit_inputs=True`, and an optional
   `previous_receipt_id` for the chain pattern (D.1 trading critic).
5. Returns an `EerfulLLMResponse` (a real `LLMResponse` subclass, not a
   wrapper — see Q5 in the implementation plan) carrying both the model
   output and the receipt.

The `previous_receipt_id` for the chain pattern is read from
`params.provider_params["eerful.previous_receipt_id"]` when present,
otherwise from the instance attribute. After each successful call,
`self._previous_receipt_id` is set to the new receipt's id so the
next `.complete()` chains automatically — overridable per-call via
`provider_params`. Same shape for `eerful.commit_inputs`.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from jig.core.types import (
    CompletionParams,
    LLMClient,
    LLMResponse,
    Role,
    Usage,
)

from eerful.canonical import Address, Bytes32Hex
from eerful.commitment import SaltStore, compute_input_commitment, generate_salt
from eerful.errors import EvaluationClientError
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.zg.compute import ComputeResult
from eerful.zg.storage import StorageClient


class _ComputeProtocol(Protocol):
    """Structural type for the subset of `ComputeClient` we depend on.

    A nominal `ComputeClient` annotation would force every test to
    subclass it (which means standing up an `httpx.Client`, even for
    in-memory fakes). Since the only method we call is `infer_full`,
    a minimal Protocol is the right scope. The real `ComputeClient`
    satisfies it structurally; the test suite's `FakeComputeClient`
    does too without subclassing.
    """

    def infer_full(
        self,
        *,
        provider_address: Address,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ComputeResult: ...


@dataclass
class EerfulLLMResponse(LLMResponse):
    """`LLMResponse` subclass carrying the EER receipt alongside the LLM output.

    Real subclass, not a wrapper: jig's runner and pricing code do
    `isinstance(resp, LLMResponse)` and read `.tool_calls` / `.usage`
    directly, so we want type-identity preserved. Parent's fields
    (content, tool_calls, usage, latency_ms, model) are all required —
    no defaults — so appending `eer` (also required) is safe under
    Python's "no defaulted-then-non-defaulted" dataclass rule.

    A defaulted field added to `LLMResponse` upstream would silently
    break this — `eer` would have to default too, which is wrong for
    our semantics. The integration test
    `test_eerful_llm_response_is_a_llm_response_subclass` is the canary.
    """

    eer: EnhancedReceipt


class EvaluationClient(LLMClient):
    """jig `LLMClient` bound to an `EvaluatorBundle`. Every call produces
    a verifying receipt as a side effect.

    Construction is cheap (no network). Holds references to a
    `ComputeClient` (the bridge wrapper) and a `StorageClient` (for
    attestation report upload at receipt-build time). Both stay owned
    by the caller — `EvaluationClient` does not close them.

    The bound `bundle` and `evaluator_id` MUST be consistent — the
    caller is responsible for hashing the bundle's canonical bytes and
    publishing them to Storage before constructing this client. Use
    `eerful publish-evaluator` to do both in one shot.
    """

    def __init__(
        self,
        *,
        compute: _ComputeProtocol,
        storage: StorageClient,
        bundle: EvaluatorBundle,
        evaluator_id: Bytes32Hex,
        provider_address: Address,
        salt_store: SaltStore | None = None,
        commit_inputs: bool = False,
        previous_receipt_id: Bytes32Hex | None = None,
    ) -> None:
        self._compute = compute
        self._storage = storage
        self._bundle = bundle
        self._evaluator_id = evaluator_id
        self._provider_address = provider_address
        self._salt_store = salt_store
        self._commit_inputs_default = commit_inputs
        self._previous_receipt_id = previous_receipt_id

    @property
    def previous_receipt_id(self) -> Bytes32Hex | None:
        """Read-only view; updated automatically after each successful call.
        Caller may override per-call via
        `params.provider_params["eerful.previous_receipt_id"]`."""
        return self._previous_receipt_id

    async def complete(self, params: CompletionParams) -> EerfulLLMResponse:
        self._validate_params(params)
        messages = self._convert_messages(params)

        provider_params = params.provider_params or {}
        commit_inputs = bool(
            provider_params.get("eerful.commit_inputs", self._commit_inputs_default)
        )
        previous = provider_params.get(
            "eerful.previous_receipt_id", self._previous_receipt_id
        )

        # TeeML inference. ComputeClient is sync; jig's contract is async.
        # to_thread is the right hammer for v1 — chains are serial and
        # 3-deep, so the cost of an extra event-loop hop is irrelevant.
        # An `AsyncComputeClient` is the proper fix and a follow-on.
        bundle_params = self._bundle.inference_params or {}
        start = time.monotonic()
        result: ComputeResult = await asyncio.to_thread(
            self._compute.infer_full,
            provider_address=self._provider_address,
            messages=messages,
            temperature=bundle_params.get("temperature"),
            max_tokens=bundle_params.get("max_tokens"),
        )
        latency_ms = (time.monotonic() - start) * 1000.0

        # Upload the attestation report. Verify the storage-side hash
        # matches what TeeML reported — the bridge already content-checks
        # but Step 4-style attribution belongs here too (this is the
        # producer side; the verifier runs the symmetric check).
        report_hash_storage = await asyncio.to_thread(
            self._storage.upload_blob, result.attestation_report_bytes
        )
        if report_hash_storage != result.attestation_report_hash:
            raise EvaluationClientError(
                f"attestation report hash mismatch after upload: "
                f"compute reported {result.attestation_report_hash}, "
                f"storage returned {report_hash_storage}"
            )

        input_commitment = self._compute_commitment_or_none(
            params, commit_inputs, provider_params
        )
        score_block = self._parse_score_block(result.response_content)

        receipt = EnhancedReceipt.build(
            created_at=datetime.now(timezone.utc),
            evaluator_id=self._evaluator_id,
            evaluator_version=self._bundle.version,
            provider_address=self._provider_address,
            chat_id=result.chat_id,
            response_content=result.response_content,
            attestation_report_hash=result.attestation_report_hash,
            enclave_pubkey=result.enclave_pubkey,
            enclave_signature=result.enclave_signature,
            input_commitment=input_commitment,
            previous_receipt_id=previous,
            output_score_block=score_block,
        )

        # Persist the salt to the salt store (best effort, but raises if
        # the store itself is broken — silent failure here would let a
        # producer build commitments they can't reveal later, which is
        # the worst-case failure mode for §6.7 / SaltStore).
        if input_commitment is not None and self._salt_store is not None:
            self._salt_store.put(receipt.receipt_id, self._last_salt_for_commit)

        # Auto-chain: next call gets this receipt as its predecessor
        # unless the caller overrides via provider_params.
        self._previous_receipt_id = receipt.receipt_id

        return EerfulLLMResponse(
            content=result.response_content,
            tool_calls=None,
            usage=Usage(
                # Zero when the upstream provider didn't report usage;
                # `cost=None` because Phase 2 (broker credit fee) is
                # not wired yet. Both signal "we don't know" honestly.
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cost=None,
            ),
            latency_ms=latency_ms,
            model=result.model_served,
            eer=receipt,
        )

    # ---------------- internals ----------------

    def _validate_params(self, params: CompletionParams) -> None:
        """Refuse anything that would let the caller change what the bundle
        attests to.

        - `tools`: TeeML doesn't support tool calls; silently dropping
          them would produce a receipt whose response did NOT use the
          tools the caller specified. Reject loudly.
        - `system`: bundle's `system_prompt` is the criteria. A non-None
          `params.system` that differs from the bundle's prompt would
          produce a receipt that attests inference under different
          criteria than the receipt's `evaluator_id` claims.
        - `temperature` / `max_tokens`: bundle's `inference_params`
          win for receipt-comparability across producers. Conflicting
          caller values are a programming bug — refuse.
        """
        if params.tools is not None:
            raise EvaluationClientError(
                "EvaluationClient does not support tool calls — TeeML signs "
                "the response text and tools would alter what gets signed"
            )
        if params.system is not None and params.system != self._bundle.system_prompt:
            raise EvaluationClientError(
                "params.system conflicts with the bundle's system_prompt; "
                "the receipt would attest evaluation under different criteria "
                "than evaluator_id claims. Pass system=None to use the bundle's."
            )
        bundle_params = self._bundle.inference_params or {}
        for key in ("temperature", "max_tokens"):
            caller_val = getattr(params, key)
            bundle_val = bundle_params.get(key)
            if caller_val is not None and bundle_val is not None and caller_val != bundle_val:
                raise EvaluationClientError(
                    f"params.{key}={caller_val!r} conflicts with bundle's "
                    f"inference_params[{key!r}]={bundle_val!r}; remove the "
                    "caller-side override to use the bundle's value"
                )

    def _convert_messages(self, params: CompletionParams) -> list[dict[str, str]]:
        """jig `Message[]` → OpenAI-shape `[{role, content}, ...]`.

        Always prepends a system message with the bundle's
        `system_prompt` — even if the caller passed `params.system` (we
        rejected non-None mismatches above; None just means "use the
        bundle's" which is exactly what we do). The bundle's prompt is
        what the receipt attests; the system message in the wire request
        must match.
        """
        out: list[dict[str, str]] = [
            {"role": "system", "content": self._bundle.system_prompt}
        ]
        for m in params.messages:
            # Skip caller-supplied system messages; the bundle owns the
            # system slot. Tool/assistant messages flow through unchanged
            # — TeeML accepts any role the upstream provider does.
            if m.role == Role.SYSTEM:
                continue
            out.append({"role": m.role.value, "content": m.content})
        return out

    def _compute_commitment_or_none(
        self,
        params: CompletionParams,
        commit_inputs: bool,
        provider_params: dict[str, Any],
    ) -> Bytes32Hex | None:
        """When `commit_inputs` is True, derive the input bytes from the
        user-role messages and produce a §6.7 commitment.

        The salt is taken from `provider_params["eerful.salt"]` if
        present (lets a chain pin the same salt across rounds without
        plumbing a `SaltStore` everywhere); otherwise generated fresh.
        Stashes the salt on `self._last_salt_for_commit` so the
        post-receipt SaltStore.put can persist it under the receipt_id.
        """
        if not commit_inputs:
            self._last_salt_for_commit = b""  # not used; cleared for hygiene
            return None
        user_content = "\n".join(
            m.content for m in params.messages if m.role == Role.USER
        )
        input_bytes = user_content.encode("utf-8")
        salt_override = provider_params.get("eerful.salt")
        if isinstance(salt_override, bytes):
            salt = salt_override
        else:
            salt = generate_salt()
        self._last_salt_for_commit = salt
        return compute_input_commitment(input_bytes, self._evaluator_id, salt)

    def _parse_score_block(self, response_content: str) -> dict[str, Any] | None:
        """Best-effort: parse the response as JSON and return the parsed
        dict, or None if it doesn't parse / isn't a dict.

        Schema validation against `bundle.output_schema` is NOT done
        here. The receipt records what the model produced; verifiers
        run schema validation at Step 3. If we rejected non-conforming
        scores at construction time, the producer would have no
        receipt of the failed evaluation — that's the wrong default.
        """
        try:
            parsed = json.loads(response_content)
        except (ValueError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed
