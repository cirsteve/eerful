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

from eerful.canonical import Address, Bytes32Hex, is_bytes32_hex, to_lower_hex
from eerful.commitment import SaltStore, compute_input_commitment, generate_salt
from eerful.errors import EvaluationClientError, TrustViolation
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.zg.compute import ComputeResult
from eerful.zg.storage import StorageClient, UploadResult


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
        evaluator_storage_root: Bytes32Hex | None = None,
    ) -> None:
        # Fail fast on a stale or wrong `evaluator_id`. A mismatched
        # value would silently produce receipts whose Step 2 verification
        # fails downstream — far from the construction site, with
        # confusing attribution. Cheaper to refuse here. Caller still
        # passes both `bundle` and `evaluator_id` (rather than us
        # deriving from bundle) so the API matches the published Day 4
        # plan; the validation just makes the redundancy load-bearing.
        bundle_id = bundle.evaluator_id()
        if evaluator_id != bundle_id:
            raise EvaluationClientError(
                f"evaluator_id {evaluator_id!r} does not match "
                f"bundle.evaluator_id() {bundle_id!r}"
            )
        self._compute = compute
        self._storage = storage
        self._bundle = bundle
        self._evaluator_id = evaluator_id
        self._provider_address = provider_address
        self._salt_store = salt_store
        self._commit_inputs_default = commit_inputs
        self._previous_receipt_id = previous_receipt_id
        # Resolve the bundle's storage_root: either trust the caller's
        # override (publish-evaluator already uploaded; we have the root
        # from the side-file or environment) or upload on construction.
        # Bundle bytes are small (<4KB) and `getOrUpload` short-circuits
        # on cache hit, so the construct-time upload is effectively free
        # for re-runs. The override path keeps tests (and producers who
        # don't have storage at construct time) ergonomic.
        if evaluator_storage_root is None:
            upload = storage.upload_blob(bundle.canonical_bytes())
            if upload.content_hash != evaluator_id:
                # Byzantine evidence: storage's returned hash diverges
                # from the bytes we sent. Same shape as the
                # `_publish_evaluator` defense-in-depth check in cli.py;
                # raise the same exception type so the integrity
                # attribution is consistent across surfaces (caller
                # treating `EvaluationClientError` as a parameter-bug
                # signal would mishandle this).
                raise TrustViolation(
                    f"bundle upload returned content_hash {upload.content_hash} "
                    f"but bundle.evaluator_id()={evaluator_id} (canonical "
                    "encoder drift or storage byte tampering)"
                )
            self._evaluator_storage_root: Bytes32Hex = upload.storage_root
        else:
            canonical = to_lower_hex(evaluator_storage_root)
            if not is_bytes32_hex(canonical):
                raise EvaluationClientError(
                    f"evaluator_storage_root must be 0x-prefixed 64-char hex, "
                    f"got {evaluator_storage_root!r}"
                )
            self._evaluator_storage_root = canonical
        # Per-instance lock serializes `complete()` calls so the
        # auto-chain pattern (`self._previous_receipt_id` read at start,
        # write at end) is concurrency-safe. Without this, two
        # `await asyncio.gather(client.complete(p1), client.complete(p2))`
        # calls could both read the same predecessor and produce a
        # branched chain. Producers wanting parallel independent
        # receipts should use multiple client instances or pin
        # `eerful.previous_receipt_id` per-call.
        self._chain_lock = asyncio.Lock()

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
        commit_inputs = self._resolve_commit_inputs(provider_params)

        # The chain lock makes the read-of-predecessor / produce-receipt /
        # write-successor sequence atomic. Without it, concurrent
        # `complete()` calls on the same client could both read the
        # same predecessor (branching the chain) or both write at the
        # end (last-write-wins). The lock costs a per-instance
        # serialization, which matches the chain pattern's natural
        # shape (sequential v1 -> v2 -> v3); parallel-independent
        # users should use multiple client instances.
        async with self._chain_lock:
            previous = self._resolve_previous_receipt_id(provider_params)

            # Bundle's inference_params win when set, but caller-passed
            # values flow through when the bundle leaves them unset.
            # Spec §6.5 frames inference_params as informational, so a
            # bundle that omits `temperature` is signalling "use whatever
            # the caller wants." `_validate_params` already rejects
            # bundle/caller conflicts; this resolves the caller-only case.
            bundle_params = self._bundle.inference_params or {}
            effective_temperature = bundle_params.get(
                "temperature", params.temperature
            )
            effective_max_tokens = bundle_params.get(
                "max_tokens", params.max_tokens
            )

            # TeeML inference. ComputeClient is sync; jig's contract is async.
            # to_thread is the right hammer for v1 — chains are serial and
            # 3-deep, so the cost of an extra event-loop hop is irrelevant.
            # An `AsyncComputeClient` is the proper fix and a follow-on.
            start = time.monotonic()
            result: ComputeResult = await asyncio.to_thread(
                self._compute.infer_full,
                provider_address=self._provider_address,
                messages=messages,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
            latency_ms = (time.monotonic() - start) * 1000.0

            # Upload the attestation report. Verify the storage-side hash
            # matches what TeeML reported — the bridge already content-checks
            # but Step 4-style attribution belongs here too (this is the
            # producer side; the verifier runs the symmetric check).
            report_upload: UploadResult = await asyncio.to_thread(
                self._storage.upload_blob, result.attestation_report_bytes
            )
            if report_upload.content_hash != result.attestation_report_hash:
                # Byzantine evidence: same shape as the constructor's
                # bundle-upload check above. The compute provider's
                # report bytes hashed to one value, storage returned a
                # different one — never trust the storage URI in that
                # case.
                raise TrustViolation(
                    f"attestation report hash mismatch after upload: "
                    f"compute reported {result.attestation_report_hash}, "
                    f"storage returned {report_upload.content_hash}"
                )

            input_commitment = self._compute_commitment_or_none(
                params, commit_inputs, provider_params
            )
            score_block = self._parse_score_block(result.response_content)

            receipt = EnhancedReceipt.build(
                created_at=datetime.now(timezone.utc),
                evaluator_id=self._evaluator_id,
                evaluator_storage_root=self._evaluator_storage_root,
                evaluator_version=self._bundle.version,
                provider_address=self._provider_address,
                chat_id=result.chat_id,
                response_content=result.response_content,
                attestation_report_hash=result.attestation_report_hash,
                attestation_storage_root=report_upload.storage_root,
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

    def _resolve_commit_inputs(self, provider_params: dict[str, Any]) -> bool:
        """Read `eerful.commit_inputs` from provider_params with strict
        type validation. Rejects truthy-coerced values (e.g. the strings
        `"false"` / `"0"`, the integer `1`) — silently flipping a
        commitment on or off based on a YAML-loaded string would be a
        nasty surprise for the producer."""
        if "eerful.commit_inputs" not in provider_params:
            return self._commit_inputs_default
        value = provider_params["eerful.commit_inputs"]
        if not isinstance(value, bool):
            raise EvaluationClientError(
                f"provider_params['eerful.commit_inputs'] must be bool, "
                f"got {type(value).__name__}={value!r}"
            )
        return value

    def _resolve_previous_receipt_id(
        self, provider_params: dict[str, Any]
    ) -> Bytes32Hex | None:
        """Read `eerful.previous_receipt_id` from provider_params with
        strict shape validation: must be `None` or a Bytes32Hex string
        (`0x` + 64 lowercase hex chars).

        The shape check matters here, not just at receipt construction
        time. `EnhancedReceipt.build` will reject a malformed value via
        pydantic, but the error surfaces deep in the receipt module
        with no breadcrumbs back to `provider_params`. Catching it at
        the call boundary tells the caller exactly what they passed
        wrong, where, and why.
        """
        if "eerful.previous_receipt_id" not in provider_params:
            return self._previous_receipt_id
        value = provider_params["eerful.previous_receipt_id"]
        if value is None:
            return None
        if not isinstance(value, str):
            raise EvaluationClientError(
                f"provider_params['eerful.previous_receipt_id'] must be a "
                f"Bytes32Hex string or None, got {type(value).__name__}={value!r}"
            )
        try:
            canonical = to_lower_hex(value)
        except (TypeError, ValueError) as exc:
            raise EvaluationClientError(
                f"provider_params['eerful.previous_receipt_id'] is not a "
                f"valid hex string: {value!r}"
            ) from exc
        if not is_bytes32_hex(canonical):
            raise EvaluationClientError(
                f"provider_params['eerful.previous_receipt_id'] must be "
                f"0x-prefixed 64-char hex (32 bytes), got {value!r}"
            )
        return canonical

    def _resolve_salt(self, provider_params: dict[str, Any]) -> bytes | None:
        """Read `eerful.salt` from provider_params with strict type
        validation: must be `bytes` (non-empty) if present, `None` (or
        absent) to signal "generate a fresh one."

        Silently regenerating on a wrong-type input would be a
        footgun: a producer who explicitly passed `eerful.salt="abc"`
        thinking they pinned a salt would instead get a random one
        per call, breaking the chain pattern's stable input commitment
        without any error. Loud failure is the right default.

        Empty bytes (`b""`) are also rejected. An empty salt provides
        zero entropy, defeating §6.7's brute-force-protection purpose
        — same reasoning as `commitment.generate_salt(0)` raising
        ValueError. A producer who genuinely wants no salt can call
        `compute_input_commitment` directly with whatever bytes they
        want; the EvaluationClient surface refuses to construct a
        receipt under a known-broken commitment.
        """
        if "eerful.salt" not in provider_params:
            return None
        value = provider_params["eerful.salt"]
        if value is None:
            return None
        if not isinstance(value, bytes):
            raise EvaluationClientError(
                f"provider_params['eerful.salt'] must be bytes or None, "
                f"got {type(value).__name__}={value!r}"
            )
        if len(value) == 0:
            raise EvaluationClientError(
                "provider_params['eerful.salt'] must be non-empty bytes; "
                "an empty salt provides zero entropy and defeats §6.7's "
                "brute-force-reversal protection"
            )
        return value

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
        # Truthy check (not `is not None`): `tools=[]` from a serializer
        # that emits "no tools" as an empty list is semantically the
        # same as `tools=None` — no tool calls would happen — so
        # accept it. Only a non-empty list is the unsupported case.
        if params.tools:
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

        Salt resolution: `_resolve_salt` returns the caller-pinned salt
        (lets a chain pin one across rounds without plumbing a
        `SaltStore` everywhere) or None to signal generate-fresh.
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
        # `is None` rather than truthy: `_resolve_salt` already rejects
        # empty bytes, so we'd never see `b""` here; the explicit check
        # documents that None means "generate fresh" and any other
        # value (always non-empty by construction) should be used as-is.
        salt = self._resolve_salt(provider_params)
        if salt is None:
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
