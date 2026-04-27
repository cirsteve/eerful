"""Shared fixtures for the jig-adapter test suite.

The two recurring needs across `test_evaluation_client.py`,
`test_evaluation_grader.py`, and `test_pipeline_integration.py` are:

- A `FakeComputeClient` that synthesizes a `ComputeResult` with a
  *real* secp256k1 signature so the resulting receipt round-trips
  through `verify_step_6_enclave_signature`. Without this, every test
  would have to repeat the signing dance.
- A small `EvaluatorBundle` with a numeric `output_schema` plus a
  matching JSON response body, so the grader's score-extraction path
  has something realistic to work with.

`MockStorageClient` lives in `eerful.zg.storage` already; tests import
it directly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from eth_keys import keys
from eth_utils import keccak

from eerful.canonical import to_lower_hex
from eerful.evaluator import EvaluatorBundle
from eerful.zg.compute import ComputeResult


def _sign_personal(text: str, privkey_bytes: bytes = b"\x42" * 32) -> tuple[str, str, str]:
    """EIP-191 personal_sign over `text`. Returns (pubkey_hex,
    signature_hex, address_hex). Mirrors the helper in
    test_zg_compute.py — tests in this directory need the same
    primitive without importing from the sibling test module."""
    text_bytes = text.encode("utf-8")
    msg_hash = keccak(
        b"\x19Ethereum Signed Message:\n" + str(len(text_bytes)).encode() + text_bytes
    )
    pk = keys.PrivateKey(privkey_bytes)
    sig = pk.sign_msg_hash(msg_hash)
    return (
        to_lower_hex(pk.public_key.to_bytes()),
        "0x" + sig.to_bytes().hex(),
        to_lower_hex(pk.public_key.to_canonical_address()),
    )


class FakeComputeClient:
    """Stand-in for `eerful.zg.compute.ComputeClient`.

    Returns a ComputeResult whose `enclave_pubkey` and
    `enclave_signature` are a real secp256k1 keypair signing the
    response body — so `verify_step_6_enclave_signature` passes on
    receipts produced through this client. Tests can override the
    response_content via `set_next_response` to exercise score-block
    parsing variants.

    Tracks calls so tests can assert what messages were sent without
    plumbing a separate spy.
    """

    def __init__(self, *, response_content: str = '{"score": 0.7}') -> None:
        self._response_content = response_content
        self.calls: list[dict[str, Any]] = []
        self._counter = 0

    def set_next_response(self, content: str) -> None:
        self._response_content = content

    def infer_full(
        self,
        *,
        provider_address: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ComputeResult:
        self._counter += 1
        chat_id = f"chat-{self._counter:03d}"
        # Sign the response we're about to return so step 6 verifies.
        pubkey, sig, address = _sign_personal(self._response_content)
        # Synthesize an attestation report; actual content doesn't
        # matter because Step 5 isn't exercised here.
        report_bytes = json.dumps(
            {"quote": "fake", "tcb_info": {"compose_hash": "00" * 32}}
        ).encode()
        report_hash = "0x" + hashlib.sha256(report_bytes).hexdigest()
        self.calls.append(
            {
                "provider_address": provider_address,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_id": chat_id,
            }
        )
        return ComputeResult(
            chat_id=chat_id,
            response_content=self._response_content,
            model_served="zai-org/GLM-5-FP8",
            provider_endpoint="https://provider.test/v1",
            enclave_pubkey=pubkey,
            enclave_signature=sig,
            signing_address=address,
            attestation_report_bytes=report_bytes,
            attestation_report_hash=report_hash,
            input_tokens=42,
            output_tokens=21,
        )


SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["risk", "novelty", "commentary"],
    "properties": {
        "risk": {"type": "number", "minimum": 0, "maximum": 1},
        "novelty": {"type": "number", "minimum": 0, "maximum": 1},
        "commentary": {"type": "string"},
    },
}


@pytest.fixture
def trading_critic_bundle() -> EvaluatorBundle:
    """A small but realistic critic bundle with two numeric dimensions
    plus a string commentary field. Used to exercise score extraction
    (numeric fields → Score, string fields → ignored)."""
    return EvaluatorBundle(
        version="trading-critic@0.1.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="You are a critic. Score the strategy. Output JSON.",
        output_schema=SCHEMA,
        inference_params={"temperature": 0.0, "max_tokens": 500},
    )


@pytest.fixture
def critic_response_json() -> str:
    """Canonical critic output body — what `FakeComputeClient` returns
    by default. Validates against `trading_critic_bundle.output_schema`
    so Step 3 verification passes."""
    return json.dumps(
        {"risk": 0.4, "novelty": 0.7, "commentary": "Solid mean-reversion."}
    )


@pytest.fixture
def fake_compute(critic_response_json: str) -> FakeComputeClient:
    return FakeComputeClient(response_content=critic_response_json)
