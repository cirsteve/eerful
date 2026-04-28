"""Integration test for the D.1 trading-critic demo.

Exercises `examples.trading_critic.demo.run_demo` end-to-end against
in-process fakes:

- `FakeComputeClient` from `tests/jig/conftest.py` (real secp256k1
  signature so Step 6 verifies)
- `MockStorageClient` from `eerful.zg.storage`

The bundle used here intentionally has NO `accepted_compose_hashes`
so Step 5 reports `gating="skipped"` rather than failing — the fake
report bytes wouldn't match a real testnet allowlist anyway. The
real-testnet recording lives in the maintainer's manual run of
`demo.py` against Provider 1 before submission (gitignored receipts).

What this test pins:

- `run_demo` produces exactly three receipts.
- Each receipt round-trips through `verify_receipt` (Steps 1-3 + 6,
  plus Step 5 compose-hash subset against the no-allowlist bundle).
- Chain linkage: None → v1 → v2 → v3.
- All three share one non-None `input_commitment` (the §6.7
  chain-pattern invariant the demo is built around).
- All three name the same `evaluator_id`.
- Receipts are persisted to the receipts directory.
- A second `run_demo` call against a fresh storage instance starting
  from the existing salt produces matching commitments — pins that the
  salt persistence is load-bearing for chain stability across runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Make `examples/trading_critic/demo.py` importable without packaging it.
_EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "trading_critic"
sys.path.insert(0, str(_EXAMPLES))

from demo import run_demo  # noqa: E402

from eerful.commitment import SaltStore  # noqa: E402
from eerful.evaluator import EvaluatorBundle  # noqa: E402
from eerful.verify import verify_receipt  # noqa: E402
from eerful.zg.storage import MockStorageClient  # noqa: E402

from tests.jig.conftest import FakeComputeClient  # noqa: E402


_PROVIDER = "0x" + "b" * 40

_CRITIC_RESPONSE_V1 = json.dumps(
    {
        "risk": 0.7,
        "novelty": 0.3,
        "robustness": 0.4,
        "overall": 0.45,
        "commentary": "MA crossover with no stops; full-size positions concentrate risk.",
    }
)
_CRITIC_RESPONSE_V2 = json.dumps(
    {
        "risk": 0.5,
        "novelty": 0.4,
        "robustness": 0.6,
        "overall": 0.55,
        "commentary": "ATR-based sizing closes the position-risk gap; still single-asset.",
    }
)
_CRITIC_RESPONSE_V3 = json.dumps(
    {
        "risk": 0.4,
        "novelty": 0.65,
        "robustness": 0.75,
        "overall": 0.65,
        "commentary": "Regime filter + contrarian sub-strategy gives multi-regime coverage.",
    }
)


class _ScriptedComputeClient(FakeComputeClient):
    """FakeComputeClient that returns a different scripted response per
    call — so v1/v2/v3 receipts have distinct `output_score_block`s.

    Without this, every call would return the same `response_content`,
    producing receipts whose `chat_id` differs (counter-incremented in
    the parent class) but whose score blocks are identical — the
    chain's narrative would lose information. We're not testing critic
    quality (that's the manual real-testnet recording); we are
    testing that scripted responses flow through `run_demo`'s
    receipt-construction loop without distortion.
    """

    def __init__(self) -> None:
        super().__init__(response_content=_CRITIC_RESPONSE_V1)
        self._scripted = [_CRITIC_RESPONSE_V1, _CRITIC_RESPONSE_V2, _CRITIC_RESPONSE_V3]
        self._call_idx = 0

    def infer_full(self, **kwargs: Any) -> Any:
        if self._call_idx < len(self._scripted):
            self._response_content = self._scripted[self._call_idx]
            self._call_idx += 1
        return super().infer_full(**kwargs)


@pytest.fixture
def demo_bundle() -> EvaluatorBundle:
    """Bundle without `accepted_compose_hashes` — the FakeComputeClient's
    synthesized report bytes won't match a real testnet allowlist, so
    we let Step 5 land in `gating="skipped"` and the demo's chain
    properties remain testable."""
    return EvaluatorBundle(
        version="trading-critic@0.1.0-test",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="You are a critic. Reply JSON.",
        output_schema={
            "type": "object",
            "required": ["risk", "novelty", "robustness", "overall", "commentary"],
            "additionalProperties": False,
            "properties": {
                "risk": {"type": "number", "minimum": 0, "maximum": 1},
                "novelty": {"type": "number", "minimum": 0, "maximum": 1},
                "robustness": {"type": "number", "minimum": 0, "maximum": 1},
                "overall": {"type": "number", "minimum": 0, "maximum": 1},
                "commentary": {"type": "string", "minLength": 1},
            },
        },
        inference_params={"temperature": 0.0, "max_tokens": 2000},
    )


def _write_strategies(strategies_dir: Path) -> None:
    """Three minimally-distinct strategy files. Content doesn't drive
    the test — `_ScriptedComputeClient` produces fixed responses
    regardless — but `run_demo` reads the files, so they must exist."""
    strategies_dir.mkdir(parents=True, exist_ok=True)
    (strategies_dir / "v1.md").write_text("# v1\nMA crossover only.\n")
    (strategies_dir / "v2.md").write_text("# v2\nMA crossover + stops.\n")
    (strategies_dir / "v3.md").write_text("# v3\nv2 + regime filter.\n")


def test_run_demo_produces_three_chained_receipts(
    tmp_path: Path, demo_bundle: EvaluatorBundle
) -> None:
    strategies_dir = tmp_path / "strategies"
    receipts_dir = tmp_path / "receipts"
    salt_path = tmp_path / ".salt"
    _write_strategies(strategies_dir)

    storage = MockStorageClient()
    compute = _ScriptedComputeClient()
    salt_store = SaltStore(salt_path)

    receipts = run_demo(
        compute=compute,
        storage=storage,
        bundle=demo_bundle,
        provider_address=_PROVIDER,
        strategies_dir=strategies_dir,
        receipts_dir=receipts_dir,
        salt_store=salt_store,
    )

    assert len(receipts) == 3

    # Chain linkage
    assert receipts[0].previous_receipt_id is None
    assert receipts[1].previous_receipt_id == receipts[0].receipt_id
    assert receipts[2].previous_receipt_id == receipts[1].receipt_id

    # Commitment stability — the §6.7 invariant the demo is built around
    commitments = {r.input_commitment for r in receipts}
    assert len(commitments) == 1
    assert None not in commitments

    # Bundle stability
    assert {r.evaluator_id for r in receipts} == {demo_bundle.evaluator_id()}

    # Score block round-trips through the receipt
    assert receipts[0].output_score_block == json.loads(_CRITIC_RESPONSE_V1)
    assert receipts[1].output_score_block == json.loads(_CRITIC_RESPONSE_V2)
    assert receipts[2].output_score_block == json.loads(_CRITIC_RESPONSE_V3)

    # On-disk persistence
    for v in ("v1", "v2", "v3"):
        assert (receipts_dir / f"{v}.json").exists()


def test_run_demo_receipts_pass_offline_verification(
    tmp_path: Path, demo_bundle: EvaluatorBundle
) -> None:
    """Each produced receipt must round-trip through `verify_receipt`
    (Steps 1-3 + 5 compose-hash subset + 6) on the bytes the demo
    uploaded. Catches regressions where the demo writes receipts that
    look right structurally but don't actually verify."""
    strategies_dir = tmp_path / "strategies"
    receipts_dir = tmp_path / "receipts"
    salt_path = tmp_path / ".salt"
    _write_strategies(strategies_dir)

    storage = MockStorageClient()
    compute = _ScriptedComputeClient()
    salt_store = SaltStore(salt_path)

    receipts = run_demo(
        compute=compute,
        storage=storage,
        bundle=demo_bundle,
        provider_address=_PROVIDER,
        strategies_dir=strategies_dir,
        receipts_dir=receipts_dir,
        salt_store=salt_store,
    )

    bundle_bytes = demo_bundle.canonical_bytes()
    for receipt in receipts:
        # `report_bytes=None` skips Step 5 — FakeComputeClient's report
        # is not a parseable real attestation. Steps 1, 2, 3, 6 still
        # run, which is what we care about for the demo's offline
        # verification story.
        result = verify_receipt(receipt, bundle_bytes, report_bytes=None)
        assert result.bundle.evaluator_id() == demo_bundle.evaluator_id()


def test_run_demo_persists_salt_across_invocations(
    tmp_path: Path, demo_bundle: EvaluatorBundle
) -> None:
    """A second `run_demo` against fresh storage but the SAME SaltStore
    must produce receipts whose `input_commitment` matches the first
    run's. This is what `.salt` is for in production: across re-runs,
    the chain's input identity stays stable.

    Sanity check that the salt-persistence path isn't accidentally
    regenerating the salt every run (which would silently break chain
    semantics: every re-run would produce a new commitment, and a
    verifier's reveal of an old commitment would fail).
    """
    strategies_dir = tmp_path / "strategies"
    receipts_dir_a = tmp_path / "receipts_a"
    receipts_dir_b = tmp_path / "receipts_b"
    salt_path = tmp_path / ".salt"
    _write_strategies(strategies_dir)

    salt_store = SaltStore(salt_path)

    receipts_a = run_demo(
        compute=_ScriptedComputeClient(),
        storage=MockStorageClient(),
        bundle=demo_bundle,
        provider_address=_PROVIDER,
        strategies_dir=strategies_dir,
        receipts_dir=receipts_dir_a,
        salt_store=salt_store,
    )
    receipts_b = run_demo(
        compute=_ScriptedComputeClient(),
        storage=MockStorageClient(),
        bundle=demo_bundle,
        provider_address=_PROVIDER,
        strategies_dir=strategies_dir,
        receipts_dir=receipts_dir_b,
        salt_store=salt_store,
    )

    # Receipts differ overall (created_at, chat_id, signatures all change)
    # but the commitment must match — that's the contract.
    assert receipts_a[0].input_commitment is not None
    assert receipts_a[0].input_commitment == receipts_b[0].input_commitment
    assert receipts_a[1].input_commitment == receipts_b[1].input_commitment
    assert receipts_a[2].input_commitment == receipts_b[2].input_commitment


def test_run_demo_uploads_bundle_and_reports(
    tmp_path: Path, demo_bundle: EvaluatorBundle
) -> None:
    """The demo must upload the bundle's canonical bytes AND each
    attestation report so a verifier with the receipt can fetch them
    by storage_root. This is the producer-side half of the spec §7.1
    Steps 2 and 4 contract.
    """
    strategies_dir = tmp_path / "strategies"
    receipts_dir = tmp_path / "receipts"
    salt_path = tmp_path / ".salt"
    _write_strategies(strategies_dir)

    storage = MockStorageClient()
    compute = _ScriptedComputeClient()
    salt_store = SaltStore(salt_path)

    receipts = run_demo(
        compute=compute,
        storage=storage,
        bundle=demo_bundle,
        provider_address=_PROVIDER,
        strategies_dir=strategies_dir,
        receipts_dir=receipts_dir,
        salt_store=salt_store,
    )

    # Bundle is fetchable at evaluator_id / evaluator_storage_root
    fetched_bundle = storage.download_blob(
        receipts[0].evaluator_id, receipts[0].evaluator_storage_root
    )
    assert fetched_bundle == demo_bundle.canonical_bytes()

    # Each receipt's report is fetchable at its
    # attestation_report_hash / attestation_storage_root pair
    for receipt in receipts:
        report = storage.download_blob(
            receipt.attestation_report_hash, receipt.attestation_storage_root
        )
        assert len(report) > 0
