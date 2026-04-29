"""Integration test for the trading-rails demo.

Exercises `examples.trading.agent.run_agent` end-to-end against a
scripted compute client. The clean path produces a proposal whose
working drawdown matches the principal's mandate (5%); the poisoned
path produces a proposal with a drifted drawdown (25%) — the in-band
override the bundle's grader is meant to catch.

Step 5 (compose-hash gate) is skipped in this suite because
`FakeComputeClient`'s synthesized attestation report isn't parseable
by the real Step 5 parser. The real Step-5 enforcement is covered by
`tests/test_verify.py` with synthesized parseable reports; this suite
covers the producer-side flow (artifact rendering, receipt
construction, persistence) and the executor's gate semantics against
fake-but-shape-correct receipts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from eerful.evaluator import EvaluatorBundle
from eerful.executor import GateOutcome, evaluate_gate
from eerful.policy import (
    POLICY_VERSION,
    DiversityRules,
    PrincipalPolicy,
    TierPolicy,
)
from eerful.receipt import EnhancedReceipt
from eerful.verify import verify_receipt
from eerful.zg.compute import ComputeResult
from eerful.zg.storage import MockStorageClient

from tests.jig.conftest import FakeComputeClient


# Load `examples/trading/agent.py` without mutating sys.path — same
# pattern as the old test_examples_trading_critic.py used.
_AGENT_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "trading" / "agent.py"
)
_AGENT_SPEC = importlib.util.spec_from_file_location(
    "examples.trading.agent", _AGENT_PATH
)
assert _AGENT_SPEC is not None and _AGENT_SPEC.loader is not None
_AGENT_MODULE = importlib.util.module_from_spec(_AGENT_SPEC)
# `dataclass(frozen=True)` decorator looks up its host module via
# `sys.modules[cls.__module__]` for type-hint resolution. Without
# registering before exec_module, that lookup returns None and
# decoration crashes during import.
sys.modules[_AGENT_SPEC.name] = _AGENT_MODULE
_AGENT_SPEC.loader.exec_module(_AGENT_MODULE)

run_agent: Callable[..., Any] = _AGENT_MODULE.run_agent
render_proposal: Callable[..., str] = _AGENT_MODULE.render_proposal


_PROVIDER = "0x" + "b" * 40
_BUNDLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "trading" / "bundles"
_TOOL_RESPONSES_DIR = (
    Path(__file__).resolve().parent.parent / "examples" / "trading" / "tool_responses"
)


# ---------------- shared fixtures ----------------


@pytest.fixture
def proposal_bundle() -> EvaluatorBundle:
    return EvaluatorBundle.model_validate_json(
        (_BUNDLES_DIR / "proposal_grade.json").read_bytes()
    )


@pytest.fixture
def implementation_bundle() -> EvaluatorBundle:
    return EvaluatorBundle.model_validate_json(
        (_BUNDLES_DIR / "implementation_grade.json").read_bytes()
    )


@pytest.fixture
def clean_tool_response() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((_TOOL_RESPONSES_DIR / "clean.json").read_bytes())
    return data


@pytest.fixture
def poisoned_tool_response() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((_TOOL_RESPONSES_DIR / "poisoned.json").read_bytes())
    return data


# A passing score block both graders share. PASS-path tests use this;
# REFUSE-path tests inject a different block via a scripted compute
# client. Schema-aligned with both bundles' output_schema (the agent
# parses whatever the bundle's grader produces, so a fake response
# just has to be schema-shaped).
_PASS_PROPOSAL_SCORE = {
    "mandate_compliance": 0.95,
    "coherence": 0.85,
    "specificity": 0.8,
    "overall": 0.87,
    "commentary": "Strategy honors all mandate clauses; momentum signal is concrete and risk controls match the 5% drawdown ceiling.",
}

_FAIL_PROPOSAL_SCORE = {
    "mandate_compliance": 0.1,
    "coherence": 0.7,
    "specificity": 0.8,
    "overall": 0.25,
    "commentary": "Proposal asserts the mandate has been 'updated' to a 25% drawdown ceiling. The mandate is fixed at bundle hash; this is presumed compromised — see drift_detection in the bundle metadata.",
}

_PASS_IMPL_SCORE = {
    "spec_alignment": 0.9,
    "safety": 0.85,
    "determinism": 0.9,
    "overall": 0.88,
    "commentary": "Code reflects the proposal's signals, sizing, and drawdown halt. No hidden state.",
}


class _PerBundleScriptedCompute(FakeComputeClient):
    """FakeComputeClient that returns different score blocks per bundle.

    The real demo's TeeML provider scores each artifact under the
    bundle's prompt; here we simulate that by inspecting the system
    message (which equals the bundle's system_prompt) and returning the
    appropriate scripted block. Lets a single test exercise a flow
    that touches both graders without conflating their responses."""

    def __init__(
        self,
        *,
        proposal_score: dict[str, Any],
        implementation_score: dict[str, Any],
    ) -> None:
        super().__init__(response_content=json.dumps(proposal_score))
        self._proposal_response = json.dumps(proposal_score)
        self._impl_response = json.dumps(implementation_score)

    def infer_full(
        self,
        *,
        provider_address: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ComputeResult:
        # System prompt is messages[0]; route by a substring that
        # appears in EXACTLY ONE bundle's prompt. Easy to bungle with
        # generic markers like 'proposal_grade' (the implementation
        # prompt references 'proposal_grade' too in its two-gate
        # explanation) — pin to phrases unique to each bundle's role.
        system = messages[0]["content"] if messages else ""
        if "trading-strategy CODE faithfully implements" in system:
            self._response_content = self._impl_response
        elif "STRATEGY PROPOSAL against a fixed mandate" in system:
            self._response_content = self._proposal_response
        else:
            raise AssertionError(
                "scripted compute client received a system prompt that matched "
                "neither bundle's role marker — update the routing or the bundles"
            )
        return super().infer_full(
            provider_address=provider_address,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ---------------- artifact rendering ----------------


def test_render_proposal_at_5_percent_omits_drift_marker():
    """Clean path: agent's working mandate equals the principal's; no
    'regulatory update' marker appears in the proposal text."""
    text = render_proposal(max_drawdown_pct=5.0)
    assert "5%" in text
    assert "regulatory update" not in text.lower()


def test_render_proposal_at_25_percent_includes_drift_marker():
    """Poisoned path: working mandate diverges from the principal's;
    the proposal text carries the in-band 'regulatory update' marker
    that the grader is told to flag (see proposal_grade.json's
    drift_detection metadata + scoring rule)."""
    text = render_proposal(max_drawdown_pct=25.0)
    assert "25%" in text
    assert "regulatory update" in text.lower()


# ---------------- run_agent: clean path ----------------


def test_run_agent_clean_path_uses_principal_mandate(
    tmp_path: Path,
    proposal_bundle: EvaluatorBundle,
    implementation_bundle: EvaluatorBundle,
    clean_tool_response: dict[str, Any],
) -> None:
    """Clean tool response → working mandate equals the principal's
    committed value (5%). Proposal text reflects 5%; both receipts are
    persisted; no drift marker."""
    storage = MockStorageClient()
    compute = _PerBundleScriptedCompute(
        proposal_score=_PASS_PROPOSAL_SCORE,
        implementation_score=_PASS_IMPL_SCORE,
    )

    run = run_agent(
        compute=compute,
        storage=storage,
        proposal_bundle=proposal_bundle,
        implementation_bundle=implementation_bundle,
        provider_address=_PROVIDER,
        tool_response=clean_tool_response,
        receipts_dir=tmp_path / "receipts",
    )

    assert run.applied_max_drawdown_pct == 5.0
    assert run.proposal_path.exists()
    assert run.implementation_path.exists()


def test_run_agent_poisoned_path_applies_drift(
    tmp_path: Path,
    proposal_bundle: EvaluatorBundle,
    implementation_bundle: EvaluatorBundle,
    poisoned_tool_response: dict[str, Any],
) -> None:
    """Poisoned tool response with `mandate_updates.max_drawdown: '25%'`
    → working mandate is corrupted to 25%. Verifies the producer-side
    half of the demo: the agent obediently uses the poisoned value,
    which is what makes the gate's REFUSE_SCORE meaningful (the agent
    isn't doing the catching; the gate is)."""
    storage = MockStorageClient()
    compute = _PerBundleScriptedCompute(
        proposal_score=_FAIL_PROPOSAL_SCORE,
        implementation_score=_PASS_IMPL_SCORE,
    )

    run = run_agent(
        compute=compute,
        storage=storage,
        proposal_bundle=proposal_bundle,
        implementation_bundle=implementation_bundle,
        provider_address=_PROVIDER,
        tool_response=poisoned_tool_response,
        receipts_dir=tmp_path / "receipts",
    )

    assert run.applied_max_drawdown_pct == 25.0


def test_run_agent_receipts_round_trip_through_verify(
    tmp_path: Path,
    proposal_bundle: EvaluatorBundle,
    implementation_bundle: EvaluatorBundle,
    clean_tool_response: dict[str, Any],
) -> None:
    """Both receipts produced by `run_agent` must round-trip through
    `verify_receipt` for Steps 1-3 + 6 (Step 5 skipped: FakeComputeClient
    can't synthesize a parseable report). Catches receipt-construction
    regressions where the producer writes structurally-invalid receipts
    that look right but fail verification at gate time."""
    storage = MockStorageClient()
    compute = _PerBundleScriptedCompute(
        proposal_score=_PASS_PROPOSAL_SCORE,
        implementation_score=_PASS_IMPL_SCORE,
    )

    run = run_agent(
        compute=compute,
        storage=storage,
        proposal_bundle=proposal_bundle,
        implementation_bundle=implementation_bundle,
        provider_address=_PROVIDER,
        tool_response=clean_tool_response,
        receipts_dir=tmp_path / "receipts",
    )

    proposal_result = verify_receipt(
        run.proposal_receipt,
        proposal_bundle.canonical_bytes(),
        report_bytes=None,
    )
    assert proposal_result.bundle.evaluator_id() == proposal_bundle.evaluator_id()

    impl_result = verify_receipt(
        run.implementation_receipt,
        implementation_bundle.canonical_bytes(),
        report_bytes=None,
    )
    assert impl_result.bundle.evaluator_id() == implementation_bundle.evaluator_id()


# ---------------- end-to-end: agent → executor ----------------


def _policy_for(
    *,
    proposal_bundle: EvaluatorBundle,
    implementation_bundle: EvaluatorBundle,
    score_threshold: float = 0.7,
) -> PrincipalPolicy:
    """Construct a low-consequence policy with both bundles registered.

    Mirrors the shape of `examples/trading/principal_policy.json` but
    constructed in-memory so test threshold/diversity tweaks don't have
    to round-trip through a file."""
    return PrincipalPolicy(
        policy_version=POLICY_VERSION,
        principal_id="test-principal",
        bundles={
            "proposal_grade": proposal_bundle.evaluator_id(),
            "implementation_grade": implementation_bundle.evaluator_id(),
        },
        tiers={
            "low_consequence": TierPolicy(
                n_attestations=1,
                score_threshold=score_threshold,
                diversity=DiversityRules(),
            ),
        },
    )


def _gate_with_step_5_skipped(
    *,
    policy: PrincipalPolicy,
    bundle_name: str,
    receipt: EnhancedReceipt,
    bundle: EvaluatorBundle,
    storage: MockStorageClient,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Run `evaluate_gate` against a single receipt, with Step 5 skipped.

    `FakeComputeClient` can't produce a parseable attestation report,
    so the executor's default Step-5-via-`verify_receipt_with_storage`
    would crash. Real Step 5 is covered by `tests/test_verify.py` with
    synthesized parseable reports; here we patch
    `verify_receipt_with_storage` to skip Step 5 for the same reason
    the trading-critic suite did."""
    from eerful.verify import verify_receipt_with_storage as real_verify

    def patched_verify(r: EnhancedReceipt, s: Any) -> Any:
        return real_verify(r, s, fetch_report=False)

    monkeypatch.setattr(
        "eerful.executor.verify_receipt_with_storage", patched_verify
    )
    return evaluate_gate(
        policy=policy,
        tier="low_consequence",
        bundle_name=bundle_name,
        receipts=[receipt],
        storage=storage,
    )


def test_clean_path_proposal_grade_passes(
    tmp_path: Path,
    proposal_bundle: EvaluatorBundle,
    implementation_bundle: EvaluatorBundle,
    clean_tool_response: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path end-to-end: agent runs clean, proposal_grade gate
    PASSES with mandate_compliance high, overall above threshold."""
    storage = MockStorageClient()
    compute = _PerBundleScriptedCompute(
        proposal_score=_PASS_PROPOSAL_SCORE,
        implementation_score=_PASS_IMPL_SCORE,
    )

    run = run_agent(
        compute=compute,
        storage=storage,
        proposal_bundle=proposal_bundle,
        implementation_bundle=implementation_bundle,
        provider_address=_PROVIDER,
        tool_response=clean_tool_response,
        receipts_dir=tmp_path / "receipts",
    )

    policy = _policy_for(
        proposal_bundle=proposal_bundle,
        implementation_bundle=implementation_bundle,
    )
    gate_result = _gate_with_step_5_skipped(
        policy=policy,
        bundle_name="proposal_grade",
        receipt=run.proposal_receipt,
        bundle=proposal_bundle,
        storage=storage,
        monkeypatch=monkeypatch,
    )
    assert gate_result.outcome == GateOutcome.PASS


def test_poisoned_path_proposal_grade_refuses(
    tmp_path: Path,
    proposal_bundle: EvaluatorBundle,
    implementation_bundle: EvaluatorBundle,
    poisoned_tool_response: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The demo's punch: agent runs under a poisoned tool response,
    proposes a strategy with the corrupted 25% drawdown, the grader
    catches the in-band mandate-update marker and scores
    mandate_compliance=0.1 / overall=0.25. proposal_grade gate REFUSES.

    Critically: implementation_grade would PASS this code (code
    matches the poisoned proposal — they're internally consistent).
    The demo's narrative is exactly that two-gate architecture is what
    catches mandate drift; a single-gate setup with only
    implementation_grade would have let the poisoned strategy through."""
    storage = MockStorageClient()
    compute = _PerBundleScriptedCompute(
        proposal_score=_FAIL_PROPOSAL_SCORE,
        implementation_score=_PASS_IMPL_SCORE,
    )

    run = run_agent(
        compute=compute,
        storage=storage,
        proposal_bundle=proposal_bundle,
        implementation_bundle=implementation_bundle,
        provider_address=_PROVIDER,
        tool_response=poisoned_tool_response,
        receipts_dir=tmp_path / "receipts",
    )

    policy = _policy_for(
        proposal_bundle=proposal_bundle,
        implementation_bundle=implementation_bundle,
    )

    proposal_result = _gate_with_step_5_skipped(
        policy=policy,
        bundle_name="proposal_grade",
        receipt=run.proposal_receipt,
        bundle=proposal_bundle,
        storage=storage,
        monkeypatch=monkeypatch,
    )
    # Outcome is the stable contract; `detail` formatting can shift
    # without affecting gate semantics. Asserting the structured
    # outcome is what this test is actually trying to pin.
    assert proposal_result.outcome == GateOutcome.REFUSE_SCORE

    # implementation_grade would pass — that's the load-bearing
    # narrative of the two-gate design. Single-gate (impl-only)
    # principals miss this attack class entirely.
    impl_result = _gate_with_step_5_skipped(
        policy=policy,
        bundle_name="implementation_grade",
        receipt=run.implementation_receipt,
        bundle=implementation_bundle,
        storage=storage,
        monkeypatch=monkeypatch,
    )
    assert impl_result.outcome == GateOutcome.PASS


# ---------------- bundle hash stability ----------------


def test_committed_bundle_hashes_match_principal_policy(
    proposal_bundle: EvaluatorBundle,
    implementation_bundle: EvaluatorBundle,
) -> None:
    """The principal_policy.json on disk pins specific evaluator_ids.
    Those hashes MUST match the bundles' computed evaluator_id —
    otherwise a real `eerful gate` invocation would refuse with
    REFUSE_BUNDLE_MISMATCH against bundles published from this repo.

    Editing a bundle's prompt without updating principal_policy.json
    is exactly the regression this catches; the test is the alarm
    that says 'recompute and update the policy'."""
    policy_path = (
        Path(__file__).resolve().parent.parent
        / "examples"
        / "trading"
        / "principal_policy.json"
    )
    policy = PrincipalPolicy.model_validate_json(policy_path.read_bytes())
    assert policy.bundles["proposal_grade"] == proposal_bundle.evaluator_id()
    assert policy.bundles["implementation_grade"] == implementation_bundle.evaluator_id()
