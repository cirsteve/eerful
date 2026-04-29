"""PrincipalPolicy — what attestation a principal demands per tier.

Spec design: §3 of the executor + multi-attestation rails design
(local planning artifact).

A policy is external to bundles: bundles are the public criteria
(content-addressed by `evaluator_id`); the policy answers "how much
attestation does this principal require for this consequence." A
principal commits to bundle hashes and per-tier requirements *before*
the agent runs; the executor (`eerful.executor.evaluate_gate`) refuses
any gate call whose receipts don't satisfy the named tier's rules.

Naming convention: schema fields use generic names (`low_consequence`
/ `high_consequence`, `proposal_grade` / `implementation_grade`)
because the executor is domain-agnostic. Domain-specific demos pick
names that read naturally in their domain (e.g. `paper` / `live`,
`strat_grade` / `code_grade`) — same machinery, different labels.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eerful.canonical import Bytes32Hex, is_bytes32_hex
from eerful.zg.attestation import DeclaredComposeCategory


POLICY_VERSION: Literal["0.5"] = "0.5"
"""Current PrincipalPolicy schema version. Mirrors `eerful.receipt.SPEC_VERSION`
in lockstep — a policy authored under EER v0.5 carries `policy_version="0.5"`,
and the executor refuses mismatching versions rather than guess at semantics."""


ScoreAggregation = Literal["all_must_pass"]
"""How the executor combines per-receipt scores into a tier-level pass/fail.

v0.5 ships `all_must_pass` only — every receipt's `overall` ≥
`score_threshold`. `median` and `threshold_of_passers` are deferred
(per the rails design): they raise "why did this pass when a TEE
said no" questions you don't want at demo time."""


class DiversityRules(BaseModel):
    """Per-tier diversity requirements applied to the receipt set.

    Without diversity rules, an N-attestation tier can be trivially
    satisfied by running the same provider N times — the cryptographic
    cost of compromising one TEE buys access to all N receipts. Diversity
    is what forces the attacker to compromise N independent enclaves.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    distinct_signers: bool = False
    """If true, every receipt in the gate set must carry a different
    `tee_signer_address` (derived from the receipt's `enclave_pubkey` —
    each TEE instance generates its own enclave-born key by construction)."""

    distinct_compose_hashes: bool = False
    """If true, every receipt must come from a different attested
    compose-hash. Strictly stronger than `distinct_signers` (two enclaves
    running the same compose share the compose-hash even when their
    signer addresses differ), at the cost of supply: fewer TEE composes
    in the wild than TEE instances."""


class TierPolicy(BaseModel):
    """One consequence tier's attestation requirements.

    A pipeline gating both `proposal_grade` and `implementation_grade`
    needs `n_attestations` valid receipts *for each* — `n_attestations`
    is per-gate-call, not per-pipeline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_attestations: int = Field(ge=1)
    """Distinct valid receipts required to pass this tier's gate. >=1.
    With `distinct_signers: false`, N receipts from the same provider
    satisfies the count — the diversity rule is what gives N>1 actual
    security."""

    score_threshold: float = Field(ge=0.0, le=1.0)
    """Minimum `output_score_block.overall` (or schema-defined rank
    dimension) that each receipt must meet. Inclusive."""

    required_categories: list[DeclaredComposeCategory] | None = None
    """§8.2 category allowlist for this tier. When set, every receipt's
    declared compose-category (from `Step5Result.declared_entry.category`)
    must be in this list. `None` means no category enforcement (the
    POC default — substrate-independent). Production uses typically set
    `["A"]` for high-consequence and may relax to `["A", "B"]` for
    low-consequence."""

    diversity: DiversityRules = DiversityRules()
    """Diversity rules. Defaults to all-False (no diversity enforced).
    Set `distinct_signers: true` to make N>1 attestations meaningful."""

    @model_validator(mode="after")
    def _validate_required_categories(self) -> TierPolicy:
        """An empty list has no canonical 'no enforcement' meaning — `None`
        is the canonical absence form (mirrors `EvaluatorBundle`'s
        `accepted_compose_hashes` rule). Without this, two equivalent
        policies (one with `[]`, one with `None`) would canonical-JSON
        differently and a future content-addressed policy hash would
        diverge."""
        if self.required_categories is not None and len(self.required_categories) == 0:
            raise ValueError(
                "required_categories must be omitted (None) rather than empty; "
                "an empty list has no canonical 'no enforcement' form."
            )
        return self


class PrincipalPolicy(BaseModel):
    """A principal's full attestation policy across all bundles and tiers.

    Authored once per principal/deployment and shipped to the executor.
    The executor reads `bundles` to know which `evaluator_id` corresponds
    to which logical name and `tiers` to know what attestation each
    consequence level demands.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: Literal["0.5"]
    """EER spec version this policy targets. Matches
    `eerful.policy.POLICY_VERSION`. The executor refuses mismatching
    versions."""

    principal_id: str
    """Free-form identifier for the principal (e.g. a DID
    `did:pkh:eip155:16601:0x...` or an EVM address). Not parsed by the
    executor — informational. v0.6 may tighten to a structured form."""

    bundles: dict[str, Bytes32Hex]
    """Logical evaluator name → content-addressed bundle hash. The
    executor refuses any receipt whose `evaluator_id` does not match
    the hash registered here under the gate's `bundle_name`. Empty
    dict is rejected — a policy with no bundles can gate nothing."""

    tiers: dict[str, TierPolicy]
    """Consequence tier name → TierPolicy. The executor's `tier`
    parameter is looked up here. Empty dict is rejected."""

    score_aggregation: ScoreAggregation = "all_must_pass"
    """How per-receipt scores combine into a tier-level pass/fail.
    v0.5 ships `all_must_pass` only."""

    @model_validator(mode="after")
    def _validate_invariants(self) -> PrincipalPolicy:
        """Enforce invariants the type aliases can't express:

        - `bundles` and `tiers` are non-empty (a policy with neither
          can gate nothing).
        - Each `bundles` value is a syntactically valid 32-byte hex
          string. The `Bytes32Hex` BeforeValidator lowercases but
          doesn't bound length; without this check, a 31-byte hash
          would silently miss the executor's lookup against the
          receipt's `evaluator_id`.
        """
        if len(self.bundles) == 0:
            raise ValueError("bundles must contain at least one entry")
        if len(self.tiers) == 0:
            raise ValueError("tiers must contain at least one entry")
        for name, h in self.bundles.items():
            if not is_bytes32_hex(h):
                raise ValueError(
                    f"bundles[{name!r}] is not a valid 32-byte hex string: {h!r}"
                )
        return self
