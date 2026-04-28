"""PrincipalPolicy schema + invariants.

The executor (`tests/test_executor.py`) tests how the policy *consumes*
receipts. This file pins what a valid PrincipalPolicy looks like and
what construction-time invariants it enforces — independent of any
gate evaluation.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from eerful.policy import (
    POLICY_VERSION,
    DiversityRules,
    PrincipalPolicy,
    TierPolicy,
)


_BUNDLE_HASH_A = "0x" + "a" * 64
_BUNDLE_HASH_B = "0x" + "b" * 64


def _tier(**overrides: Any) -> TierPolicy:
    fields: dict[str, Any] = dict(
        n_attestations=1,
        score_threshold=0.6,
    )
    fields.update(overrides)
    return TierPolicy(**fields)


def _policy(**overrides: Any) -> PrincipalPolicy:
    fields: dict[str, Any] = dict(
        policy_version=POLICY_VERSION,
        principal_id="did:pkh:eip155:16602:0x" + "1" * 40,
        bundles={"proposal_grade": _BUNDLE_HASH_A},
        tiers={"low_consequence": _tier()},
    )
    fields.update(overrides)
    return PrincipalPolicy(**fields)


# ---------------- DiversityRules ----------------


def test_diversity_rules_defaults_to_all_false():
    """Default `DiversityRules()` enforces nothing — a tier with no
    diversity overrides means N receipts from one provider satisfy
    the count. Locks down the "no enforcement by default" semantics
    so a tier omitting `diversity` doesn't accidentally become strict."""
    d = DiversityRules()
    assert d.distinct_signers is False
    assert d.distinct_compose_hashes is False


def test_diversity_rules_frozen():
    d = DiversityRules()
    with pytest.raises(ValidationError):
        d.distinct_signers = True


# ---------------- TierPolicy ----------------


def test_tier_policy_defaults():
    """Required fields construct with sane defaults: no required_categories
    (None), default DiversityRules (all-False)."""
    t = _tier()
    assert t.n_attestations == 1
    assert t.score_threshold == 0.6
    assert t.required_categories is None
    assert t.diversity.distinct_signers is False


def test_tier_policy_n_attestations_must_be_at_least_1():
    """N=0 is nonsensical — a gate with zero required receipts always
    passes. Pydantic ge=1 catches this at construction."""
    with pytest.raises(ValidationError):
        _tier(n_attestations=0)


def test_tier_policy_score_threshold_in_range():
    """Threshold outside [0, 1] is impossible to satisfy or trivially
    satisfied. Bound at construction time."""
    with pytest.raises(ValidationError):
        _tier(score_threshold=-0.1)
    with pytest.raises(ValidationError):
        _tier(score_threshold=1.1)


def test_tier_policy_required_categories_none_means_no_enforcement():
    """`None` is the canonical 'no enforcement' form — POC default.
    The executor short-circuits the category check when this is None."""
    assert _tier().required_categories is None


def test_tier_policy_required_categories_empty_list_rejected():
    """Mirrors EvaluatorBundle's accepted_compose_hashes rule: empty list
    has no canonical 'no enforcement' meaning, `None` does. Without this
    rule, two policies that mean the same thing would canonical-JSON
    differently."""
    with pytest.raises(ValidationError) as exc:
        _tier(required_categories=[])
    assert "required_categories" in str(exc.value)


def test_tier_policy_required_categories_accepts_abc():
    """A/B/C are the only valid declared categories per spec §8.2 (mirrors
    `DeclaredComposeCategory`). 'unknown' is heuristic-only and rejected."""
    t = _tier(required_categories=["A", "B"])
    assert t.required_categories == ["A", "B"]


def test_tier_policy_required_categories_rejects_unknown():
    with pytest.raises(ValidationError):
        _tier(required_categories=["unknown"])


def test_tier_policy_extra_forbidden():
    with pytest.raises(ValidationError):
        TierPolicy(
            n_attestations=1, score_threshold=0.5, surprise=1  # type: ignore[call-arg]
        )


def test_tier_policy_frozen():
    t = _tier()
    with pytest.raises(ValidationError):
        t.n_attestations = 4


def test_tier_policy_diversity_constructs_inline():
    t = _tier(diversity=DiversityRules(distinct_signers=True))
    assert t.diversity.distinct_signers is True


# ---------------- PrincipalPolicy ----------------


def test_policy_constructs_with_minimum_viable_fields():
    p = _policy()
    assert p.policy_version == POLICY_VERSION
    assert "proposal_grade" in p.bundles
    assert "low_consequence" in p.tiers
    # Default aggregation
    assert p.score_aggregation == "all_must_pass"


def test_policy_rejects_wrong_version():
    """Spec asymmetric break: a v0.5 executor refuses non-0.5 policies.
    Catches a silent-upgrade trap where a v0.4 policy file is fed to a
    v0.5 executor and read with v0.5 semantics."""
    with pytest.raises(ValidationError):
        _policy(policy_version="0.4")


def test_policy_rejects_empty_bundles():
    """A policy with zero bundles can gate nothing — surface as a
    construction error rather than a silent-no-op gate."""
    with pytest.raises(ValidationError) as exc:
        _policy(bundles={})
    assert "bundles" in str(exc.value).lower()


def test_policy_rejects_empty_tiers():
    """Symmetric to bundles: a policy with zero tiers has nothing to
    look up."""
    with pytest.raises(ValidationError) as exc:
        _policy(tiers={})
    assert "tiers" in str(exc.value).lower()


def test_policy_rejects_invalid_bundle_hash():
    """`Bytes32Hex` BeforeValidator lowercases but doesn't bound length.
    Pydantic should still surface the bad input at construction —
    otherwise an executor lookup would silently miss."""
    with pytest.raises(ValidationError):
        _policy(bundles={"proposal_grade": "0xabc"})  # too short


def test_policy_lowercase_normalizes_bundle_hashes():
    """Receipts carry lowercase `evaluator_id`; the policy's bundle
    hashes must lowercase too so dict lookups match. Without
    normalization, the executor would compare `0xABC...` to `0xabc...`
    and miss."""
    upper = _BUNDLE_HASH_A.upper()
    p = _policy(bundles={"proposal_grade": upper})
    assert p.bundles["proposal_grade"] == _BUNDLE_HASH_A


def test_policy_extra_forbidden():
    with pytest.raises(ValidationError):
        PrincipalPolicy(
            policy_version=POLICY_VERSION,
            principal_id="x",
            bundles={"x": _BUNDLE_HASH_A},
            tiers={"y": _tier()},
            surprise=1,  # type: ignore[call-arg]
        )


def test_policy_frozen():
    p = _policy()
    with pytest.raises(ValidationError):
        p.principal_id = "different"


def test_policy_round_trips_through_json():
    """The executor will load policies from disk; canonical JSON
    round-trip must be lossless. Catches a regression where a
    round-tripped policy fails to construct again."""
    p = _policy(
        bundles={"proposal_grade": _BUNDLE_HASH_A, "implementation_grade": _BUNDLE_HASH_B},
        tiers={
            "low_consequence": _tier(n_attestations=1, score_threshold=0.6),
            "high_consequence": _tier(
                n_attestations=4,
                score_threshold=0.7,
                required_categories=["A"],
                diversity=DiversityRules(distinct_signers=True),
            ),
        },
    )
    raw = p.model_dump_json()
    p2 = PrincipalPolicy.model_validate_json(raw)
    assert p2 == p


def test_policy_rejects_non_default_score_aggregation():
    """v0.5 ships `all_must_pass` only. `median` and other modes are
    deferred — a policy declaring them must fail at construction time
    so a v0.5 executor doesn't silently apply the wrong aggregation."""
    with pytest.raises(ValidationError):
        _policy(score_aggregation="median")
