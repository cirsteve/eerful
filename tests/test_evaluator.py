"""EvaluatorBundle content hashing and field validation (spec §6.5)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from eerful.evaluator import EvaluatorBundle


def _bundle(**overrides: Any) -> EvaluatorBundle:
    fields: dict[str, Any] = dict(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
    )
    fields.update(overrides)
    return EvaluatorBundle(**fields)


def test_evaluator_id_deterministic():
    assert _bundle().evaluator_id() == _bundle().evaluator_id()


def test_evaluator_id_changes_on_field_change():
    assert _bundle().evaluator_id() != _bundle(system_prompt="rate it harshly").evaluator_id()


def test_evaluator_id_is_bytes32_hex():
    eid = _bundle().evaluator_id()
    assert eid.startswith("0x")
    assert len(eid) == 66


def test_canonical_bytes_keys_sorted():
    b = _bundle()
    raw = b.canonical_bytes()
    assert raw.index(b'"model_identifier"') < raw.index(b'"system_prompt"')
    assert raw.index(b'"system_prompt"') < raw.index(b'"version"')


def test_accepted_compose_hashes_optional():
    assert _bundle().accepted_compose_hashes is None
    h = "0x" + "a" * 64
    assert _bundle(accepted_compose_hashes=[h]).accepted_compose_hashes == [h]


def test_accepted_compose_hashes_changes_evaluator_id():
    assert _bundle().evaluator_id() != _bundle(accepted_compose_hashes=["0x" + "a" * 64]).evaluator_id()


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        EvaluatorBundle(version="v", model_identifier="m", system_prompt="p", surprise=1)  # type: ignore[call-arg]


def test_uppercase_compose_hashes_normalized_on_construction():
    """Spec §6.4 mandates lowercase hex; bundles built with uppercase
    `accepted_compose_hashes` must hash to the same `evaluator_id` as
    bundles built with lowercase."""
    h_upper = "0x" + "A" * 64
    h_lower = "0x" + "a" * 64
    b_upper = _bundle(accepted_compose_hashes=[h_upper])
    b_lower = _bundle(accepted_compose_hashes=[h_lower])
    assert b_upper.accepted_compose_hashes == [h_lower]
    assert b_upper.evaluator_id() == b_lower.evaluator_id()


def test_accepted_compose_hashes_empty_list_rejected():
    """Spec §6.5 docstring: empty list has no canonical form; absence (None)
    is the only canonical 'no gating'. Without this rule, evaluator_id
    would diverge between [] (encodes as `[]`) and None (encodes as
    `null`) for what publishers intend as the same bundle."""
    with pytest.raises(ValidationError) as exc:
        _bundle(accepted_compose_hashes=[])
    assert "accepted_compose_hashes" in str(exc.value)


def test_accepted_compose_hashes_wrong_length_rejected():
    """Bytes32Hex requires exactly 64 hex chars; the BeforeValidator only
    lowercases, so a model_validator MUST enforce the length invariant or
    short hashes propagate into Step 5's allowlist comparison."""
    short = "0x" + "a" * 63  # 63 hex chars; valid hex, wrong length
    with pytest.raises(ValidationError):
        _bundle(accepted_compose_hashes=[short])


def test_accepted_compose_hashes_canonical_form_is_none():
    """Receipts derive evaluator_id from canonical_bytes(); we lock down that
    the canonical 'no gating' form is None (not an absent key, not []),
    since bundle hash drift breaks every receipt's receipt_id chain."""
    b = _bundle()
    assert b.accepted_compose_hashes is None
    assert b'"accepted_compose_hashes":null' in b.canonical_bytes()


def test_evaluator_id_stable_when_field_unset():
    """Bundle hash must be byte-identical across runs/processes when the
    field is unset; this is the load-bearing invariant for receipt_id
    stability across every receipt produced under this bundle."""
    expected = _bundle().evaluator_id()
    for _ in range(3):
        assert _bundle().evaluator_id() == expected
