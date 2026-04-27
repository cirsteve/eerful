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
