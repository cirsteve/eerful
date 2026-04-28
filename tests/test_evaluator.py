"""EvaluatorBundle content hashing and field validation (spec §6.5)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from eerful.evaluator import ComposeHashEntry, EvaluatorBundle


_PROVIDER = "0x" + "b" * 40
_HASH_A = "0x" + "a" * 64
_HASH_B = "0x" + "c" * 64


def _entry(**overrides: Any) -> ComposeHashEntry:
    fields: dict[str, Any] = dict(
        hash=_HASH_A,
        category="A",
        provider_address=_PROVIDER,
    )
    fields.update(overrides)
    return ComposeHashEntry(**fields)


def _bundle(**overrides: Any) -> EvaluatorBundle:
    fields: dict[str, Any] = dict(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
    )
    fields.update(overrides)
    return EvaluatorBundle(**fields)


# ---------------- ComposeHashEntry ----------------


def test_compose_hash_entry_constructs_with_all_fields():
    e = ComposeHashEntry(
        hash=_HASH_A, category="A", provider_address=_PROVIDER, notes="vLLM --model GLM-5"
    )
    assert e.hash == _HASH_A
    assert e.category == "A"
    assert e.provider_address == _PROVIDER
    assert e.notes == "vLLM --model GLM-5"


def test_compose_hash_entry_notes_optional():
    assert _entry().notes is None


def test_compose_hash_entry_frozen():
    e = _entry()
    with pytest.raises(ValidationError):
        e.hash = _HASH_B


def test_compose_hash_entry_extra_forbidden():
    with pytest.raises(ValidationError):
        ComposeHashEntry(
            hash=_HASH_A,
            category="A",
            provider_address=_PROVIDER,
            surprise=1,  # type: ignore[call-arg]
        )


def test_compose_hash_entry_uppercase_normalized():
    """Bytes32Hex/Address BeforeValidator lowercases at construction so the
    canonical bundle bytes don't depend on caller-side casing — without this,
    bundles built from upper- and lowercase inputs would hash to different
    evaluator_ids despite carrying identical semantic content."""
    e = ComposeHashEntry(
        hash="0x" + "A" * 64, category="A", provider_address="0x" + "B" * 40
    )
    assert e.hash == "0x" + "a" * 64
    assert e.provider_address == "0x" + "b" * 40


def test_compose_hash_entry_rejects_short_hash():
    """Bytes32Hex's BeforeValidator only lowercases; the model_validator
    enforces 32-byte length. Without it a 31-byte hash would propagate
    into Step 5's lookup and silently match nothing."""
    with pytest.raises(ValidationError):
        ComposeHashEntry(
            hash="0x" + "a" * 63, category="A", provider_address=_PROVIDER
        )


def test_compose_hash_entry_rejects_short_provider_address():
    with pytest.raises(ValidationError):
        ComposeHashEntry(
            hash=_HASH_A, category="A", provider_address="0x" + "b" * 39
        )


def test_compose_hash_entry_rejects_invalid_category():
    with pytest.raises(ValidationError):
        ComposeHashEntry(
            hash=_HASH_A, category="D", provider_address=_PROVIDER  # type: ignore[arg-type]
        )


# ---------------- EvaluatorBundle ----------------


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
    entries = [_entry()]
    assert _bundle(accepted_compose_hashes=entries).accepted_compose_hashes == entries


def test_accepted_compose_hashes_changes_evaluator_id():
    assert _bundle().evaluator_id() != _bundle(accepted_compose_hashes=[_entry()]).evaluator_id()


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        EvaluatorBundle(version="v", model_identifier="m", system_prompt="p", surprise=1)  # type: ignore[call-arg]


def test_accepted_compose_hashes_empty_list_rejected():
    """Spec §6.5 docstring: empty list has no canonical form; absence (None)
    is the only canonical 'no gating'. Without this rule, evaluator_id
    would diverge between [] (encodes as `[]`) and None (encodes as
    `null`) for what publishers intend as the same bundle."""
    with pytest.raises(ValidationError) as exc:
        _bundle(accepted_compose_hashes=[])
    assert "accepted_compose_hashes" in str(exc.value)


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


def test_evaluator_id_stable_for_entry_list():
    """Same property as `_stable_when_field_unset` but with a populated
    allowlist — the nested-model serialization must canonicalize stably,
    not pick up dict-iteration ordering or object identity."""
    expected = _bundle(accepted_compose_hashes=[_entry()]).evaluator_id()
    for _ in range(3):
        assert _bundle(accepted_compose_hashes=[_entry()]).evaluator_id() == expected


def test_evaluator_id_changes_when_entry_category_changes():
    """The publisher's category declaration is part of the bundle's
    cryptographic commitment — changing it must produce a new evaluator_id
    so a receiver can't silently reclassify a bundle's accepted composes
    without the bundle hash changing."""
    a = _bundle(accepted_compose_hashes=[_entry(category="A")]).evaluator_id()
    b = _bundle(accepted_compose_hashes=[_entry(category="B")]).evaluator_id()
    assert a != b


def test_accepted_compose_hashes_rejects_duplicate_hashes():
    """Step 5 picks the first matching entry; duplicates would make
    category/provider resolution order-dependent and let a publisher
    silently reclassify the same compose by re-listing it. Bundle
    construction must reject this at validation time."""
    other_provider = "0x" + "e" * 40
    with pytest.raises(ValidationError) as exc:
        _bundle(
            accepted_compose_hashes=[
                _entry(hash=_HASH_A, category="A"),
                _entry(hash=_HASH_A, category="B", provider_address=other_provider),
            ]
        )
    assert "duplicate" in str(exc.value).lower()
    assert _HASH_A in str(exc.value)


def test_accepted_compose_hashes_rejects_duplicate_after_case_normalization():
    """Two entries that differ only in hash casing are duplicates after
    BeforeValidator lowercasing — uniqueness must hold over the canonical
    form, not the input form, otherwise the rule is bypassed by a
    publisher feeding mixed-case duplicates."""
    with pytest.raises(ValidationError) as exc:
        _bundle(
            accepted_compose_hashes=[
                _entry(hash=_HASH_A),
                _entry(hash=_HASH_A.upper()),
            ]
        )
    assert "duplicate" in str(exc.value).lower()


def test_entry_field_keys_sorted_in_canonical_form():
    """Every level of the canonical JSON must be key-sorted (spec §6.4).
    This is what makes evaluator_id cross-implementation-stable when
    entries get serialized inside the bundle."""
    b = _bundle(
        accepted_compose_hashes=[
            ComposeHashEntry(
                hash=_HASH_A, category="A", provider_address=_PROVIDER, notes="x"
            )
        ]
    )
    raw = b.canonical_bytes()
    # Inside an entry: category < hash < notes < provider_address (lex order).
    cat_idx = raw.index(b'"category"')
    hash_idx = raw.index(b'"hash"')
    notes_idx = raw.index(b'"notes"')
    provider_idx = raw.index(b'"provider_address"')
    assert cat_idx < hash_idx < notes_idx < provider_idx
