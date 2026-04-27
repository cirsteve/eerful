"""Canonical JSON encoding (spec §6.4). These tests pin the byte-identical
guarantee that receipts and evaluator content hashes depend on."""

from __future__ import annotations

import pytest

from eerful.canonical import (
    canonical_json_bytes,
    is_address,
    is_bytes32_hex,
    is_bytes_hex,
    to_lower_hex,
)


def test_keys_sorted_at_top_level():
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_keys_sorted_recursively():
    obj = {"b": {"d": 1, "c": 2}, "a": 3}
    assert canonical_json_bytes(obj) == b'{"a":3,"b":{"c":2,"d":1}}'


def test_no_insignificant_whitespace():
    assert canonical_json_bytes({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_null_for_none():
    assert canonical_json_bytes({"x": None}) == b'{"x":null}'


def test_utf8_not_ascii_escaped():
    assert canonical_json_bytes({"k": "résumé"}) == '{"k":"résumé"}'.encode()


def test_byte_identical_across_runs():
    obj = {"a": 1, "b": "two", "c": [3, 4], "d": {"e": 5}}
    assert canonical_json_bytes(obj) == canonical_json_bytes(obj)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": bad})


def test_to_lower_hex_bytes():
    assert to_lower_hex(b"\x01\x02") == "0x0102"


def test_to_lower_hex_uppercase_string():
    assert to_lower_hex("0xABCD") == "0xabcd"


def test_to_lower_hex_no_prefix():
    assert to_lower_hex("ABCD") == "0xabcd"


def test_to_lower_hex_invalid():
    with pytest.raises(ValueError):
        to_lower_hex("0xnothex")


def test_predicates():
    assert is_bytes32_hex("0x" + "a" * 64)
    assert not is_bytes32_hex("0x" + "a" * 63)
    assert is_address("0x" + "b" * 40)
    assert not is_address("0x" + "b" * 41)
    assert is_bytes_hex("0xabcd")
    assert is_bytes_hex("0x")
