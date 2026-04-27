"""
Canonical JSON serialization and the hex-string type aliases used across
every model in eerful.

The output of `canonical_json_bytes` is part of the EER protocol surface
(see spec §6.4). Any change to its serialization rules invalidates every
receipt and every evaluator hash produced before the change. Bump
coordinated with a spec version bump.
"""

from __future__ import annotations

import json
import re
from typing import Any

Bytes32Hex = str
"""0x-prefixed 64-character lowercase hex string. Spec §6.1."""

BytesHex = str
"""0x-prefixed lowercase hex string of variable length. Used for fields
whose byte width depends on the underlying cryptographic primitive
(e.g. enclave_pubkey, enclave_signature). Spec §6.1, amendment a."""

Address = str
"""0x-prefixed 40-character hex string. EVM address."""

ZERO_BYTES32: Bytes32Hex = "0x" + "00" * 32

_HEX_RE = re.compile(r"^0x[0-9a-f]*$")
_BYTES32_RE = re.compile(r"^0x[0-9a-f]{64}$")
_ADDRESS_RE = re.compile(r"^0x[0-9a-f]{40}$")


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON serialization for canonical signing and hashing.

    Rules (spec §6.4):
    - Keys sorted lexicographically at every depth
    - No insignificant whitespace
    - UTF-8 encoded output
    - NaN and infinities rejected
    - `null` for absent optional fields, not omission (caller's responsibility)
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def to_lower_hex(value: bytes | str) -> BytesHex:
    """Canonicalize bytes or a hex string to 0x-prefixed lowercase hex.

    Accepts raw bytes, or hex with or without a 0x prefix, or already-canonical
    input. Used at construction boundaries so downstream code can rely on a
    single representation. Empty bytes / "0x" / "" all canonicalize to "0x".
    """
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if not isinstance(value, str):
        raise TypeError(f"to_lower_hex expected bytes or str, got {type(value).__name__}")
    s = value.lower()
    if s.startswith("0x"):
        s = s[2:]
    if s and not all(c in "0123456789abcdef" for c in s):
        raise ValueError(f"not a hex string: {value!r}")
    return "0x" + s


def is_bytes32_hex(s: str) -> bool:
    return bool(_BYTES32_RE.match(s))


def is_bytes_hex(s: str) -> bool:
    return bool(_HEX_RE.match(s))


def is_address(s: str) -> bool:
    return bool(_ADDRESS_RE.match(s))
