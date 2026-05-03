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
from typing import Annotated, Any

from eth_keys import keys
from pydantic import BeforeValidator


def _normalize_hex_at_boundary(value: object) -> object:
    """Pydantic BeforeValidator: canonicalize hex inputs at construction.

    Strings and bytes flow through `to_lower_hex`; everything else
    (including None for Optional fields) passes through so pydantic's own
    type checks raise the appropriate error. Spec §6.4 mandates lowercase
    hex; enforcing it at field-construction is what makes
    `receipt_id`/`evaluator_id` derivation cross-implementation-stable.
    """
    if isinstance(value, (bytes, str)):
        return to_lower_hex(value)
    return value


_HexNormalizer = BeforeValidator(_normalize_hex_at_boundary)

Bytes32Hex = Annotated[str, _HexNormalizer]
"""0x-prefixed 64-character lowercase hex string. Spec §6.1.

Field-level uses pick up the BeforeValidator automatically; non-field
uses (return-type annotations, type aliases) treat it as `str`."""

BytesHex = Annotated[str, _HexNormalizer]
"""0x-prefixed lowercase hex string of variable length. Used for fields
whose byte width depends on the underlying cryptographic primitive
(e.g. enclave_pubkey, enclave_signature). Spec §6.1, amendment a."""

Address = Annotated[str, _HexNormalizer]
"""0x-prefixed 40-character hex string. EVM address."""

ZERO_BYTES32: str = "0x" + "00" * 32

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


def tee_signer_address_from_pubkey(pubkey_hex: BytesHex) -> Address:
    """Derive the EVM address of an enclave's signing key from its pubkey.

    The 0G TeeML attestation report's `report_data` field carries the
    EVM address of the enclave-born signing key (spec §6.7); the
    pubkey-to-address derivation is keccak256 of the 64-byte X||Y
    public key, last 20 bytes. `enclave_pubkey` in the receipt is
    already in X||Y form (no SEC1 0x04 prefix), so this matches.

    Used by the gate's `distinct_signers` diversity rule and by Step 5b
    (§7.1) to bind the receipt's claimed pubkey to the attested signer
    in `report_data`. Two enclaves with different signing keys produce
    different addresses; two on-chain identities sharing one enclave
    (the Provider 15+16 fixture in `research/day1_attestation_findings.md`)
    produce the same address — diversity caught.
    """
    canonical = to_lower_hex(pubkey_hex)
    pubkey_bytes = bytes.fromhex(canonical.removeprefix("0x"))
    if len(pubkey_bytes) != 64:
        raise ValueError(
            f"enclave_pubkey must be 64 bytes (X||Y, no SEC1 prefix), got {len(pubkey_bytes)}"
        )
    pub = keys.PublicKey(pubkey_bytes)
    return to_lower_hex(pub.to_canonical_address())
