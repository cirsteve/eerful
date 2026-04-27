"""Input commitment construction (spec §6.7) + producer-side salt store.

The commitment is `sha256(input_bytes || evaluator_id || salt)`. It binds
a receipt to a private input identity without disclosing the input. EER
does not specify a single canonicalization for `input_bytes` — the
producer chooses it (raw blob, content hash of an opaque file, Merkle
root over a structured artifact) and documents the choice in the
evaluator bundle's metadata or in a higher-layer registry.

`evaluator_id` is folded into the hash so the same input under two
different evaluators produces two different commitments. Without this
binding a producer could reuse the same commitment across evaluators in
ways that confuse linkage. Per §6.7, this is the recommended construction
even though alternative constructions (commit-reveal schemes, Merkle
proofs over input components) are explicitly permitted.

`SaltStore` is a producer convenience, not part of the protocol. The
salt MUST be retained if the producer wants to later reveal the input —
without the salt, the commitment is one-way and the input cannot be
proven against it. Lost-salt commitments are recoverable as identity
(linking receipts about the same input) but not as proofs (showing what
the input actually was).
"""

from __future__ import annotations

import json
import secrets
from hashlib import sha256
from pathlib import Path
from typing import Any

from eerful.canonical import Bytes32Hex, is_bytes32_hex, to_lower_hex


def compute_input_commitment(
    input_bytes: bytes,
    evaluator_id: Bytes32Hex,
    salt: bytes,
) -> Bytes32Hex:
    """Compute `sha256(input_bytes || evaluator_id || salt)` per spec §6.7.

    `evaluator_id` is hashed as raw 32 bytes (the underlying digest), not
    as its 0x-hex string. This matches the `||` (concatenation) semantics
    in the spec: we're concatenating the *digest bytes*, not the textual
    representation. Using the hex string would silently produce different
    commitments depending on case (`0xab...` vs `0xAB...`) even though
    receipts canonicalize to lowercase, so a verifier reconstructing the
    commitment from the receipt's `evaluator_id` would have to commit to
    a textual encoding — fragile. Raw bytes are the unambiguous form.

    `salt` SHOULD be at least 32 bytes (`generate_salt`'s default) for
    the §6.7 brute-force-reversal protection to apply meaningfully —
    shorter salts admit precomputation against low-entropy inputs.
    """
    try:
        canonical_eid = to_lower_hex(evaluator_id)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"evaluator_id must be 0x-prefixed 64-char lowercase hex, got {evaluator_id!r}"
        ) from e
    if not is_bytes32_hex(canonical_eid):
        raise ValueError(
            f"evaluator_id must be 0x-prefixed 64-char lowercase hex, got {evaluator_id!r}"
        )
    eid_bytes = bytes.fromhex(canonical_eid[2:])
    digest = sha256(input_bytes + eid_bytes + salt).hexdigest()
    return "0x" + digest


def generate_salt(n_bytes: int = 32) -> bytes:
    """Cryptographically random salt suitable for `compute_input_commitment`.

    Default 32 bytes matches the spec's recommendation: enough entropy
    to make brute-force reversal for low-entropy inputs (e.g. enum-like
    strategy identifiers) computationally infeasible.
    """
    if n_bytes <= 0:
        raise ValueError(f"n_bytes must be positive, got {n_bytes}")
    return secrets.token_bytes(n_bytes)


class SaltStore:
    """File-backed JSON store mapping `receipt_id` → (salt, input_path).

    Producer-side convenience for retaining the salt across the
    chain-of-receipts pattern (e.g. trading-critic v1 → v2 → v3 all share
    a stable input commitment). NOT part of the protocol — verifiers
    never see this file.

    Storage format is plain JSON keyed by `receipt_id` (a Bytes32Hex
    string), with `salt` stored as 0x-hex and `input_path` as an
    optional string pointer back to where the producer's input lives.
    Both fields go through canonical normalization on read so a
    hand-edited file is still consumable.

    Concurrency: this class assumes a single producer process. Concurrent
    `put`s race on the file write. Producers running multi-process
    pipelines should serialize access externally (or call the lower-level
    `compute_input_commitment` and manage their own salts).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def put(
        self,
        receipt_id: Bytes32Hex,
        salt: bytes,
        input_path: str | None = None,
    ) -> None:
        """Persist `salt` (and optional `input_path`) under `receipt_id`.

        Overwrites any prior entry under the same receipt_id silently —
        receipt_ids are sha256 outputs so collisions imply a tampered
        chain, not a normal overwrite case. If you're seeing one,
        something else is wrong; we don't guard the overwrite.
        """
        try:
            canonical_id = to_lower_hex(receipt_id)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"receipt_id must be 0x-prefixed 64-char lowercase hex, got {receipt_id!r}"
            ) from e
        if not is_bytes32_hex(canonical_id):
            raise ValueError(
                f"receipt_id must be 0x-prefixed 64-char lowercase hex, got {receipt_id!r}"
            )
        data = self._read_all()
        data[canonical_id] = {
            "salt": "0x" + salt.hex(),
            "input_path": input_path,
        }
        self._write_all(data)

    def get(self, receipt_id: Bytes32Hex) -> tuple[bytes, str | None]:
        """Return `(salt_bytes, input_path)` for `receipt_id`.

        Raises `KeyError` if no entry exists for the given receipt_id.
        Callers reconstructing a commitment after the fact rely on this
        to surface "salt was never stored" loudly rather than silently
        producing a wrong commitment.
        """
        canonical_id = to_lower_hex(receipt_id)
        data = self._read_all()
        entry = data.get(canonical_id)
        if entry is None:
            raise KeyError(f"no salt entry for receipt_id {canonical_id}")
        salt_hex = entry.get("salt")
        if not isinstance(salt_hex, str):
            raise ValueError(
                f"salt store entry for {canonical_id} missing 'salt' field"
            )
        salt = bytes.fromhex(to_lower_hex(salt_hex)[2:])
        input_path = entry.get("input_path")
        if input_path is not None and not isinstance(input_path, str):
            raise ValueError(
                f"salt store entry for {canonical_id} has non-string input_path"
            )
        return salt, input_path

    def _read_all(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"salt store at {self._path} is not valid JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(f"salt store at {self._path} is not a JSON object")
        return raw

    def _write_all(self, data: dict[str, dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys keeps the file diffable across put() calls; an
        # accidentally-checked-in salt file (gitignored, but mistakes
        # happen) at least produces a clean diff for incident response.
        self._path.write_text(
            json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )


__all__ = [
    "SaltStore",
    "compute_input_commitment",
    "generate_salt",
]
