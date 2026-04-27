"""§6.7 input-commitment construction + producer-side SaltStore."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from eerful.commitment import SaltStore, compute_input_commitment, generate_salt


_EID = "0x" + "ab" * 32  # arbitrary 32-byte evaluator_id
_OTHER_EID = "0x" + "cd" * 32


# ---------------- compute_input_commitment ----------------


def test_commitment_matches_spec_construction():
    """Commitment is `sha256(input_bytes || evaluator_id_bytes || salt)`
    where evaluator_id is hashed as raw 32 bytes (the digest), not as
    its 0x-hex string. Spec §6.7."""
    input_bytes = b"my private input"
    salt = b"\x01" * 32
    expected_eid_bytes = bytes.fromhex(_EID[2:])
    expected = (
        "0x"
        + hashlib.sha256(input_bytes + expected_eid_bytes + salt).hexdigest()
    )
    assert compute_input_commitment(input_bytes, _EID, salt) == expected


def test_commitment_is_deterministic():
    """Same inputs → same output across calls. Producer relies on this
    to reconstruct the commitment when salts are persisted."""
    salt = b"\x42" * 32
    a = compute_input_commitment(b"strategy-identity", _EID, salt)
    b = compute_input_commitment(b"strategy-identity", _EID, salt)
    assert a == b


def test_commitment_changes_when_salt_changes():
    """The §6.7 brute-force-reversal protection: same input, different
    salt → different commitment."""
    salt_a = b"\x01" * 32
    salt_b = b"\x02" * 32
    a = compute_input_commitment(b"input", _EID, salt_a)
    b = compute_input_commitment(b"input", _EID, salt_b)
    assert a != b


def test_commitment_changes_when_evaluator_changes():
    """The §6.7 evaluator-binding rule: same input + same salt under a
    different evaluator → different commitment. Without this, a producer
    could reuse one commitment across evaluators in confusing ways."""
    salt = b"\x42" * 32
    a = compute_input_commitment(b"input", _EID, salt)
    b = compute_input_commitment(b"input", _OTHER_EID, salt)
    assert a != b


def test_commitment_changes_when_input_changes():
    salt = b"\x42" * 32
    a = compute_input_commitment(b"input one", _EID, salt)
    b = compute_input_commitment(b"input two", _EID, salt)
    assert a != b


def test_commitment_evaluator_id_case_insensitive():
    """Receipts canonicalize evaluator_id to lowercase, but a hand-built
    commitment shouldn't silently produce a different value if the
    caller passes uppercase. We canonicalize at the function boundary."""
    salt = b"\x00" * 32
    upper = "0x" + "AB" * 32
    lower = "0x" + "ab" * 32
    assert compute_input_commitment(b"x", upper, salt) == compute_input_commitment(
        b"x", lower, salt
    )


def test_commitment_rejects_non_bytes32_evaluator_id():
    """An evaluator_id that isn't a valid Bytes32Hex string fails fast.
    Catches caller bugs (e.g. passing the bundle's `version` by mistake,
    or a truncated hex)."""
    salt = b"\x00" * 32
    with pytest.raises(ValueError, match="evaluator_id"):
        compute_input_commitment(b"x", "0xnothex", salt)
    with pytest.raises(ValueError, match="evaluator_id"):
        compute_input_commitment(b"x", "0xab", salt)  # too short


def test_commitment_accepts_empty_input():
    """Spec §6.7 doesn't forbid empty input — a producer might commit to
    a sentinel like `b""` if their input identity is implicit. Sanity:
    the function returns *some* commitment without crashing."""
    out = compute_input_commitment(b"", _EID, b"\x00" * 32)
    assert out.startswith("0x")
    assert len(out) == 66


# ---------------- generate_salt ----------------


def test_generate_salt_default_length_is_32_bytes():
    assert len(generate_salt()) == 32


def test_generate_salt_respects_n_bytes():
    assert len(generate_salt(16)) == 16
    assert len(generate_salt(64)) == 64


def test_generate_salt_is_random():
    """Two calls produce different bytes — at 32 bytes the collision
    probability is ~2^-128, so this is effectively a guarantee."""
    assert generate_salt() != generate_salt()


def test_generate_salt_rejects_zero_or_negative():
    """Zero-byte salt would silently produce a deterministic commitment
    keyed only on (input, evaluator) — defeats §6.7's protection. We
    refuse it at the boundary."""
    with pytest.raises(ValueError):
        generate_salt(0)
    with pytest.raises(ValueError):
        generate_salt(-1)


# ---------------- SaltStore ----------------


def _receipt_id(suffix: str = "00") -> str:
    return "0x" + suffix * 32


def test_salt_store_put_get_roundtrip(tmp_path: Path) -> None:
    store = SaltStore(tmp_path / "salts.json")
    salt = b"\x77" * 32
    store.put(_receipt_id("aa"), salt, input_path="strategies/v1.md")
    out_salt, out_path = store.get(_receipt_id("aa"))
    assert out_salt == salt
    assert out_path == "strategies/v1.md"


def test_salt_store_get_missing_raises_keyerror(tmp_path: Path) -> None:
    """Lost-salt scenario — silently returning a default would let a
    producer build a wrong commitment. Surface loud."""
    store = SaltStore(tmp_path / "salts.json")
    with pytest.raises(KeyError):
        store.get(_receipt_id("ff"))


def test_salt_store_handles_missing_file(tmp_path: Path) -> None:
    """First read against a nonexistent store returns an empty mapping,
    not an OSError — producers don't have to pre-create the file."""
    store = SaltStore(tmp_path / "doesnt-exist.json")
    with pytest.raises(KeyError):
        store.get(_receipt_id("aa"))


def test_salt_store_put_creates_parent_directories(tmp_path: Path) -> None:
    """Producers point at `examples/trading_critic/.salt` by convention
    — the parent dir may not yet exist on a fresh clone. put() creates it."""
    nested = tmp_path / "deeply" / "nested" / "salts.json"
    store = SaltStore(nested)
    store.put(_receipt_id("aa"), b"\x01" * 32)
    assert nested.exists()
    assert nested.parent.exists()


def test_salt_store_put_overwrites_existing_entry(tmp_path: Path) -> None:
    """Receipts have unique receipt_ids in normal operation — an
    overwrite means the producer is re-running with a fresh salt for
    the same receipt_id, which is fine. We don't guard against it."""
    store = SaltStore(tmp_path / "salts.json")
    store.put(_receipt_id("aa"), b"\x01" * 32, input_path="v1.md")
    store.put(_receipt_id("aa"), b"\x02" * 32, input_path="v1-rerun.md")
    salt, path = store.get(_receipt_id("aa"))
    assert salt == b"\x02" * 32
    assert path == "v1-rerun.md"


def test_salt_store_input_path_is_optional(tmp_path: Path) -> None:
    store = SaltStore(tmp_path / "salts.json")
    store.put(_receipt_id("aa"), b"\x01" * 32)
    salt, path = store.get(_receipt_id("aa"))
    assert salt == b"\x01" * 32
    assert path is None


def test_salt_store_persists_across_instances(tmp_path: Path) -> None:
    """A new SaltStore pointing at the same file reads the same data.
    Producers commonly construct the store fresh per run."""
    path = tmp_path / "salts.json"
    SaltStore(path).put(_receipt_id("aa"), b"\x01" * 32, "v1.md")
    salt, ip = SaltStore(path).get(_receipt_id("aa"))
    assert salt == b"\x01" * 32
    assert ip == "v1.md"


def test_salt_store_canonicalizes_receipt_id_case(tmp_path: Path) -> None:
    """Get with uppercase receipt_id hits the entry put with lowercase
    (and vice-versa). Receipts canonicalize to lowercase but a hand
    query with uppercase shouldn't silently miss."""
    store = SaltStore(tmp_path / "salts.json")
    store.put("0x" + "ab" * 32, b"\x01" * 32)
    salt, _ = store.get("0x" + "AB" * 32)
    assert salt == b"\x01" * 32


def test_salt_store_rejects_invalid_receipt_id_on_put(tmp_path: Path) -> None:
    """Invalid receipt_id at put-time is a programming error — refuse
    early so the file never accumulates junk keys."""
    store = SaltStore(tmp_path / "salts.json")
    with pytest.raises(ValueError, match="receipt_id"):
        store.put("0xnothex", b"\x01" * 32)


def test_salt_store_file_is_human_readable_json(tmp_path: Path) -> None:
    """Producers may inspect the salt store by hand. Confirm the file
    is sorted-key indented JSON, not pickled or otherwise opaque."""
    path = tmp_path / "salts.json"
    SaltStore(path).put(_receipt_id("aa"), b"\x01" * 32, "v1.md")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == {
        "0x" + "aa" * 32: {
            "salt": "0x" + "01" * 32,
            "input_path": "v1.md",
        },
    }


def test_salt_store_corrupt_file_raises_value_error(tmp_path: Path) -> None:
    """A non-JSON file at the salt store path is operator surprise (typo,
    accidental edit) — surface as ValueError, not silent pass."""
    path = tmp_path / "salts.json"
    path.write_text("not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        SaltStore(path).get(_receipt_id("aa"))


def test_salt_store_write_is_atomic_prior_file_survives_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-write must NOT corrupt the prior file. Simulate by
    monkeypatching `os.replace` to raise after the new bytes are
    fsynced — the previous SaltStore contents should still be readable
    and the orphan tempfile should be cleaned up.

    Lost-salt is unrecoverable per the module docstring; truncating the
    salt file on a SIGKILL would destroy every persisted reveal in one
    shot, which is exactly the failure mode atomicity is for."""
    path = tmp_path / "salts.json"
    store = SaltStore(path)
    store.put(_receipt_id("aa"), b"\x01" * 32, "v1.md")

    import os as _os

    real_replace = _os.replace

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated crash mid-replace")

    monkeypatch.setattr("eerful.commitment.os.replace", _boom)
    with pytest.raises(OSError, match="simulated crash"):
        store.put(_receipt_id("bb"), b"\x02" * 32, "v2.md")
    monkeypatch.setattr("eerful.commitment.os.replace", real_replace)

    # Prior entry intact: the put-bb attempt did not corrupt put-aa.
    salt, ip = SaltStore(path).get(_receipt_id("aa"))
    assert salt == b"\x01" * 32
    assert ip == "v1.md"

    # Tempfile was cleaned up on the failure — no `.salts.json.*.tmp`
    # orphans linger (they would gradually fill the directory under
    # repeated crashes).
    leftovers = list(tmp_path.glob(".*.tmp")) + list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"orphan tempfiles not cleaned: {leftovers}"


def test_salt_store_normal_write_leaves_no_tempfile(tmp_path: Path) -> None:
    """Successful write leaves only the destination — no `.tmp` siblings."""
    path = tmp_path / "salts.json"
    SaltStore(path).put(_receipt_id("aa"), b"\x01" * 32)
    SaltStore(path).put(_receipt_id("bb"), b"\x02" * 32)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["salts.json"], f"unexpected files: {files}"


def test_salt_store_top_level_array_rejected(tmp_path: Path) -> None:
    """A salt file whose top level is a JSON array is malformed — the
    schema is `dict[receipt_id, entry]`."""
    path = tmp_path / "salts.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="not a JSON object"):
        SaltStore(path).get(_receipt_id("aa"))


# ---------------- Integration: commitment ↔ SaltStore round-trip ----------------


def test_commitment_reconstructed_from_salt_store_matches_original(tmp_path: Path) -> None:
    """Producer flow: generate salt, compute commitment, persist salt
    by receipt_id, later reload salt and reconstruct the same
    commitment. This is the load-bearing pattern for the trading-critic
    chain (v1 → v2 → v3 share a stable input commitment)."""
    rid = _receipt_id("aa")
    input_bytes = b"strategy-identity:market-neutral-v0"

    salt = generate_salt()
    commitment = compute_input_commitment(input_bytes, _EID, salt)

    store = SaltStore(tmp_path / "salts.json")
    store.put(rid, salt, input_path="strategies/v1.md")

    # Later, possibly in a different process — reload the salt and
    # confirm the reconstruction is byte-identical.
    salt_back, _ = SaltStore(tmp_path / "salts.json").get(rid)
    assert compute_input_commitment(input_bytes, _EID, salt_back) == commitment
