"""CLI smoke tests — argparse wiring + publish-evaluator + verify behavior.

`verify_receipt` and the storage-aware orchestrator have their own coverage
in test_verify.py. This file focuses on:

- publish-evaluator's canonical-bytes upload contract
- dry-run path
- error mapping (storage error, trust violation, bundle validation)
- summary lines that surface §6.5 / §8 caveats
- verify subcommand wiring: storage-by-default, --bundle/--report
  overrides, --skip-step-5, loopback guard, mutual-exclusion errors
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from eth_keys import keys
from eth_utils import keccak

from eerful.canonical import tee_signer_address_from_pubkey
from eerful.cli import _is_loopback_bridge_url, _publish_evaluator, main
from eerful.errors import StorageError, TrustViolation
from eerful.evaluator import ComposeHashEntry, EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.zg.storage import MockStorageClient, UploadResult


_PROVIDER_ADDR = "0x" + "d" * 40


def _entry_dict(hash_hex: str, *, category: str = "A") -> dict[str, Any]:
    """Wire-format ComposeHashEntry for `_bundle_bytes` callers — the JSON
    that `EvaluatorBundle.model_validate_json` expects when the input is
    a hand-rolled bundle dict (not a model)."""
    return {
        "hash": hash_hex,
        "category": category,
        "provider_address": _PROVIDER_ADDR,
        "notes": None,
    }


def _bundle_bytes(**overrides: Any) -> bytes:
    payload: dict[str, Any] = {
        "version": "trading-critic@1.0.0",
        "model_identifier": "zai-org/GLM-5-FP8",
        "system_prompt": "rate it",
    }
    payload.update(overrides)
    return json.dumps(payload).encode("utf-8")


# ---------------- _publish_evaluator (inner function) ----------------


def test_publish_evaluator_uploads_canonical_bytes_and_returns_evaluator_id():
    bundle_bytes = _bundle_bytes()
    storage = MockStorageClient()
    eid, sroot, bundle = _publish_evaluator(bundle_bytes, storage)
    assert eid == bundle.evaluator_id()
    # Mock invariant: storage_root == content_hash; tier 2 contract is that
    # the publisher gets a non-None storage_root back.
    assert sroot == eid
    # Storage should hold the *canonical* encoding, not the input bytes
    # (the input may not be canonical-form JSON).
    assert storage.download_blob(eid, sroot) == bundle.canonical_bytes()


def test_publish_evaluator_idempotent_uploads_dont_duplicate():
    """Same bundle uploaded twice maps to the same evaluator_id and
    download still works after the second call."""
    bundle_bytes = _bundle_bytes()
    storage = MockStorageClient()
    eid1, sroot1, _ = _publish_evaluator(bundle_bytes, storage)
    eid2, sroot2, _ = _publish_evaluator(bundle_bytes, storage)
    assert eid1 == eid2
    assert sroot1 == sroot2
    assert storage.download_blob(eid1, sroot1) is not None


def test_publish_evaluator_rejects_storage_hash_mismatch():
    """If storage returns a different hash than bundle.evaluator_id(),
    it's a TrustViolation — either canonical encoding drifted between
    publisher and storage, or storage tampered with bytes mid-flight."""

    class _LyingStorage:
        def upload_blob(self, data: bytes) -> UploadResult:
            bogus = "0x" + "0" * 64
            return UploadResult(content_hash=bogus, storage_root=bogus)

        def download_blob(self, content_hash: str, storage_root: str) -> bytes:
            raise NotImplementedError

    with pytest.raises(TrustViolation, match="canonical encoder drift|tampering"):
        _publish_evaluator(_bundle_bytes(), _LyingStorage())


def test_publish_evaluator_propagates_storage_errors():
    """Bridge unavailability / network errors surface unwrapped — the
    CLI dispatcher catches them at the boundary."""

    class _BrokenStorage:
        def upload_blob(self, data: bytes) -> UploadResult:
            raise StorageError("bridge offline")

        def download_blob(self, content_hash: str, storage_root: str) -> bytes:
            raise NotImplementedError

    with pytest.raises(StorageError, match="bridge offline"):
        _publish_evaluator(_bundle_bytes(), _BrokenStorage())


# ---------------- main() — argparse wiring ----------------


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    """Run main() with captured stdout/stderr. Returns (exit_code, stdout, stderr)."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


def test_publish_evaluator_dry_run_prints_evaluator_id(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(_bundle_bytes())
    expected_eid = EvaluatorBundle.model_validate_json(_bundle_bytes()).evaluator_id()

    rc, out, err = _run_main(
        ["publish-evaluator", "--bundle", str(bundle_path), "--dry-run"]
    )
    assert rc == 0, err
    assert "dry run" in out
    assert expected_eid in out
    assert "trading-critic@1.0.0" in out


def test_publish_evaluator_dry_run_surfaces_no_allowlist_caveat(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(_bundle_bytes())

    rc, out, _ = _run_main(
        ["publish-evaluator", "--bundle", str(bundle_path), "--dry-run"]
    )
    assert rc == 0
    assert "accepted_compose_hashes: not set" in out
    assert "§8" in out


def test_publish_evaluator_dry_run_surfaces_allowlist_count(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(
        _bundle_bytes(
            accepted_compose_hashes=[
                _entry_dict("0x" + "a" * 64),
                _entry_dict("0x" + "b" * 64),
            ]
        )
    )

    rc, out, _ = _run_main(
        ["publish-evaluator", "--bundle", str(bundle_path), "--dry-run"]
    )
    assert rc == 0
    assert "accepted_compose_hashes: 2 entries" in out
    assert "§6.5" in out


def test_publish_evaluator_missing_bundle_returns_2(tmp_path: Path) -> None:
    rc, _, err = _run_main(
        ["publish-evaluator", "--bundle", str(tmp_path / "missing.json"), "--dry-run"]
    )
    assert rc == 2
    assert "failed to load bundle" in err


def test_publish_evaluator_invalid_bundle_returns_1(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bad.json"
    bundle_path.write_bytes(b"not valid json at all")

    rc, _, err = _run_main(
        ["publish-evaluator", "--bundle", str(bundle_path), "--dry-run"]
    )
    assert rc == 1
    assert "does not validate" in err


def test_publish_evaluator_rejects_empty_allowlist(tmp_path: Path) -> None:
    """EvaluatorBundle's model validator forbids accepted_compose_hashes=[];
    the CLI should surface that as a validation failure, not crash."""
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(_bundle_bytes(accepted_compose_hashes=[]))

    rc, _, err = _run_main(
        ["publish-evaluator", "--bundle", str(bundle_path), "--dry-run"]
    )
    assert rc == 1
    assert "does not validate" in err


def test_main_rejects_unknown_subcommand() -> None:
    with pytest.raises(SystemExit):
        _run_main(["nonexistent-command"])


# ---------------- loopback bridge guard ----------------


def test_publish_evaluator_rejects_remote_bridge_url(tmp_path: Path) -> None:
    """Without --allow-remote-bridge, a non-loopback URL must not result
    in any upload — bundles uploaded through a malicious bridge would
    leak the publisher's evaluator definition before they meant to."""
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(_bundle_bytes())

    rc, _, err = _run_main(
        [
            "publish-evaluator",
            "--bundle",
            str(bundle_path),
            "--bridge-url",
            "http://attacker.example.com:7878",
        ]
    )
    assert rc == 2
    assert "non-loopback bridge" in err
    assert "--allow-remote-bridge" in err


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://127.0.0.1:7878", True),
        ("http://localhost:7878", True),
        ("http://[::1]:7878", True),
        ("https://localhost", True),
        # Anywhere in 127.0.0.0/8 is RFC-loopback (delegated to
        # ipaddress.is_loopback); 127.0.1.1 is the Debian/Ubuntu default
        # /etc/hosts entry, 127.0.0.2 is a common alt-binding for tests.
        ("http://127.0.1.1:7878", True),
        ("http://127.0.0.2:7878", True),
        ("http://attacker.example.com:7878", False),
        ("http://10.0.0.5:7878", False),
        ("http://192.168.1.50:7878", False),
        ("http://0.0.0.0:7878", False),  # NOT loopback — listens on all ifaces
        # Non-IP-literal hostnames other than "localhost" force opt-in,
        # even ones that look intranet-y. Operator awareness over DNS
        # trust at this layer.
        ("http://bridge.internal:7878", False),
        ("not a url at all", False),
    ],
)
def test_is_loopback_bridge_url(url: str, expected: bool) -> None:
    """Unit test the host parser directly so the loopback contract is
    pinned independent of networking. 0.0.0.0 explicitly is NOT
    treated as loopback — a service bound to 0.0.0.0 listens on every
    interface, so a client connecting to it could still leak."""
    assert _is_loopback_bridge_url(url) is expected


def test_publish_evaluator_remote_with_opt_in_passes_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --allow-remote-bridge, the loopback guard yields. We verify
    that by monkeypatching BridgeStorageClient to record the URL it
    was given — testing the guard's pass-through, not the network."""
    captured: dict[str, str] = {}

    class _FakeBridge:
        def __init__(self, *, bridge_url: str) -> None:
            captured["url"] = bridge_url

        def __enter__(self) -> "_FakeBridge":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def upload_blob(self, data: bytes) -> UploadResult:
            # Return the canonical hash so _publish_evaluator's
            # defense-in-depth check passes; we want to verify the
            # guard, not exercise upload-mismatch handling. Mirror
            # the mock's storage_root == content_hash convention.
            bundle = EvaluatorBundle.model_validate_json(data)
            eid = bundle.evaluator_id()
            return UploadResult(content_hash=eid, storage_root=eid)

        def download_blob(self, content_hash: str, storage_root: str) -> bytes:
            raise NotImplementedError

    monkeypatch.setattr("eerful.cli.BridgeStorageClient", _FakeBridge)

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(_bundle_bytes())

    rc, out, err = _run_main(
        [
            "publish-evaluator",
            "--bundle",
            str(bundle_path),
            "--bridge-url",
            "http://offsite.example:7878",
            "--allow-remote-bridge",
        ]
    )
    assert rc == 0, err
    assert captured["url"] == "http://offsite.example:7878"
    assert "uploaded" in out


def test_publish_evaluator_dry_run_skips_loopback_check(tmp_path: Path) -> None:
    """Dry-run never connects, so the loopback guard does not apply.
    Surface design: telling someone they need --allow-remote-bridge to
    *not connect* would be confusing."""
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(_bundle_bytes())

    rc, out, _ = _run_main(
        [
            "publish-evaluator",
            "--bundle",
            str(bundle_path),
            "--bridge-url",
            "http://attacker.example.com:7878",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert "dry run" in out


# ---------------- verify subcommand ----------------


_TEST_PRIVKEY = b"\x42" * 32


def _sign_personal(text: str) -> tuple[str, str]:
    text_bytes = text.encode("utf-8")
    msg_hash = keccak(
        b"\x19Ethereum Signed Message:\n" + str(len(text_bytes)).encode() + text_bytes
    )
    pk = keys.PrivateKey(_TEST_PRIVKEY)
    sig = pk.sign_msg_hash(msg_hash)
    return "0x" + pk.public_key.to_bytes().hex(), "0x" + sig.to_bytes().hex()


def _build_report_bytes(
    compose_str: str = "model=zai-org/GLM-5-FP8",
    *,
    report_data_address: str | None = None,
) -> tuple[bytes, str]:
    """Build a minimal attestation report and return (bytes, compose_hash).

    `report_data_address` controls Step 5b's pubkey-binding check.
    Default (None) → empty report_data; pass the address derived from
    a receipt's pubkey to make Step 5b pass.
    """
    app_compose = {
        "docker_compose_file": (
            "services:\n  vllm:\n    image: vllm/vllm-openai:nightly\n"
            f"    command: --{compose_str}\n"
        ),
    }
    raw = json.dumps(app_compose, sort_keys=True)
    real_hash = hashlib.sha256(raw.encode()).hexdigest()
    event_log: list[dict[str, Any]] = [
        {
            "imr": 3,
            "event_type": 134217729,
            "digest": "00" * 48,
            "event": "compose-hash",
            "event_payload": real_hash,
        }
    ]
    tcb = {
        "compose_hash": real_hash,
        "event_log": event_log,
        "app_compose": raw,
    }
    if report_data_address is not None:
        addr = report_data_address.lower()
        if not addr.startswith("0x"):
            addr = "0x" + addr
        rd_bytes = addr.encode("ascii") + b"\x00" * (64 - len(addr))
        report_data = base64.b64encode(rd_bytes).decode()
    else:
        report_data = ""
    envelope = {
        "quote": "00",
        "event_log": json.dumps(event_log),
        "report_data": report_data,
        "vm_config": "{}",
        "tcb_info": json.dumps(tcb),
        "nvidia_payload": {},
    }
    return json.dumps(envelope).encode(), "0x" + real_hash.lower()


def _make_receipt_and_artifacts(
    *,
    accepted_compose_hashes: list[str] | None = None,
) -> tuple[EnhancedReceipt, bytes, bytes]:
    """Build a verifying receipt + the bundle bytes + the report bytes
    that storage should hold for the verify subcommand to succeed.

    Mock convention: `storage_root == content_hash` for both
    artifacts, so the receipt's `evaluator_storage_root` /
    `attestation_storage_root` equal `evaluator_id` /
    `attestation_report_hash`."""
    bundle_kwargs: dict[str, Any] = dict(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
    )
    if accepted_compose_hashes is not None:
        bundle_kwargs["accepted_compose_hashes"] = [
            ComposeHashEntry(
                hash=h, category="A", provider_address=_PROVIDER_ADDR
            )
            for h in accepted_compose_hashes
        ]
    bundle = EvaluatorBundle(**bundle_kwargs)
    bundle_canonical = bundle.canonical_bytes()
    pubkey, sig = _sign_personal("hello")
    report_bytes, _ = _build_report_bytes(
        report_data_address=tee_signer_address_from_pubkey(pubkey),
    )
    rh = "0x" + hashlib.sha256(report_bytes).hexdigest()
    receipt = EnhancedReceipt.build(
        created_at=datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc),
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle.evaluator_id(),
        evaluator_version=bundle.version,
        provider_address="0x" + "b" * 40,
        chat_id="chat-123",
        response_content="hello",
        attestation_report_hash=rh,
        attestation_storage_root=rh,
        enclave_pubkey=pubkey,
        enclave_signature=sig,
    )
    return receipt, bundle_canonical, report_bytes


def _patch_bridge_with_storage(
    monkeypatch: pytest.MonkeyPatch, storage: MockStorageClient
) -> dict[str, Any]:
    """Replace BridgeStorageClient in the CLI with a context-manager
    wrapper around `storage`. Captures the bridge_url for assertions."""
    captured: dict[str, Any] = {}

    class _BridgeWrapper:
        def __init__(self, *, bridge_url: str) -> None:
            captured["url"] = bridge_url
            self._storage = storage

        def __enter__(self) -> MockStorageClient:
            return self._storage

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr("eerful.cli.BridgeStorageClient", _BridgeWrapper)
    return captured


def test_verify_fetches_bundle_and_report_from_storage_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default verify path: no --bundle, no --report. Storage holds both
    artifacts; the CLI fetches by content hash and verifies through
    Steps 1–3 + 5 (gating skipped, no allowlist) + 6."""
    receipt, bundle_canonical, report_bytes = _make_receipt_and_artifacts()
    storage = MockStorageClient()
    storage.upload_blob(bundle_canonical)
    storage.upload_blob(report_bytes)

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, out, err = _run_main(["verify", str(receipt_path)])
    assert rc == 0, err
    assert "OK" in out
    assert "trading-critic@1.0.0" in out
    # Default: no allowlist on the bundle → gating skipped
    assert "compose-hash gating: skipped" in out


def test_verify_storage_path_enforces_allowlist_when_bundle_declares_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Storage-fetched bundle declares an allowlist that includes the
    attested compose-hash → Step 5 gating reads `enforced` in the output."""
    _, _, report_bytes = _make_receipt_and_artifacts()
    real_hash = "0x" + hashlib.sha256(
        json.dumps(
            {
                "docker_compose_file": (
                    "services:\n  vllm:\n    image: vllm/vllm-openai:nightly\n"
                    "    command: --model=zai-org/GLM-5-FP8\n"
                ),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    receipt, bundle_canonical, _ = _make_receipt_and_artifacts(
        accepted_compose_hashes=[real_hash]
    )
    storage = MockStorageClient()
    storage.upload_blob(bundle_canonical)
    storage.upload_blob(report_bytes)

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, out, err = _run_main(["verify", str(receipt_path)])
    assert rc == 0, err
    assert "compose-hash gating: enforced" in out
    assert "allowlist match" in out


def test_verify_skip_step_5_does_not_fetch_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--skip-step-5 means we never look up the report. Storage holding
    only the bundle is sufficient; the CLI completes Steps 1–3 + 6."""
    receipt, bundle_canonical, _ = _make_receipt_and_artifacts()
    storage = MockStorageClient()
    storage.upload_blob(bundle_canonical)
    # Report intentionally NOT in storage.
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, out, err = _run_main(["verify", str(receipt_path), "--skip-step-5"])
    assert rc == 0, err
    # Single explanatory line, not a contradictory pair.
    assert "Step 5: skipped (--skip-step-5)" in out
    assert "Step 5: not run" not in out


def test_verify_falls_back_to_bundle_override_without_bridge(tmp_path: Path) -> None:
    """If both --bundle and --skip-step-5 are given, the CLI does not
    open a bridge connection at all. We assert this implicitly: no
    monkeypatch on BridgeStorageClient, and the CLI still succeeds.
    A regression that opens the bridge anyway would raise a connection
    error here."""
    receipt, bundle_canonical, _ = _make_receipt_and_artifacts()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(bundle_canonical)

    rc, out, err = _run_main(
        [
            "verify",
            str(receipt_path),
            "--bundle",
            str(bundle_path),
            "--skip-step-5",
        ]
    )
    assert rc == 0, err
    assert "OK" in out


def test_verify_uses_report_override_with_bundle_from_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed-source verification: bundle from storage, report from local
    file. Step 5 runs against the file's bytes; storage is touched only
    for the bundle."""
    receipt, bundle_canonical, report_bytes = _make_receipt_and_artifacts()
    storage = MockStorageClient()
    storage.upload_blob(bundle_canonical)
    # Report NOT uploaded to storage; we'll pass --report.

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())
    report_path = tmp_path / "report.json"
    report_path.write_bytes(report_bytes)

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, out, err = _run_main(
        ["verify", str(receipt_path), "--report", str(report_path)]
    )
    assert rc == 0, err
    assert "compose-hash gating" in out


def test_verify_no_bridge_path_also_hash_checks_report_override(
    tmp_path: Path,
) -> None:
    """The fully-offline path (--bundle + --report, no bridge) must also
    enforce Step 4's content check on the report file. Otherwise
    `eerful verify --bundle X --report Y` could silently accept a
    mismatched report. Regression: prior to the round-3 fix, the
    no-bridge branch in `_cmd_verify` called `verify_receipt` directly
    without the check."""
    receipt, bundle_canonical, _ = _make_receipt_and_artifacts()

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(bundle_canonical)

    # Different report bytes than the receipt commits to.
    other_report, _ = _build_report_bytes("model=other")
    other_path = tmp_path / "wrong-report.json"
    other_path.write_bytes(other_report)

    rc, _, err = _run_main(
        [
            "verify",
            str(receipt_path),
            "--bundle",
            str(bundle_path),
            "--report",
            str(other_path),
        ]
    )
    assert rc == 1
    assert "verification step 4" in err
    assert "content hash mismatch" in err


def test_verify_report_override_mismatch_fails_at_step_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--report` file whose sha256 doesn't match
    `receipt.attestation_report_hash` must fail at Step 4 — otherwise
    the override turns into a way to bypass report binding entirely
    (Step 5 might still parse and pass the allowlist check on the
    wrong report)."""
    receipt, bundle_canonical, _ = _make_receipt_and_artifacts()
    storage = MockStorageClient()
    storage.upload_blob(bundle_canonical)

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())
    # Different report bytes than the one the receipt commits to.
    other_report, _ = _build_report_bytes("model=other")
    other_path = tmp_path / "wrong-report.json"
    other_path.write_bytes(other_report)

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, _, err = _run_main(
        ["verify", str(receipt_path), "--report", str(other_path)]
    )
    assert rc == 1
    assert "verification step 4" in err
    assert "content hash mismatch" in err
    assert "--report file hashes" in err


def test_verify_report_and_skip_step_5_are_mutually_exclusive(tmp_path: Path) -> None:
    receipt, bundle_canonical, report_bytes = _make_receipt_and_artifacts()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(bundle_canonical)
    report_path = tmp_path / "report.json"
    report_path.write_bytes(report_bytes)

    rc, _, err = _run_main(
        [
            "verify",
            str(receipt_path),
            "--bundle",
            str(bundle_path),
            "--report",
            str(report_path),
            "--skip-step-5",
        ]
    )
    assert rc == 2
    assert "mutually exclusive" in err


def test_verify_rejects_remote_bridge_url(tmp_path: Path) -> None:
    """Same loopback guard as publish-evaluator: a non-loopback bridge
    URL is refused unless --allow-remote-bridge is given. Important for
    verify too — verifier privacy depends on which queries leak."""
    receipt, _, _ = _make_receipt_and_artifacts()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    rc, _, err = _run_main(
        [
            "verify",
            str(receipt_path),
            "--bridge-url",
            "http://attacker.example.com:7878",
        ]
    )
    assert rc == 2
    assert "non-loopback bridge" in err
    assert "--allow-remote-bridge" in err


def test_verify_remote_bridge_with_opt_in_passes_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, bundle_canonical, report_bytes = _make_receipt_and_artifacts()
    storage = MockStorageClient()
    storage.upload_blob(bundle_canonical)
    storage.upload_blob(report_bytes)

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    captured = _patch_bridge_with_storage(monkeypatch, storage)

    rc, out, err = _run_main(
        [
            "verify",
            str(receipt_path),
            "--bridge-url",
            "http://offsite.example:7878",
            "--allow-remote-bridge",
        ]
    )
    assert rc == 0, err
    assert captured["url"] == "http://offsite.example:7878"
    assert "OK" in out


def test_verify_surfaces_step_attribution_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A storage-fetched bundle whose hash mismatches surfaces as Step 2
    failure, not a generic error. Important for ops triage: the spec
    step number tells you which assumption broke."""
    receipt, bundle_canonical, report_bytes = _make_receipt_and_artifacts()
    storage = MockStorageClient()
    # Upload a *different* bundle under a *different* hash, but don't
    # upload the receipt's actual bundle. Storage lookup by
    # receipt.evaluator_id will miss → Step 2.
    storage.upload_blob(b"some unrelated bytes")
    storage.upload_blob(report_bytes)

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, _, err = _run_main(["verify", str(receipt_path)])
    assert rc == 1
    assert "verification step 2" in err

    # Bonus assertion to make sure we're verifying the right thing —
    # the unrelated bytes haven't been silently treated as the bundle.
    assert "evaluator bundle not retrievable" in err

    # The local bundle bytes should still be valid (sanity check that
    # the receipt isn't broken — the failure is purely Step 2 fetch).
    assert bundle_canonical not in err.encode()


# ---------------- gate subcommand ----------------


def _make_gate_artifacts(
    *,
    accepted_compose_hashes: list[ComposeHashEntry] | None = None,
    output_score_block: dict[str, Any] | None = None,
) -> tuple[EnhancedReceipt, EvaluatorBundle, bytes]:
    """Build a receipt + bundle + report bytes for gate-subcommand tests.

    Differs from `_make_receipt_and_artifacts` (which targets the verify
    subcommand) in two ways: it surfaces the bundle (so tests can register
    its `evaluator_id` in a policy), and it includes a default
    `output_score_block` so the gate's all_must_pass score check succeeds
    on the PASS path."""
    bundle = EvaluatorBundle(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
        accepted_compose_hashes=accepted_compose_hashes,
    )
    pubkey, sig = _sign_personal("hello")
    report_bytes, _ = _build_report_bytes(
        report_data_address=tee_signer_address_from_pubkey(pubkey),
    )
    rh = "0x" + hashlib.sha256(report_bytes).hexdigest()
    score = output_score_block if output_score_block is not None else {"overall": 0.8}
    receipt = EnhancedReceipt.build(
        created_at=datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc),
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle.evaluator_id(),
        evaluator_version=bundle.version,
        provider_address="0x" + "b" * 40,
        chat_id="chat-1",
        response_content="hello",
        attestation_report_hash=rh,
        attestation_storage_root=rh,
        enclave_pubkey=pubkey,
        enclave_signature=sig,
        output_score_block=score,
    )
    return receipt, bundle, report_bytes


def _write_policy(
    tmp_path: Path,
    *,
    bundle_id: str,
    n_attestations: int = 1,
    score_threshold: float = 0.6,
) -> Path:
    policy_dict = {
        "policy_version": "0.5",
        "principal_id": "test",
        "bundles": {"proposal_grade": bundle_id},
        "tiers": {
            "low_consequence": {
                "n_attestations": n_attestations,
                "score_threshold": score_threshold,
                "diversity": {"distinct_signers": False, "distinct_compose_hashes": False},
            }
        },
        "score_aggregation": "all_must_pass",
    }
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy_dict))
    return p


def test_gate_passes_minimum_viable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single receipt, no allowlist, score above threshold, storage holds
    the bundle + report → exit 0, PASS line in stdout, canonical_set_hash
    surfaced."""
    receipt, bundle, report_bytes = _make_gate_artifacts()
    storage = MockStorageClient()
    storage.upload_blob(bundle.canonical_bytes())
    storage.upload_blob(report_bytes)

    policy_path = _write_policy(tmp_path, bundle_id=bundle.evaluator_id())
    receipt_path = tmp_path / "r1.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, out, err = _run_main(
        [
            "gate",
            "--policy",
            str(policy_path),
            "--tier",
            "low_consequence",
            "--bundle",
            "proposal_grade",
            "--receipt",
            str(receipt_path),
        ]
    )
    assert rc == 0, err
    assert "PASS" in out
    assert "canonical_set_hash" in out
    assert "1 supplied / 1 required" in out


def test_gate_refuses_with_exit_1_on_score_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Score below threshold → REFUSE_SCORE → exit 1, REFUSE line in stderr,
    detail surfaces the gap. Distinct from exit 2 (wiring error)."""
    receipt, bundle, report_bytes = _make_gate_artifacts(
        output_score_block={"overall": 0.4}
    )
    storage = MockStorageClient()
    storage.upload_blob(bundle.canonical_bytes())
    storage.upload_blob(report_bytes)

    policy_path = _write_policy(
        tmp_path, bundle_id=bundle.evaluator_id(), score_threshold=0.7
    )
    receipt_path = tmp_path / "r1.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, _, err = _run_main(
        [
            "gate",
            "--policy",
            str(policy_path),
            "--tier",
            "low_consequence",
            "--bundle",
            "proposal_grade",
            "--receipt",
            str(receipt_path),
        ]
    )
    assert rc == 1
    assert "REFUSE" in err
    assert "refuse_score" in err
    assert "0.4" in err


def test_gate_exits_2_on_unknown_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PolicyError surfaces as exit 2 (wiring bug), not 1 (refusal). Lets
    a CI integration distinguish 'rails working as designed, refused' from
    'someone misconfigured the policy'."""
    receipt, bundle, _ = _make_gate_artifacts()
    storage = MockStorageClient()
    storage.upload_blob(bundle.canonical_bytes())

    policy_path = _write_policy(tmp_path, bundle_id=bundle.evaluator_id())
    receipt_path = tmp_path / "r1.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, _, err = _run_main(
        [
            "gate",
            "--policy",
            str(policy_path),
            "--tier",
            "high_consequence",  # not in policy
            "--bundle",
            "proposal_grade",
            "--receipt",
            str(receipt_path),
        ]
    )
    assert rc == 2
    assert "policy wiring error" in err
    assert "high_consequence" in err


def test_gate_exits_2_on_unknown_bundle_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, bundle, _ = _make_gate_artifacts()
    storage = MockStorageClient()
    storage.upload_blob(bundle.canonical_bytes())

    policy_path = _write_policy(tmp_path, bundle_id=bundle.evaluator_id())
    receipt_path = tmp_path / "r1.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, _, err = _run_main(
        [
            "gate",
            "--policy",
            str(policy_path),
            "--tier",
            "low_consequence",
            "--bundle",
            "nonexistent",  # not in policy.bundles
            "--receipt",
            str(receipt_path),
        ]
    )
    assert rc == 2
    assert "policy wiring error" in err
    assert "nonexistent" in err


def test_gate_exits_2_on_missing_policy_file(tmp_path: Path) -> None:
    rc, _, err = _run_main(
        [
            "gate",
            "--policy",
            str(tmp_path / "nope.json"),
            "--tier",
            "low_consequence",
            "--bundle",
            "proposal_grade",
            "--receipt",
            str(tmp_path / "any.json"),
        ]
    )
    assert rc == 2
    assert "failed to load policy" in err


def test_gate_exits_2_on_invalid_policy_json(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(b"not valid json")
    receipt_path = tmp_path / "r.json"
    receipt_path.write_bytes(b"{}")  # any value; we won't reach receipt loading

    rc, _, err = _run_main(
        [
            "gate",
            "--policy",
            str(policy_path),
            "--tier",
            "low_consequence",
            "--bundle",
            "proposal_grade",
            "--receipt",
            str(receipt_path),
        ]
    )
    assert rc == 2
    assert "policy does not validate" in err


def test_gate_exits_2_on_invalid_receipt_json(tmp_path: Path) -> None:
    """Bad receipt file fails before any storage lookup. Distinguishable
    from a refusal at exit-code level."""
    bundle_id = "0x" + "a" * 64
    policy_path = _write_policy(tmp_path, bundle_id=bundle_id)
    receipt_path = tmp_path / "r.json"
    receipt_path.write_bytes(b"{not valid}")

    rc, _, err = _run_main(
        [
            "gate",
            "--policy",
            str(policy_path),
            "--tier",
            "low_consequence",
            "--bundle",
            "proposal_grade",
            "--receipt",
            str(receipt_path),
        ]
    )
    assert rc == 2
    assert "does not validate" in err


def test_gate_refuses_non_loopback_bridge_url(tmp_path: Path) -> None:
    """The loopback guard applies to gate just like verify and publish.
    Without --allow-remote-bridge, a non-loopback URL is exit 2.

    Uses a valid receipt + policy so the failure surfaces *only* on the
    loopback check (not on input validation that happens earlier)."""
    receipt, bundle, _ = _make_gate_artifacts()
    policy_path = _write_policy(tmp_path, bundle_id=bundle.evaluator_id())
    receipt_path = tmp_path / "r.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    rc, _, err = _run_main(
        [
            "gate",
            "--policy",
            str(policy_path),
            "--tier",
            "low_consequence",
            "--bundle",
            "proposal_grade",
            "--receipt",
            str(receipt_path),
            "--bridge-url",
            "http://attacker.example.com:7878",
        ]
    )
    assert rc == 2
    assert "non-loopback bridge" in err


def test_gate_exits_2_when_evaluate_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defensive: if `evaluate_gate` raises ValueError that the executor
    didn't translate to a refuse outcome (e.g. a future code path the
    internal catch doesn't cover), the CLI must exit 2 with a controlled
    message — not crash with a traceback at the operator surface."""
    receipt, bundle, _ = _make_gate_artifacts()
    storage = MockStorageClient()
    storage.upload_blob(bundle.canonical_bytes())

    policy_path = _write_policy(tmp_path, bundle_id=bundle.evaluator_id())
    receipt_path = tmp_path / "r1.json"
    receipt_path.write_bytes(receipt.model_dump_json().encode())

    _patch_bridge_with_storage(monkeypatch, storage)

    def _raise_value_error(**_kwargs: Any) -> None:
        raise ValueError("synthesized: pubkey is wrong length")

    monkeypatch.setattr("eerful.cli.evaluate_gate", _raise_value_error)

    rc, _, err = _run_main(
        [
            "gate",
            "--policy",
            str(policy_path),
            "--tier",
            "low_consequence",
            "--bundle",
            "proposal_grade",
            "--receipt",
            str(receipt_path),
        ]
    )
    assert rc == 2
    assert "invalid input" in err
    assert "pubkey is wrong length" in err


def test_gate_passes_with_n2_distinct_signers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: two receipts from distinct signing keys, tier requires
    distinct_signers=true and n_attestations=2, score above threshold →
    PASS. Locks down the multi-receipt path through the CLI."""
    bundle = EvaluatorBundle(
        version="trading-critic@1.0.0",
        model_identifier="zai-org/GLM-5-FP8",
        system_prompt="rate it",
    )
    response_content = "hello"
    pk_a = keys.PrivateKey(b"\x42" * 32)
    pk_b = keys.PrivateKey(b"\x77" * 32)
    text_bytes = response_content.encode()
    msg_hash = keccak(
        b"\x19Ethereum Signed Message:\n" + str(len(text_bytes)).encode() + text_bytes
    )
    pubkey_a = "0x" + pk_a.public_key.to_bytes().hex()
    pubkey_b = "0x" + pk_b.public_key.to_bytes().hex()
    sig_a = "0x" + pk_a.sign_msg_hash(msg_hash).to_bytes().hex()
    sig_b = "0x" + pk_b.sign_msg_hash(msg_hash).to_bytes().hex()

    # Distinct signers → distinct enclaves → distinct attestation reports.
    report_a, _ = _build_report_bytes(
        "model=zai-org/GLM-5-FP8",
        report_data_address=tee_signer_address_from_pubkey(pubkey_a),
    )
    report_b, _ = _build_report_bytes(
        "model=zai-org/GLM-5-FP8-other",
        report_data_address=tee_signer_address_from_pubkey(pubkey_b),
    )
    rh_a = "0x" + hashlib.sha256(report_a).hexdigest()
    rh_b = "0x" + hashlib.sha256(report_b).hexdigest()
    storage = MockStorageClient()
    storage.upload_blob(bundle.canonical_bytes())
    storage.upload_blob(report_a)
    storage.upload_blob(report_b)

    common: dict[str, Any] = dict(
        created_at=datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc),
        evaluator_id=bundle.evaluator_id(),
        evaluator_storage_root=bundle.evaluator_id(),
        evaluator_version=bundle.version,
        provider_address="0x" + "b" * 40,
        response_content=response_content,
        output_score_block={"overall": 0.8},
    )
    r1 = EnhancedReceipt.build(
        chat_id="chat-1",
        enclave_pubkey=pubkey_a,
        enclave_signature=sig_a,
        attestation_report_hash=rh_a,
        attestation_storage_root=rh_a,
        **common,
    )
    r2 = EnhancedReceipt.build(
        chat_id="chat-2",
        enclave_pubkey=pubkey_b,
        enclave_signature=sig_b,
        attestation_report_hash=rh_b,
        attestation_storage_root=rh_b,
        **common,
    )
    r1_path = tmp_path / "r1.json"
    r2_path = tmp_path / "r2.json"
    r1_path.write_bytes(r1.model_dump_json().encode())
    r2_path.write_bytes(r2.model_dump_json().encode())

    policy_dict: dict[str, Any] = {
        "policy_version": "0.5",
        "principal_id": "test",
        "bundles": {"proposal_grade": bundle.evaluator_id()},
        "tiers": {
            "high_consequence": {
                "n_attestations": 2,
                "score_threshold": 0.6,
                "diversity": {"distinct_signers": True, "distinct_compose_hashes": False},
            }
        },
        "score_aggregation": "all_must_pass",
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy_dict))

    _patch_bridge_with_storage(monkeypatch, storage)

    rc, out, err = _run_main(
        [
            "gate",
            "--policy",
            str(policy_path),
            "--tier",
            "high_consequence",
            "--bundle",
            "proposal_grade",
            "--receipt",
            str(r1_path),
            "--receipt",
            str(r2_path),
        ]
    )
    assert rc == 0, err
    assert "PASS" in out
    assert "2 supplied / 2 required" in out
