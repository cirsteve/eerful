"""CLI smoke tests — argparse wiring + publish-evaluator behavior.

The verify subcommand has its own behavioral coverage in test_verify.py
(those exercise the underlying verify_receipt directly). This file
focuses on:

- publish-evaluator's canonical-bytes upload contract
- dry-run path
- error mapping (storage error, trust violation, bundle validation)
- summary lines that surface §6.5 / §8 caveats
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from eerful.cli import _is_loopback_bridge_url, _publish_evaluator, main
from eerful.errors import StorageError, TrustViolation
from eerful.evaluator import EvaluatorBundle
from eerful.zg.storage import MockStorageClient


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
    eid, bundle = _publish_evaluator(bundle_bytes, storage)
    assert eid == bundle.evaluator_id()
    # Storage should hold the *canonical* encoding, not the input bytes
    # (the input may not be canonical-form JSON).
    assert storage.download_blob(eid) == bundle.canonical_bytes()


def test_publish_evaluator_idempotent_uploads_dont_duplicate():
    """Same bundle uploaded twice maps to the same evaluator_id and
    download still works after the second call."""
    bundle_bytes = _bundle_bytes()
    storage = MockStorageClient()
    eid1, _ = _publish_evaluator(bundle_bytes, storage)
    eid2, _ = _publish_evaluator(bundle_bytes, storage)
    assert eid1 == eid2
    assert storage.download_blob(eid1) is not None


def test_publish_evaluator_rejects_storage_hash_mismatch():
    """If storage returns a different hash than bundle.evaluator_id(),
    it's a TrustViolation — either canonical encoding drifted between
    publisher and storage, or storage tampered with bytes mid-flight."""

    class _LyingStorage:
        def upload_blob(self, data: bytes) -> str:
            return "0x" + "0" * 64  # always lies

        def download_blob(self, content_hash: str) -> bytes:
            raise NotImplementedError

    with pytest.raises(TrustViolation, match="canonical encoder drift|tampering"):
        _publish_evaluator(_bundle_bytes(), _LyingStorage())


def test_publish_evaluator_propagates_storage_errors():
    """Bridge unavailability / network errors surface unwrapped — the
    CLI dispatcher catches them at the boundary."""

    class _BrokenStorage:
        def upload_blob(self, data: bytes) -> str:
            raise StorageError("bridge offline")

        def download_blob(self, content_hash: str) -> bytes:
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
                "0x" + "a" * 64,
                "0x" + "b" * 64,
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

        def upload_blob(self, data: bytes) -> str:
            # Return the canonical hash so _publish_evaluator's
            # defense-in-depth check passes; we want to verify the
            # guard, not exercise upload-mismatch handling.
            bundle = EvaluatorBundle.model_validate_json(data)
            return bundle.evaluator_id()

        def download_blob(self, content_hash: str) -> bytes:
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
