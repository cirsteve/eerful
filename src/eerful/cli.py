"""eerful CLI.

The `verify` subcommand runs the spec §7.1 verification algorithm against
an on-disk receipt and prints a human-readable verdict. Step 4 (fetching
the attestation report from 0G Storage) is not yet wired here — the
caller passes `--report` with the report bytes on disk for now. When 0G
Storage integration lands, Step 4 will run from `attestation_report_hash`
without an explicit flag.

The `publish-evaluator` subcommand uploads a bundle to 0G Storage via the
local zg-bridge so verifiers can fetch it by `evaluator_id` (the bundle's
content hash). The bundle bytes that get uploaded are the canonical
encoding produced by `EvaluatorBundle.canonical_bytes()` — uploading the
on-disk file directly would break Step 2 verification if the source file
is not byte-identical to canonical form.

The `evaluate` subcommand lands with the jig adapter.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from eerful.canonical import Bytes32Hex
from eerful.errors import StorageError, TrustViolation, VerificationError
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.verify import VerificationResult, verify_receipt
from eerful.zg.storage import BridgeStorageClient, StorageClient


_DEFAULT_BRIDGE_URL = "http://127.0.0.1:7878"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback_bridge_url(bridge_url: str) -> bool:
    """True iff `bridge_url`'s host is loopback.

    The bridge holds the wallet key and bundles uploaded through it are
    signed-and-paid actions; sending them to a non-loopback host without
    explicit operator opt-in is the same class of risk as binding the
    bridge itself to a non-loopback interface (which the bridge already
    refuses by default, see services/zg-bridge/server.ts). The CLI side
    of the same boundary needs the same guard — the bridge's bind-host
    check protects it from listening on a public interface, but does
    NOT stop a misconfigured client from connecting to one.
    """
    host = urlparse(bridge_url).hostname
    return host is not None and host in _LOOPBACK_HOSTS


def _category_blurb(category: str) -> str:
    """Map §8.2 category to a one-line human description.

    Kept in this module (not in `attestation.py`) so the protocol layer
    stays category-symbol-only — the prose framing belongs to the
    user-facing surface."""
    if category == "A":
        return (
            "§8 Category A — bound launch string. The attested compose names "
            "this model; the launch-time identifier is cryptographically bound."
        )
    if category == "B":
        return (
            "§8 Category B — unrelated compose. The attested compose does not "
            "reference this model; the model claim has no attestation backing."
        )
    if category == "C":
        return (
            "§8 Category C — centralized passthrough. The attested compose runs "
            "a broker proxy; the model is served from a non-attested backend."
        )
    return "§8 category — unknown (compose did not match A/B/C heuristics)."


def _print_verification_result(result: VerificationResult) -> None:
    print("OK — Steps 1–3 passed.")
    print(f"  evaluator: {result.bundle.version} ({result.bundle.model_identifier})")

    step5 = result.step5
    if step5 is None:
        print("  Step 5: not run (no --report supplied; pass --report to enforce §6.5).")
        print(
            "  caveat: without Step 5, model-environment binding is unverified — "
            "the receipt is integrity-checked but the §8 category is unknown."
        )
        return

    print(f"  attested compose-hash: {step5.compose_hash}")
    print(f"  compose-hash gating: {step5.gating}")
    if step5.gating == "enforced":
        print(
            f"  ✓ allowlist match — compose-hash is in evaluator bundle's "
            f"accepted_compose_hashes ({len(result.bundle.accepted_compose_hashes or [])} entries)."
        )
    else:
        # gating=="skipped" — bundle declared no allowlist; surface the §8 caveat.
        print(
            "  caveat: bundle did not declare accepted_compose_hashes; "
            "model-identity binding rests on protocol-level attestation alone (§8)."
        )
    print(f"  {_category_blurb(step5.category)}")


def _cmd_verify(args: argparse.Namespace) -> int:
    receipt_path: Path = args.receipt
    bundle_path: Path = args.bundle
    report_path: Path | None = args.report

    try:
        receipt = EnhancedReceipt.model_validate_json(receipt_path.read_bytes())
    except Exception as e:
        print(f"failed to load receipt at {receipt_path}: {e}", file=sys.stderr)
        return 2

    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as e:
        print(f"failed to load bundle at {bundle_path}: {e}", file=sys.stderr)
        return 2

    if report_path is not None:
        try:
            report_bytes: bytes | None = report_path.read_bytes()
        except OSError as e:
            print(f"failed to load report at {report_path}: {e}", file=sys.stderr)
            return 2
    else:
        report_bytes = None

    try:
        result = verify_receipt(receipt, bundle_bytes, report_bytes)
    except VerificationError as e:
        print(f"FAIL — verification step {e.step}: {e.reason}", file=sys.stderr)
        return 1

    _print_verification_result(result)
    return 0


def _publish_evaluator(
    bundle_bytes: bytes,
    storage: StorageClient,
) -> tuple[Bytes32Hex, EvaluatorBundle]:
    """Validate and upload a bundle. Returns (storage-side content hash, bundle).

    The upload sends the bundle's canonical encoding (sort-keys etc),
    not the on-disk bytes verbatim. A bundle file authored by hand may
    have whitespace / key-order differences from canonical form; verifiers
    re-derive `evaluator_id` from canonical bytes during Step 2, so the
    publisher must upload canonical bytes for fetched-bundle ↔
    receipt.evaluator_id round-trips to match.

    The caller's defense-in-depth check: storage's returned hash MUST
    equal `bundle.evaluator_id()`. A mismatch means the storage backend
    served back different bytes than what was sent (or our canonical
    encoder disagreed with itself between calls — the test suite catches
    that elsewhere). Surfaces as a TrustViolation in either case.
    """
    bundle = EvaluatorBundle.model_validate_json(bundle_bytes)
    canonical = bundle.canonical_bytes()
    expected_id = bundle.evaluator_id()
    uploaded = storage.upload_blob(canonical)
    if uploaded != expected_id:
        raise TrustViolation(
            f"upload returned {uploaded} but bundle.evaluator_id()={expected_id} "
            "(canonical encoder drift or storage byte tampering)"
        )
    return uploaded, bundle


def _cmd_publish_evaluator(args: argparse.Namespace) -> int:
    bundle_path: Path = args.bundle
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as e:
        print(f"failed to load bundle at {bundle_path}: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        try:
            bundle = EvaluatorBundle.model_validate_json(bundle_bytes)
        except Exception as e:
            print(f"bundle does not validate: {e}", file=sys.stderr)
            return 1
        _print_publish_summary(bundle.evaluator_id(), bundle, uploaded=False)
        return 0

    bridge_url = args.bridge_url
    if not _is_loopback_bridge_url(bridge_url) and not args.allow_remote_bridge:
        print(
            f"refusing to send bundle to non-loopback bridge {bridge_url!r}. "
            "Re-run with --allow-remote-bridge if this is intentional and you "
            "trust the network path.",
            file=sys.stderr,
        )
        return 2
    try:
        with BridgeStorageClient(bridge_url=bridge_url) as storage:
            evaluator_id, bundle = _publish_evaluator(bundle_bytes, storage)
    except (StorageError, TrustViolation) as e:
        print(f"publish failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # Bundle validation, JSON parse, etc — caller-fixable input issues.
        print(f"bundle does not validate: {e}", file=sys.stderr)
        return 1

    _print_publish_summary(evaluator_id, bundle, uploaded=True)
    return 0


def _print_publish_summary(
    evaluator_id: Bytes32Hex,
    bundle: EvaluatorBundle,
    *,
    uploaded: bool,
) -> None:
    if uploaded:
        print("OK — evaluator bundle uploaded to 0G Storage.")
    else:
        print("OK — bundle validates (dry run; nothing uploaded).")
    print(f"  evaluator_id: {evaluator_id}")
    print(f"  version: {bundle.version}")
    print(f"  model_identifier: {bundle.model_identifier}")
    allowlist = bundle.accepted_compose_hashes
    if allowlist:
        print(
            f"  accepted_compose_hashes: {len(allowlist)} entries "
            "(verifiers will enforce §6.5 compose-hash gating)."
        )
    else:
        print(
            "  accepted_compose_hashes: not set "
            "(no compose-hash gating; see §8 for what verifiers can prove)."
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eerful")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="verify a receipt against an evaluator bundle and report")
    v.add_argument("receipt", type=Path, help="path to receipt JSON")
    v.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="path to evaluator bundle bytes (canonical JSON; matches receipt.evaluator_id)",
    )
    v.add_argument(
        "--report",
        type=Path,
        default=None,
        help="path to attestation report JSON (optional; enables Step 5 compose-hash gating)",
    )
    v.set_defaults(func=_cmd_verify)

    pub = sub.add_parser(
        "publish-evaluator",
        help="upload an evaluator bundle to 0G Storage via the local zg-bridge",
    )
    pub.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="path to evaluator bundle JSON (any valid encoding; canonical bytes are uploaded)",
    )
    pub.add_argument(
        "--bridge-url",
        type=str,
        default=os.environ.get("EERFUL_0G_BRIDGE_URL", _DEFAULT_BRIDGE_URL),
        help=(
            "URL of the zg-bridge (default: $EERFUL_0G_BRIDGE_URL or "
            f"{_DEFAULT_BRIDGE_URL}). Non-loopback URLs are refused unless "
            "--allow-remote-bridge is passed."
        ),
    )
    pub.add_argument(
        "--allow-remote-bridge",
        action="store_true",
        help=(
            "opt out of the loopback-only bridge guard. Required to upload "
            "bundles to a bridge running off-host. Mirrors the bridge's own "
            "EERFUL_0G_BRIDGE_BIND_HOST_I_UNDERSTAND opt-in."
        ),
    )
    pub.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the bundle and print evaluator_id without uploading",
    )
    pub.set_defaults(func=_cmd_publish_evaluator)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
