"""eerful CLI.

The `verify` subcommand runs the spec §7.1 verification algorithm against
an on-disk receipt and prints a human-readable verdict. By default it
fetches both the evaluator bundle (Step 2) and the attestation report
(Step 4) from 0G Storage by content hash via the local zg-bridge —
matching the spec's normative form. `--bundle` and `--report` allow
overriding either artifact with a local file (offline verification, dev
loops, cached blobs); `--skip-step-5` lets a verifier without storage
access for the report skip Step 5 entirely while still running Steps
1–3 + 6.

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
import hashlib
import ipaddress
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from eerful.canonical import Bytes32Hex
from eerful.errors import StorageError, TrustViolation, VerificationError
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.verify import (
    VerificationResult,
    fetch_evaluator_bundle_bytes,
    verify_receipt,
    verify_receipt_with_storage,
    verify_step_4_attestation_report,
)
from eerful.zg.storage import BridgeStorageClient, StorageClient


_DEFAULT_BRIDGE_URL = "http://127.0.0.1:7878"


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

    Loopback recognition delegates to `ipaddress.ip_address(...).is_loopback`
    for IP literals so the entire 127.0.0.0/8 range counts (including
    127.0.1.1, the default `/etc/hosts` entry on Debian/Ubuntu). The
    "localhost" hostname is matched explicitly — `ip_address("localhost")`
    raises, and we don't want to do DNS at this layer (a hostile resolver
    could map `localhost` to a public IP, but a hostile resolver could
    also map any name we whitelist; the right defense is operator
    awareness via --allow-remote-bridge for any non-IP-literal off-list).
    """
    host = urlparse(bridge_url).hostname
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Hostname that isn't an IP literal and isn't "localhost":
        # treat as remote. Forces `--allow-remote-bridge` for things
        # like `bridge.internal`, which is the conservative default.
        return False


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


def _print_verification_result(
    result: VerificationResult, *, skip_step_5: bool = False
) -> None:
    print("OK — Steps 1–3 passed.")
    print(f"  evaluator: {result.bundle.version} ({result.bundle.model_identifier})")

    step5 = result.step5
    if step5 is None:
        if skip_step_5:
            print("  Step 5: skipped (--skip-step-5).")
        else:
            # Library callers can pass `report_bytes=None` directly to
            # `verify_receipt`; the CLI never lands here without
            # --skip-step-5 (storage fetch is the default). Keep the
            # generic message for the library-call shape.
            print("  Step 5: not run (no attestation report supplied).")
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
    bundle_path: Path | None = args.bundle
    report_path: Path | None = args.report
    skip_step_5: bool = args.skip_step_5

    try:
        receipt = EnhancedReceipt.model_validate_json(receipt_path.read_bytes())
    except Exception as e:
        print(f"failed to load receipt at {receipt_path}: {e}", file=sys.stderr)
        return 2

    # Determine which artifacts must come from storage. If a local file
    # was given for an artifact, use it; otherwise fetch from the bridge.
    # `--skip-step-5` short-circuits the report fetch entirely.
    bundle_from_file: bytes | None = None
    if bundle_path is not None:
        try:
            bundle_from_file = bundle_path.read_bytes()
        except OSError as e:
            print(f"failed to load bundle at {bundle_path}: {e}", file=sys.stderr)
            return 2

    report_from_file: bytes | None = None
    if report_path is not None:
        if skip_step_5:
            print(
                "--report and --skip-step-5 are mutually exclusive: "
                "--report supplies the bytes Step 5 needs.",
                file=sys.stderr,
            )
            return 2
        try:
            report_from_file = report_path.read_bytes()
        except OSError as e:
            print(f"failed to load report at {report_path}: {e}", file=sys.stderr)
            return 2
        # Step 4's content check applies to file-supplied bytes too —
        # otherwise --report becomes a way to bypass report binding
        # entirely. Doing it here at read time means BOTH the
        # bridge-fetched-bundle-but-file-report path and the all-files
        # no-bridge path are covered by the same check.
        try:
            _check_report_override_hash(receipt, report_from_file)
        except VerificationError as e:
            print(f"FAIL — verification step {e.step}: {e.reason}", file=sys.stderr)
            return 1

    need_bridge = bundle_from_file is None or (
        report_from_file is None and not skip_step_5
    )

    try:
        if need_bridge:
            bridge_url = args.bridge_url
            if not _is_loopback_bridge_url(bridge_url) and not args.allow_remote_bridge:
                print(
                    f"refusing to query non-loopback bridge {bridge_url!r}. "
                    "Re-run with --allow-remote-bridge if this is intentional "
                    "and you trust the network path.",
                    file=sys.stderr,
                )
                return 2
            with BridgeStorageClient(bridge_url=bridge_url) as storage:
                result = _verify_with_overrides(
                    receipt,
                    storage,
                    bundle_override=bundle_from_file,
                    report_override=report_from_file,
                    skip_step_5=skip_step_5,
                )
        else:
            # Both artifacts (or bundle + skip-step-5) from local files —
            # no bridge needed. Run the pure pipeline directly. Report
            # bytes were already hash-checked above, bundle bytes get
            # the Step 2 hash check inside `verify_receipt`.
            assert bundle_from_file is not None  # narrowed by need_bridge
            result = verify_receipt(
                receipt,
                bundle_from_file,
                None if skip_step_5 else report_from_file,
            )
    except VerificationError as e:
        print(f"FAIL — verification step {e.step}: {e.reason}", file=sys.stderr)
        return 1

    _print_verification_result(result, skip_step_5=skip_step_5)
    return 0


def _verify_with_overrides(
    receipt: EnhancedReceipt,
    storage: StorageClient,
    *,
    bundle_override: bytes | None,
    report_override: bytes | None,
    skip_step_5: bool,
) -> VerificationResult:
    """Run verification, mixing storage fetches with caller-supplied bytes.

    The matrix:
      - bundle_override given: use those bytes; else fetch from storage.
      - report_override given: use those bytes; else fetch from storage,
        unless `skip_step_5` is set, in which case Step 5 is skipped.

    File-supplied bytes get the same Step-2/Step-4 content-hash check
    that the storage path does — a producer mistake (or a malicious
    file) that hands us a report whose hash doesn't match the receipt's
    `attestation_report_hash` must surface as `VerificationError(step=4)`,
    not as a silent pass through Step 5. Bundle override is similarly
    checked at Step 2 (`verify_step_2_evaluator_bundle` does the hash +
    parse on whatever bytes we hand it).

    When neither override is set, this collapses to
    `verify_receipt_with_storage` exactly. The override path lets a
    verifier mix sources (e.g. cached bundle on disk + storage report)
    without losing per-step error attribution.
    """
    if bundle_override is None and report_override is None:
        return verify_receipt_with_storage(
            receipt, storage, fetch_report=not skip_step_5
        )

    bundle_bytes = (
        bundle_override
        if bundle_override is not None
        else fetch_evaluator_bundle_bytes(receipt, storage)
    )
    if skip_step_5:
        report_bytes: bytes | None = None
    elif report_override is not None:
        # File-supplied report bytes are hash-checked at read time in
        # `_cmd_verify` (so the no-bridge path is covered too), so we
        # can pass them through here.
        report_bytes = report_override
    else:
        report_bytes = verify_step_4_attestation_report(receipt, storage)
    return verify_receipt(receipt, bundle_bytes, report_bytes)


def _check_report_override_hash(
    receipt: EnhancedReceipt, report_bytes: bytes
) -> None:
    """Raise VerificationError(step=4) if file-supplied report bytes
    don't hash to receipt.attestation_report_hash.

    Step 4 is "fetch report from storage and confirm content hash."
    A file-supplied report skipped the fetch but the content check
    still applies — without it `--report` would become a way to bypass
    report binding entirely (Step 5 might parse and pass the allowlist
    check on the wrong report). Centralized here so every CLI code path
    that loads a report from disk goes through the same check.
    """
    actual = "0x" + hashlib.sha256(report_bytes).hexdigest()
    if actual != receipt.attestation_report_hash:
        raise VerificationError(
            step=4,
            reason=(
                f"attestation report content hash mismatch: receipt names "
                f"{receipt.attestation_report_hash}, --report file hashes "
                f"to {actual}"
            ),
        )


def _publish_evaluator(
    bundle_bytes: bytes,
    storage: StorageClient,
) -> tuple[Bytes32Hex, Bytes32Hex, EvaluatorBundle]:
    """Validate and upload a bundle. Returns (evaluator_id, storage_root, bundle).

    The upload sends the bundle's canonical encoding (sort-keys etc),
    not the on-disk bytes verbatim. A bundle file authored by hand may
    have whitespace / key-order differences from canonical form; verifiers
    re-derive `evaluator_id` from canonical bytes during Step 2, so the
    publisher must upload canonical bytes for fetched-bundle ↔
    receipt.evaluator_id round-trips to match.

    The caller's defense-in-depth check: storage's returned content_hash
    MUST equal `bundle.evaluator_id()`. A mismatch means the storage
    backend served back different bytes than what was sent (or our
    canonical encoder disagreed with itself between calls — the test
    suite catches that elsewhere). Surfaces as a TrustViolation in
    either case. `storage_root` is what receipts will carry as
    `evaluator_storage_root` so any verifier can fetch the bundle.
    """
    bundle = EvaluatorBundle.model_validate_json(bundle_bytes)
    canonical = bundle.canonical_bytes()
    expected_id = bundle.evaluator_id()
    upload = storage.upload_blob(canonical)
    if upload.content_hash != expected_id:
        raise TrustViolation(
            f"upload returned {upload.content_hash} but bundle.evaluator_id()={expected_id} "
            "(canonical encoder drift or storage byte tampering)"
        )
    return upload.content_hash, upload.storage_root, bundle


def _write_published_side_file(
    bundle_path: Path,
    evaluator_id: Bytes32Hex,
    evaluator_storage_root: Bytes32Hex,
    bundle: EvaluatorBundle,
) -> Path:
    """Write `<bundle>.published.json` next to the bundle file.

    Downstream tooling (notably the trading-critic demo) discovers the
    bundle's `(evaluator_id, evaluator_storage_root)` tuple from this
    side-file rather than re-uploading the bundle every run. Co-located
    with the bundle so `Path("bundle.json")` → `Path("bundle.json.published.json")`
    is the simplest possible discovery rule.
    """
    side_path = bundle_path.with_name(bundle_path.name + ".published.json")
    side_path.write_text(
        json.dumps(
            {
                "evaluator_id": evaluator_id,
                "evaluator_storage_root": evaluator_storage_root,
                "evaluator_version": bundle.version,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return side_path


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
        # Dry run never uploaded, so no storage_root to report or persist.
        _print_publish_summary(
            bundle.evaluator_id(), evaluator_storage_root=None, bundle=bundle, uploaded=False
        )
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
            evaluator_id, evaluator_storage_root, bundle = _publish_evaluator(
                bundle_bytes, storage
            )
    except (StorageError, TrustViolation) as e:
        print(f"publish failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # Bundle validation, JSON parse, etc — caller-fixable input issues.
        print(f"bundle does not validate: {e}", file=sys.stderr)
        return 1

    side_path: Path | None = None
    if not args.no_side_file:
        try:
            side_path = _write_published_side_file(
                bundle_path, evaluator_id, evaluator_storage_root, bundle
            )
        except OSError as e:
            print(f"warning: failed to write side-file: {e}", file=sys.stderr)
    _print_publish_summary(
        evaluator_id,
        evaluator_storage_root=evaluator_storage_root,
        bundle=bundle,
        uploaded=True,
        side_file=side_path,
    )
    return 0


def _print_publish_summary(
    evaluator_id: Bytes32Hex,
    *,
    evaluator_storage_root: Bytes32Hex | None,
    bundle: EvaluatorBundle,
    uploaded: bool,
    side_file: Path | None = None,
) -> None:
    if uploaded:
        print("OK — evaluator bundle uploaded to 0G Storage.")
    else:
        print("OK — bundle validates (dry run; nothing uploaded).")
    print(f"  evaluator_id: {evaluator_id}")
    if evaluator_storage_root is not None:
        print(f"  evaluator_storage_root: {evaluator_storage_root}")
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
    if side_file is not None:
        print(f"  side-file: {side_file}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eerful")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser(
        "verify",
        help=(
            "verify a receipt; fetches the evaluator bundle and attestation "
            "report from 0G Storage by content hash by default"
        ),
    )
    v.add_argument("receipt", type=Path, help="path to receipt JSON")
    v.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help=(
            "override: load the evaluator bundle from this local file "
            "instead of fetching it from storage"
        ),
    )
    v.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "override: load the attestation report from this local file "
            "instead of fetching it from storage"
        ),
    )
    v.add_argument(
        "--skip-step-5",
        action="store_true",
        help=(
            "skip Step 5 (compose-hash gate); does not fetch the report. "
            "Use when the report is unavailable from storage and you only "
            "need integrity + signature verification"
        ),
    )
    v.add_argument(
        "--bridge-url",
        type=str,
        default=os.environ.get("EERFUL_0G_BRIDGE_URL", _DEFAULT_BRIDGE_URL),
        help=(
            "URL of the zg-bridge (default: $EERFUL_0G_BRIDGE_URL or "
            f"{_DEFAULT_BRIDGE_URL}). Non-loopback URLs are refused unless "
            "--allow-remote-bridge is passed."
        ),
    )
    v.add_argument(
        "--allow-remote-bridge",
        action="store_true",
        help=(
            "opt out of the loopback-only bridge guard. Required to fetch "
            "from a bridge running off-host."
        ),
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
    pub.add_argument(
        "--no-side-file",
        action="store_true",
        help=(
            "skip writing <bundle>.published.json. The side-file persists "
            "evaluator_id and evaluator_storage_root next to the bundle for "
            "downstream tooling; opt out for CI/automation flows that "
            "consume the values directly from stdout."
        ),
    )
    pub.set_defaults(func=_cmd_publish_evaluator)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
