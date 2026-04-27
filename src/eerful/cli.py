"""eerful CLI.

The `verify` subcommand runs the spec §7.1 verification algorithm against
an on-disk receipt and prints a human-readable verdict. Step 4 (fetching
the attestation report from 0G Storage) is not yet wired here — the
caller passes `--report` with the report bytes on disk for now. When 0G
Storage integration lands, Step 4 will run from `attestation_report_hash`
without an explicit flag.

Other subcommands (`publish-evaluator`, `evaluate`) land alongside the
remaining Step 4 wiring and the jig adapter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eerful.errors import VerificationError
from eerful.receipt import EnhancedReceipt
from eerful.verify import VerificationResult, verify_receipt


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

    bundle_bytes = bundle_path.read_bytes()
    report_bytes = report_path.read_bytes() if report_path is not None else None

    try:
        result = verify_receipt(receipt, bundle_bytes, report_bytes)
    except VerificationError as e:
        print(f"FAIL — verification step {e.step}: {e.reason}", file=sys.stderr)
        return 1

    _print_verification_result(result)
    return 0


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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
