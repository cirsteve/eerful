"""verify_chain — D.1 chain + commitment assertions.

Distinct from `eerful verify`, which checks one receipt in isolation.
This script adds the demo-specific properties:

- All three receipts pass §7.1 verification end-to-end (Steps 1-6),
  with Step 5's compose-hash gate enforced AND the §8.2 category
  reaching A.
- Chain links: `previous_receipt_id` forms `None → v1 → v2 → v3`.
- Input-commitment stability: all three share the same non-None
  commitment (the §6.7 chain-pattern invariant the demo is designed
  around).
- Bundle stability: all three name the same evaluator_id.

Bridge prerequisite: the local `services/zg-bridge/` must be
running. The script honors the same loopback-only guard as `eerful
verify`; use `--allow-remote-bridge` to opt out.

Usage:
    uv run python examples/trading_critic/verify_chain.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from eerful.cli import _is_loopback_bridge_url
from eerful.errors import VerificationError
from eerful.receipt import EnhancedReceipt
from eerful.verify import verify_receipt_with_storage
from eerful.zg.storage import BridgeStorageClient

_HERE = Path(__file__).resolve().parent
_RECEIPTS_DIR = _HERE / "receipts"

_VERSIONS: tuple[str, str, str] = ("v1", "v2", "v3")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _load_chain(receipts_dir: Path) -> list[EnhancedReceipt]:
    receipts: list[EnhancedReceipt] = []
    for v in _VERSIONS:
        path = receipts_dir / f"{v}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path} — run `python examples/trading_critic/demo.py` first"
            )
        receipts.append(EnhancedReceipt.model_validate_json(path.read_bytes()))
    return receipts


def _assert_chain_links(receipts: list[EnhancedReceipt]) -> None:
    """v1's predecessor is None; v2 → v1; v3 → v2."""
    if receipts[0].previous_receipt_id is not None:
        raise AssertionError(
            f"v1.previous_receipt_id is {receipts[0].previous_receipt_id!r}, "
            "expected None"
        )
    if receipts[1].previous_receipt_id != receipts[0].receipt_id:
        raise AssertionError(
            f"v2.previous_receipt_id is {receipts[1].previous_receipt_id!r}, "
            f"expected {receipts[0].receipt_id!r}"
        )
    if receipts[2].previous_receipt_id != receipts[1].receipt_id:
        raise AssertionError(
            f"v3.previous_receipt_id is {receipts[2].previous_receipt_id!r}, "
            f"expected {receipts[1].receipt_id!r}"
        )


def _assert_commitment_stable(receipts: list[EnhancedReceipt]) -> None:
    """All three receipts share one non-None input_commitment.

    This is the §6.7 chain-pattern invariant the demo is built around:
    the strategy *identity* is stable across iterations even though the
    strategy *text* changes.
    """
    commitments = {r.input_commitment for r in receipts}
    if len(commitments) != 1:
        raise AssertionError(
            f"input_commitments differ across receipts: "
            f"{[r.input_commitment for r in receipts]}"
        )
    if None in commitments:
        raise AssertionError(
            "all receipts have input_commitment=None — the demo's chain "
            "pattern requires a stable commitment (see spec §6.7)"
        )


def _assert_bundle_stable(receipts: list[EnhancedReceipt]) -> None:
    ids = {r.evaluator_id for r in receipts}
    if len(ids) != 1:
        raise AssertionError(f"receipts name different evaluators: {ids}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify_chain")
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=_RECEIPTS_DIR,
        help=f"directory holding v1.json/v2.json/v3.json (default: {_RECEIPTS_DIR})",
    )
    parser.add_argument(
        "--bridge-url",
        type=str,
        default=None,
        help=(
            "bridge URL (default: $EERFUL_0G_BRIDGE_URL or "
            "http://127.0.0.1:$EERFUL_0G_BRIDGE_PORT or :7878)"
        ),
    )
    parser.add_argument(
        "--allow-remote-bridge",
        action="store_true",
        help="opt out of the loopback-only bridge guard (mirrors the CLI flag)",
    )
    args = parser.parse_args(argv)

    repo_root = _HERE.parent.parent
    _load_dotenv(repo_root / ".env")

    bridge_url = args.bridge_url or os.environ.get(
        "EERFUL_0G_BRIDGE_URL",
        f"http://127.0.0.1:{os.environ.get('EERFUL_0G_BRIDGE_PORT', '7878')}",
    )

    if not _is_loopback_bridge_url(bridge_url) and not args.allow_remote_bridge:
        print(
            f"refusing to query non-loopback bridge {bridge_url!r}. "
            "Re-run with --allow-remote-bridge if this is intentional "
            "and you trust the network path.",
            file=sys.stderr,
        )
        return 2

    try:
        receipts = _load_chain(args.receipts_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    print(f"== loaded {len(receipts)} receipts from {args.receipts_dir}")

    # Step 1-6 verification per receipt. Any failure surfaces with the
    # spec-step number that broke; we exit 1 with attribution.
    try:
        with BridgeStorageClient(bridge_url=bridge_url) as storage:
            for v, receipt in zip(_VERSIONS, receipts):
                result = verify_receipt_with_storage(receipt, storage)
                if result.step5 is None:
                    raise AssertionError(
                        f"{v}: Step 5 did not run — verify_receipt_with_storage "
                        "should always run it under the storage path"
                    )
                if result.step5.gating != "enforced":
                    raise AssertionError(
                        f"{v}: compose-hash gating is {result.step5.gating!r}, "
                        "expected 'enforced' — bundle.json must declare "
                        "accepted_compose_hashes for the chain demo"
                    )
                if result.step5.category != "A":
                    raise AssertionError(
                        f"{v}: §8.2 category is {result.step5.category!r}, "
                        "expected 'A' — provider's compose no longer names the "
                        "model in its launch string"
                    )
                print(f"  {v}: ✓ Steps 1-6 pass; gating=enforced; category=A")
    except VerificationError as e:
        print(f"FAIL — verification step {e.step}: {e.reason}", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"FAIL — chain assertion: {e}", file=sys.stderr)
        return 1

    try:
        _assert_chain_links(receipts)
        _assert_commitment_stable(receipts)
        _assert_bundle_stable(receipts)
    except AssertionError as e:
        print(f"FAIL — chain assertion: {e}", file=sys.stderr)
        return 1

    bundle_version = receipts[0].evaluator_version
    chain_arrow = " -> ".join(r.receipt_id for r in receipts)
    score_progression = [r.output_score_block for r in receipts]

    print()
    print(f"All {len(receipts)} receipts verified.")
    print(f"  evaluator: {bundle_version} (evaluator_id={receipts[0].evaluator_id})")
    print("  compose-hash gating: enforced (§8.2 Category A)")
    print(f"  chain: {chain_arrow}")
    print(f"  input_commitment (stable): {receipts[0].input_commitment}")
    print("  score progression:")
    for v, scores in zip(_VERSIONS, score_progression):
        if scores is None:
            print(f"    {v}: (no score block)")
            continue
        dims = {k: scores[k] for k in ("risk", "novelty", "robustness", "overall") if k in scores}
        print(f"    {v}: {dims}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
