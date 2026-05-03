"""Standalone compromised-agent forgery — mints an EnhancedReceipt
without going through the TEE.

Same forgery logic as `agent_multi.py --forge` but as a one-shot
utility: takes a borrow source, produces a single forged receipt,
done. Useful for unit-test-style demos and for forging against a
specific bundle without spinning up the full AXL flow.

For the recording's third arc, prefer `agent_multi.py --forge` —
it's structurally parallel to the first two arcs (same explorer +
refiner + AXL traffic) and produces both proposal + implementation
forged receipts in one shot.

Usage:
    python forge_attempt.py \\
        --borrow-receipt examples/trading/receipts/proposal.json \\
        --bundle examples/trading/bundles/proposal_grade.json \\
        --score 0.95 \\
        --out examples/trading/receipts/forged.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Local sibling import — _forge.py lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _forge import fake_signer_for, forge_receipt  # noqa: E402

import os  # noqa: E402

from eerful.evaluator import EvaluatorBundle  # noqa: E402
from eerful.receipt import EnhancedReceipt  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forge_attempt",
        description="Mint a forged EnhancedReceipt to demonstrate §7.1 Step 5b refusal.",
    )
    parser.add_argument(
        "--borrow-receipt",
        type=Path,
        required=True,
        help=(
            "path to a prior valid receipt — the forger steals its "
            "attestation_report_hash, evaluator_id, and storage roots "
            "to make the forgery plausible"
        ),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("examples/trading/bundles/proposal_grade.json"),
        help=(
            "path to the EvaluatorBundle this receipt claims to be for. "
            "Default: proposal_grade.json. The forger doesn't actually "
            "evaluate against the bundle — it just needs the bundle's "
            "evaluator_id to embed in the receipt."
        ),
    )
    parser.add_argument(
        "--score",
        type=float,
        default=0.95,
        help="overall score to claim (the forger's lie). Default 0.95.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="path to write the forged receipt JSON",
    )
    parser.add_argument(
        "--response-content",
        type=str,
        default=None,
        help=(
            "hand-crafted response_content to sign. Default: a JSON "
            "object containing the forged score block — close enough "
            "in shape to a real grader response that Step 3 (output "
            "schema) passes."
        ),
    )
    args = parser.parse_args(argv)

    try:
        borrow = EnhancedReceipt.model_validate_json(args.borrow_receipt.read_bytes())
    except OSError as e:
        print(f"failed to read borrow receipt at {args.borrow_receipt}: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"borrow receipt does not validate: {e}", file=sys.stderr)
        return 2

    try:
        bundle = EvaluatorBundle.model_validate_json(args.bundle.read_bytes())
    except (OSError, Exception) as e:
        print(f"failed to load bundle at {args.bundle}: {e}", file=sys.stderr)
        return 2

    privkey_bytes = os.urandom(32)
    fake_signer = fake_signer_for(privkey_bytes)

    try:
        produced = forge_receipt(
            borrow=borrow,
            bundle=bundle,
            receipts_dir=args.out.parent,
            out_name=args.out.name,
            score=args.score,
            response_content=args.response_content,
            privkey_bytes=privkey_bytes,
        )
    except OSError as e:
        print(f"failed to write forged receipt at {args.out}: {e}", file=sys.stderr)
        return 2

    print("forged receipt minted:", produced.path, file=sys.stderr)
    print(f"  borrowed attestation:   {borrow.attestation_report_hash[:18]}...", file=sys.stderr)
    print(f"  forged signer address:  {fake_signer}", file=sys.stderr)
    print(f"  output_score_block:     overall={args.score}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "the receipt is internally consistent — Step 6's signature recovery",
        file=sys.stderr,
    )
    print(
        "passes against the forged pubkey. it will fail verification at",
        file=sys.stderr,
    )
    print(
        "§7.1 Step 5b: the forged pubkey-derived address does not match",
        file=sys.stderr,
    )
    print(
        "the EVM address baked into the attestation report's report_data.",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
