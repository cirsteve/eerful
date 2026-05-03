"""Multi-agent trading agent — drives the AXL explorer/refiner round
trip, writes proposal.md + implementation.py, generates 2 receipts via
0G TEE, prints next steps.

Topology:
  - This script runs on otto (developer machine, has zg-bridge + eerful)
  - SSH tunnel forwards otto:9002 → gil:9002 (so /send + /recv on
    localhost go through gil's AXL node — gil is what louie sees as
    the sender peer)
  - Refiner runs on louie as a daemon (`python3 refiner.py`)
  - AXL (Yggdrasil) routes between gil and louie over the LAN

  agent_multi.py (otto) ─SSH tunnel─→ gil's AXL ──Yggdrasil──→ louie's AXL ─→ refiner.py
                                                                            ↓
                                       proposal.md + implementation.py ←──┘
                                                ↓
                                           _produce_receipt × 2
                                       (otto's local zg-bridge → 0G TEE)
                                                ↓
                                     receipts/{proposal,implementation}.json

The "two machines" the hackathon framing wants are gil + louie running
AXL peers. Otto is just the dev box driving the demo + minting
receipts. Strip otto away and the same architecture would have the
explorer logic running on gil directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Set up to import sibling explorer module + parent agent.py's _produce_receipt
HERE = Path(__file__).resolve().parent
TRADING_DIR = HERE.parent  # examples/trading/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRADING_DIR))

from _forge import forge_receipt  # noqa: E402
from agent import _produce_receipt, _PRINCIPAL_MANDATE_MAX_DRAWDOWN_PCT  # noqa: E402
from eerful._emit import emit_event  # noqa: E402
from eerful.canonical import Address, tee_signer_address_from_pubkey  # noqa: E402
from eerful.errors import ComputeError, TrustViolation  # noqa: E402
from eerful.evaluator import EvaluatorBundle  # noqa: E402
from eerful.receipt import EnhancedReceipt  # noqa: E402
from eerful.zg.bridge_init import bridge_init  # noqa: E402
from eerful.zg.compute import ComputeClient  # noqa: E402
from eerful.zg.storage import BridgeStorageClient  # noqa: E402

from explorer import explore_and_refine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [agent_multi] %(message)s")
log = logging.getLogger(__name__)


# Refiner's AXL peer ID — bake into the demo. Any topology change
# regenerates this; reflect updates here when re-deploying nodes.
DEFAULT_REFINER_PEER_ID = "a6c6caaa05c82ce497e5b1026b13920fccfad027c2cca733ededc88fd61d8974"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def main(argv: list[str] | None = None) -> int:
    repo_root = TRADING_DIR.parent.parent
    _load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(prog="agent_multi")
    parser.add_argument(
        "--tool-response",
        type=Path,
        default=TRADING_DIR / "tool_responses" / "clean.json",
        help="path to tool response JSON the explorer consumes",
    )
    parser.add_argument(
        "--refiner-peer",
        type=str,
        default=DEFAULT_REFINER_PEER_ID,
        help="ed25519 peer ID of the refiner (default: louie)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Optuna trial budget passed to refiner",
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=TRADING_DIR / "receipts",
        help="output dir for receipts",
    )
    parser.add_argument(
        "--bundles-dir",
        type=Path,
        default=TRADING_DIR / "bundles",
    )
    parser.add_argument(
        "--bridge-url",
        type=str,
        default=os.environ.get(
            "EERFUL_0G_BRIDGE_URL",
            f"http://127.0.0.1:{os.environ.get('EERFUL_0G_BRIDGE_PORT', '7878')}",
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.environ.get("EERFUL_0G_COMPUTE_PROVIDER_ADDRESS"),
    )
    parser.add_argument(
        "--skip-bridge-init",
        action="store_true",
        help="skip ledger top-up + acknowledge (when already done)",
    )
    parser.add_argument(
        "--forge",
        action="store_true",
        help=(
            "compromised-agent mode: run the full AXL flow normally, "
            "but at the receipt-minting moment SKIP the TEE call and "
            "mint forged receipts locally. The receipts pass Steps "
            "1–4, 5 (compose-hash), and 6 individually but fail Step "
            "5b — pubkey doesn't match the borrowed attestation's "
            "report_data. Demonstrates the cryptographic refusal "
            "class for the third demo arc."
        ),
    )
    parser.add_argument(
        "--forge-borrow-dir",
        type=Path,
        default=None,
        help=(
            "(only used with --forge) directory holding the prior "
            "valid receipts the forger steals attestation+evaluator "
            "pointers from. Reads proposal.json + implementation.json "
            "(one borrow source per bundle being forged — each forge "
            "needs its own bundle's evaluator pointers, otherwise the "
            "gate refuses with REFUSE_BUNDLE_MISMATCH instead of the "
            "Step 5b binding refusal we want). Defaults to receipts_dir."
        ),
    )
    args = parser.parse_args(argv)

    if not args.provider:
        print("EERFUL_0G_COMPUTE_PROVIDER_ADDRESS unset; pass --provider or set env", file=sys.stderr)
        return 2

    try:
        tool_response = json.loads(args.tool_response.read_bytes())
    except (OSError, json.JSONDecodeError) as e:
        print(f"failed to read tool response at {args.tool_response}: {e}", file=sys.stderr)
        return 2
    if not isinstance(tool_response, dict):
        print(f"tool response at {args.tool_response} must be a JSON object", file=sys.stderr)
        return 2

    # ---- forge mode: pre-load the borrow sources BEFORE rendering ----
    # The borrow sources are read first so we can fail fast (missing/
    # invalid files) without doing the AXL round trip. It's also
    # important to read before writing — the default borrow paths are
    # the same paths agent_multi will overwrite when minting.
    borrow_proposal: EnhancedReceipt | None = None
    borrow_implementation: EnhancedReceipt | None = None
    if args.forge:
        borrow_dir = args.forge_borrow_dir or args.receipts_dir
        borrow_proposal_path = borrow_dir / "proposal.json"
        borrow_impl_path = borrow_dir / "implementation.json"
        for borrow_path, label in (
            (borrow_proposal_path, "proposal"),
            (borrow_impl_path, "implementation"),
        ):
            if not borrow_path.exists():
                print(
                    f"--forge needs a {label} borrow source at {borrow_path} "
                    "(run the clean arc first to populate it).",
                    file=sys.stderr,
                )
                return 2
        try:
            borrow_proposal = EnhancedReceipt.model_validate_json(
                borrow_proposal_path.read_bytes()
            )
            borrow_implementation = EnhancedReceipt.model_validate_json(
                borrow_impl_path.read_bytes()
            )
        except OSError as e:
            print(f"failed to read borrow receipts in {borrow_dir}: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"borrow receipts do not validate: {e}", file=sys.stderr)
            return 2

    log.info("== AXL multi-agent trading demo ==")
    log.info("tool response: %s", args.tool_response.name)
    log.info("refiner peer:  %s...", args.refiner_peer[:16])

    emit_event(
        source="agent_multi",
        kind="run_started",
        tool_response_path=str(args.tool_response),
        tool_response_name=args.tool_response.name,
        refiner_peer_prefix=args.refiner_peer[:16],
    )

    # ---- 1. AXL round trip: explorer → refiner → explorer ----
    artifacts = explore_and_refine(
        refiner_peer_id=args.refiner_peer,
        tool_response=tool_response,
        n_trials=args.n_trials,
    )

    # Parse the baked-in drawdown once. A missing/malformed line is a
    # producer bug (the explorer template is fixed text), so fail clean
    # with a clear error rather than crashing on StopIteration.
    dd_line = next(
        (
            line
            for line in artifacts.implementation_py.splitlines()
            if line.startswith("MAX_DRAWDOWN_PCT =")
        ),
        None,
    )
    if dd_line is None:
        log.error("implementation_py is missing MAX_DRAWDOWN_PCT line")
        return 2
    try:
        applied_max_dd = float(dd_line.split("=", 1)[1].strip())
    except (IndexError, ValueError) as e:
        log.error("could not parse MAX_DRAWDOWN_PCT from %r: %s", dd_line, e)
        return 2
    drift_marker = (
        " (DRIFT — poisoned tool response)"
        if applied_max_dd != _PRINCIPAL_MANDATE_MAX_DRAWDOWN_PCT
        else ""
    )
    log.info("agent's working max_drawdown: %.0f%%%s", applied_max_dd, drift_marker)
    log.info("refiner sharpe: %.3f params: %s", artifacts.sharpe, artifacts.best_params)

    emit_event(
        source="agent_multi",
        kind="working_mandate",
        max_drawdown_pct=applied_max_dd,
        principal_mandate_pct=_PRINCIPAL_MANDATE_MAX_DRAWDOWN_PCT,
        drift=applied_max_dd != _PRINCIPAL_MANDATE_MAX_DRAWDOWN_PCT,
        sharpe=artifacts.sharpe,
    )

    # ---- 2. write artifacts to disk for human inspection ----
    args.receipts_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = args.receipts_dir / "proposal.md"
    impl_path = args.receipts_dir / "implementation.py"
    proposal_path.write_text(artifacts.proposal_md)
    impl_path.write_text(artifacts.implementation_py)
    log.info("artifacts: %s, %s", proposal_path, impl_path)

    # ---- 3. feed both into the existing _produce_receipt flow ----
    proposal_bundle = EvaluatorBundle.model_validate_json(
        (args.bundles_dir / "proposal_grade.json").read_bytes()
    )
    implementation_bundle = EvaluatorBundle.model_validate_json(
        (args.bundles_dir / "implementation_grade.json").read_bytes()
    )

    impl_artifact_text = (
        f"PROPOSAL:\n{artifacts.proposal_md}\n\nCODE:\n{artifacts.implementation_py}\n"
    )

    if args.forge:
        # ---- 3b. compromised-agent path: skip TEE, mint forged receipts ----
        # The agent has gone through honest motions up to this point —
        # explorer drafted, refiner swept, AXL traffic happened. At the
        # receipt-minting moment, instead of calling the TEE, the
        # compromised agent generates its own keypair locally and signs
        # a hand-crafted high-score receipt. Each receipt steals
        # attestation+evaluator pointers from a prior valid receipt for
        # the same bundle — Steps 2/4/5 pass, Step 5b fails.
        assert borrow_proposal is not None  # populated above when --forge
        assert borrow_implementation is not None
        log.info(
            "FORGE mode — skipping TEE. minting forged receipts using attestation %s...",
            borrow_proposal.attestation_report_hash[:18],
        )
        proposal_receipt = forge_receipt(
            borrow=borrow_proposal,
            bundle=proposal_bundle,
            receipts_dir=args.receipts_dir,
            out_name="proposal.json",
        )
        emit_event(
            source="agent_multi",
            kind="receipt_minted",
            bundle="proposal_grade",
            receipt_id=proposal_receipt.receipt.receipt_id,
            score_block=proposal_receipt.receipt.output_score_block,
            path=str(proposal_receipt.path),
            forged=True,
        )
        implementation_receipt = forge_receipt(
            borrow=borrow_implementation,
            bundle=implementation_bundle,
            receipts_dir=args.receipts_dir,
            out_name="implementation.json",
        )
        emit_event(
            source="agent_multi",
            kind="receipt_minted",
            bundle="implementation_grade",
            receipt_id=implementation_receipt.receipt.receipt_id,
            score_block=implementation_receipt.receipt.output_score_block,
            path=str(implementation_receipt.path),
            forged=True,
        )
        log.info(
            "forged signer address: %s",
            tee_signer_address_from_pubkey(proposal_receipt.receipt.enclave_pubkey),
        )
        log.info(
            "borrowed attested signer: %s",
            "(from report's report_data — won't match)",
        )
        log.info("proposal receipt:       %s (FORGED)", proposal_receipt.path)
        log.info("implementation receipt: %s (FORGED)", implementation_receipt.path)
        log.info("")
        log.info("Next: gate each forged receipt against the principal's policy.")
        log.info("Expect REFUSE_INVALID_RECEIPT at §7.1 Step 5b.")
        return 0

    log.info("calling 0G TEE (proposal_grade + implementation_grade)")
    with ComputeClient(bridge_url=args.bridge_url) as compute, BridgeStorageClient(
        bridge_url=args.bridge_url
    ) as storage:
        if not args.skip_bridge_init:
            try:
                bridge_init(compute, args.provider)
            except ComputeError as e:
                log.error("bridge_init failed: %s", e)
                return 2

        try:
            proposal_receipt = _produce_receipt(
                compute=compute,
                storage=storage,
                bundle=proposal_bundle,
                provider_address=Address(args.provider),
                artifact_text=artifacts.proposal_md,
                receipts_dir=args.receipts_dir,
                out_name="proposal.json",
            )
            emit_event(
                source="agent_multi",
                kind="receipt_minted",
                bundle="proposal_grade",
                receipt_id=proposal_receipt.receipt.receipt_id,
                score_block=proposal_receipt.receipt.output_score_block,
                path=str(proposal_receipt.path),
            )
            implementation_receipt = _produce_receipt(
                compute=compute,
                storage=storage,
                bundle=implementation_bundle,
                provider_address=Address(args.provider),
                artifact_text=impl_artifact_text,
                receipts_dir=args.receipts_dir,
                out_name="implementation.json",
            )
            emit_event(
                source="agent_multi",
                kind="receipt_minted",
                bundle="implementation_grade",
                receipt_id=implementation_receipt.receipt.receipt_id,
                score_block=implementation_receipt.receipt.output_score_block,
                path=str(implementation_receipt.path),
            )
        except (ComputeError, TrustViolation, ValueError) as e:
            log.error("receipt production failed: %s", e)
            return 2

    log.info("proposal receipt:       %s", proposal_receipt.path)
    log.info("implementation receipt: %s", implementation_receipt.path)
    log.info("")
    log.info("Next: gate each receipt against the principal's policy.")
    log.info(
        "  uv run eerful gate --policy %s --tier low_consequence --bundle proposal_grade --receipt %s",
        TRADING_DIR / "principal_policy.json",
        proposal_receipt.path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
