"""Day 2 smoke — testnet qwen-2.5-7b end-to-end.

Prereqs:
  1. `cd services/zg-bridge && npm install && npm run start` (in another terminal).
  2. `eerful/.env` populated with EERFUL_0G_PRIVATE_KEY, EERFUL_0G_RPC,
     EERFUL_0G_BRIDGE_PORT, EERFUL_0G_COMPUTE_PROVIDER_ADDRESS.
  3. Wallet has ≥ ~1.05 0G on testnet (broker MIN_LOCKED_BALANCE = 1 0G
     plus a small fee buffer). Faucet: https://hub.0g.ai/faucet.

Run:
  uv run python examples/smoke_testnet.py

What it does:
  1. Pings /healthz to confirm the bridge has loaded the wallet.
  2. Calls /admin/acknowledge (idempotent — first run pays gas, later
     runs no-op).
  3. Sends one chat completion via /compute/inference + /compute/signature
     + /compute/attestation, all stitched by ComputeClient.infer_full.
  4. Builds an EnhancedReceipt over a one-off EvaluatorBundle.
  5. Runs verify_through_step_3 (Steps 4-7 land Day 3+).
  6. Prints the receipt JSON + verification verdict.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.verify import verify_through_step_3
from eerful.zg.compute import ComputeClient


def _load_dotenv(path: Path) -> None:
    """Tiny KEY=VALUE .env loader — no python-dotenv dep."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    _load_dotenv(repo_root / ".env")

    bridge_port = os.environ.get("EERFUL_0G_BRIDGE_PORT", "7878")
    bridge_url = f"http://127.0.0.1:{bridge_port}"
    provider_address = os.environ.get(
        "EERFUL_0G_COMPUTE_PROVIDER_ADDRESS",
        "0xa48f01287233509FD694a22Bf840225062E67836",  # testnet qwen-2.5-7b
    )

    print(f"== bridge {bridge_url}, provider {provider_address}")

    with ComputeClient(bridge_url=bridge_url) as client:
        # ---- Step 0: health
        h = client.healthz()
        print(f"  bridge wallet={h['wallet']} chain={h['chain_id']}")

        # ---- Step 1: acknowledge (idempotent)
        ack = client.acknowledge(provider_address)
        print(
            f"  acknowledge: tee_signer={ack['tee_signer_address']} "
            f"already={ack['already_acknowledged']}"
        )

        # ---- Step 2: bundle
        bundle = EvaluatorBundle(
            version="day-2-smoke@0.1.0",
            model_identifier="qwen/qwen-2.5-7b-instruct",
            system_prompt="You are a smoke-test helper. Reply briefly.",
        )
        evaluator_id = bundle.evaluator_id()
        print(f"  evaluator_id={evaluator_id}")

        # ---- Step 3: full TeeML call
        print("  calling infer_full ...")
        result = client.infer_full(
            provider_address=provider_address,
            messages=[
                {"role": "system", "content": bundle.system_prompt},
                {"role": "user", "content": "Say 'pong' and nothing else."},
            ],
            temperature=0.0,
            max_tokens=16,
        )
        print(f"  chat_id={result.chat_id}")
        print(f"  signing_address={result.signing_address}")
        print(f"  attestation_report_hash={result.attestation_report_hash}")
        print(f"  response_content (signed) = {result.response_content!r}")

    # ---- Step 4: build receipt
    receipt = EnhancedReceipt.build(
        created_at=datetime.now(timezone.utc),
        evaluator_id=evaluator_id,
        evaluator_version=bundle.version,
        provider_address=provider_address,
        chat_id=result.chat_id,
        response_content=result.response_content,
        attestation_report_hash=result.attestation_report_hash,
        enclave_pubkey=result.enclave_pubkey,
        enclave_signature=result.enclave_signature,
    )
    print(f"  receipt_id={receipt.receipt_id}")

    # ---- Step 5: verify Steps 1-3
    bundle_bytes = bundle.canonical_bytes()
    verify_through_step_3(receipt, bundle_bytes)
    print("  verify_through_step_3 PASSED")

    # ---- Step 6: print the receipt JSON for posterity
    print("\n--- receipt JSON ---")
    print(json.dumps(json.loads(receipt.model_dump_json()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
