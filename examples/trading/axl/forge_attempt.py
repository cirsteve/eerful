"""Compromised-agent forgery — mints an EnhancedReceipt without going
through the TEE.

Demonstrates the cryptographic refusal at §7.1 Step 5b. The first two
demo arcs (clean + poisoned) both assume an honest TEE call: the receipt
was really minted inside an enclave; the only attack is corrupted *input*.
This arc shows what happens when the agent itself is compromised — it
skips the TEE entirely, generates a fresh secp256k1 keypair locally,
and signs a hand-crafted receipt claiming any score it wants.

What the rails do:
- Step 4 (attestation fetch): passes — we point at a real attestation
  borrowed from a prior valid receipt.
- Step 5 (compose-hash): passes when the borrowed attestation's
  compose-hash satisfies the bundle's allowlist (it does, since the
  borrowed receipt itself passed Step 5).
- Step 6 (signature recovery): passes — the forger signed their own
  response with their own key, the math is internally consistent.
- Step 5b (pubkey ↔ report_data binding): FAILS. The forger's locally
  generated key derives to an EVM address that doesn't match the one
  the real enclave baked into the attestation's report_data field.
  `eerful gate` refuses with REFUSE_INVALID_RECEIPT.

That refusal is the cryptographic teeth of "enclave-born key" — without
Step 5b, this forgery slips through Steps 1-6 individually and the
"the receipt was signed by an enclave" claim is unverifiable in
single-receipt verification.

Usage:
    python forge_attempt.py \\
        --borrow-receipt examples/trading/receipts/proposal.json \\
        --score 0.95 \\
        --out examples/trading/receipts/forged.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from eth_keys import keys
from eth_utils import keccak

from eerful.canonical import tee_signer_address_from_pubkey
from eerful.receipt import EnhancedReceipt


def _eip191_personal_sign(text: str, privkey_bytes: bytes) -> tuple[str, str]:
    """EIP-191 personal_sign over `text`, mirroring the 0G TeeML
    enclave's signing. Returns (pubkey_hex, signature_hex)."""
    msg = text.encode("utf-8")
    msg_hash = keccak(b"\x19Ethereum Signed Message:\n" + str(len(msg)).encode() + msg)
    pk = keys.PrivateKey(privkey_bytes)
    sig = pk.sign_msg_hash(msg_hash)
    return "0x" + pk.public_key.to_bytes().hex(), "0x" + sig.to_bytes().hex()


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
        borrow_raw = args.borrow_receipt.read_bytes()
    except OSError as e:
        print(f"failed to read borrow receipt at {args.borrow_receipt}: {e}", file=sys.stderr)
        return 2
    try:
        borrow = EnhancedReceipt.model_validate_json(borrow_raw)
    except Exception as e:
        print(f"borrow receipt does not validate: {e}", file=sys.stderr)
        return 2

    # -------- the forgery --------

    # 1. Generate a fresh secp256k1 keypair LOCALLY. This key has no
    #    relationship to any enclave; nothing has attested it.
    privkey_bytes = os.urandom(32)
    fake_signer = tee_signer_address_from_pubkey(
        "0x" + keys.PrivateKey(privkey_bytes).public_key.to_bytes().hex()
    )

    # 2. Hand-craft a high-scoring score block (the agent's lie).
    forged_score: dict = {
        "mandate_compliance": float(args.score),
        "coherence": float(args.score),
        "specificity": float(args.score),
        "overall": float(args.score),
        "commentary": (
            "Strategy fully complies with all mandate clauses. Coherent, "
            "specific, well-aligned with the principal's stated goals."
        ),
    }

    # 3. Pick a response_content. Defaults to a JSON dump of the forged
    #    score block — that's the format real graders return, so Step 3's
    #    output schema check passes too.
    response_content = args.response_content or json.dumps(forged_score, sort_keys=True)

    # 4. Sign with the fake key. Step 6 will pass: the signature recovers
    #    to the (forged) pubkey we'll put in the receipt — internal
    #    consistency.
    fake_pubkey, fake_sig = _eip191_personal_sign(response_content, privkey_bytes)

    # 5. Build the forged receipt. Borrow the attestation + evaluator
    #    pointers from the legitimate receipt so Steps 2 + 4 + 5
    #    (compose-hash) pass too. Step 5b is the only one that catches it.
    forged = EnhancedReceipt.build(
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
        evaluator_id=borrow.evaluator_id,
        evaluator_storage_root=borrow.evaluator_storage_root,
        evaluator_version=borrow.evaluator_version,
        provider_address=borrow.provider_address,
        chat_id="forged-" + os.urandom(4).hex(),
        response_content=response_content,
        attestation_report_hash=borrow.attestation_report_hash,
        attestation_storage_root=borrow.attestation_storage_root,
        enclave_pubkey=fake_pubkey,
        enclave_signature=fake_sig,
        output_score_block=forged_score,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(forged.model_dump_json())

    # Operator narration. Goes to stderr so stdout stays clean for any
    # downstream `jq` consumer that wants to read the forged receipt
    # back from --out.
    print("forged receipt minted:", args.out, file=sys.stderr)
    print(f"  borrowed attestation:   {borrow.attestation_report_hash[:18]}...", file=sys.stderr)
    print(f"  forged signer address:  {fake_signer}", file=sys.stderr)
    print(f"  output_score_block:     overall={forged_score['overall']}", file=sys.stderr)
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
