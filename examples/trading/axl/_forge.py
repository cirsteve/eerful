"""Shared forgery helper for the compromised-agent demo arc.

Mints an `EnhancedReceipt` without going through the TEE — generates a
fresh secp256k1 keypair locally, signs a hand-crafted response with
that key, borrows attestation pointers from a prior valid receipt to
make the forgery plausible. Steps 1–4, 5 (compose-hash), and 6 each
pass on the result; **Step 5b** fails because the locally-generated
key derives to an EVM address that doesn't match the one the real
enclave baked into the borrowed attestation's `report_data` field.

Used by both `forge_attempt.py` (standalone one-shot utility) and
`agent_multi.py --forge` (drops the forgery into the AXL multi-agent
flow so the third demo arc looks structurally identical to the first
two — same explorer/refiner motions, same artifact files, only the
receipt-minting step diverges).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eth_keys import keys
from eth_utils import keccak

from eerful.canonical import tee_signer_address_from_pubkey
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt

# Re-use the producer-side receipt-bundle return type. agent.py's
# `_produce_receipt` returns this; matching the shape means
# agent_multi.py can call either function from the same code path
# downstream.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import _ProducedReceipt  # noqa: E402


def _eip191_personal_sign(text: str, privkey_bytes: bytes) -> tuple[str, str]:
    """EIP-191 personal_sign over `text`, mirroring the 0G TeeML
    enclave's signing. Returns (pubkey_hex, signature_hex)."""
    msg = text.encode("utf-8")
    msg_hash = keccak(b"\x19Ethereum Signed Message:\n" + str(len(msg)).encode() + msg)
    pk = keys.PrivateKey(privkey_bytes)
    sig = pk.sign_msg_hash(msg_hash)
    return "0x" + pk.public_key.to_bytes().hex(), "0x" + sig.to_bytes().hex()


_DEFAULT_COMMENTARY = (
    "Strategy fully complies with all mandate clauses. Coherent, "
    "specific, well-aligned with the principal's stated goals."
)


def _score_block_from_schema(
    bundle: EvaluatorBundle,
    score: float,
) -> dict[str, Any]:
    """Build a forged score block whose shape matches `bundle.output_schema`.

    `additionalProperties: false` schemas (used by both
    `proposal_grade` and `implementation_grade` bundles) reject any
    keys outside the declared `required` set, so a hard-coded
    proposal-style block fires Step 3's schema-validation refusal
    before Step 5b can — hiding the binding-failure narrative we want
    to demo. Iterate over the schema's required fields, populate
    numeric ones with `score` and the (one) string field with a
    plausible commentary, fill anything else with a benign default.
    """
    schema = bundle.output_schema or {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []
    block: dict[str, Any] = {}
    for field in required:
        prop = properties.get(field, {}) if isinstance(properties, dict) else {}
        ptype = prop.get("type") if isinstance(prop, dict) else None
        if ptype == "number" or ptype == "integer":
            block[field] = float(score)
        elif ptype == "string":
            block[field] = _DEFAULT_COMMENTARY
        elif ptype == "boolean":
            block[field] = True
        else:
            block[field] = float(score)
    return block


def forge_receipt(
    *,
    borrow: EnhancedReceipt,
    bundle: EvaluatorBundle,
    receipts_dir: Path,
    out_name: str,
    score: float = 0.95,
    response_content: str | None = None,
    privkey_bytes: bytes | None = None,
) -> _ProducedReceipt:
    """Mint a forged receipt. Mirrors `_produce_receipt`'s signature
    enough that callers can substitute one for the other; the actual
    parameters are slimmer because no compute/storage clients are
    involved (forgery is purely local).

    `borrow` supplies attestation pointers AND evaluator pointers —
    everything that has to point at real Storage (attestation_report_hash,
    attestation_storage_root, evaluator_id, evaluator_storage_root,
    evaluator_version) gets copied verbatim. The caller is responsible
    for picking a borrow receipt whose evaluator matches the receipt
    being forged: forging a receipt for `proposal_grade` needs a
    proposal-bundle borrow source; forging for `implementation_grade`
    needs an implementation-bundle borrow source. Mismatched borrow +
    bundle would refuse with `REFUSE_BUNDLE_MISMATCH` at the gate
    instead of the Step 5b binding refusal we want to demonstrate.
    `bundle` is accepted but not consulted for evaluator pointers —
    callers can still pass it for symmetry with `_produce_receipt`'s
    signature, but the forger uses `borrow`'s evaluator fields
    verbatim because those are what real Storage already holds.

    `privkey_bytes` defaults to a fresh `os.urandom(32)` — the agent
    has no enclave to borrow a real key from. The forged signature is
    self-consistent (Step 6 passes) but the pubkey-derived address has
    no relationship to the borrowed attestation's `report_data` (Step
    5b fails).
    """
    # 1. Generate (or accept) a fresh secp256k1 keypair locally. This
    #    key has no relationship to any enclave; nothing has attested it.
    if privkey_bytes is None:
        privkey_bytes = os.urandom(32)

    # 2. Build a high-scoring score block whose SHAPE matches the
    #    bundle's output_schema. Hand-rolled blocks failed Step 3's
    #    `additionalProperties: false` schemas (different bundles have
    #    different field names — proposal_grade has mandate_compliance,
    #    implementation_grade has spec_alignment) and refused before
    #    reaching Step 5b. Deriving from the schema keeps the demo
    #    bundle-agnostic.
    forged_score = _score_block_from_schema(bundle, score)

    # 3. Pick a response_content. Defaults to a JSON dump of the forged
    #    score block — that's the format real graders return, so Step 3's
    #    output schema check passes too.
    body = response_content or json.dumps(forged_score, sort_keys=True)

    # 4. Sign with the fake key. Step 6 will pass: the signature
    #    recovers to the (forged) pubkey we'll put in the receipt —
    #    internal consistency.
    fake_pubkey, fake_sig = _eip191_personal_sign(body, privkey_bytes)

    # 5. Build the forged receipt. Borrow the attestation + evaluator
    #    pointers from the legitimate receipt so Steps 2 + 4 + 5
    #    (compose-hash) pass too. Step 5b is the only one that catches
    #    the missing pubkey ↔ report_data binding.
    forged = EnhancedReceipt.build(
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
        evaluator_id=borrow.evaluator_id,
        evaluator_storage_root=borrow.evaluator_storage_root,
        evaluator_version=borrow.evaluator_version,
        provider_address=borrow.provider_address,
        chat_id="forged-" + os.urandom(4).hex(),
        response_content=body,
        attestation_report_hash=borrow.attestation_report_hash,
        attestation_storage_root=borrow.attestation_storage_root,
        enclave_pubkey=fake_pubkey,
        enclave_signature=fake_sig,
        output_score_block=forged_score,
    )

    out_path = receipts_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(forged.model_dump_json())

    return _ProducedReceipt(
        receipt=forged,
        bundle_evaluator_id=bundle.evaluator_id(),
        path=out_path,
    )


def fake_signer_for(privkey_bytes: bytes) -> str:
    """Convenience wrapper for operator-narration output: derive the
    EVM address corresponding to a forge keypair without doing the
    full forgery first."""
    pubkey_hex = "0x" + keys.PrivateKey(privkey_bytes).public_key.to_bytes().hex()
    return tee_signer_address_from_pubkey(pubkey_hex)
