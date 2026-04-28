"""Trading-critic demo orchestrator.

End-to-end: produces a chain of three EERs, one per hand-authored
trading-strategy iteration (`strategies/v1.md` → `v2.md` → `v3.md`),
all linked by `previous_receipt_id` and sharing a stable
`input_commitment` keyed off the strategy identity (spec §6.7).

Mirrors `examples/smoke_testnet.py`'s bridge-init pattern but uses
`EnhancedReceipt.build` directly rather than `EvaluationClient`. Why:
`EvaluationClient._compute_commitment_or_none` derives `input_bytes`
from the joined user messages, which would produce three different
commitments (one per strategy text) and defeat the chain-stability
story the demo is built around. Adding an `eerful.input_bytes`
override would be a Track C extension; we sidestep it by owning the
receipt-construction loop.

Bridge prerequisite: the local `services/zg-bridge/` must be running.
The first action `compute.healthz()` will surface a clean error
message if it isn't.

Usage:
    uv run python examples/trading_critic/demo.py [--verbose]

`--verbose` dumps the full critic commentary alongside the score
block. Default is the score block only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from eerful.canonical import Address, Bytes32Hex, canonical_json_bytes
from eerful.commitment import SaltStore, compute_input_commitment, generate_salt
from eerful.errors import ComputeError
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt
from eerful.zg.bridge_init import bridge_init
from eerful.zg.compute import ComputeClient, ComputeResult
from eerful.zg.storage import BridgeStorageClient, StorageClient


class _ComputeProtocol(Protocol):
    """Structural type for the `infer_full` shape `run_demo` consumes.

    The production `ComputeClient` satisfies it, and `FakeComputeClient`
    in `tests/jig/conftest.py` satisfies it without subclassing —
    matches the same pattern `EvaluationClient` uses to keep the test
    surface ergonomic."""

    def infer_full(
        self,
        *,
        provider_address: Address,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ComputeResult: ...

_HERE = Path(__file__).resolve().parent
_BUNDLE_PATH = _HERE / "bundle.json"
_STRATEGIES_DIR = _HERE / "strategies"
_RECEIPTS_DIR = _HERE / "receipts"
_SALT_STORE_PATH = _HERE / ".salt"

_VERSIONS: tuple[str, str, str] = ("v1", "v2", "v3")

_STRATEGY_IDENTITY = "trading-strategy-v0"
"""Stable identity used to derive the input commitment shared across
v1/v2/v3 receipts. NOT the strategy *text* — the iterations differ in
text but share an identity in the spec §6.7 sense."""

_SALT_ANCHOR_KEY: Bytes32Hex = "0x" + hashlib.sha256(b"trading-critic-anchor").hexdigest()
"""Synthetic receipt_id-shaped key under which the chain's stable salt
is persisted in the SaltStore. Not a real receipt_id; just a stable
Bytes32Hex slot the SaltStore API accepts."""

_DEFAULT_PROVIDER: Address = "0xd9966e13a6026Fcca4b13E7ff95c94DE268C471C"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _strategy_identity_bytes() -> bytes:
    """Canonical-JSON encoding of the strategy identity (spec §6.7).

    A verifier reconstructing the commitment from the salt + identity
    must produce the same bytes; canonical JSON is what makes that
    cross-implementation stable. See `bundle.json` →
    `metadata.input_commitment_construction`."""
    return canonical_json_bytes({"strategy_identity": _STRATEGY_IDENTITY})


def _resolve_or_generate_salt(salt_store: SaltStore) -> bytes:
    """Read the chain's salt from the store, or generate + persist one."""
    try:
        salt, _ = salt_store.get(_SALT_ANCHOR_KEY)
        return salt
    except KeyError:
        salt = generate_salt()
        salt_store.put(_SALT_ANCHOR_KEY, salt, input_path=str(_HERE))
        return salt


def _existing_chain_consistent(
    receipts_dir: Path, *, evaluator_id: Bytes32Hex
) -> bool:
    """Idempotency check: do all three receipts already exist on disk,
    link forward consistently, AND match the current evaluator_id? If
    yes, demo can exit cleanly without re-billing the broker.

    A partial state (one or two receipts present) returns False — the
    chain has to be re-run from v1 because each receipt's id depends on
    its predecessor's id, and broker calls are not idempotent at the
    wallet layer.

    The `evaluator_id` check matters when the maintainer reruns
    `bundle_inspect.py` and repins `accepted_compose_hashes`: the new
    bundle has a new evaluator_id, so the on-disk receipts (produced
    against the old bundle) no longer verify under the current bundle.
    Without this check, demo would short-circuit to exit 0 and leave
    an unverifiable chain in `receipts/`.
    """
    paths = [receipts_dir / f"{v}.json" for v in _VERSIONS]
    if not all(p.exists() for p in paths):
        return False
    try:
        receipts = [
            EnhancedReceipt.model_validate_json(p.read_bytes()) for p in paths
        ]
    except Exception:
        return False
    if any(r.evaluator_id != evaluator_id for r in receipts):
        return False
    if receipts[0].previous_receipt_id is not None:
        return False
    if receipts[1].previous_receipt_id != receipts[0].receipt_id:
        return False
    if receipts[2].previous_receipt_id != receipts[1].receipt_id:
        return False
    # All three must share the same input_commitment for the chain
    # story to hold; otherwise the producer ran with a stale .salt
    # between iterations.
    commitments = {r.input_commitment for r in receipts}
    if len(commitments) != 1 or None in commitments:
        return False
    return True


_PLACEHOLDER_COMPOSE_HASH: Bytes32Hex = "0x" + ("0" * 64)
"""Sentinel value `bundle.json` ships with — the maintainer's
`bundle_inspect.py --confirm-compose-hash` flow swaps in the live
hash before publishing. Demo refuses to run while the placeholder is
in place to avoid producing receipts that fail Step 5 against any
real provider."""


def _parse_score_block(response_content: str, version: str) -> dict[str, object]:
    """Strict JSON parse — the bundle's system_prompt mandates JSON-only
    output. A failure here is a content/critic-misbehavior signal that
    the maintainer needs to see, not paper over.

    Distinct from `EvaluationClient._parse_score_block`'s permissive
    parse: the demo owns the receipt construction loop and we want
    early-loud failure during prompt iteration. Step 3 verification on
    the persisted receipt would catch a missing required field, but
    this earlier check surfaces the *raw critic response* in the error
    message — much more useful for debugging the system_prompt.
    """
    try:
        parsed = json.loads(response_content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"strategy {version}: critic response is not valid JSON.\n"
            f"  parse error: {e}\n"
            f"  raw response: {response_content!r}"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"strategy {version}: critic response parsed as {type(parsed).__name__}, "
            f"not a dict.\n  raw response: {response_content!r}"
        )
    return parsed


def run_demo(
    *,
    compute: _ComputeProtocol,
    storage: StorageClient,
    bundle: EvaluatorBundle,
    provider_address: Address,
    strategies_dir: Path,
    receipts_dir: Path,
    salt_store: SaltStore,
    verbose: bool = False,
) -> list[EnhancedReceipt]:
    """Inner loop: produce three receipts under the given evaluator
    against three strategy markdown files.

    Callable from tests with `FakeComputeClient` + `MockStorageClient`;
    `main()` wires up the production `ComputeClient` and
    `BridgeStorageClient` and calls this. Returns the produced
    receipts in order so callers can do post-hoc assertions.

    The bundle is uploaded to storage at the start of the run
    (idempotent — `getOrUpload`-style backends short-circuit on cache
    hit). The returned `storage_root` is what populates each receipt's
    `evaluator_storage_root` (spec §6.1, Tier 2).
    """
    receipts_dir.mkdir(parents=True, exist_ok=True)

    evaluator_id = bundle.evaluator_id()
    bundle_upload = storage.upload_blob(bundle.canonical_bytes())
    if bundle_upload.content_hash != evaluator_id:
        raise RuntimeError(
            f"bundle upload returned content_hash {bundle_upload.content_hash}, "
            f"expected {evaluator_id} — canonical encoder drift or storage tampering"
        )
    evaluator_storage_root = bundle_upload.storage_root

    salt = _resolve_or_generate_salt(salt_store)
    input_commitment = compute_input_commitment(
        _strategy_identity_bytes(), evaluator_id, salt
    )

    bundle_params = bundle.inference_params or {}
    temperature = bundle_params.get("temperature")
    max_tokens = bundle_params.get("max_tokens")

    receipts: list[EnhancedReceipt] = []
    previous: Bytes32Hex | None = None
    for version in _VERSIONS:
        strategy_text = (strategies_dir / f"{version}.md").read_text()
        messages = [
            {"role": "system", "content": bundle.system_prompt},
            {"role": "user", "content": strategy_text},
        ]
        result: ComputeResult = compute.infer_full(
            provider_address=provider_address,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        report_upload = storage.upload_blob(result.attestation_report_bytes)
        if report_upload.content_hash != result.attestation_report_hash:
            raise RuntimeError(
                f"strategy {version}: attestation report upload returned "
                f"content_hash {report_upload.content_hash}, expected "
                f"{result.attestation_report_hash} (compute reported one hash, "
                "storage returned another — byzantine evidence)"
            )

        score_block = _parse_score_block(result.response_content, version)

        receipt = EnhancedReceipt.build(
            created_at=datetime.now(timezone.utc),
            evaluator_id=evaluator_id,
            evaluator_storage_root=evaluator_storage_root,
            evaluator_version=bundle.version,
            provider_address=provider_address,
            chat_id=result.chat_id,
            response_content=result.response_content,
            attestation_report_hash=result.attestation_report_hash,
            attestation_storage_root=report_upload.storage_root,
            enclave_pubkey=result.enclave_pubkey,
            enclave_signature=result.enclave_signature,
            input_commitment=input_commitment,
            previous_receipt_id=previous,
            output_score_block=score_block,
        )

        (receipts_dir / f"{version}.json").write_text(
            receipt.model_dump_json(indent=2) + "\n"
        )
        receipts.append(receipt)
        previous = receipt.receipt_id

        # Print per-version summary. With --verbose, include the full
        # commentary; otherwise just the four numeric dimensions for
        # quick monotonic-improvement scanning.
        score_dims = {
            k: score_block[k]
            for k in ("risk", "novelty", "robustness", "overall")
            if k in score_block
        }
        print(f"  {version}: receipt_id={receipt.receipt_id}")
        print(f"      scores={score_dims}")
        if verbose and "commentary" in score_block:
            print(f"      commentary: {score_block['commentary']}")

    return receipts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="dump full critic commentary alongside score blocks",
    )
    args = parser.parse_args(argv)

    repo_root = _HERE.parent.parent
    _load_dotenv(repo_root / ".env")

    bridge_port = os.environ.get("EERFUL_0G_BRIDGE_PORT", "7878")
    bridge_url = f"http://127.0.0.1:{bridge_port}"
    provider_address = os.environ.get(
        "EERFUL_0G_COMPUTE_PROVIDER_ADDRESS", _DEFAULT_PROVIDER
    )

    print(f"== bridge {bridge_url}, provider {provider_address}")

    bundle = EvaluatorBundle.model_validate_json(_BUNDLE_PATH.read_bytes())
    print(f"  evaluator: {bundle.version} ({bundle.model_identifier})")
    print(f"  evaluator_id: {bundle.evaluator_id()}")

    if _existing_chain_consistent(_RECEIPTS_DIR, evaluator_id=bundle.evaluator_id()):
        print(
            "  receipts/{v1,v2,v3}.json already exist, link consistently, "
            "and match the current evaluator_id — exiting cleanly. "
            "Delete the receipts/ directory to rerun."
        )
        return 0

    # Preflight local inputs BEFORE any broker call burns gas. A
    # missing v2.md / v3.md is a common setup miss (they're gitignored
    # and require `author_strategies.py` to draft); a still-placeholder
    # `accepted_compose_hashes` would produce receipts that fail Step
    # 5 against Provider 1. Both cost nothing to detect and would
    # otherwise cost a v1 inference + ledger top-up before failing.
    if bundle.accepted_compose_hashes == [_PLACEHOLDER_COMPOSE_HASH]:
        print(
            "  ✗ bundle.json still pins the all-zero placeholder compose-hash.\n"
            "      Run `python examples/trading_critic/bundle_inspect.py "
            "--confirm-compose-hash`,\n"
            "      paste the printed value into bundle.json's "
            "accepted_compose_hashes,\n"
            "      then `eerful publish-evaluator --bundle "
            "examples/trading_critic/bundle.json`.",
            file=sys.stderr,
        )
        return 2

    missing = [
        str(_STRATEGIES_DIR / f"{v}.md")
        for v in _VERSIONS
        if not (_STRATEGIES_DIR / f"{v}.md").exists()
    ]
    if missing:
        print(
            "  ✗ missing strategy files (v2/v3 are gitignored — draft them "
            "with author_strategies.py):\n      "
            + "\n      ".join(missing),
            file=sys.stderr,
        )
        return 2

    salt_store = SaltStore(_SALT_STORE_PATH)

    with ComputeClient(bridge_url=bridge_url) as compute:
        # R-D1.2 cold-boot guard: bridge_init re-raises ComputeError
        # with a message pointing at the bridge README, so a
        # connection-refused / wallet-not-loaded failure surfaces with
        # the recovery action rather than a raw traceback. Done
        # immediately so we don't burn the broker acknowledge call
        # before realizing the bridge isn't up.
        try:
            status = bridge_init(compute, provider_address)
        except ComputeError as e:
            print(f"  ✗ {e}", file=sys.stderr)
            return 2
        print(f"  bridge wallet={status.wallet} chain={status.chain_id}")
        print(
            f"  ledger: created={status.ledger_created} "
            f"balance={status.total_balance_0g:.4f} 0G"
        )
        print(
            f"  acknowledge: tee_signer={status.tee_signer_address} "
            f"already={status.already_acknowledged}"
        )

        with BridgeStorageClient(bridge_url=bridge_url) as storage:
            receipts = run_demo(
                compute=compute,
                storage=storage,
                bundle=bundle,
                provider_address=provider_address,
                strategies_dir=_STRATEGIES_DIR,
                receipts_dir=_RECEIPTS_DIR,
                salt_store=salt_store,
                verbose=args.verbose,
            )

    print()
    print(f"  chain: {' -> '.join(r.receipt_id for r in receipts)}")
    print(f"  input_commitment (stable): {receipts[0].input_commitment}")
    print(f"  receipts written to {_RECEIPTS_DIR}/")
    print("  next: python examples/trading_critic/verify_chain.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
