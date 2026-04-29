"""Executor — the hard gate that fronts consequential actions.

Spec design: `specs/executor-and-rails-design.md` §4.

`evaluate_gate` checks a set of receipts against a `PrincipalPolicy`'s
named tier. PASS only if every check below succeeds, in spec order; on
any failure, returns the first refuse outcome with a human-readable
detail (the CLI prints it verbatim).

  1. Receipt count ≥ tier.n_attestations
  2. Every receipt's evaluator_id matches policy.bundles[bundle_name]
  3. Every receipt verifies via §7.1 Steps 1–6 (storage-fetched bundle
     and attestation report)
  4. Every receipt's declared compose-category is in tier.required_categories
     (when set; the bundle's allowlist enforcement at Step 5 already
     gates the compose-hash side)
  5. Diversity rules satisfied (distinct_signers / distinct_compose_hashes)
  6. Score aggregation passes (all_must_pass: every overall ≥ threshold)

The function is domain-agnostic: `tier` and `bundle_name` are plain
strings looked up against the policy. Trading, code review, RFC
compliance, etc. all flow through the same six checks.

This module never prints. Logging is the CLI's responsibility.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from eth_keys import keys

from eerful.canonical import Address, Bytes32Hex, BytesHex, to_lower_hex
from eerful.errors import PolicyError, VerificationError
from eerful.policy import PrincipalPolicy
from eerful.receipt import EnhancedReceipt
from eerful.verify import VerificationResult, verify_receipt_with_storage
from eerful.zg.storage import StorageClient

__all__ = [
    "GateOutcome",
    "GateResult",
    "canonical_set_hash",
    "evaluate_gate",
    "tee_signer_address_from_pubkey",
]


class GateOutcome(str, Enum):
    """One of seven terminal states for a gate evaluation. PASS plus six
    refuse outcomes corresponding to the six checks. CLI surfaces the
    enum value verbatim so a downstream watcher can pattern-match on
    string literals (`refuse_score`, etc.)."""

    PASS = "pass"
    REFUSE_INSUFFICIENT_RECEIPTS = "refuse_insufficient_receipts"
    REFUSE_BUNDLE_MISMATCH = "refuse_bundle_mismatch"
    REFUSE_INVALID_RECEIPT = "refuse_invalid_receipt"
    REFUSE_DIVERSITY = "refuse_diversity"
    REFUSE_CATEGORY = "refuse_category"
    REFUSE_SCORE = "refuse_score"


@dataclass(frozen=True)
class GateResult:
    """Outcome of a single `evaluate_gate` call.

    `canonical_set_hash` is set on PASS only — it's the content-addressed
    anchor for the (sorted) receipt set, suitable for use as
    `previous_receipt_id` in any downstream chain composition. None on
    refuse, where there's nothing legitimate to anchor.
    """

    outcome: GateOutcome
    tier: str
    bundle_name: str
    receipts_supplied: int
    receipts_required: int
    detail: str
    canonical_set_hash: Bytes32Hex | None


def tee_signer_address_from_pubkey(pubkey_hex: BytesHex) -> Address:
    """Derive the EVM address of an enclave's signing key from its pubkey.

    The 0G TeeML attestation report's `report_data` field carries the
    EVM address of the enclave-born signing key (spec §6.7); the
    pubkey-to-address derivation is keccak256 of the 64-byte X||Y
    public key, last 20 bytes. `enclave_pubkey` in the receipt is
    already in X||Y form (no SEC1 0x04 prefix), so this matches.

    Used by the `distinct_signers` diversity rule. Two enclaves with
    different signing keys produce different addresses; two on-chain
    identities sharing one enclave (the Provider 15+16 fixture in
    `research/day1_attestation_findings.md`) produce the same address —
    diversity caught.
    """
    canonical = to_lower_hex(pubkey_hex)
    pubkey_bytes = bytes.fromhex(canonical.removeprefix("0x"))
    if len(pubkey_bytes) != 64:
        raise ValueError(
            f"enclave_pubkey must be 64 bytes (X||Y, no SEC1 prefix), got {len(pubkey_bytes)}"
        )
    pub = keys.PublicKey(pubkey_bytes)
    return to_lower_hex(pub.to_canonical_address())


def canonical_set_hash(receipts: Sequence[EnhancedReceipt]) -> Bytes32Hex:
    """Content-addressed hash of an N-receipt set.

    Dedupe by `receipt_id`, sort, newline-join, sha256. True set
    semantics — two callers passing the same receipts in different
    orders, or one caller passing a duplicate by accident, all agree on
    the same hash. Used as `previous_receipt_id` when a downstream
    action chains off a multi-attestation gate.

    Spec note: `specs/executor-and-rails-design.md` §5.4 calls this
    `canonical_set_hash` and uses a `sha256:` prefix for human
    legibility; we use the codebase's `0x...` Bytes32Hex convention so
    the value can flow into `previous_receipt_id` (which is typed as
    `Bytes32Hex`) without a format adapter.
    """
    leaves = sorted({r.receipt_id for r in receipts})
    digest = hashlib.sha256("\n".join(leaves).encode("utf-8")).hexdigest()
    return "0x" + digest


def _refuse(
    *,
    outcome: GateOutcome,
    tier: str,
    bundle_name: str,
    receipts_supplied: int,
    receipts_required: int,
    detail: str,
) -> GateResult:
    """Construct a refuse `GateResult` with `canonical_set_hash=None`.
    Centralized to keep the seven outcome paths uniform."""
    return GateResult(
        outcome=outcome,
        tier=tier,
        bundle_name=bundle_name,
        receipts_supplied=receipts_supplied,
        receipts_required=receipts_required,
        detail=detail,
        canonical_set_hash=None,
    )


def evaluate_gate(
    *,
    policy: PrincipalPolicy,
    tier: str,
    bundle_name: str,
    receipts: Sequence[EnhancedReceipt],
    storage: StorageClient,
) -> GateResult:
    """Evaluate the six-check gate sequence; return PASS or first refuse.

    `tier` and `bundle_name` must be keys in `policy.tiers` and
    `policy.bundles` respectively — a missing key is a programming bug
    (caller wiring), not a gate refusal, and raises `PolicyError`. All
    other failure modes return a `GateResult` with the appropriate
    refuse outcome.
    """
    if tier not in policy.tiers:
        raise PolicyError(
            f"tier {tier!r} not in policy.tiers (have: {sorted(policy.tiers)})"
        )
    if bundle_name not in policy.bundles:
        raise PolicyError(
            f"bundle_name {bundle_name!r} not in policy.bundles (have: {sorted(policy.bundles)})"
        )

    tier_policy = policy.tiers[tier]
    expected_evaluator_id = policy.bundles[bundle_name]
    n_required = tier_policy.n_attestations
    n_supplied = len(receipts)
    distinct_ids = {r.receipt_id for r in receipts}
    n_distinct = len(distinct_ids)

    # Check 1: distinct receipt count. Counts unique `receipt_id`s, not raw
    # input length — without this, a caller can satisfy an N-attestation
    # tier by passing the same receipt N times when diversity rules are
    # off. The diversity rules (signers / compose-hashes) are about
    # provider-level distinctness; receipt-level distinctness is the more
    # fundamental floor.
    if n_distinct < n_required:
        if n_supplied != n_distinct:
            detail = (
                f"received {n_supplied} receipt(s) but only {n_distinct} distinct "
                f"by receipt_id; tier {tier!r} requires {n_required}"
            )
        else:
            detail = (
                f"received {n_supplied} receipt(s), tier {tier!r} requires {n_required}"
            )
        return _refuse(
            outcome=GateOutcome.REFUSE_INSUFFICIENT_RECEIPTS,
            tier=tier,
            bundle_name=bundle_name,
            receipts_supplied=n_supplied,
            receipts_required=n_required,
            detail=detail,
        )

    # Check 2: every receipt names the policy-pinned evaluator_id.
    # Catches a swapped-bundle attempt where receipts came from a
    # different evaluator than the principal committed to.
    for r in receipts:
        if r.evaluator_id != expected_evaluator_id:
            return _refuse(
                outcome=GateOutcome.REFUSE_BUNDLE_MISMATCH,
                tier=tier,
                bundle_name=bundle_name,
                receipts_supplied=n_supplied,
                receipts_required=n_required,
                detail=(
                    f"receipt {r.receipt_id} declares evaluator_id "
                    f"{r.evaluator_id}, policy.bundles[{bundle_name!r}] "
                    f"requires {expected_evaluator_id}"
                ),
            )

    # Check 3: every receipt verifies via §7.1 Steps 1–6.
    # Done once per receipt; results retained because Steps 4–5 read the
    # storage-fetched bundle and the parsed compose-hash, and we don't
    # want to fetch+parse a second time for the category and diversity
    # checks below.
    verification_results: list[VerificationResult] = []
    for r in receipts:
        try:
            verification_results.append(verify_receipt_with_storage(r, storage))
        except VerificationError as e:
            return _refuse(
                outcome=GateOutcome.REFUSE_INVALID_RECEIPT,
                tier=tier,
                bundle_name=bundle_name,
                receipts_supplied=n_supplied,
                receipts_required=n_required,
                detail=(
                    f"receipt {r.receipt_id} failed §7.1 Step {e.step}: {e.reason}"
                ),
            )
        except (ValueError, TypeError) as e:
            # Storage-layer hex/length validators (e.g.
            # `BridgeStorageClient.download_blob`) raise `ValueError` on
            # a malformed `evaluator_storage_root` /
            # `attestation_storage_root` — Pydantic's `Bytes32Hex`
            # BeforeValidator only lowercases, so length-bad inputs reach
            # storage and surface here. Translate to a refuse outcome so
            # the gate fails closed with attribution rather than crashing.
            return _refuse(
                outcome=GateOutcome.REFUSE_INVALID_RECEIPT,
                tier=tier,
                bundle_name=bundle_name,
                receipts_supplied=n_supplied,
                receipts_required=n_required,
                detail=(
                    f"receipt {r.receipt_id} is malformed or has invalid "
                    f"verification inputs: {e}"
                ),
            )

    # Check 4: declared category is in tier.required_categories (when set).
    # Step 5 already enforced compose-hash ∈ bundle.accepted_compose_hashes;
    # the category half is the publisher's declaration on the matched entry.
    # If the bundle had no allowlist (gating="skipped"), there's no declared
    # category to check — that's a tier-vs-bundle mismatch, treated as
    # REFUSE_CATEGORY (the policy demanded a category enforcement the
    # bundle can't provide).
    if tier_policy.required_categories is not None:
        for r, result in zip(receipts, verification_results, strict=True):
            entry = result.step5.declared_entry if result.step5 is not None else None
            if entry is None:
                return _refuse(
                    outcome=GateOutcome.REFUSE_CATEGORY,
                    tier=tier,
                    bundle_name=bundle_name,
                    receipts_supplied=n_supplied,
                    receipts_required=n_required,
                    detail=(
                        f"receipt {r.receipt_id} has no declared compose-category "
                        f"(bundle declared no accepted_compose_hashes); "
                        f"tier {tier!r} requires categories {tier_policy.required_categories}"
                    ),
                )
            if entry.category not in tier_policy.required_categories:
                return _refuse(
                    outcome=GateOutcome.REFUSE_CATEGORY,
                    tier=tier,
                    bundle_name=bundle_name,
                    receipts_supplied=n_supplied,
                    receipts_required=n_required,
                    detail=(
                        f"receipt {r.receipt_id} declared category {entry.category!r}, "
                        f"tier {tier!r} requires {tier_policy.required_categories}"
                    ),
                )

    # Check 5: diversity rules. The rule that gives N>1 attestations
    # actual security — without it, an attacker compromising one TEE has
    # the same access as N legitimate signers.
    if tier_policy.diversity.distinct_signers:
        signers = [tee_signer_address_from_pubkey(r.enclave_pubkey) for r in receipts]
        if len(set(signers)) != len(signers):
            return _refuse(
                outcome=GateOutcome.REFUSE_DIVERSITY,
                tier=tier,
                bundle_name=bundle_name,
                receipts_supplied=n_supplied,
                receipts_required=n_required,
                detail=(
                    f"distinct_signers required but {n_supplied - len(set(signers))} "
                    f"duplicate signer(s) found across the receipt set"
                ),
            )

    if tier_policy.diversity.distinct_compose_hashes:
        compose_hashes: list[str] = []
        for r, result in zip(receipts, verification_results, strict=True):
            if result.step5 is None:
                # Defensive: `verify_receipt_with_storage` defaults to
                # fetch_report=True so step5 is always populated when
                # `evaluate_gate` calls it (gating="skipped" still
                # surfaces a Step5Result with a populated compose_hash).
                # This branch only fires if a future caller wires the
                # executor against a verifier path that suppresses
                # Step 5 entirely — kept as a fail-closed guard rather
                # than an `assert`.
                return _refuse(
                    outcome=GateOutcome.REFUSE_DIVERSITY,
                    tier=tier,
                    bundle_name=bundle_name,
                    receipts_supplied=n_supplied,
                    receipts_required=n_required,
                    detail=(
                        f"distinct_compose_hashes required but receipt {r.receipt_id} "
                        f"has no Step 5 result (report fetch was suppressed)"
                    ),
                )
            compose_hashes.append(result.step5.compose_hash)
        if len(set(compose_hashes)) != len(compose_hashes):
            return _refuse(
                outcome=GateOutcome.REFUSE_DIVERSITY,
                tier=tier,
                bundle_name=bundle_name,
                receipts_supplied=n_supplied,
                receipts_required=n_required,
                detail=(
                    f"distinct_compose_hashes required but "
                    f"{n_supplied - len(set(compose_hashes))} duplicate compose-hash(es) "
                    f"found across the receipt set"
                ),
            )

    # Check 6: score aggregation. v0.5 ships `all_must_pass` only —
    # every receipt's `output_score_block.overall` must meet the
    # threshold. PrincipalPolicy's Literal type already constrained
    # `score_aggregation` to `"all_must_pass"`, but we still branch on
    # it explicitly so a v0.6 expansion (median, threshold_of_passers)
    # surfaces here as a clear unimplemented case.
    threshold = tier_policy.score_threshold
    if policy.score_aggregation == "all_must_pass":
        for r in receipts:
            score_block = r.output_score_block
            if score_block is None:
                return _refuse(
                    outcome=GateOutcome.REFUSE_SCORE,
                    tier=tier,
                    bundle_name=bundle_name,
                    receipts_supplied=n_supplied,
                    receipts_required=n_required,
                    detail=(
                        f"receipt {r.receipt_id} has no output_score_block; "
                        f"all_must_pass requires a numeric 'overall' on each receipt"
                    ),
                )
            overall = score_block.get("overall")
            if (
                isinstance(overall, bool)
                or not isinstance(overall, (int, float))
                or not math.isfinite(overall)
            ):
                # `bool` is `int` in Python; reject explicitly to catch
                # JSON-decoded `true/false` masquerading as a score.
                # `math.isfinite` rejects NaN and ±Infinity: NaN
                # comparisons always return False (so `NaN < threshold`
                # is False and would silently PASS), and `+Infinity <
                # threshold` is also False — both bypass the threshold
                # check below without an explicit finite-check guard.
                return _refuse(
                    outcome=GateOutcome.REFUSE_SCORE,
                    tier=tier,
                    bundle_name=bundle_name,
                    receipts_supplied=n_supplied,
                    receipts_required=n_required,
                    detail=(
                        f"receipt {r.receipt_id} output_score_block.overall is "
                        f"{overall!r}, not a finite numeric value"
                    ),
                )
            if overall < threshold:
                return _refuse(
                    outcome=GateOutcome.REFUSE_SCORE,
                    tier=tier,
                    bundle_name=bundle_name,
                    receipts_supplied=n_supplied,
                    receipts_required=n_required,
                    detail=(
                        f"receipt {r.receipt_id} overall {overall} < "
                        f"tier {tier!r} threshold {threshold}"
                    ),
                )

    # All six checks passed — anchor the receipt set.
    # `detail` is outcome-neutral; the CLI prefixes with the human label
    # (`PASS — ...`) based on `outcome`. Avoids the double-prefix the
    # CLI's `_print_gate_result` would otherwise produce.
    return GateResult(
        outcome=GateOutcome.PASS,
        tier=tier,
        bundle_name=bundle_name,
        receipts_supplied=n_supplied,
        receipts_required=n_required,
        detail=(
            f"{n_supplied} receipt(s) under bundle {bundle_name!r}, "
            f"tier {tier!r}"
        ),
        canonical_set_hash=canonical_set_hash(receipts),
    )
