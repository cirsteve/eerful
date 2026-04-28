"""Idempotent cold-boot dance for the local zg-bridge.

Three call sites do the same three-step sequence before any TeeML
inference call: `healthz` (cold-boot guard), `add_ledger` (top up the
broker sub-account), `acknowledge` (one-time per provider). Extracting
the dance here keeps `examples/smoke_testnet.py`,
`examples/trading_critic/demo.py`, and
`examples/trading_critic/bundle_inspect.py --score-test` consistent —
in particular the "bridge not reachable" error message points at the
same recovery action everywhere.

Returns a typed status dataclass instead of printing, so callers retain
control over output (CLI scripts can format human-friendly lines;
tests can assert against fields). The helper itself is library-shaped:
one function, no I/O beyond what `ComputeClient` already does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from eerful.canonical import Address
from eerful.errors import ComputeError
from eerful.zg.compute import ComputeClient

__all__ = ["BridgeInitStatus", "bridge_init"]


@dataclass(frozen=True)
class BridgeInitStatus:
    """Result of the cold-boot dance — every field a caller might want
    to surface to its user. Frozen so accidental mutation can't silently
    drift one caller's view from another's."""

    wallet: Address
    chain_id: int
    ledger_created: bool
    """True when this `add_ledger` call created a fresh sub-account.
    False on subsequent runs — the call topped up the existing one."""
    total_balance_0g: float
    tee_signer_address: Address
    already_acknowledged: bool
    """True when the provider was already acknowledged before this
    call. False on the first run for a given (wallet, provider) pair."""


def bridge_init(
    compute: ComputeClient,
    provider_address: Address,
    *,
    bridge_url: str,
    ledger_amount: float | None = None,
) -> BridgeInitStatus:
    """Run the bridge cold-boot dance: healthz → add_ledger → acknowledge.

    All three steps are idempotent (healthz is read-only; add_ledger
    tops up the existing sub-account; acknowledge no-ops if already
    done), so this is safe to call on every script run.

    `ledger_amount` defaults to `$EERFUL_0G_LEDGER_DEPOSIT` (or 1.1
    if unset). The 1.1 default sits just above the broker's
    `MIN_LOCKED_BALANCE = 1 0G`; smaller values fail at the first
    inference call with "balance below minimum lock".

    Cold-boot failure (healthz raises `ComputeError`) re-raises with
    a friendlier message that points at the bridge README. Callers
    typically catch `ComputeError` and exit non-zero with the message
    on stderr.
    """
    try:
        h = compute.healthz()
    except ComputeError as e:
        # Re-raise so callers see one consistent recovery hint instead
        # of each one duplicating the same "see services/zg-bridge/
        # README.md and run `npm run dev`" string.
        raise ComputeError(
            f"bridge not reachable at {bridge_url}: {e}; "
            f"see services/zg-bridge/README.md and run `npm run dev`"
        ) from e

    # Resolve the deposit amount with explicit validation at the helper
    # boundary. A malformed `EERFUL_0G_LEDGER_DEPOSIT` would otherwise
    # raise a raw ValueError mid-init; a zero or negative value would
    # be accepted here and surface much later as the broker's
    # opaque "balance below minimum lock" message at first inference.
    raw_amount: float | str = (
        ledger_amount
        if ledger_amount is not None
        else os.environ.get("EERFUL_0G_LEDGER_DEPOSIT", "1.1")
    )
    try:
        amount = float(raw_amount)
    except (TypeError, ValueError) as e:
        raise ComputeError(
            f"invalid ledger amount {raw_amount!r}: expected a positive "
            "number of 0G (set EERFUL_0G_LEDGER_DEPOSIT or pass "
            "ledger_amount=)"
        ) from e
    if amount <= 0:
        raise ComputeError(
            f"invalid ledger amount {amount}: must be > 0 0G "
            "(broker's MIN_LOCKED_BALANCE is 1.0; default is 1.1)"
        )
    ledger = compute.add_ledger(amount)
    ack = compute.acknowledge(provider_address)

    return BridgeInitStatus(
        wallet=h["wallet"],
        chain_id=int(h["chain_id"]),
        ledger_created=bool(ledger["created"]),
        total_balance_0g=float(ledger["total_balance_0g"]),
        tee_signer_address=ack["tee_signer_address"],
        already_acknowledged=bool(ack["already_acknowledged"]),
    )
