"""bridge_init() — hermetic tests for the cold-boot dance helper.

`httpx.MockTransport` mirrors the pattern in `test_zg_compute.py`. Live
exercise lives in the example scripts (`smoke_testnet.py`, `demo.py`,
`bundle_inspect.py --score-test`).

What this pins:

- healthz `ComputeError` is rewrapped with the standard "see bridge
  README" recovery hint and the bridge URL is read off the client
  itself (not a separate kwarg, so a caller can't make the message
  lie).
- happy-path returns a `BridgeInitStatus` with every field populated
  from the corresponding bridge response.
- `EERFUL_0G_LEDGER_DEPOSIT` and the `ledger_amount=` kwarg are
  validated at the helper boundary: malformed strings, non-positive
  values, and non-finite values (NaN / Infinity) all raise
  `ComputeError` rather than slipping through to `add_ledger`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest

from eerful.errors import ComputeError
from eerful.zg.bridge_init import BridgeInitStatus, bridge_init
from eerful.zg.compute import ComputeClient


@contextmanager
def _make_client(handler: httpx.MockTransport) -> Iterator[ComputeClient]:
    http = httpx.Client(transport=handler)
    try:
        yield ComputeClient(bridge_url="http://bridge.test", http=http)
    finally:
        http.close()


def _ok_handler(
    *,
    healthz: dict[str, Any] | None = None,
    ledger: dict[str, Any] | None = None,
    ack: dict[str, Any] | None = None,
) -> httpx.MockTransport:
    """Build a MockTransport that returns canonical-shape responses on
    each of the three endpoints `bridge_init` calls. Tests pass overrides
    for any field they want to assert on."""

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/healthz":
            return httpx.Response(
                200,
                json=healthz
                or {
                    "status": "ok",
                    "wallet": "0xabc",
                    "chain_id": 16602,
                    "rpc": "rpc.test",
                    "broker_initialized": True,
                },
            )
        if path == "/admin/add-ledger":
            return httpx.Response(
                200,
                json=ledger
                or {
                    "created": False,
                    "total_balance_neuron": "1100000000000000000",
                    "available_balance_neuron": "1000000000000000000",
                    "total_balance_0g": 1.1,
                },
            )
        if path == "/admin/acknowledge":
            return httpx.Response(
                200,
                json=ack
                or {
                    "provider_address": "0x" + "b" * 40,
                    "tee_signer_address": "0x" + "c" * 40,
                    "already_acknowledged": False,
                },
            )
        return httpx.Response(404, json={"error": "no_handler", "path": path})

    return httpx.MockTransport(handle)


# ---------------- happy path ----------------


def test_bridge_init_returns_populated_status():
    """Every BridgeInitStatus field comes off the corresponding response."""
    with _make_client(_ok_handler()) as compute:
        status = bridge_init(compute, "0x" + "b" * 40)
    assert isinstance(status, BridgeInitStatus)
    assert status.wallet == "0xabc"
    assert status.chain_id == 16602
    assert status.ledger_created is False
    assert status.total_balance_0g == pytest.approx(1.1)
    assert status.tee_signer_address == "0x" + "c" * 40
    assert status.already_acknowledged is False


def test_bridge_init_calls_endpoints_in_order():
    """healthz must precede add_ledger and acknowledge — otherwise a bad
    bridge could be hit with the broker calls before the cold-boot
    guard catches it."""
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"wallet": "0xabc", "chain_id": 1})
        if request.url.path == "/admin/add-ledger":
            return httpx.Response(
                200,
                json={"created": True, "total_balance_0g": 1.1, "total_balance_neuron": "0", "available_balance_neuron": "0"},
            )
        if request.url.path == "/admin/acknowledge":
            return httpx.Response(
                200,
                json={"provider_address": "x", "tee_signer_address": "y", "already_acknowledged": True},
            )
        return httpx.Response(404)

    with _make_client(httpx.MockTransport(handle)) as compute:
        bridge_init(compute, "0x" + "b" * 40)

    assert seen == ["/healthz", "/admin/add-ledger", "/admin/acknowledge"]


# ---------------- healthz failure rewrap ----------------


def test_bridge_init_rewraps_healthz_failure_with_readme_pointer():
    """A 500 from healthz must surface as ComputeError pointing at the
    bridge README, with the bridge URL pulled off the client (not a
    separate kwarg that could lie)."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(500, json={"status": "error", "error": "wallet not loaded"})
        return httpx.Response(404)

    with _make_client(httpx.MockTransport(handle)) as compute:
        with pytest.raises(ComputeError) as exc:
            bridge_init(compute, "0x" + "b" * 40)
    msg = str(exc.value)
    assert "bridge not reachable at http://bridge.test" in msg
    assert "services/zg-bridge/README.md" in msg
    assert "npm run dev" in msg


def test_bridge_init_rewraps_healthz_transport_error():
    """A connection refused (httpx.ConnectError) is what the user
    actually sees most often — the bridge process isn't running. Same
    rewrap path as a 500."""

    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _make_client(httpx.MockTransport(handle)) as compute:
        with pytest.raises(ComputeError) as exc:
            bridge_init(compute, "0x" + "b" * 40)
    assert "bridge not reachable at http://bridge.test" in str(exc.value)


# ---------------- ledger_amount validation ----------------


def test_bridge_init_default_ledger_amount_is_1_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env, no kwarg → 1.1 default. Verified by inspecting what the
    helper sends to /admin/add-ledger."""
    monkeypatch.delenv("EERFUL_0G_LEDGER_DEPOSIT", raising=False)
    captured: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"wallet": "x", "chain_id": 1})
        if request.url.path == "/admin/add-ledger":
            captured["body"] = request.read().decode()
            return httpx.Response(
                200,
                json={"created": False, "total_balance_0g": 1.1, "total_balance_neuron": "0", "available_balance_neuron": "0"},
            )
        return httpx.Response(
            200,
            json={"provider_address": "x", "tee_signer_address": "y", "already_acknowledged": False},
        )

    with _make_client(httpx.MockTransport(handle)) as compute:
        bridge_init(compute, "0x" + "b" * 40)
    assert '"amount_0g":1.1' in captured["body"]


def test_bridge_init_env_var_drives_ledger_amount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EERFUL_0G_LEDGER_DEPOSIT", "2.5")
    captured: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"wallet": "x", "chain_id": 1})
        if request.url.path == "/admin/add-ledger":
            captured["body"] = request.read().decode()
            return httpx.Response(
                200,
                json={"created": False, "total_balance_0g": 2.5, "total_balance_neuron": "0", "available_balance_neuron": "0"},
            )
        return httpx.Response(
            200,
            json={"provider_address": "x", "tee_signer_address": "y", "already_acknowledged": False},
        )

    with _make_client(httpx.MockTransport(handle)) as compute:
        bridge_init(compute, "0x" + "b" * 40)
    assert '"amount_0g":2.5' in captured["body"]


def test_bridge_init_kwarg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EERFUL_0G_LEDGER_DEPOSIT", "9.9")
    captured: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"wallet": "x", "chain_id": 1})
        if request.url.path == "/admin/add-ledger":
            captured["body"] = request.read().decode()
            return httpx.Response(
                200,
                json={"created": False, "total_balance_0g": 1.5, "total_balance_neuron": "0", "available_balance_neuron": "0"},
            )
        return httpx.Response(
            200,
            json={"provider_address": "x", "tee_signer_address": "y", "already_acknowledged": False},
        )

    with _make_client(httpx.MockTransport(handle)) as compute:
        bridge_init(compute, "0x" + "b" * 40, ledger_amount=1.5)
    assert '"amount_0g":1.5' in captured["body"]


def test_bridge_init_rejects_non_numeric_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in `.env` (`EERFUL_0G_LEDGER_DEPOSIT=abc`) must surface as
    ComputeError at the helper boundary — otherwise a raw ValueError
    leaks mid-init with a confusing line number."""
    monkeypatch.setenv("EERFUL_0G_LEDGER_DEPOSIT", "abc")
    with _make_client(_ok_handler()) as compute:
        with pytest.raises(ComputeError, match="invalid ledger amount"):
            bridge_init(compute, "0x" + "b" * 40)


@pytest.mark.parametrize("amount", [-1.0, 0.0])
def test_bridge_init_rejects_non_positive(amount: float) -> None:
    """≤ 0 fails at helper. The bridge would later surface as 'balance
    below minimum lock' on the first inference; far from the source."""
    with _make_client(_ok_handler()) as compute:
        with pytest.raises(ComputeError, match="must be > 0"):
            bridge_init(compute, "0x" + "b" * 40, ledger_amount=amount)


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
def test_bridge_init_rejects_non_finite(amount: float) -> None:
    """NaN/Infinity slip past `<= 0` (NaN comparisons are always False
    under IEEE 754); explicit `math.isfinite` catches them. Without
    this, a NaN would be JSON-encoded and shipped to the bridge,
    failing in some confusing downstream way."""
    assert not math.isfinite(amount)
    with _make_client(_ok_handler()) as compute:
        with pytest.raises(ComputeError, match="finite"):
            bridge_init(compute, "0x" + "b" * 40, ledger_amount=amount)
