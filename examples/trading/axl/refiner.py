"""Refiner agent — runs on louie. Listens on AXL, receives strategy
drafts from the explorer, runs an Optuna sweep on the toy backtest,
returns the best params + Sharpe.

This is the cheap-compute layer. No LLM. Pure parameter optimization
against the explorer's submitted strategy module. The eerful pitch is
that the AXL channel is unattested — the gate at the end is what
gives the system its security properties.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import optuna

# Local sibling import — refiner.py and toy_backtest.py live next to each other
sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_backtest import run_strategy  # noqa: E402
from transport import recv_envelope, send_envelope  # noqa: E402

# emit_event lives in `eerful._emit` in the source tree; on louie, the
# deployed daemon doesn't have eerful installed (just httpx, optuna,
# numpy, pandas — see axl/README.md). The sibling fallback handles that
# deploy: scp `src/eerful/_emit.py` next to refiner.py and the local
# import wins.
try:
    from eerful._emit import emit_event  # noqa: E402
except ImportError:  # pragma: no cover — deploy-shape fallback
    from _emit import emit_event  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [refiner] %(message)s")
log = logging.getLogger(__name__)

# Reduce Optuna's chatter — every trial logs by default.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optional peer allowlist: when set, only drafts from this peer ID are
# accepted. Demo runs without it (logs a warning instead of refusing) so
# new operators don't have to figure out peer IDs to get a first run; in
# any production-shaped deployment this MUST be set, since module_code
# below executes whatever the sender ships.
ALLOWED_EXPLORER_PEER_ID = os.environ.get("AXL_EXPLORER_PEER_ID")

# AXL's X-From-Peer-Id header truncates the public key after ~14 bytes
# (28 hex chars) and pads the rest with `f`. Comparing against the full
# 64-char `our_public_key` from /topology fails even when peers match.
# Use a conservative 24-char prefix for the allowlist check — well
# below the truncation point and still gives 96 bits of identification.
_PEER_PREFIX_LEN = 24


def _peer_matches(received: str, expected: str | None) -> bool:
    if not expected:
        return False
    return received[:_PEER_PREFIX_LEN] == expected[:_PEER_PREFIX_LEN]

# Hard cap on Optuna trials — caller-supplied via the AXL envelope, but
# the sender is not trusted (it's the same channel that ships executable
# code; we already constrain risk by allowlisting peers, but a buggy
# explorer could still pin CPU here). 200 is plenty for a single
# strategy sweep.
_MAX_TRIALS = 200


def _build_objective(module_code: str, params_space: dict[str, list]):
    """Each entry in params_space is [type, low, high] or [type, choices]
    matching gecko's PARAMS_SPACE grammar. Optuna maps each to suggest_*."""

    def objective(trial: optuna.Trial) -> float:
        params: dict = {}
        for name, spec in params_space.items():
            if not isinstance(spec, list) or not spec:
                continue
            kind = spec[0]
            if kind == "int" and len(spec) >= 3:
                params[name] = trial.suggest_int(name, int(spec[1]), int(spec[2]))
            elif kind == "float" and len(spec) >= 3:
                params[name] = trial.suggest_float(name, float(spec[1]), float(spec[2]))
            elif kind == "categorical" and len(spec) >= 2:
                params[name] = trial.suggest_categorical(name, spec[1])
        return run_strategy(module_code, params)

    return objective


_PROGRESS_EVERY = 5


def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Optuna `callbacks=[...]` runs this after each trial. Throttled
    to one emit per `_PROGRESS_EVERY` trials so 20-trial sweeps don't
    flood the SSE wire with 20 near-identical events."""
    n = trial.number + 1
    if n % _PROGRESS_EVERY != 0:
        return
    try:
        best_value = study.best_value
        best_params = study.best_params
    except ValueError:
        # First few trials may not have produced a valid value yet.
        return
    emit_event(
        source="refiner",
        kind="optuna_progress",
        trial=n,
        best_sharpe=float(best_value),
        params=best_params,
    )


def handle_request(strategy_spec: str, module_code: str, params_space: dict, n_trials: int = 20) -> dict:
    """Run one Optuna study against the submitted strategy. Returns
    {best_params, sharpe, n_trials}. If no trial produces a positive
    Sharpe, returns the best (worst) one anyway — the demo recording
    still wants to *see* the sweep complete and ship a result."""
    log.info("starting study: %d trials, params=%s", n_trials, list(params_space.keys()))
    emit_event(
        source="refiner",
        kind="optuna_started",
        n_trials=n_trials,
        params=list(params_space.keys()),
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(
        _build_objective(module_code, params_space),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_progress_callback],
    )
    best = study.best_trial
    log.info("study done — best sharpe=%.3f params=%s", best.value, best.params)
    emit_event(
        source="refiner",
        kind="optuna_finished",
        sharpe=float(best.value),
        best_params=best.params,
    )
    return {"best_params": best.params, "sharpe": float(best.value), "n_trials": n_trials}


def main() -> int:
    log.info("listening on AXL — waiting for strategy drafts")
    while True:
        recv = recv_envelope(timeout_sec=300)
        if recv is None:
            log.info("no message in 5min — still listening")
            continue
        from_peer, payload = recv
        if ALLOWED_EXPLORER_PEER_ID and not _peer_matches(from_peer, ALLOWED_EXPLORER_PEER_ID):
            log.warning(
                "rejecting envelope from unallowed peer %s (allowlist=%s)",
                from_peer[:_PEER_PREFIX_LEN],
                ALLOWED_EXPLORER_PEER_ID[:_PEER_PREFIX_LEN],
            )
            continue
        if not ALLOWED_EXPLORER_PEER_ID:
            log.warning(
                "AXL_EXPLORER_PEER_ID unset — accepting drafts from any peer "
                "(%s). Set the env var in any production deployment.",
                from_peer[:16],
            )
        if payload.get("kind") != "STRATEGY_DRAFT":
            log.warning("ignoring envelope of kind %r from %s", payload.get("kind"), from_peer[:16])
            continue
        log.info("STRATEGY_DRAFT received from %s", from_peer[:16])

        # Fresh run_id per draft so the SPA's run-boundary logic groups
        # this request's events together (without trying to thread the
        # explorer's run_id through the AXL envelope).
        os.environ["EERFUL_DEMO_RUN_ID"] = "refiner-" + os.urandom(4).hex()
        emit_event(
            source="refiner",
            kind="axl_recv",
            envelope_kind="STRATEGY_DRAFT",
            peer_id_prefix=from_peer[:16],
            n_trials=int(payload.get("n_trials", 20)),
        )

        try:
            # Clamp n_trials so a malformed/malicious caller can't pin
            # CPU. Bound matches typical single-strategy sweep budgets.
            requested_trials = int(payload.get("n_trials", 20))
            n_trials = max(1, min(requested_trials, _MAX_TRIALS))
            if n_trials != requested_trials:
                log.info("clamped n_trials %d → %d", requested_trials, n_trials)
            result = handle_request(
                strategy_spec=payload["strategy_spec"],
                module_code=payload["module_code"],
                params_space=payload["params_space"],
                n_trials=n_trials,
            )
            response = {"kind": "OPTIMIZATION_RESULT", **result}
        except Exception as e:
            log.exception("study failed")
            response = {"kind": "OPTIMIZATION_ERROR", "error": str(e)}
        try:
            send_envelope(dest_peer_id=from_peer, payload=response)
            log.info("response sent to %s", from_peer[:16])
            emit_event(
                source="refiner",
                kind="axl_send",
                envelope_kind=response.get("kind", "?"),
                peer_id_prefix=from_peer[:16],
            )
        except Exception:
            # Don't crash the daemon if a single response can't be
            # delivered (peer offline, AXL bridge restarted, etc.) —
            # just log and continue serving the next request.
            log.exception("failed to send response to %s", from_peer[:16])


if __name__ == "__main__":
    sys.exit(main() or 0)
