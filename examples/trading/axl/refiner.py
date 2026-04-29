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
import sys
from pathlib import Path

import optuna

# Local sibling import — refiner.py and toy_backtest.py live next to each other
sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_backtest import run_strategy  # noqa: E402
from transport import recv_envelope, send_envelope  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [refiner] %(message)s")
log = logging.getLogger(__name__)

# Reduce Optuna's chatter — every trial logs by default.
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def handle_request(strategy_spec: str, module_code: str, params_space: dict, n_trials: int = 20) -> dict:
    """Run one Optuna study against the submitted strategy. Returns
    {best_params, sharpe, n_trials}. If no trial produces a positive
    Sharpe, returns the best (worst) one anyway — the demo recording
    still wants to *see* the sweep complete and ship a result."""
    log.info("starting study: %d trials, params=%s", n_trials, list(params_space.keys()))
    study = optuna.create_study(direction="maximize")
    study.optimize(_build_objective(module_code, params_space), n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    log.info("study done — best sharpe=%.3f params=%s", best.value, best.params)
    return {"best_params": best.params, "sharpe": float(best.value), "n_trials": n_trials}


def main() -> int:
    log.info("listening on AXL — waiting for strategy drafts")
    while True:
        recv = recv_envelope(timeout_sec=300)
        if recv is None:
            log.info("no message in 5min — still listening")
            continue
        from_peer, payload = recv
        if payload.get("kind") != "STRATEGY_DRAFT":
            log.warning("ignoring envelope of kind %r from %s", payload.get("kind"), from_peer[:16])
            continue
        log.info("STRATEGY_DRAFT received from %s", from_peer[:16])
        try:
            result = handle_request(
                strategy_spec=payload["strategy_spec"],
                module_code=payload["module_code"],
                params_space=payload["params_space"],
                n_trials=int(payload.get("n_trials", 20)),
            )
            response = {"kind": "OPTIMIZATION_RESULT", **result}
        except Exception as e:
            log.exception("study failed")
            response = {"kind": "OPTIMIZATION_ERROR", "error": str(e)}
        send_envelope(dest_peer_id=from_peer, payload=response)
        log.info("response sent to %s", from_peer[:16])


if __name__ == "__main__":
    sys.exit(main() or 0)
