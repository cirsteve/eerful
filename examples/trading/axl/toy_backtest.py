"""Toy backtest harness for the AXL multi-agent demo.

Synthetic price data — deterministic seed, no external deps. The shape
is BTC/ETH/SOL perpetuals at 4-hour candles, ~6 months of history.
That's enough to compute a non-trivial Sharpe; the demo doesn't need
realistic market regimes, only a well-defined objective for Optuna to
optimize against.

The function `run_strategy` is what the refiner calls inside its
Optuna objective: takes the strategy module (loaded from explorer's
text) plus a params dict, returns Sharpe.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

_COINS = ("BTC", "ETH", "SOL")
_PERIODS = 1000  # ~6 months of 4-hour candles
_SEED = 42


def synth_prices() -> dict[str, pd.DataFrame]:
    """Deterministic synthetic price + funding history per coin."""
    rng = np.random.default_rng(_SEED)
    out: dict[str, pd.DataFrame] = {}
    for i, coin in enumerate(_COINS):
        # Per-coin drift/vol mimics rough perp behavior — BTC less volatile,
        # SOL more, ETH in between. Matters for Optuna to find non-trivial
        # parameter optima rather than scoring all params identically.
        drift = (0.0001, 0.00015, 0.0002)[i]
        vol = (0.015, 0.022, 0.030)[i]
        rets = rng.normal(drift, vol, size=_PERIODS)
        close = 100.0 * np.exp(np.cumsum(rets))
        funding = rng.normal(0.0, 0.0002, size=_PERIODS)
        ts = pd.date_range("2025-10-01", periods=_PERIODS, freq="4h")
        out[coin] = pd.DataFrame({"timestamp": ts, "close": close, "funding": funding}).set_index("timestamp")
    return out


def _load_module_from_text(name: str, code: str) -> ModuleType:
    """Materialize the explorer's submitted module string into an import.

    Writes to a temp path, builds a module spec, executes. Module survives
    one Optuna sweep then gets garbage-collected. No persistent state."""
    tmp_path = Path(f"/tmp/{name}.py")
    tmp_path.write_text(code)
    spec = importlib.util.spec_from_file_location(name, tmp_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def run_strategy(module_code: str, params: dict) -> float:
    """Backtest synthesis: load strategy, generate signals + positions
    per coin, compute aggregate equity curve Sharpe.

    Returns Sharpe (higher = better). Negative or NaN means refuse-this-
    parameter-set and Optuna will steer away."""
    mod = _load_module_from_text("toy_strategy", module_code)
    if not hasattr(mod, "generate_signals"):
        return -1.0  # malformed module → bad params
    prices = synth_prices()
    equity_returns = []
    lookback = int(params.get("lookback_periods", 42))
    if lookback < 2 or lookback >= _PERIODS - 10:
        return -1.0
    for coin, df in prices.items():
        try:
            df_with_signal = mod.generate_signals(df.copy(), params)
        except Exception:
            return -1.0
        if "signal" not in df_with_signal.columns:
            return -1.0
        # Per-coin position returns: signal aligned to next-period return
        ret = df_with_signal["close"].pct_change().shift(-1)
        position = df_with_signal["signal"].shift(1).fillna(0.0)
        coin_returns = (ret * position).dropna()
        equity_returns.append(coin_returns)
    portfolio = sum(equity_returns) / len(equity_returns)
    if portfolio.std() == 0 or np.isnan(portfolio.std()):
        return 0.0
    annualized = portfolio.mean() / portfolio.std() * np.sqrt(252 * 6)  # 6 4h-periods/day
    return float(annualized) if np.isfinite(annualized) else 0.0
