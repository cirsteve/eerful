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
import tempfile
import uuid
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

    Uses a UUID-suffixed module name + tempfile so concurrent calls don't
    collide on `sys.modules` or stomp `/tmp/<name>.py`. The temp file is
    cleaned up after exec; the module is removed from `sys.modules` to
    avoid bleed across sweeps."""
    unique_name = f"{name}_{uuid.uuid4().hex}"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=f"{name}_",
        delete=False,
    ) as f:
        f.write(code)
        tmp_path = Path(f.name)
    try:
        spec = importlib.util.spec_from_file_location(unique_name, tmp_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not build spec for {unique_name}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(unique_name, None)
        return mod
    finally:
        tmp_path.unlink(missing_ok=True)


def run_strategy(module_code: str, params: dict) -> float:
    """Backtest synthesis: load strategy, generate signals + positions
    per coin, compute aggregate equity curve Sharpe.

    Returns Sharpe (higher = better). Any error during module load,
    signal generation, or column access returns -1.0 — Optuna treats
    that as a failed trial and steers away. Wrapping is broad here on
    purpose: this runs untrusted (well, refiner-allowlisted but
    arbitrary-shape) module code and we're optimizing, not auditing."""
    try:
        mod = _load_module_from_text("toy_strategy", module_code)
    except Exception:
        return -1.0
    if not hasattr(mod, "generate_signals"):
        return -1.0  # malformed module → bad params
    lookback = int(params.get("lookback_periods", 42))
    if lookback < 2 or lookback >= _PERIODS - 10:
        return -1.0
    prices = synth_prices()
    equity_returns = []
    for _coin, df in prices.items():
        try:
            df_with_signal = mod.generate_signals(df.copy(), params)
            if not hasattr(df_with_signal, "columns") or "signal" not in df_with_signal.columns:
                return -1.0
            if "close" not in df_with_signal.columns:
                return -1.0
            # Per-coin position returns: signal aligned to next-period return
            ret = df_with_signal["close"].pct_change().shift(-1)
            position = df_with_signal["signal"].shift(1).fillna(0.0)
            coin_returns = (ret * position).dropna()
        except Exception:
            return -1.0
        equity_returns.append(coin_returns)
    portfolio = sum(equity_returns) / len(equity_returns)
    if portfolio.std() == 0 or np.isnan(portfolio.std()):
        return 0.0
    annualized = portfolio.mean() / portfolio.std() * np.sqrt(252 * 6)  # 6 4h-periods/day
    return float(annualized) if np.isfinite(annualized) else 0.0
