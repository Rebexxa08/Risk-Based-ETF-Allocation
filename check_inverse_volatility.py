
from pathlib import Path

import pandas as pd

from etf_risk_allocation.strategies import (
    inverse_volatility_weights
)
from etf_risk_allocation.backtest import (
    backtest_dynamic_weights
)


# 1. Projektpfade
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"


# 2. ETF-Renditen laden
returns = pd.read_csv(
    DATA / "returns_daily.csv",
    index_col=0,
    parse_dates=True
)

returns = returns.sort_index()
returns = returns[["Equity", "Bonds", "Gold"]].dropna()


# 3. Inverse-Volatility-Gewichte berechnen
weights = inverse_volatility_weights(
    returns,
    lookback=252
)


# 4. Backtest-Zeitraum festlegen
start_date = pd.Timestamp("2019-01-03")

dates = returns.index.intersection(weights.index)
dates = dates[dates >= start_date]

returns_bt = returns.loc[dates]
weights_bt = weights.loc[dates]


# 5. Backtest ausführen
result = backtest_dynamic_weights(
    returns=returns_bt,
    target_weights=weights_bt,
    initial_capital=10_000,
    cost_rate=0.001
)


# 6. Ergebnisse speichern
result.to_csv(
    DATA / "inverse_volatility_backtest.csv"
)

weights_bt.to_csv(
    DATA / "inverse_volatility_weights.csv",
    index_label="Date"
)


# 7. Ergebnisse ausgeben
print("\n--- Inverse-Volatility-Backtest ---")

print("Start:", result.index.min().date())
print("Ende:", result.index.max().date())

print(
    "Endkapital:",
    round(result["portfolio_value"].iloc[-1], 2),
    "EUR"
)

print(
    "Gesamtrendite:",
    round(
        (result["portfolio_value"].iloc[-1] / 10_000 - 1)
        * 100,
        2
    ),
    "%"
)

print(
    "Rebalancings:",
    int(result["rebalanced"].sum())
)

print(
    "Transaktionskosten:",
    round(result["trading_cost_eur"].sum(), 2),
    "EUR"
)

print("\nLetzte Zielgewichte:")
print(weights_bt.tail())


import numpy as np

print("\n--- Plausibilitätschecks ---")

weight_cols = [
    "equity_weight",
    "bonds_weight",
    "gold_weight"
]

# Test 1: Zielgewichte summieren sich zu 100 %
np.testing.assert_allclose(
    weights_bt.sum(axis=1).to_numpy(),
    1.0,
    atol=1e-10
)
print("Test 1 bestanden: Zielgewichte summieren sich zu 1.")

# Test 2: Tatsächliche Gewichte summieren sich zu 100 %
np.testing.assert_allclose(
    result[weight_cols].sum(axis=1).to_numpy(),
    1.0,
    atol=1e-10
)
print("Test 2 bestanden: Portfolio ist vollständig investiert.")

# Test 3: Portfoliowert und Nettorendite stimmen überein
np.testing.assert_allclose(
    result["portfolio_value"].pct_change().iloc[1:],
    result["net_return"].iloc[1:],
    atol=1e-10
)
print("Test 3 bestanden: Nettorenditen sind konsistent.")

# Test 4: Rebalancing-Gewichte entsprechen Zielgewichten
rebalance_dates = result.index[result["rebalanced"]]

np.testing.assert_allclose(
    result.loc[rebalance_dates, weight_cols].to_numpy(),
    weights_bt.loc[rebalance_dates].to_numpy(),
    atol=1e-10
)
print("Test 4 bestanden: Rebalancing korrekt.")

# Test 5: Keine negativen Gewichte
assert (result[weight_cols] >= 0).all().all()
print("Test 5 bestanden: Keine Short-Positionen.")
