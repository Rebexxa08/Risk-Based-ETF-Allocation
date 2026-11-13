
from pathlib import Path

import numpy as np
import pandas as pd

from etf_risk_allocation.strategies import (
    minimum_variance_weights
)
from etf_risk_allocation.backtest import (
    backtest_dynamic_weights
)


# 1. Projektpfade
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"

assets = ["Equity", "Bonds", "Gold"]
initial_capital = 10_000

# 2. Renditedaten laden
returns = pd.read_csv(
    DATA / "returns_daily.csv",
    index_col=0,
    parse_dates=True
)

returns = returns[assets].sort_index().dropna()

# 3. Minimum-Variance-Gewichte berechnen
weights = minimum_variance_weights(
    returns,
    lookback=252
)

# 4. Backtestzeitraum festlegen
start_date = pd.Timestamp("2019-01-03")

dates = returns.index.intersection(weights.index)
dates = dates[dates >= start_date]

returns_bt = returns.loc[dates]
weights_bt = weights.loc[dates]

# 5. Backtest ausführen
result = backtest_dynamic_weights(
    returns=returns_bt,
    target_weights=weights_bt,
    initial_capital=initial_capital,
    cost_rate=0.001
)

# 6. Ergebnisse speichern
result.to_csv(
    DATA / "minimum_variance_backtest.csv",
    index_label="Date"
)

weights_bt.to_csv(
    DATA / "minimum_variance_weights.csv",
    index_label="Date"
)

# 7. Ergebnisse anzeigen
print("\n--- Minimum-Variance-Backtest ---")

print("Start:", result.index.min().date())
print("Ende:", result.index.max().date())

end_capital = result["portfolio_value"].iloc[-1]

print("Endkapital:", round(end_capital, 2), "EUR")
print(
    "Gesamtrendite:",
    round((end_capital / initial_capital - 1) * 100, 2),
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


# 8. Plausibilitätschecks
print("\n--- Plausibilitätschecks ---")

weight_cols = [
    "equity_weight",
    "bonds_weight",
    "gold_weight"
]

# Test 1: Zielgewichte summieren sich zu 1
np.testing.assert_allclose(
    weights_bt.sum(axis=1).to_numpy(),
    1.0,
    atol=1e-10
)
print("Test 1 bestanden: Zielgewichte summieren sich zu 1.")

# Test 2: Keine negativen Zielgewichte
assert (weights_bt >= 0).all().all()
assert (weights_bt <= 1).all().all()
print("Test 2 bestanden: Gewichtsgrenzen eingehalten.")

# Test 3: Portfolio bleibt vollständig investiert
np.testing.assert_allclose(
    result[weight_cols].sum(axis=1).to_numpy(),
    1.0,
    atol=1e-10
)
print("Test 3 bestanden: Portfolio vollständig investiert.")

# Test 4: Portfoliowert und Nettorendite konsistent
np.testing.assert_allclose(
    result["portfolio_value"].pct_change().iloc[1:],
    result["net_return"].iloc[1:],
    atol=1e-10
)
print("Test 4 bestanden: Nettorenditen konsistent.")

# Test 5: Rebalancing übernimmt Zielgewichte
rebalance_dates = result.index[result["rebalanced"]]

np.testing.assert_allclose(
    result.loc[rebalance_dates, weight_cols].to_numpy(),
    weights_bt.loc[rebalance_dates, assets].to_numpy(),
    atol=1e-10
)
print("Test 5 bestanden: Rebalancing korrekt.")

# Test 6: Varianz nicht größer als bei Gleichgewichtung
first_date = weights.index[0]
first_position = returns.index.get_loc(first_date)

history = returns.iloc[
    first_position - 252:first_position
]

covariance = history.cov().to_numpy()

w_opt = weights.loc[first_date].to_numpy()
w_equal = np.full(len(assets), 1 / len(assets))

var_opt = w_opt @ covariance @ w_opt
var_equal = w_equal @ covariance @ w_equal

assert var_opt <= var_equal + 1e-12
print("Test 6 bestanden: Varianzoptimierung plausibel.")

# Test 7: Kein Look-Ahead Bias beim ersten Gewicht
changed_returns = returns.loc[:first_date].copy()
changed_returns.loc[first_date, "Equity"] += 0.5

changed_weights = minimum_variance_weights(
    changed_returns,
    lookback=252
)

np.testing.assert_allclose(
    changed_weights.iloc[0].to_numpy(),
    weights.iloc[0].to_numpy(),
    atol=1e-10
)
print("Test 7 bestanden: Kein Look-Ahead Bias.")

print("\nAlle Tests erfolgreich bestanden.")

print("\n--- Kovarianzanalyse ---")

last_date = weights.index[-1]
last_position = returns.index.get_loc(last_date)

history = returns.iloc[
    last_position - 252:last_position
]

covariance = history.cov()

print("\nAnnualisierte Kovarianzmatrix:")
print((covariance * 252).round(6))

print("\nAnnualisierte Einzelvolatilitäten:")
volatility = np.sqrt(
    np.diag(covariance.to_numpy()) * 252
)

print(
    pd.Series(
        volatility,
        index=assets
    ).round(4)
)

print("\nKorrelationsmatrix:")
print(history.corr().round(4))