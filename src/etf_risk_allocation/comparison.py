
import numpy as np
import pandas as pd

from .strategies import (
    weights_60_40,
    equal_weight_weights,
    inverse_volatility_weights,
    minimum_variance_weights,
)
from .backtest import backtest_dynamic_weights


ASSETS = ["Equity", "Bonds", "Gold"]

INITIAL_CAPITAL = 10_000.0
COST_RATE = 0.001
LOOKBACK = 252
BACKTEST_START = "2019-01-03"


def constant_weight_frame(
    index: pd.DatetimeIndex,
    weights: pd.Series,
) -> pd.DataFrame:
    """
    Wandelt feste Strategiegewichte in eine
    tägliche Zielgewichtungs-Zeitreihe um.
    """
    weights = weights.reindex(
        ASSETS, fill_value=0.0
    )

    return pd.DataFrame(
        {
            asset: float(weights[asset])
            for asset in ASSETS
        },
        index=index,
    )


def run_all_backtests(
    returns: pd.DataFrame,
):
    """
    Führt alle vier Strategien mit identischen
    Daten und Backtesting-Annahmen aus.

    Returns
    -------
    backtests : dict
        Vollständige Ergebnisse pro Strategie.

    portfolio_values : DataFrame
        Tägliche Portfoliovermögen.

    daily_returns : DataFrame
        Tägliche Nettorenditen.

    summary : DataFrame
        Zusammenfassung der Backtests.
    """

    # 1. Gemeinsames ETF-Universum
    returns = returns[ASSETS].copy()
    returns = returns.sort_index()

    if (
        returns.empty
        or returns.index.has_duplicates
        or not isinstance(
            returns.index, pd.DatetimeIndex
        )
    ):
        raise ValueError(
            "Ungültiger Rendite-Datensatz."
        )

    if not np.isfinite(
        returns.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Renditen enthalten NaN oder Inf."
        )

    # 2. Strategiegewichte bestimmen
    # Wichtig: Hier noch alle historischen
    # Daten einschließlich 2018 verwenden.

    target_weights = {
        "60_40": constant_weight_frame(
            returns.index,
            weights_60_40(),
        ),
        "equal_weight": constant_weight_frame(
            returns.index,
            equal_weight_weights(),
        ),
        "inverse_volatility": (
            inverse_volatility_weights(
                returns, lookback=LOOKBACK
            )
        ),
        "minimum_variance": (
            minimum_variance_weights(
                returns, lookback=LOOKBACK
            )
        ),
    }

    # 3. Gemeinsamen Starttermin bestimmen
    # Erst wenn alle vier Strategien
    # gültige Gewichte liefern, beginnen wir.

    first_dates = [
        pd.Timestamp(BACKTEST_START)
    ]

    for name, weights in target_weights.items():
        if not set(ASSETS).issubset(
            weights.columns
        ):
            raise ValueError(
                f"Asset-Spalten fehlen: {name}"
            )

        valid = weights[ASSETS].dropna()

        if valid.empty:
            raise ValueError(
                f"Keine gültigen Gewichte: {name}"
            )

        first_dates.append(valid.index.min())

    common_start = max(first_dates)

    common_returns = returns.loc[
        common_start:
    ].copy()

    if common_returns.empty:
        raise ValueError(
            "Kein gemeinsamer Zeitraum."
        )

    # 4. Gewichte für alle Handelstage
    # synchronisieren und validieren.

    aligned_weights = {}

    for name, weights in target_weights.items():
        w = weights.reindex(
            common_returns.index
        )[ASSETS].astype(float)

        if not np.isfinite(w.to_numpy()).all():
            raise ValueError(
                f"Fehlende Gewichte: {name}"
            )

        if (w.to_numpy() < -1e-8).any():
            raise ValueError(
                f"Short-Position gefunden: {name}"
            )

        if (w.to_numpy() > 1 + 1e-8).any():
            raise ValueError(
                f"Leverage gefunden: {name}"
            )

        if not np.allclose(
            w.sum(axis=1).to_numpy(),
            1.0,
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(
                f"Gewichtssumme ungleich 1: {name}"
            )

        aligned_weights[name] = w

    # 5. Alle Strategien durch dieselbe
    # bereits implementierte Engine schicken.

    backtests = {}

    for name, weights in aligned_weights.items():
        result = backtest_dynamic_weights(
            returns=common_returns,
            target_weights=weights,
            initial_capital=INITIAL_CAPITAL,
            cost_rate=COST_RATE,
        )

        if not result.index.equals(
            common_returns.index
        ):
            raise ValueError(
                f"Abweichender Zeitindex: {name}"
            )

        backtests[name] = result

    # 6. Einheitliche Rebalancing-Termine
    reference = next(
        iter(backtests.values())
    )["rebalanced"].astype(bool)

    for name, result in backtests.items():
        flags = result["rebalanced"].astype(
            bool
        )

        if not flags.equals(reference):
            raise ValueError(
                f"Rebalancing weicht ab: {name}"
            )

    # 7. Vergleichbare Portfolio-Zeitreihen
    portfolio_values = pd.DataFrame({
        name: result["portfolio_value"]
        for name, result in backtests.items()
    })

    daily_returns = pd.DataFrame({
        name: result["net_return"]
        for name, result in backtests.items()
    })

    # 8. Übersicht der Ergebnisse
    summary_rows = []

    for name, result in backtests.items():
        final_value = float(
            result["portfolio_value"].iloc[-1]
        )

        total_return = (
            final_value / INITIAL_CAPITAL - 1
        ) * 100

        summary_rows.append({
            "Strategy": name,
            "End Capital EUR": final_value,
            "Total Return %": total_return,
            "Rebalancings": int(
                result["rebalanced"].sum()
            ),
            "Trading Costs EUR": float(
                result["trading_cost_eur"].sum()
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    ).set_index("Strategy")

    return (
        backtests,
        portfolio_values,
        daily_returns,
        summary,
    )
