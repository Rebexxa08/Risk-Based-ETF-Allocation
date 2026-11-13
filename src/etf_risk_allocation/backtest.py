
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .strategies import (
    weights_60_40,
    constant_mix_daily_returns,
)


def backtest_monthly_rebalanced(
    returns: pd.DataFrame,
    target_weights: pd.Series,
    initial_capital: float = 10_000.0,
    cost_rate: float = 0.001,
) -> pd.DataFrame:
    """Backtest mit täglichen Renditen und monatlichem Rebalancing."""

    if returns.empty:
        raise ValueError("Keine Renditedaten vorhanden.")

    if (
        not isinstance(returns.index, pd.DatetimeIndex)
        or not returns.index.is_monotonic_increasing
        or returns.index.has_duplicates
    ):
        raise ValueError("Der Datumsindex muss sortiert und eindeutig sein.")

    if returns.isna().any().any() or (returns <= -1).any().any():
        raise ValueError("Ungültige oder fehlende Renditen.")

    if initial_capital <= 0 or not 0 <= cost_rate < 1:
        raise ValueError("Ungültige Backtest-Parameter.")

    target = target_weights.reindex(returns.columns)

    if (
        target.isna().any()
        or (target < 0).any()
        or not np.isclose(target.sum(), 1.0)
    ):
        raise ValueError("Die Zielgewichte müssen sich zu 1 summieren.")

    wealth = float(initial_capital)
    weights = target.copy()
    dates = returns.index
    records = []

    for i, (date, daily_returns) in enumerate(returns.iterrows()):

        # 1. Portfolio-Wert vor der Tagesrendite
        wealth_previous = wealth

        # 2. Tagesrendite mit den aktuellen Gewichten
        gross_return = float(
            (weights * daily_returns).sum()
        )
        wealth *= (1 + gross_return)

        # 3. Gewichte nach der Marktbewegung aktualisieren
        weights = (
            weights * (1 + daily_returns)
            / (1 + gross_return)
        )

        # 4. Prüfen, ob dies der letzte Handelstag
        #    des Monats innerhalb des Backtests ist
        month_end = (
            i < len(dates) - 1
            and date.to_period("M")
            != dates[i + 1].to_period("M")
        )

        turnover = 0.0
        trading_cost = 0.0

        if month_end:
            # 5. Gehandeltes Volumen berechnen
            turnover = float(
                (target - weights).abs().sum()
            )

            # 6. Transaktionskosten berücksichtigen
            trading_cost = wealth * turnover * cost_rate
            wealth -= trading_cost

            # 7. Zielgewichte für den nächsten Tag setzen
            weights = target.copy()

        # 8. Tagesergebnisse speichern
        records.append({
            "Date": date,
            "gross_return": gross_return,
            "net_return": wealth / wealth_previous - 1,
            "portfolio_value": wealth,
            "equity_weight": weights["Equity"],
            "bonds_weight": weights["Bonds"],
            "rebalanced": month_end,
            "turnover": turnover,
            "trading_cost_eur": trading_cost,
        })

    return pd.DataFrame.from_records(
        records
    ).set_index("Date")


def backtest_dynamic_weights(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 10_000.0,
    cost_rate: float = 0.001
) -> pd.DataFrame:
    """
    Backtest mit dynamischen Zielgewichten.

    - Tägliche Portfoliorenditen
    - Monatliches Rebalancing
    - Gewichtungsdrift zwischen Rebalancings
    - Transaktionskosten
    """

    # 1. Daten vorbereiten
    returns = returns.sort_index()
    target_weights = target_weights.sort_index()

    assets = list(target_weights.columns)
    returns = returns[assets]

    if not returns.index.equals(target_weights.index):
        raise ValueError("Indizes stimmen nicht überein.")

    if returns.isna().any().any():
        raise ValueError("Fehlende Renditen vorhanden.")

    if (
        target_weights.isna().any().any()
        or (target_weights < 0).any().any()
    ):
        raise ValueError("Ungültige Zielgewichte.")

    if not (
        (target_weights.sum(axis=1) - 1).abs() < 1e-8
    ).all():
        raise ValueError("Gewichte müssen sich zu 1 summieren.")

    if initial_capital <= 0 or cost_rate < 0:
        raise ValueError("Ungültige Backtest-Parameter.")

    # 2. Portfolio initialisieren
    capital = float(initial_capital)
    weights = target_weights.iloc[0].copy()

    dates = returns.index
    records = []

    # 3. Täglichen Backtest durchführen
    for i, date in enumerate(dates):

        capital_before = capital
        asset_returns = returns.loc[date]

        # Brutto-Portfoliorendite
        gross_return = float(
            (weights * asset_returns).sum()
        )

        # Portfolioentwicklung
        capital = capital_before * (1 + gross_return)

        # Gewichtungsdrift durch Kursveränderungen
        drifted_weights = (
            weights * (1 + asset_returns)
            / (1 + gross_return)
        )

        # 4. Monatsende erkennen
        rebalance = (
            i < len(dates) - 1
            and date.to_period("M")
            != dates[i + 1].to_period("M")
        )

        turnover = 0.0
        trading_cost = 0.0

        # 5. Monatliches Rebalancing
        if rebalance:

            target = target_weights.loc[date]

            # Gehandeltes Volumen
            turnover = float(
                (target - drifted_weights).abs().sum()
            )

            # Transaktionskosten
            trading_cost = capital * turnover * cost_rate

            capital -= trading_cost

            # Neue Zielgewichte übernehmen
            weights = target.copy()

        else:
            weights = drifted_weights

        # 6. Nettorendite
        net_return = capital / capital_before - 1

        # 7. Ergebnisse speichern
        row = {
            "Date": date,
            "gross_return": gross_return,
            "net_return": net_return,
            "portfolio_value": capital,
            "rebalanced": rebalance,
            "turnover": turnover,
            "trading_cost_eur": trading_cost
        }

        for asset in assets:
            row[f"{asset.lower()}_weight"] = float(
                weights[asset]
            )

        records.append(row)

    return pd.DataFrame(records).set_index("Date")


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"

    # Daten aus Schritt 3 laden
    returns = pd.read_csv(
        data_dir / "returns_daily.csv",
        index_col=0,
        parse_dates=True,
    )

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Gemeinsamen Backtestzeitraum verwenden
    start_date = pd.Timestamp(
        metadata["first_backtest_day"]
    )

    returns = returns.loc[
        start_date:, ["Equity", "Bonds"]
    ].astype(float)

    # Monatliches Rebalancing
    results = backtest_monthly_rebalanced(
        returns=returns,
        target_weights=weights_60_40(),
        initial_capital=10_000.0,
        cost_rate=0.001,
    )

    # Vergleich: theoretisch täglich konstantes 60/40
    results["daily_mix_return"] = (
        constant_mix_daily_returns(returns)
    )

    results["daily_mix_value"] = (
        10_000.0 *
        (1 + results["daily_mix_return"]).cumprod()
    )

    # Ergebnisse speichern
    output_path = data_dir / "benchmark_60_40.csv"
    results.to_csv(output_path, index_label="Date")

    print(f"Backtestbeginn: {results.index[0].date()}")
    print(f"Backtestende: {results.index[-1].date()}")
    print(
        f"Endwert 60/40: "
        f"{results['portfolio_value'].iloc[-1]:.2f} EUR"
    )
    print(
        f"Rebalancings: "
        f"{int(results['rebalanced'].sum())}"
    )
    print(
        f"Transaktionskosten: "
        f"{results['trading_cost_eur'].sum():.2f} EUR"
    )
    print(f"Gespeichert unter: {output_path}")


if __name__ == "__main__":
    main()
