
from pathlib import Path

import pandas as pd

from etf_risk_allocation.comparison import (
    run_all_backtests,
)


def main():

    # Historische ETF-Renditen laden
    data_path = Path(
        "data/processed/returns_daily.csv"
    )

    returns = pd.read_csv(
        data_path,
        index_col="Date",
        parse_dates=True,
    )

    # Vier Strategien gemeinsam simulieren
    (
        backtests,
        portfolio_values,
        daily_returns,
        summary,
    ) = run_all_backtests(returns)

    # Ausgabeordner anlegen
    output_dir = Path(
        "data/processed/backtests"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Einzelergebnisse speichern
    for name, result in backtests.items():
        result.to_csv(
            output_dir / f"{name}.csv",
            index_label="Date",
        )

    # Vergleichszeitreihen speichern
    portfolio_values.to_csv(
        output_dir / "portfolio_values.csv",
        index_label="Date",
    )

    daily_returns.to_csv(
        output_dir / "daily_returns.csv",
        index_label="Date",
    )

    summary.to_csv(
        output_dir / "summary.csv"
    )

    # Konsolenausgabe
    print("\n--- Strategy Comparison ---")
    print(
        "Start:",
        portfolio_values.index.min().date()
    )
    print(
        "Ende:",
        portfolio_values.index.max().date()
    )
    print(
        "Handelstage:",
        len(portfolio_values)
    )

    print("\n--- Ergebnisse ---")
    print(
        summary.round(2).to_string()
    )

    print("\n--- Datenprüfung ---")

    print(
        "Strategien:",
        list(portfolio_values.columns)
    )

    print(
        "Fehlende Werte:",
        int(portfolio_values.isna().sum().sum())
    )

    print(
        "Gleicher Zeitindex:",
        portfolio_values.index.equals(
            daily_returns.index
        )
    )

    print(
        "\nErgebnisse gespeichert unter:",
        output_dir,
    )


if __name__ == "__main__":
    main()
