
import numpy as np
import pandas as pd
import pytest

from etf_risk_allocation.backtest import (
    backtest_dynamic_weights,
)


INITIAL_CAPITAL = 10_000.0
COST_RATE = 0.001
ASSETS = ["Equity", "Bonds", "Gold"]


def make_data():
    """
    Künstliche Renditen über drei Monate.

    Die Renditen sind bewusst einfach,
    damit wir das Ergebnis prüfen können.
    """
    dates = pd.bdate_range(
        "2025-01-01", "2025-03-31"
    )

    returns = pd.DataFrame(
        0.0, index=dates, columns=ASSETS
    )

    returns.loc[
        "2025-01-15", "Equity"
    ] = 0.10

    returns.loc[
        "2025-02-12", "Bonds"
    ] = -0.05

    weights = pd.DataFrame(
        {
            "Equity": 0.5,
            "Bonds": 0.5,
            "Gold": 0.0,
        },
        index=dates,
    )

    return returns, weights


def run_engine(
    returns, weights, cost_rate=COST_RATE
):
    return backtest_dynamic_weights(
        returns=returns,
        target_weights=weights,
        initial_capital=INITIAL_CAPITAL,
        cost_rate=cost_rate,
    )


def test_portfolio_value_consistency():
    """
    Portfoliovermögen muss mit
    den täglichen Nettorenditen
    konsistent sein.
    """
    returns, weights = make_data()

    result = run_engine(
        returns, weights
    )

    values = result["portfolio_value"]
    net_returns = result["net_return"]

    expected = values.shift(1) * (
        1 + net_returns
    )

    np.testing.assert_allclose(
        values.iloc[1:],
        expected.iloc[1:],
        rtol=1e-8,
        atol=1e-8,
    )


def test_no_cost_without_trades():
    """
    Bei vollständig konstanten Renditen
    und konstanten Zielgewichten darf
    nach der Erstinvestition kein
    weiteres Handelsvolumen entstehen.
    """
    returns, weights = make_data()

    returns.loc[:, :] = 0.0

    result = run_engine(
        returns, weights
    )

    # Eventuelle Erstinvestition ausnehmen.
    later_costs = result[
        "trading_cost_eur"
    ].iloc[1:]

    assert np.isclose(
        later_costs.sum(), 0.0, atol=1e-8
    )


def test_transaction_cost_rate():
    """
    Höhere Gebühren dürfen bei derselben
    Handelsfolge nicht zu einem höheren
    Endvermögen führen.
    """
    returns, weights = make_data()

    free = run_engine(
        returns, weights, cost_rate=0.0
    )

    costly = run_engine(
        returns, weights, cost_rate=0.001
    )

    assert costly[
        "portfolio_value"
    ].iloc[-1] <= free[
        "portfolio_value"
    ].iloc[-1] + 1e-8

    assert np.isclose(
        free["trading_cost_eur"].sum(),
        0.0,
        atol=1e-8,
    )

    assert (
        costly["trading_cost_eur"] >= 0
    ).all()


def test_monthly_rebalancing():
    """
    Prüft das monatliche Rebalancing.

    Die Engine rebalanced am letzten
    Handelstag jedes Monats.

    Am letzten Datenpunkt ist kein
    weiteres Rebalancing erforderlich.
    """
    returns, weights = make_data()

    result = run_engine(
        returns, weights
    )

    rebalance_dates = result.index[
        result["rebalanced"].astype(bool)
    ]

    # Erwartete Rebalancing-Termine:
    # Letzter Handelstag jedes Monats,
    # außer dem letzten Monat im Datensatz.

    month_end_dates = (
        returns.groupby(
            returns.index.to_period("M")
        )
        .tail(1)
        .index
    )

    expected_dates = month_end_dates[:-1]
    pd.testing.assert_index_equal(
        rebalance_dates,
        expected_dates,
        check_names=False,
        )


def test_no_look_ahead_in_backtest():
    """
    Verändern wir eine zukünftige
    Rendite, dürfen vergangene
    Portfolioergebnisse gleich bleiben.
    """
    returns, weights = make_data()

    original = run_engine(
        returns, weights
    )

    changed_returns = returns.copy()

    future_date = pd.Timestamp(
        "2025-03-17"
    )

    changed_returns.loc[
        future_date, "Equity"
    ] = -0.40

    changed = run_engine(
        changed_returns, weights
    )

    past = original.index < future_date

    np.testing.assert_allclose(
        original.loc[
            past, "portfolio_value"
        ],
        changed.loc[
            past, "portfolio_value"
        ],
        rtol=1e-10,
        atol=1e-10,
    )


def test_positive_portfolio_value():
    """
    Der Portfoliowert muss bei diesen
    Testdaten stets positiv bleiben.
    """
    returns, weights = make_data()

    result = run_engine(
        returns, weights
    )

    assert (
        result["portfolio_value"] > 0
    ).all()

    assert np.isfinite(
        result["portfolio_value"]
    ).all()
