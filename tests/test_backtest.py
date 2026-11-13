
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


def test_exact_transaction_costs():
    """
    Prüft exakte Handelskosten.

    Start: 10.000 EUR
    Gewichte: 60 % Equity, 40 % Bonds
    Equity-Rendite: +10 %

    Erwartetes Handelsvolumen: 480 EUR
    Erwartete Kosten: 0.48 EUR
    Endvermögen: 10.599,52 EUR
    """

    dates = pd.to_datetime([
        "2025-01-30",
        "2025-01-31",
        "2025-02-03",
    ])

    returns = pd.DataFrame(
        0.0,
        index=dates,
        columns=ASSETS,
    )

    returns.loc[
        "2025-01-31", "Equity"
    ] = 0.10

    weights = pd.DataFrame(
        {
            "Equity": 0.6,
            "Bonds": 0.4,
            "Gold": 0.0,
        },
        index=dates,
    )

    result = run_engine(
        returns,
        weights,
        cost_rate=0.001,
    )

    rebalance_date = pd.Timestamp(
        "2025-01-31"
    )

    # Erwartetes Vermögen vor Rebalancing
    wealth_before = 10_600.0

    # 240 EUR Verkauf + 240 EUR Kauf
    expected_volume = 480.0

    # Handelskosten von 0.1 %
    expected_cost = 0.48

    expected_wealth = 10_599.52

    # Rebalancing findet statt
    assert result.loc[
        rebalance_date, "rebalanced"
    ]

    # Exakten Turnover prüfen
    assert result.loc[
        rebalance_date, "turnover"
    ] == pytest.approx(
        expected_volume / wealth_before
    )

    # Exakte Gebühren prüfen
    assert result.loc[
        rebalance_date, "trading_cost_eur"
    ] == pytest.approx(expected_cost)

    # Gesamtkosten prüfen
    assert result[
        "trading_cost_eur"
    ].sum() == pytest.approx(
        expected_cost
    )

    # Portfoliowert nach Kosten
    assert result.loc[
        rebalance_date, "portfolio_value"
    ] == pytest.approx(
        expected_wealth
    )


def test_exact_rebalancing():
    """
    Prüft die exakte Neugewichtung
    und deren Wirkung am Folgetag.

    Start: 10.000 EUR
    Equity/Bonds: 50/50

    31.01.: Equity +20 %
    Rebalancing auf 50/50

    03.02.: Equity -10 %, Bonds +4 %

    Erwartetes Endvermögen: 10.670 EUR
    """

    dates = pd.to_datetime([
        "2025-01-30",
        "2025-01-31",
        "2025-02-03",
    ])

    returns = pd.DataFrame(
        0.0,
        index=dates,
        columns=ASSETS,
    )

    returns.loc[
        "2025-01-31", "Equity"
    ] = 0.20

    returns.loc[
        "2025-02-03", "Equity"
    ] = -0.10

    returns.loc[
        "2025-02-03", "Bonds"
    ] = 0.04

    weights = pd.DataFrame(
        {
            "Equity": 0.5,
            "Bonds": 0.5,
            "Gold": 0.0,
        },
        index=dates,
    )

    # Ohne Gebühren, um die reine
    # Rebalancing-Wirkung zu isolieren.
    result = run_engine(
        returns,
        weights,
        cost_rate=0.0,
    )

    rebalance_date = pd.Timestamp(
        "2025-01-31"
    )

    next_date = pd.Timestamp(
        "2025-02-03"
    )

    # 31.01.: 5.000 * 1.20 + 5.000
    assert result.loc[
        rebalance_date, "portfolio_value"
    ] == pytest.approx(11_000.0)

    # 500 EUR Verkauf + 500 EUR Kauf
    assert result.loc[
        rebalance_date, "turnover"
    ] == pytest.approx(
        1_000 / 11_000
    )

    # Rebalancing auf 50/50
    assert result.loc[
        rebalance_date, "equity_weight"
    ] == pytest.approx(0.5)

    assert result.loc[
        rebalance_date, "bonds_weight"
    ] == pytest.approx(0.5)

    # Rebalancing darf die Rendite
    # desselben Tages nicht verändern.
    assert result.loc[
        rebalance_date, "gross_return"
    ] == pytest.approx(0.10)

    # Folgetag nutzt neue Gewichte:
    # 0.5 * (-0.10) + 0.5 * 0.04 = -0.03
    assert result.loc[
        next_date, "gross_return"
    ] == pytest.approx(-0.03)

    # 11.000 * 0.97 = 10.670 EUR
    assert result.loc[
        next_date, "portfolio_value"
    ] == pytest.approx(10_670.0)
