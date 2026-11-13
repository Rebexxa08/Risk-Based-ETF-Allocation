
import pandas as pd


def weights_60_40() -> pd.Series:
    """Definiert die Zielgewichte des 60/40-Portfolios."""
    return pd.Series(
        {"Equity": 0.60, "Bonds": 0.40},
        dtype=float
    )


def constant_mix_daily_returns(
    returns: pd.DataFrame
) -> pd.Series:
    """
    Berechnet die täglichen Renditen bei konstanten
    Gewichten (theoretisch tägliches Rebalancing).
    """
    weights = weights_60_40()

    portfolio_returns = (
        returns[weights.index]
        .mul(weights)
        .sum(axis=1)
    )

    return portfolio_returns.rename("daily_mix_return")
