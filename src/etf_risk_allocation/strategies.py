
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


def inverse_volatility_weights(
    returns: pd.DataFrame,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Berechnet tägliche Zielgewichte nach Inverse Volatility.

    Die Volatilität basiert auf den vergangenen
    `lookback` Handelstagen.

    Die Gewichte werden um einen Handelstag
    verschoben, um Look-Ahead Bias zu vermeiden.
    """

    # 1. Anlageklassen auswählen
    assets = ["Equity", "Bonds", "Gold"]
    returns = returns[assets].copy()

    # 2. Historische Volatilität berechnen
    volatility = (
        returns
        .rolling(window=lookback, min_periods=lookback)
        .std()
        .shift(1)
    )

    # 3. Schutz vor Division durch null
    volatility = volatility.replace(0, float("nan"))

    # 4. Inverse Volatilität berechnen
    inverse_vol = 1 / volatility

    # 5. Gewichte normalisieren
    weights = inverse_vol.div(
        inverse_vol.sum(axis=1),
        axis=0
    )

    # 6. Unvollständige Beobachtungen entfernen
    weights = weights.dropna()

    return weights
