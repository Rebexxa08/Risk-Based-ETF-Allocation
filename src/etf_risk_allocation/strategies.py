
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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


def minimum_variance_weights(
    returns: pd.DataFrame,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Berechnet tägliche Minimum-Variance-Zielgewichte.

    - Rollierende Kovarianzmatrix über 252 Handelstage
    - Nur historische Renditen (kein Look-Ahead Bias)
    - Long-only: 0 <= w_i <= 1
    - Summe der Gewichte = 1
    """

    # 1. Anlageklassen auswählen
    assets = ["Equity", "Bonds", "Gold"]
    returns = returns[assets].sort_index().copy()

    # 2. Eingabedaten prüfen
    if lookback < 2 or len(returns) <= lookback:
        raise ValueError("Zu wenige Renditebeobachtungen.")

    if (
        not isinstance(returns.index, pd.DatetimeIndex)
        or returns.index.has_duplicates
    ):
        raise ValueError("Ungültiger Datumsindex.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Fehlende oder ungültige Renditen.")

    # 3. Optimierung vorbereiten
    n_assets = len(assets)

    # Startwerte: Gleichgewichtung
    x0 = np.full(n_assets, 1.0 / n_assets)

    # Keine negativen Gewichte / kein Leverage
    bounds = [(0.0, 1.0)] * n_assets

    # Nebenbedingung: Summe der Gewichte = 1
    constraints = [{
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
        "jac": lambda w: np.ones_like(w)
    }]

    weight_records = []

    # 4. Rollierende Portfoliooptimierung
    for i in range(lookback, len(returns)):

        # Nur die vergangenen 252 Handelstage
        history = returns.iloc[i - lookback:i]

        # Historische Kovarianzmatrix
        covariance = history.cov().to_numpy()

        # 5. Minimum-Variance-Optimierung
        result = minimize(
            fun=lambda w: float(
                252.0 * (w @ covariance @ w)
            ),
            x0=x0,
            jac=lambda w: 504.0 * covariance @ w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": 1e-12,
                "maxiter": 500
            }
        )

        # 6. Optimierungsergebnis prüfen
        if not result.success:
            raise RuntimeError(
                "Optimierung fehlgeschlagen am "
                f"{returns.index[i].date()}: "
                f"{result.message}"
            )

        # Numerische Rundungsfehler korrigieren
        optimal_weights = np.clip(
            result.x, 0.0, 1.0
        )
        optimal_weights /= optimal_weights.sum()

        weight_records.append(optimal_weights)

    # 7. Ergebnisse als DataFrame zurückgeben
    weights = pd.DataFrame(
        weight_records,
        index=returns.index[lookback:],
        columns=assets
    )

    return weights
