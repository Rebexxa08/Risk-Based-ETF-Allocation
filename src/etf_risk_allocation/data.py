
from pathlib import Path
from datetime import date
import json

import pandas as pd
import yfinance as yf


# --------------------------------------------------
# 1. Konfiguration
# --------------------------------------------------

TICKERS = {
    "Equity": "IUSQ.DE",
    "Bonds": "EUNA.DE",
    "Gold": "4GLD.DE",
}

START_DATE = "2018-01-01"
BACKTEST_START = "2019-01-01"
LOOKBACK = 252

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def first_day_current_month():
    """Exklusive Obergrenze: aktueller Monat wird nicht geladen."""
    today = date.today()
    return today.replace(day=1).isoformat()


# --------------------------------------------------
# 2. Kurse herunterladen
# --------------------------------------------------

def download_prices():
    """Lädt EUR-notierte, adjustierte Schlusskurse."""

    price_series = {}

    for asset, ticker in TICKERS.items():
        print(f"Lade {asset}: {ticker}")

        df = yf.download(
            ticker,
            start=START_DATE,
            end=first_day_current_month(),
            interval="1d",
            auto_adjust=True,
            actions=True,
            multi_level_index=False,
            progress=False,
        )

        if df.empty:
            raise ValueError(f"Keine Daten für {ticker}")

        if "Close" not in df.columns:
            raise ValueError(f"Close fehlt für {ticker}")

        close = df["Close"].copy()
        close.index = pd.to_datetime(close.index)
        close.index = close.index.tz_localize(None).normalize()

        close = close.sort_index()
        close = close[~close.index.duplicated(keep="last")]
        price_series[asset] = close.rename(asset)

    # Outer Join: fehlende Börsentage zunächst sichtbar lassen
    return pd.concat(
    price_series.values(),
    axis=1,
    sort=True
)

# --------------------------------------------------
# 3. Datenqualität prüfen
# --------------------------------------------------

def check_data_quality(prices):
    """Prüft Daten auf Lücken und auffällige Werte."""

    print("\n--- Datenqualität ---")

    for asset in prices.columns:
        series = prices[asset]
        first = series.first_valid_index()
        last = series.last_valid_index()

        if first is None:
            raise ValueError(f"Keine Preise für {asset}")

        active = series.loc[first:last]

        missing = active.isna().sum()
        invalid = (active.dropna() <= 0).sum()

        print(f"\n{asset}")
        print(f"Erster Kurs: {first.date()}")
        print(f"Letzter Kurs: {last.date()}")
        print(f"Fehlende Werte: {missing}")
        print(f"Ungültige Preise: {invalid}")

        if invalid:
            raise ValueError(f"Ungültige Kurse bei {asset}")

        if missing:
            print("WARNUNG: Fehlende Handelstage prüfen!")

        # Auffällige tägliche Veränderungen nur auf
        # tatsächlich aufeinanderfolgenden Beobachtungen
        returns = active.pct_change(fill_method=None)
        outliers = returns[returns.abs() > 0.20].dropna()

        if not outliers.empty:
            print("WARNUNG: Kursbewegungen über 20 %:")
            print(outliers.to_string())


# --------------------------------------------------
# 4. Bereinigung und Renditeberechnung
# --------------------------------------------------

def prepare_data(prices):
    """Synchronisiert Kurse und berechnet Tagesrenditen."""

    # Vollständige gemeinsame Kursbeobachtungen
    clean_prices = prices.dropna(how="any").copy()

    if clean_prices.empty:
        raise ValueError("Keine gemeinsamen Handelstage.")

    # Keine stillschweigende Forward-Fill-Imputation
    returns = clean_prices.pct_change(fill_method=None)
    returns = returns.iloc[1:]

    # Prüfen, ob alle Renditen endlich sind
    import numpy as np

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Nicht-endliche Renditen entdeckt.")

    # Mindestens 252 historische Renditen vor
    # dem ersten tatsächlich möglichen Backtesttag
    candidates = returns.loc[BACKTEST_START:]

    eligible = [
        day for day in candidates.index
        if returns.index.get_loc(day) >= LOOKBACK
    ]

    if not eligible:
        raise ValueError(
            "Nicht genügend Historie für den Backtest."
        )

    first_backtest_day = eligible[0]

    print("\n--- Bereinigter Datensatz ---")
    print(f"Beobachtungen (Preise): {len(clean_prices)}")
    print(f"Beobachtungen (Renditen): {len(returns)}")
    print(f"Backtest möglich ab: {first_backtest_day.date()}")

    return clean_prices, returns, first_backtest_day


# --------------------------------------------------
# 5. Daten speichern
# --------------------------------------------------

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_prices = download_prices()
    check_data_quality(raw_prices)

    clean_prices, returns, first_day = prepare_data(
        raw_prices
    )

    raw_prices.to_csv(RAW_DIR / "prices_raw.csv")

    clean_prices.to_csv(
        PROCESSED_DIR / "prices_eur.csv"
    )

    returns.to_csv(
        PROCESSED_DIR / "returns_daily.csv"
    )

    metadata = {
        "tickers": TICKERS,
        "currency": "EUR",
        "start_date": START_DATE,
        "download_end_exclusive": first_day_current_month(),
        "first_backtest_day": str(first_day.date()),
        "lookback_returns": LOOKBACK,
        "price_adjustment": "yfinance auto_adjust=True",
    }

    with open(
        PROCESSED_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nDateien erfolgreich gespeichert.")


if __name__ == "__main__":
    main()
