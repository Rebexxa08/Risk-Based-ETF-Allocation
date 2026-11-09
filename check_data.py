
import pandas as pd

# Ursprüngliche Daten laden
prices = pd.read_csv(
    "data/raw/prices_raw.csv",
    index_col=0,
    parse_dates=True
)

# Zeilen mit mindestens einem fehlenden Kurs
missing = prices[prices.isna().any(axis=1)]

print("Fehlende Kurswerte:")
print(missing.to_string())

# Anzahl fehlender Werte pro Anlageklasse
print("\nFehlende Werte je Anlageklasse:")
print(prices.isna().sum())

# Beobachtete Kursveränderungen über 20 %
print("\nAuffällige Kursbewegungen:")
returns = prices.pct_change(fill_method=None)

for asset in returns.columns:
    unusual = returns.loc[
        returns[asset].abs() > 0.20, asset
    ]
    print(f"\n{asset}:")
    print(unusual.to_string())
