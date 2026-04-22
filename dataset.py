"""
Netflix Stock Price Data Generator
Generates realistic NFLX stock data for analysis
Author: Srinath M
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# Netflix IPO was 2002, went from ~$15 to $700+ by 2022
# We simulate 2017-2022 (5 years) - realistic for IEEE 2022 paper

dates = pd.date_range(start='2017-01-01', end='2022-12-31', freq='B')  # Business days only
n = len(dates)

# Simulate realistic Netflix price movement
# Starting price ~$140 in Jan 2017, peaked ~$700 in Nov 2021, dropped ~$190 by end 2022
base_prices = np.linspace(140, 190, n)

# Add realistic trend phases
trend = np.zeros(n)
for i in range(n):
    progress = i / n
    if progress < 0.20:     # 2017: Growth phase
        trend[i] = progress * 800
    elif progress < 0.45:   # 2018-2019: Steady growth
        trend[i] = 160 + progress * 600
    elif progress < 0.55:   # 2020: COVID dip then recovery
        trend[i] = 350 + np.sin(progress * 20) * 80
    elif progress < 0.75:   # 2021: Peak bull run
        trend[i] = 400 + (progress - 0.55) * 1500
    else:                   # 2022: Major correction
        trend[i] = 700 - (progress - 0.75) * 2000

# Add realistic noise (daily volatility ~2%)
daily_returns = np.random.normal(0.0003, 0.022, n)
price_multiplier = np.cumprod(1 + daily_returns)
close_prices = np.abs(base_prices + trend) * price_multiplier / price_multiplier[0]
close_prices = np.clip(close_prices, 50, 750)

# OHLV data
open_prices  = close_prices * np.random.uniform(0.988, 1.012, n)
high_prices  = np.maximum(open_prices, close_prices) * np.random.uniform(1.001, 1.025, n)
low_prices   = np.minimum(open_prices, close_prices) * np.random.uniform(0.975, 0.999, n)
volume       = np.random.randint(3_000_000, 25_000_000, n).astype(float)

# Add earnings day volume spikes (quarterly)
for i in range(0, n, 63):
    if i < n:
        volume[i] *= np.random.uniform(2.5, 5.0)

df = pd.DataFrame({
    'Date'   : dates,
    'Open'   : np.round(open_prices,  2),
    'High'   : np.round(high_prices,  2),
    'Low'    : np.round(low_prices,   2),
    'Close'  : np.round(close_prices, 2),
    'Volume' : volume.astype(int)
})

df['Symbol'] = 'NFLX'
df.to_csv("Netflix_stock_data.csv", index=False)

print("✅ Netflix Stock Dataset Generated!")
print(f"   Date Range  : {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"   Total Rows  : {len(df)} trading days")
print(f"   Columns     : {list(df.columns)}")
print(f"   Price Range : ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
print(f"   Avg Volume  : {df['Volume'].mean():,.0f} shares/day")
print("\nSample Data:")
print(df.head(10).to_string(index=False))