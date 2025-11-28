#program 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("/content/Electric_Production.csv")

# Convert DATE column to datetime and set index
df['Date'] = pd.to_datetime(df['DATE'])
df.set_index('Date', inplace=True)

# Time series column
ts = df['IPG2211A2N']

# -----------------------------
# 1. Detect Trend Using Moving Averages
# -----------------------------
ts_ma_12 = ts.rolling(window=12).mean()   # 12-month moving average
ts_ma_24 = ts.rolling(window=24).mean()   # 24-month moving average

plt.figure(figsize=(12,5))
plt.plot(ts, label="Original Series")
plt.plot(ts_ma_12, label="12-Month Moving Avg", linewidth=2)
plt.plot(ts_ma_24, label="24-Month Moving Avg", linewidth=2)
plt.title("Trend Detection Using Moving Averages")
plt.legend()
plt.show()

# -----------------------------
# 2. ACF and PACF Plots
# -----------------------------
plt.figure(figsize=(12,5))
plot_acf(ts, lags=40)
plt.title("ACF Plot")
plt.show()

plt.figure(figsize=(12,5))
plot_pacf(ts, lags=40, method="ywm")
plt.title("PACF Plot")
plt.show()
