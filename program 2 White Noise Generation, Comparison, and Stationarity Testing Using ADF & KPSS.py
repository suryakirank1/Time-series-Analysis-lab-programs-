import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# -------------------------------
# 1. Load and Prepare Time Series
# -------------------------------
df = pd.read_csv("/content/Electric_Production.csv")

# Convert DATE column to datetime
df['Date'] = pd.to_datetime(df['DATE'])

# Set Date as index
df.set_index('Date', inplace=True)

# Select the target time series
ts = df['IPG2211A2N']

# Show first few rows
print("Original Time Series:")
print(ts.head())

# ---------------------------
# 2. Generate White Noise
# ---------------------------
np.random.seed(42)  # for reproducibility
white_noise = np.random.normal(loc=0, scale=1, size=len(ts))

# ---------------------------
# 3. Visualize both series
# ---------------------------
plt.figure(figsize=(12, 5))
plt.plot(ts, label="Original Time Series")
plt.title("Original Time Series Data")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(white_noise, label="White Noise", color='orange')
plt.title("Generated White Noise")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.show()

# Comparison plot
plt.figure(figsize=(12, 5))
plt.plot(ts[:200], label="Time Series (first 200 points)")
plt.plot(white_noise[:200], label="White Noise (first 200 points)")
plt.title("Comparison: Time Series vs White Noise")
plt.legend()
plt.show()

# ---------------------------
# 4. Augmented Dickey-Fuller Test
# ---------------------------
print("\n--------------- ADF TEST ----------------")
adf_result = adfuller(ts.dropna())
print(f"ADF Statistic : {adf_result[0]}")
print(f"p-value       : {adf_result[1]}")
print("Critical Values:", adf_result[4])

# ---------------------------
# 5. KPSS Test
# ---------------------------
print("\n--------------- KPSS TEST ----------------")
kpss_result = kpss(ts.dropna(), regression='c')  # 'c' = constant trend
print(f"KPSS Statistic : {kpss_result[0]}")
print(f"p-value        : {kpss_result[1]}")
print("Critical Values:", kpss_result[3])
