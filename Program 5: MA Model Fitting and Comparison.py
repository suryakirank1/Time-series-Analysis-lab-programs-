# Title: "Moving Average Modeling of U.S. Electric Production"

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("/content/Electric_Production.csv")

# Convert DATE column to datetime and set as index
df['Date'] = pd.to_datetime(df['DATE'])
df.set_index('Date', inplace=True)

# Time series
ts = df['IPG2211A2N']

# i. Plot ACF and PACF
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_acf(ts, lags=30, ax=plt.gca(), title="ACF of Electric Production")
plt.subplot(1,2,2)
plot_pacf(ts, lags=30, ax=plt.gca(), title="PACF of Electric Production")
plt.tight_layout()
plt.show()

# ii. Fit an MA(1) model
ma1_model = ARIMA(ts, order=(0,0,1)).fit()
print("MA(1) Summary:")
print(ma1_model.summary())

# iii. Fit a higher lag MA model (e.g., MA(3))
ma3_model = ARIMA(ts, order=(0,0,3)).fit()
print("\nMA(3) Summary:")
print(ma3_model.summary())

# iv. Compare performances using MSE
ma1_pred = ma1_model.fittedvalues
ma3_pred = ma3_model.fittedvalues

mse_ma1 = mean_squared_error(ts, ma1_pred)
mse_ma3 = mean_squared_error(ts, ma3_pred)

print(f"\nMean Squared Error - MA(1): {mse_ma1:.4f}")
print(f"Mean Squared Error - MA(3): {mse_ma3:.4f}")

if mse_ma3 < mse_ma1:
    print("Higher lag MA(3) model performs better.")
else:
    print("MA(1) model performs better.")
