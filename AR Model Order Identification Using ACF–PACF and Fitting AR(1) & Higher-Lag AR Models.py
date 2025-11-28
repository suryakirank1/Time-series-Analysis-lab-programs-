import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# ------------------------------------
# Load Dataset
# ------------------------------------
df = pd.read_csv("/content/Electric_Production.csv")

df['Date'] = pd.to_datetime(df['DATE'])
df.set_index('Date', inplace=True)

ts = df['IPG2211A2N']

# ------------------------------------
# 1. Examine ACF and PACF to determine AR order
# ------------------------------------
plt.figure(figsize=(12,5))
plot_acf(ts, lags=40)
plt.title("ACF Plot for AR Order Identification")
plt.show()

plt.figure(figsize=(12,5))
plot_pacf(ts, lags=40, method="ywm")
plt.title("PACF Plot for AR Order Identification")
plt.show()

# ------------------------------------
# Helper functions for evaluation
# ------------------------------------
def evaluate(y_true, y_pred):
    return {
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# Train-test split (simple)
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# ------------------------------------
# 2. Fit AR(1) Model
# ------------------------------------
ar1_model = AutoReg(train, lags=1).fit()
ar1_pred = ar1_model.predict(start=len(train), end=len(train)+len(test)-1)

print("\nAR(1) Performance:")
print(evaluate(test, ar1_pred))

# ------------------------------------
# 3. Fit Higher-Lag AR Models (AR-2, AR-3, AR-5)
# ------------------------------------
for lag in [2, 3, 5]:
    model = AutoReg(train, lags=lag).fit()
    pred = model.predict(start=len(train), end=len(train)+len(test)-1)

    print(f"\nAR({lag}) Performance:")
    print(evaluate(test, pred))
