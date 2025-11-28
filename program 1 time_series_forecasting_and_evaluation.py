# ---------------------------------------------
# LAB EXAM : Time Series Forecasting Techniques
# ---------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# ---------------------------------------------
# 1. LOAD YOUR DATASET
# ---------------------------------------------
# Replace with your dataset
# df must have a datetime index & a numeric column
df = pd.read_csv("your_timeseries.csv")

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

ts = df['Value']       # replace 'Value' with your column name



# ---------------------------------------------
# 2. TRAIN–TEST SPLIT
# ---------------------------------------------
n = int(len(ts)*0.8)
train = ts[:n]
test  = ts[n:]


# ---------------------------------------------
# 3. Simple Moving Average (SMA)
# ---------------------------------------------
window = 3   # can use 3 or 5 or 12 depending on data
sma_forecast = ts.rolling(window=window).mean().iloc[n:]


# ---------------------------------------------
# 4. Simple Exponential Smoothing (SES)
# ---------------------------------------------
ses_model = SimpleExpSmoothing(train).fit()
ses_forecast = ses_model.forecast(len(test))


# ---------------------------------------------
# 5. Holt-Winters (Additive Trend + Additive Seasonality)
# ---------------------------------------------
# If your data has monthly seasonality use seasonal_periods=12
hw_model = ExponentialSmoothing(train,
                                trend='add',
                                seasonal='add',
                                seasonal_periods=12).fit()

hw_forecast = hw_model.forecast(len(test))


# ---------------------------------------------
# 6. Evaluation Metrics
# ---------------------------------------------
def evaluate(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = sqrt(mse)
    return mae, mse, rmse

print("===== Evaluation Results =====")
print("SMA :", evaluate(test, sma_forecast))
print("SES :", evaluate(test, ses_forecast))
print("Holt-Winters :", evaluate(test, hw_forecast))


# ---------------------------------------------
# 7. Plot Forecasts
# ---------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(sma_forecast, label="SMA Forecast")
plt.plot(ses_forecast, label="SES Forecast")
plt.plot(hw_forecast, label="Holt-Winters Forecast")
plt.legend()
plt.title("Forecast Comparison")
plt.show()


# ---------------------------------------------
# 8. TREND & SEASONALITY IDENTIFICATION (Optional)
# ---------------------------------------------
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(ts, model='additive', period=12)
result.plot()
plt.show()
