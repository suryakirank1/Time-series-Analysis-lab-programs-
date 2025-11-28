# Title: "ARMA Modeling and Forecasting of U.S. Electric Production"

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("/content/Electric_Production.csv")

# Convert DATE column to datetime and set as index
df['Date'] = pd.to_datetime(df['DATE'])
df.set_index('Date', inplace=True)

# Time series
ts = df['IPG2211A2N']

# i. Initialize the ARMA model (ARMA(p,q) ~ ARIMA with d=0)
# Let's use ARMA(2,1) as an example
arma_model = ARIMA(ts, order=(2,0,1))

# ii. Train the model on the dataset
arma_fit = arma_model.fit()
print("ARMA(2,1) Summary:")
print(arma_fit.summary())

# iii. Generate forecasts
# Forecast for the next 12 periods (months)
forecast_steps = 12
forecast = arma_fit.predict(start=len(ts), end=len(ts)+forecast_steps-1)

# Plot original series and forecast
plt.figure(figsize=(10,5))
plt.plot(ts, label='Original Series')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title("ARMA(2,1) Forecast of Electric Production")
plt.xlabel("Date")
plt.ylabel("Electric Production")
plt.legend()
plt.show()
