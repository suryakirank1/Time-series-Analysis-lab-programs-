# Title: "ARIMA Modeling and Forecasting of U.S. Electric Production"

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

# i. Initialize the ARIMA model (example: ARIMA(2,1,2))
# p=2 (AR terms), d=1 (first difference), q=2 (MA terms)
arima_model = ARIMA(ts, order=(2,1,2))

# ii. Train the model on the dataset
arima_fit = arima_model.fit()
print("ARIMA(2,1,2) Summary:")
print(arima_fit.summary())

# iii. Generate forecasts
# Forecast for the next 12 periods (months)
forecast_steps = 12
forecast = arima_fit.predict(start=len(ts), end=len(ts)+forecast_steps-1, typ='levels')

# Plot original series and forecast
plt.figure(figsize=(10,5))
plt.plot(ts, label='Original Series')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title("ARIMA(2,1,2) Forecast of Electric Production")
plt.xlabel("Date")
plt.ylabel("Electric Production")
plt.legend()
plt.show()
