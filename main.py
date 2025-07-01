import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('data/Nat_Gas (1).csv', parse_dates=['Dates'])
df = df.sort_values('Dates')
df.set_index('Dates', inplace=True)

# Plot original data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Prices'], label='Historical')
plt.title('Monthly Natural Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price ($/MMBtu)')
plt.grid(True)
plt.legend()
plt.show()

# Fit SARIMA model
model = SARIMAX(df['Prices'], 
                order=(1,1,1),           # ARIMA part: (p,d,q)
                seasonal_order=(1,1,1,12), # Seasonal: (P,D,Q,s)
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

# Forecast next 12 months
forecast = results.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Create future date index
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    'Dates': future_dates,
    'Forecasted_Prices': forecast_mean.values
})

# Plot historical + forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Prices'], label='Historical')
plt.plot(forecast_df['Dates'], forecast_df['Forecasted_Prices'], '--o', color='darkorange', label='Forecasted')

plt.fill_between(forecast_df['Dates'],
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='orange', alpha=0.2)

plt.title('Natural Gas Prices (Historical + 12‑Month SARIMA Forecast)')
plt.xlabel('Date')
plt.ylabel('Price ($/MMBtu)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Format x-axis
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Show forecast table
print("\nForecasted Prices (SARIMA - next 12 months):")
print(forecast_df.to_string(index=False))

# Optionally save
forecast_df.to_csv('data/sarima_forecast_next_12_months.csv', index=False)