import pandas as pd
import plotly.express as px
from prophet import Prophet

# Load dataset
df = pd.read_csv("covid_data.csv")

# Filter India data
india_df = df[df['location'] == 'India'][['date', 'total_cases', 'total_deaths']]
india_df.rename(columns={'date':'Date', 'total_cases':'Confirmed', 'total_deaths':'Deaths'}, inplace=True)
india_df['Date'] = pd.to_datetime(india_df['Date'])

# Plot Confirmed and Deaths
fig = px.line(india_df, x='Date', y=['Confirmed','Deaths'],
              title="India COVID‑19 Trends", labels={'value':'Count','variable':'Category'})
fig.show()

# Forecast Confirmed cases 7 days ahead
prophet_df = india_df[['Date','Confirmed']].rename(columns={'Date':'ds','Confirmed':'y'})
m = Prophet()
m.fit(prophet_df)
future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)

# Plot forecast
fig2 = m.plot(forecast)
fig2.show()
