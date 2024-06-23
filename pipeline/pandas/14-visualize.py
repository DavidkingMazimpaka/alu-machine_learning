#!/usr/bin/env python3

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the DataFrame
df = from_file('data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the column Weighted_Price
df = df.drop(columns=['Weighted_Price'])

# Rename the column Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the DataFrame on Date
df = df.set_index('Date')

# Fill missing values as specified
df['Close'] = df['Close'].fillna(method='ffill')
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter the data to include only entries from 2017 and beyond
df = df.loc['2017-01-01':]

# Resample the data at daily intervals and aggregate the values as specified
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
plt.figure(figsize=(14, 10))

# Plot each column separately
plt.subplot(3, 1, 1)
plt.plot(df_daily.index, df_daily['High'], label='High', color='blue')
plt.plot(df_daily.index, df_daily['Low'], label='Low', color='red')
plt.plot(df_daily.index, df_daily['Open'], label='Open', color='green')
plt.plot(df_daily.index, df_daily['Close'], label='Close', color='orange')
plt.legend(loc='best')
plt.title('Bitcoin Prices (Daily)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')

plt.subplot(3, 1, 2)
plt.plot(df_daily.index, df_daily['Volume_(BTC)'], label='Volume (BTC)', color='purple')
plt.legend(loc='best')
plt.title('Bitcoin Volume (BTC) (Daily)')
plt.xlabel('Date')
plt.ylabel('Volume (BTC)')

plt.subplot(3, 1, 3)
plt.plot(df_daily.index, df_daily['Volume_(Currency)'], label='Volume (Currency)', color='brown')
plt.legend(loc='best')
plt.title('Bitcoin Volume (Currency) (Daily)')
plt.xlabel('Date')
plt.ylabel('Volume (USD)')

plt.tight_layout()
plt.show()
