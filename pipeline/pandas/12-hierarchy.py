#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the DataFrames
df1 = from_file('data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Index the DataFrames on the Timestamp columns
df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

# Filter both DataFrames to include timestamps from 1417411980 to 1417417980
start_timestamp = 1417411980
end_timestamp = 1417417980
df1_filtered = df1.loc[start_timestamp:end_timestamp]
df2_filtered = df2.loc[start_timestamp:end_timestamp]

# Concatenate the DataFrames with keys
df = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp', 'coinbase'])

# Rearrange the MultiIndex levels to have Timestamp as the first level
df = df.swaplevel(i=0, j=1).sort_index(level='Timestamp')

print(df)
