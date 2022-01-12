import pandas as pd
import numpy as np

file_data = pd.read_csv('file_data.csv')
tickers = file_data.Name.unique()

file_data['VIX'] = pd.Series(
    index=[i for i in range(len(file_data))],
    dtype='int64'
)
file_data.replace(np.nan, 0, inplace=True)
for ticker in tickers:
    ticker_data = file_data.groupby('Name').get_group(ticker)
    vix_cal = (np.log10(ticker_data.drop(['Name'], axis=1).close).diff() ** 2).rolling(5).sum() ** 0.5
    file_data['VIX'] = file_data.VIX + vix_cal

date_wise_data = file_data.groupby('date')
dates = file_data.date.unique()
for date in dates[:5]:
    date_data = file_data.groupby('date').get_group(date)
    print(date_data.head())
    print(date_data.tail())
print()
