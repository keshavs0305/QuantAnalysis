import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

file_data = pd.read_csv('file_data.csv')
tickers = file_data.Name.unique()

file_data['VIX'] = pd.Series(
    index=[i for i in range(len(file_data))],
    dtype='float64'
)
file_data.replace(np.NaN, 0, inplace=True)


vix_data = file_data[0:0]
for ticker in tickers:
    ticker_data = file_data.groupby('Name').get_group(ticker)
    vix_cal = ((np.log10(ticker_data.drop(['Name'], axis=1).close).diff() ** 2).rolling(5).sum() ** 0.5)
    vix_cal.replace(np.nan, 0, inplace=True)
    ticker_data['VIX'] += vix_cal
    vix_data = vix_data.append(ticker_data, ignore_index=True)


ret = {}
date_wise_data = vix_data.groupby('date')
dates = vix_data.date.unique()
i = 0
for date in dates:
    date_data = date_wise_data.get_group(date)
    date_data['scaled_vix'] = StandardScaler(with_mean=False).fit_transform(date_data.VIX.to_numpy().reshape(-1, 1))
    ret[i] = {}
    ret[i]['least_volatile'] = date_data.sort_values('scaled_vix').head(10)
    ret[i]['most_volatile'] = date_data.sort_values('scaled_vix', ascending=False).head(10)
    i += 1
pd.DataFrame(ret).to_csv('analytic1.csv')
