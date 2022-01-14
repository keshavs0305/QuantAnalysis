import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def read_data():
    file_data = pd.read_csv('file_data.csv')
    tickers = file_data.Name.unique()
    return file_data, tickers


def make_vix_col(file_data):
    file_data['VIX'] = pd.Series(
        index=[i for i in range(len(file_data))],
        dtype='float64'
    )
    file_data.replace(np.NaN, 0, inplace=True)
    return file_data


def cal_vix():
    x, y = read_data()
    file_data = make_vix_col(x)

    vix_data = make_vix_col(x)[0:0]
    for ticker in y:
        ticker_data = file_data.groupby('Name').get_group(ticker)
        vix_cal = ((np.log10(ticker_data.drop(['Name'], axis=1).close).diff() ** 2).rolling(5).sum() ** 0.5)
        vix_cal.replace(np.nan, 0, inplace=True)
        ticker_data['VIX'] += vix_cal
        vix_data = vix_data.append(ticker_data, ignore_index=True)
    return vix_data


def rank_volatality():
    ret = {'date': [], 'least_vol': [], 'most_vol': []}
    vix_data = cal_vix()
    date_wise_data = vix_data.groupby('date')
    dates = vix_data.date.unique()
    for date in dates:
        date_data = date_wise_data.get_group(date)
        date_data['scaled_vix'] = StandardScaler(with_mean=False).fit_transform(date_data.VIX.to_numpy().reshape(-1, 1))
        ret['date'].append(date)
        ret['least_vol'].append(list(date_data.sort_values('scaled_vix').head(10).Name))
        ret['most_vol'].append(list(date_data.sort_values('scaled_vix', ascending=False).head(10).Name))
    pd.DataFrame(ret).to_csv('daily_vix_ranks.csv')


rank_volatality()
