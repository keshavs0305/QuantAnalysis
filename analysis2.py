import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def read_data():
    return pd.read_csv('file_data.csv')


def year_wise_data(year):
    file_data = read_data()
    return file_data.loc[file_data.date.apply(lambda x: x[:4] == year)].groupby('Name')


def find_r2(df):
    scaled = StandardScaler().fit_transform(df)
    model = LinearRegression()
    model.fit(scaled[:, 0].reshape(-1, 1), scaled[:, 1].reshape(-1, 1))
    return model.score(scaled[:, 0].reshape(-1, 1), scaled[:, 1].reshape(-1, 1))


def cal_top(tickers, year_wise_groups):
    ret = {'stock1': [], 'stock2': [], 'r2_score': []}
    for i in range(len(tickers) - 1):
        g1 = year_wise_groups.get_group(tickers[i])[['date', 'close']]
        for j in range(i + 1, len(tickers)):
            g2 = year_wise_groups.get_group(tickers[j])[['date', 'close']]
            a_b = pd.merge(g1, g2, on='date', how='inner')
            a_b = a_b.drop(['date'], axis=1).diff()
            ret['stock1'].append(tickers[i])
            ret['stock2'].append(tickers[j])
            ret['r2_score'].append(find_r2(a_b.drop([0])))
    return ret


def find_top():
    years = ['2013', '2014', '2015', '2016', '2017', '2018']
    for year in years:
        year_wise_groups = year_wise_data(year)
        tickers = list(year_wise_groups.Name.unique().index)
        ret = cal_top(tickers, year_wise_groups)
        pd.DataFrame(ret).sort_values('r2_score', ascending=False)[:5].to_csv('top5_' + year + '.csv')


find_top()
