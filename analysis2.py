import pandas as pd
import numpy as np

file_data = pd.read_csv('file_data.csv')

year_wise_groups = file_data.loc[file_data.date.apply(lambda x: x[:4] == '2013')].groupby('Name')

ret = {'stock1': [], 'stock2': [], 'corr': []}
tickers = list(year_wise_groups.Name.unique().index)
for i in range(len(tickers)-1):
    g1 = year_wise_groups.get_group(tickers[i])[['date','close']]
    for j in range(i+1,len(tickers)):
        g2 = year_wise_groups.get_group(tickers[j])[['date','close']]
        a_b = pd.merge(g1,g2,on='date',how='inner')
        a_b = a_b.drop(['date'],axis=1).diff()
        ret['stock1'].append(tickers[i])
        ret['stock2'].append(tickers[j])
        ret['corr'].append(a_b.close_x.corr(a_b.close_y))

pd.DataFrame(ret)