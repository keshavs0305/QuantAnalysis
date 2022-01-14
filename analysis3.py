import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pickle import dump
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")


def read_data():
    return pd.read_csv('file_data.csv')


def prep_data(arr, label):
    x, y = list(), list()
    for i in range(len(arr[:-1])-5):
        x.append(arr[i:i+5])
        y.append(label[i+6])
    return np.array(x), np.array(y)


def def_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(5, 5)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def train(x, y, model, ticker):
    model.fit(x, y, epochs=200)
    model.save('models/model_' + ticker + '.h5')


def train_all():
    raw_data = read_data()
    ticker_group = raw_data.groupby('Name')

    for ticker in raw_data.Name.unique():
        ticker_data = ticker_group.get_group(ticker)
        ticker_data['label'] = (ticker_data.close > ticker_data.open).apply(lambda t: int(t))

        scalar = StandardScaler()
        scale_data = scalar.fit_transform(ticker_data[['open', 'high', 'low', 'close', 'volume']])
        dump(scalar, open('models/scaler_'+ticker+'.pkl', 'wb'))

        x, y = prep_data(scale_data, ticker_data['label'].to_numpy())

        train(x, y, def_model(), ticker)

train_all()
