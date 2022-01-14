import pandas as pd
import pickle
from keras.models import load_model


def read_test_data():
    file_name = input("Input the csv test data file name: ")
    return pd.read_csv(file_name)


def make_pred():
    data = read_test_data()
    ticker = data.Name.unique()[0]
    scalar = pickle.load(open('models/scaler_'+ticker+'.pkl', 'rb'))
    x = scalar.transform(data[['open', 'high', 'low', 'close', 'volume']])[-6:-1].reshape(1,5,5)

    pred = load_model('models/model_'+ticker+'.h5').predict(x)[0][0]

    if pred <= 0.4:
        return 0
    elif pred >= 0.6:
        return 1
    else:
        return 1


make_pred()
