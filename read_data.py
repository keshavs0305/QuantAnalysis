import pandas as pd


def read_data():
    file_data = pd.read_excel("cs-1 data set.xlsx")
    file_data.to_csv('file_data.csv', index=False)
