# QuantAnalysis

Problem Satetement:
Based on historical data of S&P 500,
1. Find Weekly Volatility Index
2. Find yearly trade pairs
3. Predict movement (binary classification)

Approach:
Since data is already clean
1. I used Realized Volatility (https://www.wallstreetmojo.com/realized-volatility/) as a measure to calculate requires Volatility Index. Here daily Realized Volatility is calculated by computing the aggregate of returns over the last 5 days(change of price over a week). So this was done by simply doing some aggregations on pandas data frame of given data and then calculating a daily vix index (daily variance sumed over a week period).
2. To calculate the trade pairs, a regression was modelled between 2 stocks close price and an r2 score metric was used to define strength of relation. So top 5 r2 score pairs were calculated for each year.
3. To make a binary classification predictions, LSTM model was created using the input data with 5 time steps backwards as x-data (features) and 1 step forward as y-data (label) to avoid look ahead bias. 5 steps was choosen just to consider time effect for the past 1 week.

Findings:
1. A Realized Volatility index was calculated for each day, which informs about volatility over past 1 week (5 trading days). Based on this index, 10 most volatile and 10 least volatile stocks were found on daily basis. The output is saved in csv file named 'daily_vix_ranks.csv'.
2. After modelling a regression for each pair and sorting them by r2 scores, top 5 pairs were saved in 6 csv files each named like 'top5_(year).csv'
3. For each stock, the LSTM models are saved in 'models' folder by name like 'model_(ticker).h5'. Apart from this, a scaling function was used for each stock, which are also saved in same folder. Both model files and scalar function file will be used while making predictions.

Challenges and Opinions:
1. As per the study done, the ticks data is more useful to calculate VIX index.
2. For calculating pairs, iterating the regression modelling over 500 stocks was a very very time consuming method. It took around 30 mins for each year calculations.

Conclusion:
analysis1.py will give you the Volatility stocks as asked in a csv file name 'daily_vix_ranks.csv'.
analysis2.py will give you the best pairs for each year in a separate csv file named similar to 'top5_(year).csv'.
analysis3.py will train the LSTM model for each stock and save the models into model folder.
make_pred.py file will be used to make the predictions. The test data should contain atleast 6 time step data and the latest time date will be predicted eliminating latest data into input to avoid look ahead bias.
VIX and pairs are found as asked. LSTM models are saved and a script is ready for making predictions.

Retrospective:
In the LSTM, a lot many number of features could be added with respect to technical indicators.


How to run the predictions?
The make_pred.py file is designed to take the test data as input in csv format. A filename (ex. 'testdata.csv') is to be given as input when make_pred.py is run. The test data is supposed to have Open, High, Low, Close, Volume and ticker symbol columns in a time series manner (similar to data provided). The predictions are made for the latest day in the data. And a 5 step data before the latest day is given to the LSTM model avoiding the look ahead bias.

note: Ignore all the .ipynb files
