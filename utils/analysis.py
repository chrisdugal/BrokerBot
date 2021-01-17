"""
Perform analysis tasks using algorithmic trading strategies and various ML models
"""

import yfinance as yf
from datetime import datetime as dt, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import math

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# testing purposes
from pprint import pprint


def is_valid_ticker(ticker):
    """ check if a ticker is valid """

    stock = yf.Ticker(ticker)
    try:
        stock.info
        return True
    except:
        return False


def get_ticker_bid_ask(ticker):
    """ get ticker bid-ask and currency """

    stock = yf.Ticker(ticker)
    info = stock.info

    return f'{format(info["bid"], ".2f")} - {format(info["ask"], ".2f")} {info["currency"]}'


def sma_crossover(stock, short_sma, long_sma):
    """
    Checks for a custom simple moving average crossover of given ticker
    """

    # get sma period lengths
    SMAs = [short_sma, long_sma]

    # pull the necessary amount of data, multiply long sma period by 3 to account for non-trading days
    start_date = dt.now() - timedelta(days=int(long_sma * 3))
    data = yf.download(stock, start=start_date, end=dt.now())

    # calculate SMAs
    for i in SMAs:
        data["SMA_" + str(i)] = data.loc[:, "Close"].rolling(window=i).mean()

    # isolate short and long SMA data
    SMA_short = data["SMA_" + str(short_sma)]
    SMA_long = data["SMA_" + str(long_sma)]

    # check for crossover
    prev_state = SMA_short.iloc[-2] > SMA_long.iloc[-2]
    curr_state = SMA_short.iloc[-1] > SMA_long.iloc[-1]

    # return result
    if prev_state != curr_state:
        if curr_state:
            return "bullish crossover"
        else:
            return "bearish crossover"
    else:
        if curr_state:
            return "no crossover, bullish zone"
        else:
            return "no crossover, bearish zone"


def risk_vs_return(stocks):
    """
    Generates a 1Y risk vs return plot for given stocks
    """

    plt.style.use("seaborn")

    # pull data for past year
    data = yf.download(stocks, period="1y")

    # isolate close prices
    data = data.loc[:, "Close"]

    # look at percentage change
    pct = data.pct_change().dropna()
    data = pct.describe().T.loc[:, ["mean", "std"]]

    # scale data for trading year
    data["mean"] = data["mean"] * 252
    data["std"] = data["std"] * np.sqrt(252)

    # plot relative risk vs return for all stocks
    data.plot.scatter(x="std", y="mean", figsize=(12, 8), s=50, fontsize=15)
    for i in data.index:
        plt.annotate(
            i, xy=(data.loc[i, "std"] + 0.002, data.loc[i, "mean"] + 0.002), size=15
        )
    plt.ylabel("return (avg % change)", fontsize=18)
    plt.xlabel("risk (% change std deviation)", fontsize=18)
    plt.title("Risk vs. Return (1 year)", fontsize=20)
    plt.savefig("temp/risk_vs_return.png")

    return "temp/risk_vs_return.png"


def svm_prediction(stock):
    """
    predict future stock price using a Support Vector Machine (Regressor)
    """

    # pull stock data for past year
    data = yf.download(stock, period="1y")

    # isolate adj close price
    data = data[["Adj Close"]]

    # how many days in the future we want to predict
    days_ahead = 1

    # create a column for dependent variable, shift adj close price up by num of days
    data["Prediction"] = data[["Adj Close"]].shift(-days_ahead)

    ## create the independent dataset

    # convert dataframe to np array
    x = np.array(data[["Adj Close"]])

    # remove trailing rows (needs to match dependent dataset)
    x = x[:-days_ahead]

    ## create the dependent dataset

    # convert dataframe to np array
    y = np.array(data["Prediction"])

    # remove trailing rows - will be NaN
    y = y[:-days_ahead]

    # split the data - 80/20 training/testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # create and train the Support Vector Machine (Regressor)
    svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)

    # test model - score (coefficient of determination r^2 of prediction)
    svm_confidence = svr_rbf.score(x_test, y_test)

    # Create and train the Linear Regression Model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # test model - score (coefficient of determination r^2 of prediction)
    lr_confidence = lr.score(x_test, y_test)

    # set x_forecast equal to trailing rows of the original dataset from adj close column
    x_forecast = np.array(data[["Adj Close"]])[-days_ahead:]

    # calculate predictions
    svm_prediction = svr_rbf.predict(x_forecast)[days_ahead - 1]
    svm_confidence_pct = svm_confidence * 100

    svm_pred = [
        format(svm_prediction, ".2f"),
        format(svm_confidence_pct, ".3f"),
    ]

    lr_prediction = lr.predict(x_forecast)[days_ahead - 1]
    lr_confidence_pct = lr_confidence * 100

    lr_pred = [
        format(lr_prediction, ".2f"),
        format(lr_confidence_pct, ".3f"),
    ]

    # return results
    return svm_pred, lr_pred


def ann_prediction(stock):
    """
    predict future stock price using an artificial recurrent neural network (LSTM)
    """

    # pull the maximum amount of stock data
    data = yf.download(stock, period="5y")

    # create numpy array with close price
    dataset = np.array(data[["Close"]])

    # isolate rows to train the model on - 80%
    training_data_len = math.ceil(len(dataset) * 0.8)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # create the training dataset
    train_data = scaled_data[0:training_data_len, :]

    # split the data, x = feature, y = target
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])

    # convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # reshape the data to fit LSTM requirements
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # create the testing dataset
    # use 60 days worth of data
    test_data = scaled_data[training_data_len - 60 :, :]

    # create the datasets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, 0])

    # convert the data to a numpy array
    x_test = np.array(x_test)

    # reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # calculate RMSPE - root mean square percentage error
    rmspe = np.sqrt(np.mean(np.square(((y_test - predictions) / y_test)), axis=0))

    # plot the data - testing
    """
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid["Predictions"] = predictions

    plt.figure(figsize=(16, 8))
    plt.title("ANN - " + yf.Ticker(stock).info["shortName"])
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price", fontsize=18)
    plt.plot(train["Close"])
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Train", "Validation", "Predictions"], loc="lower right")
    plt.savefig("temp/ann.png")
    """

    # get new copy of stock data
    new_data = yf.download(stock, period="5y").filter(["Close"])

    # get the last 60 days of close prices in an array
    last_60 = new_data[-60:].values

    # scale the data from 0 to 1
    last_60_scaled = scaler.transform(last_60)

    # create empty list
    x_test = []

    # append last 60 days
    x_test.append(last_60_scaled)

    # convert to numpy array to use with LSTM model
    x_test = np.array(x_test)

    # reshape data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # get predicted price
    prediction = scaler.inverse_transform(model.predict(x_test))
    prediction = prediction[0][0]

    # calculate confidence from RMSPE
    confidence_pct = 100 - rmspe[0] * 100

    ann_pred = [format(prediction, ".2f"), format(confidence_pct, ".3f")]

    # return results
    return ann_pred
