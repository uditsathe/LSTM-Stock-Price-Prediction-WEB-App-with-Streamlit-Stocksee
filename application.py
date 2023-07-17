import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import streamlit as st

st.markdown("___")
st.markdown("<h1 style='text-align: center; color: red;'>StockSee</h1>", unsafe_allow_html=True)
st.title("Stock Price Predictor")
Ticker = st.text_input("Enter Stock Ticker", '^NSEI')
# ^NSEI is set as default input

frame = yf.Ticker(Ticker)
while True:
    try:
        company = frame.info['longName']
        break
    except:
        st.subheader("Entered Ticker value is invalid.  Try again...")
        quit()
# tickName = yf.Ticker(Ticker)
data = frame.history(period="max")
data = data.tail(2520)
data10Year = data
data.dropna(axis=0, inplace=True)
data = data.drop(columns=['Dividends', 'Stock Splits'])
lastFive = data.tail()
figTrend = plt.figure(figsize=(14, 8))
plt.plot(lastFive.Close)
st.subheader("5 day performance of "+company)
st.pyplot(figTrend)

#computing moving averages of 100 days and 200 days
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()

#now plotting 100 day moving average
fig100ma , bx= plt.subplots()
bx.plot(ma100.tail(300), color='blue', label='100 DMA')
bx.plot(data.tail(300).Close, color='black', label='Close')
bx.legend(loc='upper left')
st.subheader("100 Day Moving Average "+company)
st.pyplot(fig100ma)

#now plotting a comparison graph of 100 and 200 day moving average
figMaVSMa, ax = plt.subplots()
ax.plot(data.tail(300).Close, color='black', label='Close')
ax.plot(ma100.tail(300), color='red', label='100 DMA')
ax.plot(ma200.tail(300), color='green', label='200 DMA')
ax.legend(loc='upper left')
st.subheader("100 Day VS 200 Day Moving Average"+company)
st.pyplot(figMaVSMa)

#outputting a table that shows 10 year data of the stock
st.subheader("10 year performance table of "+company)
st.write(data10Year.describe())  # makes a table of passed data

#data pre-processing follows-
data['dma'] = data['Close'] - data['Close'].shift(5)
data['dma_positive'] = np.where(data['dma'] > 0, 1, 0)
scaler = MinMaxScaler(feature_range=(0, 1))
X = data[['Open', 'Low', 'High', 'Volume', 'dma_positive']].copy()
y = data['Close'].copy()
X[['Open', 'Low', 'High', 'Volume', 'dma_positive']] = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))


def load_data(X, seq_len, train_size=0.8):
    amount_of_features = X.shape[1]
    X_mat = X.values
    sequence_length = seq_len + 1
    datanew = []

    for index in range(len(X_mat) - sequence_length):
        datanew.append(X_mat[index: index + sequence_length])

    datanew = np.array(datanew)
    train_split = int(round(train_size * datanew.shape[0]))
    train_data = datanew[:train_split, :]

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1][:, -1]

    X_test = datanew[train_split:, :-1]
    y_test = datanew[train_split:, -1][:, -1]

    X_train = np.reshape(
        X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(
        X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return X_train, y_train, X_test, y_test


window = 22
X['close'] = y
X_train, y_train, X_test, y_test = load_data(X, window)
#loading model and testing it with test data
model = load_model(
    'keras_newModel.h5')
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([y_test])
r2_tr = 1 - np.sum((trainY[0] - trainPredict[:, 0])**2) / \
    np.sum((trainY[0] - np.mean(trainPredict[:, 0]))**2)
r2_te = 1 - np.sum((testY[0] - testPredict[:, 0])**2) / \
    np.sum((testY[0] - np.mean(testPredict[:, 0]))**2)
plot_predicted = testPredict.copy()
plot_predicted = plot_predicted.reshape(499, 1)
plot_actual = testY.copy()
plot_actual = plot_actual.reshape(499, 1)
plot_predicted_train = trainPredict.copy()
plot_predicted_train = plot_predicted_train.reshape(1998, 1)
plot_actual_train = trainY.copy()
plot_actual_train = plot_actual_train.reshape(1998, 1)

#plotting test results
figPredict, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].plot(pd.DataFrame(plot_predicted_train), label='Train Predicted')
axes[0].plot(pd.DataFrame(plot_actual_train), label='Train Actual')
axes[0].legend(loc='best')
axes[1].plot(pd.DataFrame(plot_predicted), label='Test Predicted')
axes[1].plot(pd.DataFrame(plot_actual), label='Test Actual')
axes[1].legend(loc='best')
st.markdown("___")
st.caption("Given is a graph depicting LSTM Model precision:")
st.pyplot(figPredict)
st.markdown("___")

#added social media handles section of the page
st.write("GitHub [link](https://github.com/uditsathe/uditsathe.github.io)", " | ", "LinkedIn [link](https://in.linkedin.com/in/udit-sathe-b00154214?original_referer=https%3A%2F%2Fwww.google.com%2F)", " | ",
         "Twitter [link](https://twitter.com/SatheUdit)", " | ", "Instagram [link](https://www.instagram.com/uditsathe/)")


