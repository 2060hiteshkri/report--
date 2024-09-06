import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)

# Streamlit subheader
st.subheader('Next-Day Forecasting with Long-Short Term Memory (LSTM)')

# Load symbols from CSV file
csv = pd.read_csv('convertcsv.csv')
symbol = csv['symbol'].tolist()

# Creating sidebar
ticker_input = st.selectbox('Enter or Choose Crypto Coin', symbol, index=symbol.index('ETH'))

# Define date range for data fetching
start = dt.datetime.today() - dt.timedelta(5*365)
end = dt.datetime.today()

# Fetch data using yfinance
df = yf.download(ticker_input + '-INR', start=start, end=end)

# Check if the DataFrame is empty
if df.empty:
    st.error('No data available for the selected cryptocurrency.')
else:
    st.write('It will take some seconds to fit the model....')
    df.reset_index(inplace=True)

    # Creating a new DataFrame for LSTM
    eth_lstm = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    eth_lstm['Date'] = df['Date']
    eth_lstm['Close'] = df['Close']

    # Setting the index to 'Date'
    eth_lstm.index = eth_lstm.Date
    eth_lstm.drop('Date', axis=1, inplace=True)
    eth_lstm = eth_lstm.sort_index(ascending=True)

    # Creating train and test sets
    dataset = eth_lstm.values
    train = dataset[0:990, :]
    valid = dataset[990:, :]

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Creating x_train and y_train datasets
    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Building the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    print('Fitting Model')

    # Preparing the data for prediction
    inputs = eth_lstm[len(eth_lstm) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Making predictions
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    # Calculate metrics
    rms = np.sqrt(mean_squared_error(closing_price, valid))
    acc = r2_score(closing_price, valid) * 100

    # Plotting the results
    train = df[:990]
    valid = df[990:]
    valid['Predictions'] = closing_price

    st.write('#### Actual VS Predicted Prices')

    fig_preds = go.Figure()
    fig_preds.add_trace(
        go.Scatter(
            x=train['Date'],
            y=train['Close'],
            name='Training data Closing price'
        )
    )

    fig_preds.add_trace(
        go.Scatter(
            x=valid['Date'],
            y=valid['Close'],
            name='Validation data Closing price'
        )
    )

    fig_preds.add_trace(
        go.Scatter(
            x=valid['Date'],
            y=valid['Predictions'],
            name='Predicted Closing price'
        )
    )

    fig_preds.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='Predictions on Validation Data', template='gridon'
    )

    st.plotly_chart(fig_preds, use_container_width=True)

    # Forecasting the next day's closing price
    real_data = [inputs[len(inputs) - 60:len(inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    st.write('#### Next-Day Forecasting')

    with st.container():
        col_111, col_222, col_333 = st.columns(3)
        col_111.metric(f'Closing Price Prediction of the next trading day for {ticker_input} is',
                       f' $ {str(round(float(prediction), 2))}')
        col_222.metric('Accuracy of the model is', f'{str(round(float(acc), 2))} %')
