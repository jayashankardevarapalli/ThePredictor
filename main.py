from datetime import date
import streamlit as st
import pandas as pd
from pandas import DataFrame
import pandas_datareader as data
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model as lm
from sklearn.preprocessing import MinMaxScaler as mms
from tabulate import tabulate

st.title('THE PREDICTOR')
st.caption('All you need to do is enter the stock symbol, starting date and the ending date to perform analysis and Predictions!.')

# st.write('<h5></h5>',unsafe_allow_html=True)
st.write('<h5>Stock Symbol</h5>',unsafe_allow_html=True)
ui = st.text_input("Enter the stock symbol: ",'msft')

st.write('<h5>Time Period for the Dataset</h5>',unsafe_allow_html=True)
sd = st.date_input("Enter the starting date: ", date(2000,1,1))
ed = st.date_input("Enter the ending date: ", date.today())


st.write('<h5>Statistical Analysis for ',ui,'</h5>',unsafe_allow_html=True)
df = data.DataReader(ui,'yahoo',sd,ed)
df = df.reset_index()
st.write(df)
st.write(df.describe())

df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df.index = df.pop('Date')

st.write('<h5>Visualization of the Closing Price over the years</h5>',unsafe_allow_html=True)

vizcpy = plt.figure(figsize = (16,8))
plt.plot(df.index, df['Close'])
plt.ylabel('Price')
plt.xlabel('Years')
st.pyplot(vizcpy)

st.write('<h5>Visualization of Moving Average on Closing Price.</h5>',unsafe_allow_html=True)

st.write('<h6>100 Moving Average</h6>',unsafe_allow_html=True)
moving100 = df.Close.rolling(100).mean()
ma100 = plt.figure(figsize = (16,8))
plt.plot(moving100, 'r', label="100 Moving Average")
plt.plot(df.Close, 'b', label="Original Closing Price")
plt.legend()
st.pyplot(ma100)

st.write('<h6>200 Moving Average</h6>',unsafe_allow_html=True)
moving200 = df.Close.rolling(200).mean()
ma200 = plt.figure(figsize = (16,8))
plt.plot(moving200, 'r', label="200 Moving Average")
plt.plot(df.Close, 'b', label="Original Closing Price")
plt.legend()
st.pyplot(ma200)


training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
testing = pd.DataFrame(df['Close'][int(len(df)*0.80): int(len(df))])

scaler = mms(feature_range = (0,1))
training_array = scaler.fit_transform(training)

model = lm('model.h5')

n_days = training.tail(100)
pop_df = n_days.append(testing, ignore_index=True)
input_data = scaler.fit_transform(pop_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    

x_test, y_test = np.array(x_test), np.array(y_test)

y_predict = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predict *= scale_factor
y_test *= scale_factor

st.write('<h4>Predictions Vs Original Closing Price</h4>',unsafe_allow_html=True)
finfig = plt.figure(figsize=(14,8))
plt.plot(y_test, 'b', label="Original Closing Price")
plt.plot(y_predict, 'r', label="Predicted Closing Price")
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend()
st.pyplot(finfig)

