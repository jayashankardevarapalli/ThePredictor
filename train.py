import pandas as pd
from datetime import date
import pandas_datareader as data
import numpy as np
from sklearn.preprocessing import MinMaxScaler as mms
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential as seq
 
sd = date(2000,1,1)
ed = date.today()

# sn = input("Enter the stock Name: ")

df = data.DataReader('aapl', 'yahoo', sd, ed)
df = df.reset_index()
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df.index = df.pop('Date')

training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
testing = pd.DataFrame(df['Close'][int(len(df)*0.80): int(len(df))])

scaler = mms()
training_array = scaler.fit_transform(training)

x_train = [] 
y_train = []


for i in range(100, training_array.shape[0]):
    x_train.append(training_array[i-100: i])
    y_train.append(training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape)

model = seq()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) 


model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))


model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs= 50)
model.save('model.h5')