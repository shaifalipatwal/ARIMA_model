
#%%
# Importing packages 
pip install pmdarima
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
#%%
#ARIMA (Auto Regressive Integrated Moving Average) Model: 
# Reading data for ARIMA Model
# I want to predict how Sales_quantity is going to change overtime
data = pd.read_csv('C:/Users/SHAIFALI PATWAL/Desktop/Github Projects/month_value.csv')
df = data.dropna()  # Dropping missing values from the dataset
df['Period'] = pd.to_datetime(df['Period'], dayfirst=True)  # Convert 'Period' column to datetime format
df.set_index('Period', inplace=True)  # Setting 'Period' as the index of the DataFrame
df.shape # checking the shape oft he dataset
df.head() # checking the first 5 rows
#%%
# Ploting a timeseries plot
df['Sales_quantity'].plot(figsize=(10,6))  

#%%
# Finding the order for the ARIMA Model
s_order = auto_arima(df['Sales_quantity'], trace=True)   
s_order
# Here we get the model order p=2,d=1,q=2 that we'll be using for creating the model
#%%
# Splitting the Data into Training and Testing sets
from statsmodels.tsa.arima_model import ARIMA

train=df.iloc[:-20]  # training the model on the entire data set except the last 30 rows
train

test=df.iloc[-20:]  # taking last 30 rows for the testing purpose
test
#%%
# Creating ACF and PACF plots for training data
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
acf_original = plot_acf(train['Sales_quantity'])

pacf_original = plot_pacf(train['Sales_quantity'])
#%%
# fitting the model on training data with order (2,1,2)
from statsmodels.tsa.arima.model import ARIMA
model1=ARIMA(train['Sales_quantity'],order=(2,1,2))  
model1=model1.fit()
model1.summary()  #summary of training dataset

#%%
# testing the model on the test dataset
start=len(train)
end=len(train)+len(test)-1
pred=model1.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')
pred.index=df.index[start:end+1]
pred

# Plotting the prediction
pred.plot(legend=True)
test['Sales_quantity'].plot(legend=True)

# checking the mean of AvgTemp variable
test['Sales_quantity'].mean()   # We get the  mean = 24920.0 for Sales_quantity

# checking the mean squared error of the Sales_quantity variable
rmse=sqrt(mean_squared_error(pred,test['Sales_quantity']))
rmse # we get the RMSE = 8740.38

#%%
# Predicting for the future values
index_future_dates = pd.date_range(start=df.index[-1], periods=30, freq='M')  # Adjust as needed
pred = model1.predict(start=len(df), end=len(df)+29, typ='levels').rename('ARIMA Predictions')
pred.index = index_future_dates
pred
# Plotting the predictions
pred.plot(figsize=(12,5),legend=True)






