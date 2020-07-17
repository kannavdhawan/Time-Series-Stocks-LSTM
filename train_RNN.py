# Imports

import pandas as pd
import numpy as np
import random
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from matplotlib import pyplot as plt

#Loading the dataset

url='https://raw.githubusercontent.com/kannavdhawan/Time_series_stocks_LSTM/master/q2_dataset.csv?token=AHMAKJAQPICJFWARH35B4US7DEIJE'
data=pd.read_csv(url)

print(data.head(3))
# Dropping the Close/Last column 
data_no_close=data.drop([" Close/Last"],axis=1)
data_no_close.head(3)
data_no_close['Date']=pd.to_datetime(data_no_close['Date'])
print(data_no_close.head(3))
data_no_close=data_no_close.sort_values('Date')
print(data_no_close)
data_no_close=data_no_close.reset_index(drop=True)
print(data_no_close.head(3))


data_no_close=data_no_close.iloc[:,1:].values.astype('float32')
print("rand check: ",data_no_close[0])
print(data_no_close.shape)


def feature_gen(data_values,size):
  features=[]
  targets=[]
  old_days=size
  for current_day in range(old_days,len(data_values)):
    features.append(data_values[current_day-old_days:current_day,:])  #[1259*(3,4)shape]
    targets.append(data_values[current_day][1])
  return features,targets
features,targets=feature_gen(data_no_close,3)
print(features[0])
print(features[1])
print(targets[0])
print("shape check features: ",np.asarray(features).shape)
print("shape check target: ",np.asarray(targets).shape)

# shuffling the dataset
zipped_f_t=list(zip(features,targets))
random.shuffle(zipped_f_t)
len(zipped_f_t)


# splitting the dataset into train and test
tr_sz=int(0.70*len(zipped_f_t))

train, test = zipped_f_t[:tr_sz], zipped_f_t[tr_sz:]

print(train[0])


print(len(train))
print(len(test))

# flattening the arrays stored in lists to save in a csv file 
#Each feature set below is a tuple with features of shape (3,4) and an int value for target..
X_train=pd.DataFrame([feature_set[0].ravel() for feature_set in train])
X_test=pd.DataFrame([feature_set[0].ravel() for feature_set in test])
y_train=pd.DataFrame([feature_set[1] for feature_set in train])
y_test=pd.DataFrame([feature_set[1] for feature_set in test])

# print((X_train).head(2))
# print((X_test).head(2))
# print((y_train).head(2))
# print((y_test).head(2))

#Making dataframes 
train=X_train.copy()
train['Target']=y_train
test=X_test.copy()
test['Target']=y_test

print(train.head(3))
print(test.head(3))
train.to_csv("train_data_RNN.csv")
test.to_csv("test_data_RNN.csv")


