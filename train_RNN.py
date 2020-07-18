# Imports

import pandas as pd
import numpy as np
import random
import math
import pickle
import os 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from matplotlib import pyplot as plt
import tensorflow as tf

tf.random.set_seed(1337) # setting seed 

#Loading the dataset
def load_data(path):
    """

    Loads the dataset with stock prices from csv.

    Args:
        path(os path)

    Arguments:
        path(os path): path to the csv from the data folder

    Returns: 
        data_no_close: A numpy array with float values with no 'close' column.

    """
    print("Loading data..")

    # path='https://raw.githubusercontent.com/kannavdhawan/Time_series_stocks_LSTM/master/q2_dataset.csv?token=AHMAKJAQPICJFWARH35B4US7DEIJE'

    data=pd.read_csv(path)
    # print("Checking head..:\n",data.head(3))

    data_no_close=data.drop([" Close/Last"],axis=1)             # Dropping the Close/Last column 
    # print("No close ds test..:\n",data_no_close.head(3))

    data_no_close['Date']=pd.to_datetime(data_no_close['Date']) # converting date from str to datetime
    # print("Checking ds..:\n",data_no_close.head(3))

    data_no_close=data_no_close.sort_values('Date')             # Sorting df by date
    # print("sorted date ds..:\n",data_no_close)

    data_no_close=data_no_close.reset_index(drop=True)          # Reseting the reversed index after sorting.
    # print("Reset index ds..:\n",data_no_close.head(3))

    data_no_close=data_no_close.iloc[:,1:].values.astype('float32')     #converting into np array and 'float32'.  {Already floats!~Volume}
    # print("np array floats..:\n",data_no_close[0])
    # print("Shape np array=> ",data_no_close.shape)

    return data_no_close


# Generating the features
def feature_gen(data_values,size):
    """
    Generates the features for lstm with the required window size for time series input. 

    Args:
        data_values(np array)
        size(int)

    Arguments:
        data_values(np array): numpy array returned by load_data with float values. 
        size(int): window size
    
    Returns: 
        features(list): [1256, 3, 4]
        targets(list): [1256]

    """
    features=[]
    targets=[]
    old_days=size
    for current_day in range(old_days,len(data_values)):
        features.append(data_values[current_day-old_days:current_day,:])  #[1259*(3,4)shape]
        targets.append(data_values[current_day][1])

    # print("type(features):",type(features))           #list
    # print("type(features[0])",type(features[0]))      #numpy array
    # print("shape check features: ",np.asarray(features).shape) # (1256, 3, 4)
    # print("shape check target: ",np.asarray(targets).shape)    # (1256,)
    return features,targets

def preprocess(features,targets):
    
    # shuffling the dataset
    zipped_f_t=list(zip(features,targets))
    random.shuffle(zipped_f_t)                           #shuffling the zipped object with features and labels
    tr_sz=int(0.70*len(zipped_f_t))                      # getting size for splitting the dataset into train and test
    train,test = zipped_f_t[:tr_sz], zipped_f_t[tr_sz:]  # splitting the zipped object list
    # print(len(train))
    # print(len(test))

    
    """
    ** train,test : list of truple of numpy array of shape (None,3,4) and int target.

    ** This preprocessing is done to store the data in a csv file ..
    
        * Each feature set below is a tuple with features of shape (3,4)
            and an int value for target at first index.
        * we flatten the features at 0th index from shae (3,4) to 12 for each element/input. 
        
    """
    X_train=pd.DataFrame([feature_set[0].ravel() for feature_set in train])
    X_test=pd.DataFrame([feature_set[0].ravel() for feature_set in test])
    y_train=pd.DataFrame([feature_set[1] for feature_set in train])
    y_test=pd.DataFrame([feature_set[1] for feature_set in test])

    # print((X_train).head(2))
    # print((X_test).head(2))
    # print((y_train).head(2))
    # print((y_test).head(2))

    # Creating dataframes 

    train=X_train.copy()        
    train['Target']=y_train     # concatinating labels
    
    test=X_test.copy()
    test['Target']=y_test       # concatinating labels

    # print(train.head(3))
    # print(test.head(3))

    train.to_csv(os.path.join("data/","train_data_RNN.csv")) # saving csv of train and test in data/
    test.to_csv(os.path.join("data/","test_data_RNN.csv"))


"""
//CALLS FOR CREATING TIME SERIES DATASET//
"""
# Loading dataset from csv
data_no_close=load_data(os.path.join("data/","q2_dataset.csv"))

# Generating features
win_size=3
features,targets=feature_gen(data_no_close,win_size)

# Preprocessing and saving preprocessed dataset 
preprocess(features,targets)


def dataset():
    """
    Loading dataset 
    """
    train=pd.read_csv(os.path.join("data/","train_data_RNN.csv"),index_col=[0])

    print(train.head())

    #separating data and labels
    X_train,y_train=train.iloc[:,:-1],train['Target']
    print(X_train.head())
    print(y_train.head()) 


#scaling 
scalar=[]
X_train_scalar=MinMaxScaler(feature_range=(0,1))
y_train_scalar=MinMaxScaler(feature_range=(0,1))


X_train=X_train_scalar.fit_transform(X_train)
y_train=y_train_scalar.fit_transform(np.asarray(y_train).reshape(-1,1)) # To make ot 2D, reshaping ..

scalar.extend([X_train_scalar,y_train_scalar])
pickle.dump(scalar,open(os.path.join("models/","scalar.pkl"), "wb" ))
#Reshaping the dataset X_train to 3 dimensional numpy array for lstm

X_train=np.asarray(X_train).reshape(879,3,4)
print("X_train shape: ",X_train.shape)
print("y train shape:",y_train.shape)
# Defining model 
# model = Sequential()
# model.add(LSTM(64, input_shape=(3,4)))
# # model.add(LSTM(units=32, return_sequences = True))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='sgd')

model =Sequential()
model.add(LSTM(units=32, return_sequences= True, input_shape=(3,4))),

model.add(LSTM(units=10, return_sequences= False)),
model.add(Dense(units=30))
model.add(Dense(units=20))
model.add(Dense(units=10))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
# "sgd":4.48


# printing model summary 
print(model.summary())
#Training
print("Training..")
history=model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2)
model.save(os.path.join("models/","20831774_RNN_model.h5"))
#history plot
def loss(history):
    """
    plots the Training Loss vs Validation Loss
    """
   
    Train_MSE=history.history['mse']                    #Get traning loss from model.history
    Train_Loss=history.history['loss']
    
    plot_metrics=[]
    plot_metrics.extend([Train_MSE,Train_Loss])
    for metric in plot_metrics:
            
        plt.figure(figsize=(8,7))
        if metric==Train_MSE:
            metric_name="Training MSE"
            plt.plot(metric,'r')
            plt.xlabel("Epochs")
            plt.ylabel(metric_name)
            plt.title(metric_name+" at each epoch")
            plt.legend([metric_name])
            plt.show()
            name=metric_name+".png"
            plt.savefig(os.path.join("data/",name))
        elif metric==Train_Loss:
            metric_name="Training Loss"
            plt.plot(metric,'r')
            plt.xlabel("Epochs")
            plt.ylabel(metric_name)
            plt.title(metric_name+" at each epoch")
            plt.legend([metric_name])
            plt.show()
            name=metric_name+".png"
            plt.savefig(os.path.join("data/",name))

loss(history)

loss=model.evaluate(X_train,y_train)
print("Loss on Train set: ",loss)

# Just to get the RMSE for train data ..

y_pred_train = model.predict(X_train)

print("y_pred_train:",y_pred_train.shape) #np array (879,1)
print("y_train:", y_train.shape)          #np array (879,1)

#Inverting both y_train and y_pred_train 
y_pred_train = y_train_scalar.inverse_transform(y_pred_train)
y_train = y_train_scalar.inverse_transform(y_train) 
print("y_pred_train:",y_pred_train.shape) #np array (879,1)
print("y_train:", y_train.shape)          #np array (879,1)

print(y_pred_train[0])
print(y_train[0])

# calculate root mean squared error
rmse_train= math.sqrt(mean_squared_error(y_train, y_pred_train))
print("RMSE for training data:",rmse_train)



mse_train= mean_squared_error(y_train, y_pred_train)
print('MSE for training data: ',mse_train)

mae_train=mean_absolute_error(y_train, y_pred_train)
print('MAE for training data: ',mae_train)

