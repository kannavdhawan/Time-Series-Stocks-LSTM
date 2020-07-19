import os 
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import math
from utils import data_load, metric_errors
from sklearn.preprocessing import MinMaxScaler

def load_files(path_sc,path_test):
    """ loading scalar object and Test csv.
        Args:
            path(os path)
        Attributes:
            path(os path): path to the pickle object and testdata
        Returns: 
            X_test
            y_test
            X_scalar
            y_scalar
    """
    scalar=pickle.load(open(path_sc,"rb"))
    X_scalar=scalar[0]
    y_scalar=scalar[1]

    X_test,y_test=data_load(path_test)      # test data loading from utils call

    return X_test,y_test,X_scalar,y_scalar

"""scaling with same mean and sd | please uncomment if don't want to use pickkle file of saved scalar..
"""
# X_scaler_test=MinMaxScaler(feature_range=(0,1))
# y_scaler_test=MinMaxScaler(feature_range=(0,1))

# train=pd.read_csv(os.path.join("data/","train_data_RNN.csv"),index_col=[0])
# X_train=train.iloc[:,:-1]
# y_train=train['Target']

# X_train=X_scaler_test.fit_transform(X_train)
# y_train=y_scaler_test.fit_transform(np.asarray(y_train).reshape(-1,1))

# X_test=X_scaler_test.transform(X_test)
# y_test=y_scaler_test.transform(np.asarray(y_test).reshape(-1,1))

def preprocess_test(X_test,y_test,X_scalar,y_scalar):
    """ 
    Scaling test data using saved scalar from train_RNN.
    Args:
        X_test(DataFrame)
        y_test(DataFrame)
        X_scalar(scalar ob)
        y_scalar(scalar ob)

    Attributes:
        X_test(DataFrame): Test features
        y_test(DataFrame): Test labels
        X_scalar(scalar ob): Scalar having mean and sd same as train data. | for features
        y_scalar(scalar ob): scalar for labels.

    Returns:
        X_test: preprocessed time series np array ready for lstm.
        y_test: preprocessed np array having labels for test data.
    """

    X_test=X_scalar.transform(X_test)                           #Normalizing the data features with same mean and sd as of train data.
    y_test=y_scalar.transform(np.asarray(y_test).reshape(-1,1)) #Normalizing the labels and reshaping into a 2-d np array.

    # print("X_test:\n",X_test[0]," shape:",X_test.shape)         #(377,12)
    # print("y_test:\n",y_test[0]," Shape:",y_test.shape)         #(377,1)

    
    X_test=np.asarray(X_test).reshape(X_test.shape[0],3,4)      #Reshaping features to the desired shape for time series RNN.
    # print("Reshaped X_test: \n",X_test.shape)                   #(377,3,4)
    return X_test,y_test


def pred(path_model,X_test,y_test,y_scalar):
    """
    Loads the saved model and predicts the test labels. i.e. price for next day.
    calculates the loss using evaluate for test data.
    Inverse transforms the predicted and real prices

    Args:
        path_model(os path)
        X_test(np array)
        y_test(np array)
        y_scalar(scalar ob)

    Attributes:
        path_model(os path): path to the h5 RNN model
        X_test(np array): preprocessed numpy array with shape (377,3,4)
        y_test(np array): numpy array with real price labels with shape (377,1)
        y_scalar(scalar ob): scalar for target having mean and sd same as train data. 
    Returns: 
        y_pred: predicted values/price as np array which are inverse transformed to original scale
        y_test: Real values/price inverse transformed 

    """
    lstm_model=load_model(path_model)   #loading the model     
    y_pred=lstm_model.predict(X_test)   #predicting the price
    # print(y_pred.shape)

    loss=lstm_model.evaluate(X_test,y_test)
    print("Loss on Test set: ",loss)

    y_pred= y_scalar.inverse_transform(y_pred)  # converting back to real values 
    y_test= y_scalar.inverse_transform(y_test)  # converting real labels back to real values

    print("Random Testing for Test data \n Predicted: ",y_pred[0])
    print("Target: ",y_test[0])
    
    return y_pred,y_test

def plot_acc(y_pred,y_test):

    plt.figure(figsize=(8,8))
    plt.plot(y_pred, "r")
    plt.plot(y_test,"g")
    plt.legend(["Predicted Price","Real Price"])
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.savefig(os.path.join("data/","True_Predicted_plot_test.png"))
    plt.show()

# Loading scalar and csv..
X_test,y_test,X_scalar,y_scalar=load_files(os.path.join("models/","scalar.pkl"),os.path.join("data/","test_data_RNN.csv"))

# Preprocssing the test data
X_test,y_test=preprocess_test(X_test,y_test,X_scalar,y_scalar)

# Predicting the price for next day using lstm for test data and calculating loss.
y_pred,y_test=pred(os.path.join("models/","20831774_RNN_model.h5"),X_test,y_test,y_scalar)

# MAE,MSE,RMSE from utils call for test data 
metric_errors(y_test,y_pred,flag="Test")

# Plotting the curve with predicted and real values for test data
plot_acc(y_pred,y_test)