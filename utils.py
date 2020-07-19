import pandas as pd
import matplotlib.pyplot as plt
import os 
import math
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # for ignoring warnings on console 
import tensorflow as tf



def main():
    print("inside main ")

# print("l-1")

def data_load(data_path):
    """
    Loading preprocessed dataset 
    """
    print("Loading dataset..")
    dataset=pd.read_csv(data_path,index_col=[0])
    X,y=dataset.iloc[:,:-1],dataset['Target']       #separating data and labels
    return X,y

#history plot
def train_metrics_plot(history):
    """
    plots the Training Loss and training MAE at each epoch..
    """
    print("Plotting train loss and mse..")
    Train_MAE=history.history['mae']                    #Get traning loss from hist_object.history
    Train_Loss=history.history['loss']
    
    plot_metrics=[]
    plot_metrics.extend([Train_MAE,Train_Loss])
    for metric in plot_metrics:
            
        plt.figure(figsize=(8,7))
        if metric==Train_MAE:
            metric_name="Training MAE"
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

def metric_errors(real,pred,flag):
    """ calculate root mean squared error, MSE and MAE.
    """
    rmse= math.sqrt(mean_squared_error(real,pred))
    print(flag+" RMSE :",rmse)

    mse= mean_squared_error(real,pred)
    print(flag+' MSE : ',mse)

    mae=mean_absolute_error(real,pred)
    print(flag+' MAE : ',mae)


if __name__ == "__main__":
    main()