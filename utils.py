
import pandas as pd

def data_load(data_path):
    """
    Loading dataset 
    """
    dataset=pd.read_csv(data_path,index_col=[0])
    X,y=dataset.iloc[:,:-1],dataset['Target']       #separating data and labels
    return X,y