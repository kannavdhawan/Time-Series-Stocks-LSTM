import pandas as pd

def main():
    print("inside main ")

print("Imported data_load() from utils..")

def data_load(data_path):
    """
    Loading dataset 
    """
    print("Loading dataset..")
    dataset=pd.read_csv(data_path,index_col=[0])
    X,y=dataset.iloc[:,:-1],dataset['Target']       #separating data and labels
    return X,y

if __name__ == "__main__":
    main()