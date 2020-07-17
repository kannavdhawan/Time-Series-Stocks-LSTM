import os 
from keras.models import load_model





lstm_model=load_model("lstm_model.h5")
y_pred=lstm_model.predict(X_test)
lstm_model.evaluate(y_pred,y_test)