import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from numpy import mean, absolute
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU


df = pd.read_csv(r"C:\Users\ferna\OneDrive - Universidad Politécnica de Madrid\IIT\Machine Learning\Project\ProjectCSV\scaled_df_eur.csv", index_col=0)
X = df["eur_deaths"]
y = df["Price"]
total_errors = list() 

#%%
    
def evaluate_model(X_train,y_train,y_test,X_test, i):
    
    X_train_reshaped = X_train.reshape(X_train.shape[0],1)
    X_test_reshaped = X_test.reshape(X_test.shape[0],1)
    y_train_reshaped = y_train.reshape(y_train.shape[0],1)
    y_test_reshaped = y_test.reshape(y_test.shape[0],1)

    # LSTM
        
    trainX = np.reshape(X_train, (X_train_reshaped.shape[0],1,X_train_reshaped.shape[1]))
    testX = np.reshape(X_test, (X_test_reshaped.shape[0],1,X_test_reshaped.shape[1]))
    trainy = np.reshape(y_train, (y_train_reshaped.shape[0],1,y_train_reshaped.shape[1]))
    testy = np.reshape(y_test, (y_test_reshaped.shape[0],1,y_test_reshaped.shape[1]))
    
    # design network
    model = Sequential()
    model.add(GRU(50, input_shape=(trainX.shape[1],trainX.shape[2]), return_sequences=True))
    #model.add(LSTM(30, return_sequences=True))
    model.add(GRU(25, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    # Busca early-stopper.
    history = model.fit(trainX, trainy, epochs=200, validation_data=(testX, testy), verbose=2, shuffle=False)
    # plot history -- aqui hacemos el plot de train vs validation
    plt.plot(history.history['loss'], label='train', c="b")
    plt.plot(history.history['val_loss'], label='validation', c="r")
    title = "Train vs Validation -- " + str(i)
    plt.title(title)
    plt.legend()
    plt.show()
    
    # Predictions
    yhat = model.predict(testX)
    MSE = np.square(np.subtract(y_test,yhat)).mean()  
    RMSE = math.sqrt(MSE)
    
    total_errors.append(RMSE)
    
    
    testX = testX.reshape((testX.shape[0], testX.shape[2]))
    y_pred = np.append(y_train_reshaped,yhat)
    
    plt.plot(y_pred, c="r")
    plt.plot(y, c="b")
    title = "Predictions -- " + str(i)
    plt.title(title)
    plt.show()
    
    # Predicting with all the data
    X_2 = X.to_numpy()
    y_2 = y.to_numpy()
    X_2 = X_2.reshape(X_2.shape[0],1)
    y_2 = y_2.reshape(y_2.shape[0],1)
    X_2 = np.reshape(X_2, (X_2.shape[0],1,X_2.shape[1]))
    y_2 = np.reshape(y_2, (y_2.shape[0],1,y_2.shape[1]))
    
    yhat_2 = model.predict(X_2)
    
    X_2 = X_2.reshape((X_2.shape[0], X_2.shape[2]))
    y_2 = y_2.reshape((y_2.shape[0], y_2.shape[2]))
    
    plt.plot(yhat_2,c="r", label="Prediction")
    plt.plot(y_2, c="b", label="Real value")
    plt.legend()
    title = "Predicting S&P 500 values with Covid-19 deaths -- " + str(i)
    plt.title(title)
    plt.show()
    
    
    
    
#%%
def perform_k_fold(X,y,folds):
    
    elem = int(len(X)/folds)
    print(len(X))
    
    for i in range(folds):
        i = i+1
        
        if i == 1:
            X_train, y_train = X[elem:], y[elem:]
            X_test, y_test = X[:elem], y[:elem]
            X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
            evaluate_model(X_train, y_train, y_test, X_test,i)

        elif i == folds:
            X_train, y_train = X[:elem*(i-1)], y[:elem*(i-1)]
            X_test, y_test =  X[elem*(i-1):], y[elem*(i-1):]
            X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()         
            evaluate_model(X_train, y_train, y_test, X_test,i)
            
        else:
            X_train, y_train = X[:elem*(i-1)], y[:elem*(i-1)]
            X_train2, y_train2 = X[elem*(i+1):], y[elem*(i+1):]
            
            X_train = np.append(X_train, X_train2)
            y_train = np.append(y_train, y_train2)
            
            X_test, y_test = X[elem*i:elem*(i+1)], y[elem*i:elem*(i+1)]
            X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
            evaluate_model(X_train, y_train, y_test, X_test,i)
   
   

perform_k_fold(X, y, 5)
error_GRU = np.mean(total_errors)


    
    
    
    
    
    
    
    
    
    
