import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def main():
    path_to_files = str("C:/Users/ferna/OneDrive - Universidad Politécnica de Madrid/IIT/Machine Learning/Project/Delivery/data/")

    df = pd.read_csv(path_to_files + "scaled_df_eur.csv", index_col=0)
    
    X = df["eur_deaths"]
    y = df["Price"]
    plt.plot(X, c='b')
    plt.plot(y, c='r')
    plt.show()
    X_train, y_train = X[:180], y[:180]
    X_test, y_test = X[144:], y[144:]

    X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(),
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
    model.add(LSTM(50, input_shape=(trainX.shape[1],trainX.shape[2]), return_sequences=True))
    #model.add(LSTM(30, return_sequences=True))
    model.add(LSTM(25, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # Busca early-stopper.
    history = model.fit(trainX, trainy, epochs=200, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train', c="b")
    #plt.plot(history.history['val_loss'], label='validation', c="r")
    plt.legend()
    plt.show()

    #Predictions
    yhat = model.predict(testX)
    testX = testX.reshape((testX.shape[0], testX.shape[2]))
    y_pred = np.append(y_train_reshaped,yhat)

    plt.plot(y_pred, c="r")
    plt.plot(y, c="b")
    plt.show()


    # # here we can SAVE the model: model.save("PATH TO MODEL.h5")

    # Predicting with all the data
    X = X.to_numpy()
    y = y.to_numpy()
    X = X.reshape(X.shape[0],1)
    y = y.reshape(y.shape[0],1)
    X = np.reshape(X, (X.shape[0],1,X.shape[1]))
    y = np.reshape(y, (y.shape[0],1,y.shape[1]))
    
    yhat_2 = model.predict(X)
    
    X = X.reshape((X.shape[0], X.shape[2]))
    y = y.reshape((y.shape[0], y.shape[2]))

    plt.plot(yhat_2,c="r", label="Prediction")
    plt.plot(y, c="b", label="Real value")
    plt.legend()
    plt.title("Predicting S&P 500 values with Covid-19 deaths")
    plt.show()

if __name__ == "__main__":
    main()













