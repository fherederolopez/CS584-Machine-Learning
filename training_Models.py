import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR
from numpy import mean

from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model



import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\mende\OneDrive\Escritorio\CS584 - Machine Learning\Project\scaled_df_eur.csv", index_col=0)

X = df["eur_deaths"]
y = df["Price"]

X = X.to_numpy()
y = y.to_numpy()
X = X.reshape(X.shape[0],1)
y = y.reshape(y.shape[0],1)


#%%

model_SVR = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
# RandomForestRegressor(n_estimators = 1000, random_state = 42)
#svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
#svr_lin = SVR(kernel="linear", C=100, gamma="auto")
#svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)


scores_SVR = cross_val_score(model_SVR, X, y, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
# return scores
mean_SVR = mean(scores_SVR)

model_L = linear_model.Lasso(alpha=0.1)
# RandomForestRegressor(n_estimators = 1000, random_state = 42)
#svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
#svr_lin = SVR(kernel="linear", C=100, gamma="auto")
#svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)


scores_L = cross_val_score(model_L, X, y, scoring='neg_root_mean_squared_error', cv=5)
# return scores
mean_L = mean(scores_L)


