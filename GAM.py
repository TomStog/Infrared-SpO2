import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, max_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statistics import mean, median
from pygam import GAM, LinearGAM, s, f, te
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import load_model
from sklearn.model_selection import KFold
from tqdm import tqdm
from itertools import chain, combinations

if __name__ == "__main__":

    data = pd.read_csv (r'./filename_12vars_21people_mag.csv')
    predictors = ['var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12']
    outcome = ['SpO2']

    norm_param = 100
    kf = KFold(n_splits=5, random_state=None, shuffle=True)

    X = data[np.array(predictors)].values
    y = data[outcome]

    X = np.array(X)
    y = np.array(y)

    n_features = len(predictors)

    mae_total = []
    mse_total = []

    for i in range(20):
        mae = []
        mse = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index]/norm_param, y[test_index]/norm_param

            lams = np.random.rand(100, n_features)
            lams = lams * n_features - 3
            lams = np.exp(lams)

            gam = LinearGAM(n_splines=12).gridsearch(X_train, y_train, lam=lams, progress=False)
            print(gam.summary())

            # Plot section

            #titles = data.columns[0:n_features]
            #plt.figure()
            #fig, axs = plt.subplots(1,n_features,figsize=(40, 8))
            #for i, ax in enumerate(axs):
            #	XX = gam.generate_X_grid(term=i)
            #	ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
            #	ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX,   width=.95)[1], c='r', ls='--')
            #	if i == 0:
            #		ax.set_ylim(-30,30)
            #	ax.set_title(titles[i])

            y_hat = gam.predict(X_test)

            # threshold values over 100%
            #for x in range(len(y_hat)):
            #  if y_hat[x]>(100/norm_param):
            #    y_hat[x]=(100/norm_param)

            # metrics
            mae.append(mean_absolute_error(y_test, y_hat))
            mse.append(mean_squared_error(y_test, y_hat))

        mae_total.append(mean(mae)*norm_param)
        mse_total.append(mean(mse)*(norm_param**2))

    #print(predictors)
    print("Mean Absolute Error: %.3f - Mean Squared Error: %.3f" %(mean(mae_total), mean(mse_total)))
    print("Minimum Mean Squared Error: %.3f" %(min(mse_total)))
