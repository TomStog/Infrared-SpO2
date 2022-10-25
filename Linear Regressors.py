# Linear Regression Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# Non-linear Regression Algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor

import pandas as pd
from pandas import read_csv
from sklearn.metrics import mean_absolute_error, max_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
from statistics import mean, median
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from itertools import chain, combinations
from heapq import nsmallest

# "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"

def powerset(iterable):    
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

if __name__ == "__main__":

    combo_arr = []
    stuff = ['var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12']
    for i, combo in enumerate(powerset(stuff), 1):
        combo_arr.append(combo)

    my_combo = combo_arr[(len(stuff)+1):]
    #print(my_combo)

    """LOAD DATA"""

    # Read data from CSV file
    data = pd.read_csv (r'/content/filename_12vars_16people_mag.csv')

    # Define y = f(x)
    predictors_list = np.array(my_combo)
    outcome = ['SpO2']

    # Normalization Parameter
    norm_param = 100

    # Define kfold cross validation
    kf = KFold(n_splits=5, random_state=None, shuffle=True)

    """SPLIT AND PREDICT"""

    mae_total = []
    mse_total = []

    regression_list = [LinearRegression(), Ridge(), Lasso(alpha = 0.0001), Lasso(alpha = 0.001)]

    for predictors in predictors_list:
        X = data[np.array(predictors)].values
        y = data[outcome]

        X = np.array(X)
        y = np.array(y)    

        for i in range(20):
            mae = []
            mse = []
            for train_index, test_index in kf.split(X, y):
                # Split the data
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index]/norm_param, y[test_index]/norm_param

                # Insert Regression Algorithm / LinearRegression(), Ridge(), BayesianRidge(), Lasso(alpha), ElasticNet(random_state), AdaBoostRegressor(random_state, n_estimators), RandomForestRegressor(max_depth, random_state), GradientBoostingRegressor(random_state), ExtraTreesRegressor(n_estimators, random_state)
                model =  LinearRegression()
                model.fit(X_train, y_train)

                # Predict with X_test
                y_hat = model.predict(X_test)
                y_hat = y_hat.reshape(len(y_hat), 1)

                # Calculate the metric
                mae.append(mean_absolute_error(y_test, y_hat))
                mse.append(mean_squared_error(y_test, y_hat))

            mae_total.append(mean(mae)*norm_param)
            mse_total.append(mean(mse)*(norm_param**2))

    print("Mean Absolute Error: %.3f - Mean Squared Error: %.3f" %(mean(mae_total), mean(mse_total)))
    print("Minimum Mean Squared Error: %.3f" %(min(mse_total)))
    print("Predictors : ", predictors_list[mse_total.index(min(mse_total))])

    # plot MAE per combination

    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)
    plt.plot(mae_total)
    plt.title("Mean Absolute Error per Combination")
    plt.show()

    # plot MSE per combination

    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)
    plt.plot(mse_total)
    plt.title("Mean Squared Error per Combination")
    plt.show()

    # find best performance considering both MAE & MSE

    for counter in range(1,100):
    
        mae_list = nsmallest(counter, mae_total)
        mse_list = nsmallest(counter, mse_total)
        
        mae_ind=[]        
        for i in range(len(mae_list)):
            mae_ind.append(mae_total.index(mae_list[i]))

        mse_ind=[]
        for i in range(len(mse_list)):
            mse_ind.append(mse_total.index(mse_list[i]))

        my_set = set(mse_ind) & set(mae_ind)
        if not my_set:
            print("set is empty", counter)
        else:
            print("set is not empty", counter, my_set)
            break

    elem = my_set.pop()
    print(my_combo[elem])
    print("Mean Absolute Error: %.3f - Mean Squared Error: %.3f" %(mae_total[elem], mse_total[elem]))