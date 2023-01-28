# Linear Regression Algorithms
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, ElasticNet, Lasso, Lars, Lasso, LassoLars, HuberRegressor, QuantileRegressor, RANSACRegressor, TheilSenRegressor, PoissonRegressor, TweedieRegressor, GammaRegressor

# Non-linear Regression Algorithms
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor, HistGradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

import pandas as pd
from pandas import read_csv
from sklearn.metrics import mean_absolute_error, max_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
from statistics import mean, median
from sklearn.model_selection import KFold
#import matplotlib.pyplot as plt
import warnings

from itertools import chain, combinations

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
    data = pd.read_csv (r'./filename_12vars_16people_mag.csv')
    warnings.filterwarnings("ignore")
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
    mse_list = []

    for predictors in predictors_list:
      X = data[np.array(predictors)].values
      y = data[outcome]

      X = np.array(X)
      y = np.array(y)
      mae_20 = []
      mse_20 = []    

      for i in range(1):
        mae = []
        mse = []
        for train_index, test_index in kf.split(X, y):
            # Split the data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index]/norm_param, y[test_index]/norm_param

            # Insert Regression Algorithm
            #base_estimator=DecisionTreeRegressor(max_depth=10)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict with X_test
            y_hat = model.predict(X_test)
            y_hat = y_hat.reshape(len(y_hat), 1)
            
            bool_mae = np.isnan(mean_absolute_error(y_test, y_hat))
            bool_mse = np.isnan(mean_squared_error(y_test, y_hat))

            # Calculate the metric
            if bool_mae or bool_mse:
                continue
            else:
                mae.append(mean_absolute_error(y_test, y_hat))
                mse.append(mean_squared_error(y_test, y_hat))

        if (all(x <= ((2/norm_param)**2) for x in mse)):
          mae_20.append(mean(mae))
          mse_20.append(mean(mse))
        else:
          continue

      if mse_20:
        mae_total.append(mean(mae_20)*norm_param)
        mse_total.append(mean(mse_20)*(norm_param**2))
        mse_list.append(predictors)
      else:
        continue

    print("Mean Absolute Error: %.3f - Mean Squared Error: %.3f" %(mean(mae_total), mean(mse_total)))
    print("Minimum Mean Squared Error: %.3f" %(min(mse_total)))
    
    Y = mse_total
    X = mse_list

    Z = [x for _,x in sorted(zip(Y,X))]
    print(Z[0])
    print("Minimum Number of Features :", len(Z[0]))
    Z_temp = Z[:10]

    arr_num = [0] * 12

    for item in Z_temp:
      if 'var_1' in item:
        arr_num[0]=arr_num[0] + 1;
      if 'var_2' in item:
        arr_num[1]=arr_num[1] + 1;
      if 'var_3' in item:
        arr_num[2]=arr_num[2] + 1;
      if 'var_4' in item:
        arr_num[3]=arr_num[3] + 1;
      if 'var_5' in item:
        arr_num[4]=arr_num[4] + 1;
      if 'var_6' in item:
        arr_num[5]=arr_num[5] + 1;
      if 'var_7' in item:
        arr_num[6]=arr_num[6] + 1;
      if 'var_8' in item:
        arr_num[7]=arr_num[7] + 1;
      if 'var_9' in item:
        arr_num[8]=arr_num[8] + 1;
      if 'var_10' in item:
        arr_num[9]=arr_num[9] + 1;
      if 'var_11' in item:
        arr_num[10]=arr_num[10] + 1;
      if 'var_12' in item:
        arr_num[11]=arr_num[11] + 1;

    print(arr_num)
