# regression mlp model for the abalone dataset
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error, max_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
from statistics import mean, median
from sklearn.model_selection import KFold
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import warnings
import difflib

from itertools import chain, combinations

if __name__ == '__main__':

    # "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"

    def powerset(iterable):    
        s = list(iterable)  # allows duplicate elements
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    combo_arr = []
    stuff = ['var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12']
    for i, combo in enumerate(powerset(stuff), 1):
        combo_arr.append(combo)

    my_combo = combo_arr[(len(stuff)+1):]
    print(len(my_combo))

    # Read data from CSV file
    data = pd.read_csv (r'./filename_12vars_21people_mag.csv')
    warnings.filterwarnings("ignore")
    # Define y = f(x)
    predictors_list = np.array(my_combo)
    outcome = ['SpO2']

    # Normalization Parameter
    norm_param = 100

    # Define kfold cross validation
    kf = KFold(n_splits=5, random_state=None, shuffle=True)

    mae_total = []
    mse_total = []
    mse_list = []

    #regression_list = [LinearRegression(), Ridge(), Lasso(alpha = 0.0001), Lasso(alpha = 0.001)]

    for predictors in predictors_list:
        
      X = data[np.array(predictors)].values
      y = data[outcome]
      #print(predictors)

      X = np.array(X)
      y = np.array(y)
      mae_20 = []
      mse_20 = []    

      for i in range(20):
        mae = []
        mse = []
        for train_index, test_index in kf.split(X, y):
          X_train, X_test = X[train_index], X[test_index]
          y_train, y_test = y[train_index]/norm_param, y[test_index]/norm_param

          # define the keras model
          model = Sequential()
          model.add(Dropout(0.2))
          model.add(Dense(128, input_dim=len(predictors), activation='relu', kernel_initializer='he_uniform'))
          model.add(Dropout(0.2))
          model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
          model.add(Dropout(0.2))
          model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
          model.add(Dropout(0.2))
          model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
          model.add(Dense(1, activation='linear'))
          #model.summary()

          # compile the keras model
          model.compile(loss='mse', optimizer='adam')

          checkpoint_filepath = "/tmp/checkpoint"
          checkpointer = tf.keras.callbacks.ModelCheckpoint(#filepath = 'model.h5',
                                                            checkpoint_filepath,
                                                            monitor = 'val_loss', 
                                                            verbose = 0, 
                                                            save_best_only = True,
                                                            save_weights_only = True,
                                                            mode = 'min')
          callbacks = [checkpointer]

          # fit the keras model on the dataset
          model.fit(X_train, y_train, epochs=25, batch_size=5, verbose=0, validation_data = (X_test, y_test), callbacks = callbacks)
          model.load_weights(checkpoint_filepath)

          # evaluate on test set
          y_hat = model.predict(X_test)
          results = model.evaluate(X_test, y_test, verbose=0)

          # metrics
          mae.append(mean_absolute_error(y_test, y_hat))
          mse.append(results)

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
    print(Z)
    print("Minimum Number of Features :", (len(Z[0])))
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
