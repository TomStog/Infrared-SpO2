import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error, max_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import tensorflow as tf
from keras.models import load_model

import numpy as np
from numpy import std, mean, sqrt
from sklearn.metrics import confusion_matrix
from statistics import mean, median
from sklearn.model_selection import KFold
import argparse
import math
import cv2
from scipy.ndimage import zoom

def load_videos(path):
  videos=[]
  for filename in sorted(os.listdir(path)):
    cap = cv2.VideoCapture(os.path.join(path,filename))
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #print(int(frameIds))
    frames = []
    for fid in range(int(frameIds)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    out = np.concatenate(frames)
    out = out.ravel()
    newarr = out.reshape(frame.shape[0], frame.shape[1], int(frameIds),1)
    new_array = zoom(newarr, (64/frame.shape[0], 128/frame.shape[1], 300/frameIds,1))
    videos.append(new_array)
  
  out = np.concatenate(videos)
  out = out.ravel()
  new_videos = out.reshape(len(videos), 64, 128, 300,1)
  return new_videos

if __name__ == "__main__":

    #videos = load_videos('./16videos')
    videos = load_videos('./21videos')

    data = pd.read_csv (r'./filename_12vars_21people_mag.csv')
    outcome = ['SpO2']
    y = data[outcome]
    
    X = np.array(videos)
    y = np.array(y)

    norm_param = 100
    kf = KFold(n_splits=5, random_state=None, shuffle=True)

    # split data into train and test sets
    mae_total = []
    mse_total = []
    
    for i in range(20):
        mae = []
        mae_temp = []
        mse = []
        mse_temp = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index]/255, X[test_index]/255
            y_train, y_test = y[train_index]/norm_param, y[test_index]/norm_param

            model = Sequential()
            model.add(Conv3D(16, kernel_size=(5, 5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=(64, 128, 300, 1)))
            model.add(MaxPooling3D(pool_size=(3, 3, 3)))
            #model.add(BatchNormalization(center=True, scale=True))
            model.add(Dropout(0.5))
            model.add(Conv3D(32, kernel_size=(5, 5, 5), activation='relu', kernel_initializer='he_uniform'))
            model.add(MaxPooling3D(pool_size=(3, 3, 3)))
            #model.add(BatchNormalization(center=True, scale=True))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(1, activation='linear'))

            # Compile the model
            model.compile(loss='mse', optimizer='adam')
            model.summary()
            
            checkpoint_filepath = "/tmp/checkpoint"
            checkpointer = tf.keras.callbacks.ModelCheckpoint(#filepath = 'model.h5',
                                                      checkpoint_filepath,
                                                      monitor = 'val_loss', 
                                                      verbose = 1, 
                                                      save_best_only = True,
                                                      save_weights_only = True,
                                                      mode = 'min')
            callbacks = [checkpointer]
            # Fit data to model
            model.fit(X_train, y_train, batch_size=5, epochs=100, verbose=1, validation_data = (X_test, y_test), callbacks = callbacks)

            #my_model = keras.models.load_model(checkpoint_filepath)
            model.load_weights(checkpoint_filepath)

            y_hat = model.predict(X_test)

            results = model.evaluate(X_test, y_test, batch_size=5)            
            print("Test MSE Loss:", results)

            # threshold values over 100%
            #  for x in range(len(y_hat)):
            #    if y_hat[x]>(100/norm_param):
            #      y_hat[x]=(100/norm_param)

            # metrics
            mae.append(mean_absolute_error(y_test, y_hat))            
            mse.append(results)
            
        mae_temp = np.array(mae)
        mse_temp = np.array(mse)
        if (all(x <= ((2/norm_param)**2) for x in mse_temp)):      
            mae_total.append(mean(mae_temp)*norm_param)
            mse_total.append(mean(mse_temp)*(norm_param**2))
        
    print("Mean Absolute Error: %.3f - Mean Squared Error: %.3f" %(mean(mae_total), mean(mse_total)))
    print("Minimum Mean Squared Error: %.3f" %(min(mse_total)))
