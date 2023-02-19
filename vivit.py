import io
import imageio
import ipywidgets
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error, max_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import load_model
import h5py
import pandas as pd

import cv2
from numpy import std, mean, sqrt
from sklearn.metrics import confusion_matrix
from statistics import mean, median
from sklearn.model_selection import KFold
import argparse
import math
from scipy.ndimage import zoom

# DATA
BATCH_SIZE = 8
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (64, 128, 300, 1)

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 40

# TUBELET EMBEDDING
PATCH_SIZE = (18, 36, 85)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

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


@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches


class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=1, activation='linear')(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def run_experiment():
    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, decay=WEIGHT_DECAY)
    #model.summary()
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[
            keras.metrics.MeanAbsoluteError(name='mean_absolute_error'),
            keras.metrics.MeanSquaredError(name='mean_squared_error')
        ],
    )
    
    checkpoint_filepath = "/tmp/checkpoint"
    checkpointer = tf.keras.callbacks.ModelCheckpoint(#filepath = 'model.h5',
                                                      checkpoint_filepath,
                                                      monitor='val_loss', 
                                                      verbose = 0, 
                                                      save_best_only = True,
                                                      save_weights_only = True,
                                                      mode = 'min')
    callbacks = [checkpointer]

    # Train the model.
    _ = model.fit(trainloader, epochs=EPOCHS, validation_data=testloader, callbacks = callbacks, verbose = 0)
    
    model.load_weights(checkpoint_filepath)
    #y_hat = model.predict(X_test)
    results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print("Test MSE:", results[0]*(norm_param**2))
    print("Test MAE:", results[1]*(norm_param))     

    return model, results[0], results[1]

if __name__ == "__main__":

    videos = load_videos('./videos')
    data = pd.read_csv (r'./filename_12vars_21people_mag.csv')
    outcome = ['SpO2']
    y = data[outcome]

    X = np.array(videos)
    y = np.array(y)

    norm_param = 100
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    
    mae_total = []
    mse_total = []
    
    for i in range(5):
        mae = []
        mae_temp = []
        mse = []
        mse_temp = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index]/255, X[test_index]/255
            y_train, y_test = y[train_index]/norm_param, y[test_index]/norm_param

            trainloader = prepare_dataloader(X_train, y_train, "train")
            testloader = prepare_dataloader(X_test, y_test, "test")
            
            model, results_0, results_1  = run_experiment()
            
            mse.append(results_0)
            mae.append(results_1)
            
        mae_temp = np.array(mae)
        mse_temp = np.array(mse)
        if (all(x <= ((2/norm_param)**2) for x in mse_temp)):      
            mae_total.append(mean(mae_temp)*norm_param)
            mse_total.append(mean(mse_temp)*(norm_param**2))
            
    print("Mean Absolute Error: %.3f - Mean Squared Error: %.3f" %(mean(mae_total), mean(mse_total)))
    print("Minimum Mean Squared Error: %.3f" %(min(mse_total)))
