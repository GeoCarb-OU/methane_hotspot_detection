# required packages: tensorflow, keras, pandas, numpy, matplotlib, sklearn
import tensorflow as tf 
import sys
import argparse
import pickle
import pandas as pd
import numpy as np


# Model building packages
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.layers import UpSampling2D
from keras.layers import SpatialDropout2D
from keras.layers import InputLayer
from keras import Input
from keras.layers import Concatenate, Reshape
from keras import Model
from keras.models import Sequential


def create_unet_network(
    input_shape,
    n_filters = [10,5],
    kernelSize = [3,3],
    pool_size = [2,2,2],
    spatial_dropout = None,
    padding = None,
    lrate = 0.001,
    activation_convolution = 'elu',
    output_shape = 7,
    kernel = None,
):
  
# Build a sequential model 
  model1 = Sequential()

  # Add input layer 
  model1.add(InputLayer(input_shape=input_shape, name = 'input_layer'))

  # Add downsampling layers to the Unet 
  # Add convolutional layers to the Unet
  # SpatialDropout if exists and maxpooling layers if pool_size > 1
  for i,n in enumerate(n_filters): 

    model1.add(Conv2D(n, kernelSize[i], padding = padding, activation = activation_convolution, name = 'conv_down_{}'.format(i) ))

    if spatial_dropout is not None: 
      model1.add(SpatialDropout2D(spatial_dropout, name = 'spatial_drop_{}'.format(i)))
    
    if pool_size[i] > 1: 
      model1.add(MaxPooling2D(pool_size = (pool_size[i], pool_size[i]),padding = padding,  name = 'MaxPool_down_{}'.format(i) ))

  # TODO : zip filters and poolsizes and kernel sizes
  # Upsampling layers 

  for i,n in reversed(list(enumerate(n_filters))):

    model1.add(Conv2D(n, kernelSize[(len(kernelSize) - (i + 1))], padding = padding, activation = activation_convolution, name = 'conv_up_{}'.format(i) ))

    if spatial_dropout is not None: 
      model1.add(SpatialDropout2D(spatial_dropout, name = 'spatial_up_{}'.format(i)))

    if pool_size[(len(pool_size) - (i+1))] > 1: 
      model1.add(UpSampling2D(size = (pool_size[(len(pool_size) - (i+1))], pool_size[(len(pool_size) - (i+1))]), interpolation = 'nearest', name ='UpSampling_{}'.format(i)))

  # Output layer 
  model1.add(Conv2D(1 , (1,1), padding = 'same', activation = 'sigmoid', name = 'output_layer'))
  
  model1.add(Reshape(output_shape))
  # optimizer
  opt = tf.keras.optimizers.Adam(learning_rate=lrate)


  # Compile model
  model1.compile(
          optimizer=opt,
          loss='BinaryCrossentropy',
          metrics=[tf.keras.metrics.Accuracy()]
                    )

      
  return model1