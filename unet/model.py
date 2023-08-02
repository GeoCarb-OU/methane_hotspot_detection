# required packages: tensorflow, keras, pandas, numpy, matplotlib, sklearn
import tensorflow as tf 
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import copy

# Model building packages
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import InputLayer
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Reshape
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential



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
    
    # Build a Functional model 
    
    inputs = Input(input_shape)
    
    x = Conv2D(32, (3,3), padding = 'same', activation = activation_convolution, name = 'input')(inputs)
   
    x_down = []
    x_up = []
    # Add Batch Normalization
    for i in range(len(n_filters)):
        
        x = Conv2D(n_filters[i], kernelSize[i], padding = padding, activation = activation_convolution, kernel_initializer = 'truncated_normal', use_bias = True, name = 'conv_down_{}'.format(i))(x)
        print(x.shape)
        x_down.append(x)
        x = MaxPooling2D(pool_size = (2,2),padding = padding,  name = 'MaxPool_down_{}'.format(i) )(x)
        
        
    
    for i in reversed(range(len(n_filters))):
        
        x = Conv2D(n_filters[i], kernelSize[(len(kernelSize) - (i + 1))], padding = padding, use_bias = True,  activation = activation_convolution, name = 'conv_up_{}'.format(i))(x)
        x = UpSampling2D(size = (2,2), interpolation = 'nearest', name ='UpSampling_{}'.format(i))(x)
        x = Concatenate()([x_down[i],x])
        
    
    x_out = Conv2D(1 , (1,1), padding = 'same', use_bias = True,  activation = 'sigmoid', kernel_initializer = 'truncated_normal', name = 'output_layer')(x)
    
    # kernel_initializer = 'zeros',
    model = Model(inputs = [inputs], outputs = [x_out])
    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate)

    # Compile model
    model.compile(
            optimizer=opt,
            loss='BinaryCrossentropy',
            metrics=[tf.keras.metrics.BinaryAccuracy()]
                        )
    
    return model