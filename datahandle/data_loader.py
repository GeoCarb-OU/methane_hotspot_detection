## ALL IMPORTS
import numpy as np
import pandas as pd
import glob 
import netCDF4 as nc
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import trange
from math import exp
from scipy import ndimage
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from random import randint
import random
import os 
from sklearn.model_selection import train_test_split
import tensorflow as tf 

def get_data():
    XC = []
    EM = []

    for file in sorted(glob.glob('/ourdisk/hpc/geocarb/vishnupk/WRF/*_18*')):
        print(file)
        ds = nc.Dataset(file)
        XC.append(ds['CH4_ANT'][0,:,:,:] + ds['CH4_BIO'][0,:,:,:] + ds['CH4_BCK'][0,:,:,:] + ds['CH4_BBU'][0,:,:,:])
        EM.append(ds['E_CH4'][0,0,:,:])
        
        lat = np.array(ds['XLAT'])
        lon = np.array(ds['XLONG'])
    print(len(XC))
    df = pd.DataFrame(columns=['XC', 'EM', 'lat', 'lon'])
    for i in range(len(XC)):
        df = df.append({'XC': XC[i], 'EM': EM[i], 'lat': lat, 'lon': lon}, ignore_index=True)

    df.to_pickle('/ourdisk/hpc/geocarb/vishnupk/xiao_data_05_23_18.pkl')
    
    return df

def read_data(filename = '/ourdisk/hpc/geocarb/vishnupk/folds/xiao_data_12_v1.pkl'):
    if filename is None:
        filename = '/ourdisk/hpc/geocarb/vishnupk/folds/xiao_data_12_v1.pkl'
        
    df = pd.read_pickle(filename)
    return df

def data_loader(filename = None, test_size = 0.2, random_state = 42, batch_size = 8, buffer_size = 1024):
    # Read data from pickle file 
    data = read_data(filename) 
    
    # Load the data into X and y 
    X = data['XC']
    y = data['EM'] 
    
    # # Convert to numpy array 
    # X = X.astype('float32')
    # y = y.astype('float32')
    
    #Split the data into train and test 
    x_train , x_rem, y_train, y_rem = train_test_split(X, y, test_size = test_size*2, random_state = random_state) 
    
    x_test , x_val, y_test, y_val = train_test_split(x_rem, y_rem, test_size = 0.5, random_state = random_state)
    
    #Create tensorflow dataset with buffer size 1024 and batch size 8 
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
    
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    validation_dataset = validation_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
     
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)

    return train_dataset, validation_dataset, test_dataset 