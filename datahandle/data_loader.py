## ALL IMPORTS
import numpy as np
import pandas as pd
import glob 
# import netCDF4 as nc
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
# from tqdm import trange
from math import exp
from scipy import ndimage
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from random import randint
import random
import os 
# from sklearn.model_selection import train_test_split
import tensorflow as tf 

def get_data():
    l = ["00", "03", "06", "09", "12", "15", "18", "21"]
    
    for t in l:
        XC = []
        EM = []
        #for file in sorted(glob.glob('/ourdisk/hpc/geocarb/vishnupk/WRF/*_18*')):
        for file in sorted(glob.glob('/ourdisk/hpc/geocarb/vishnupk/WRF/*_' + t + '*')):
            print(file)
            ds = nc.Dataset(file)
            XC.append(ds['CH4_ANT'][0,:,:,:] + ds['CH4_BIO'][0,:,:,:] + ds['CH4_BCK'][0,:,:,:] + ds['CH4_BBU'][0,:,:,:])
            EM.append(ds['E_CH4'][0,0,:,:]) # Simulation Only 
            
            lat = np.array(ds['XLAT'])
            lon = np.array(ds['XLONG'])
        print(len(XC))
        df = pd.DataFrame(columns=['XC', 'EM', 'lat', 'lon'])
        for i in range(len(XC)):
            df = df.append({'XC': XC[i], 'EM': EM[i], 'lat': lat, 'lon': lon}, ignore_index=True)

        df.to_pickle('/ourdisk/hpc/geocarb/vishnupk/xiao_data_' + t + '.pkl')
    
    return df

def check_files(dataset_path):
    train_path = os.path.join(dataset_path, 'train.tfrecords')
    validation_path = os.path.join(dataset_path, 'validation.tfrecords')
    test_path = os.path.join(dataset_path, 'test.tfrecords')
    
    if os.path.exists(train_path) and os.path.exists(validation_path) and os.path.exists(test_path):
        return True
    else:
        return False

def read_pkl(filename = '/ourdisk/hpc/geocarb/vishnupk/folds/xiao_data_12_v1.pkl'):
    if filename is None:
        filename = '/ourdisk/hpc/geocarb/vishnupk/folds/xiao_data_12_v1.pkl'
        
    df = pd.read_pickle(filename)
    return df

def data_loader(filename = None, test_size = 0.2, random_state = 42, batch_size = 8, buffer_size = 1024, threshold = 15, repeat = False, save_dataset = False, data_path = '/ourdisk/hpc/geocarb/vishnupk/datasets/methane/12/', testing = False):
    
    data_path = data_path +'/'+ str(threshold) + '_'
    data_exists = check_files(data_path)
    print(data_exists)
    
    # get_data()
    
    if data_exists:
        train_dataset = tf.data.Dataset.load(data_path + 'train.tfrecords')
        validation_dataset = tf.data.Dataset.load(data_path + 'validation.tfrecords')
        test_dataset = tf.data.Dataset.load(data_path + 'test.tfrecords')
    
    else:
        # Read data from pickle file
        data = read_pkl(filename) 
        
        # Load the data into X and y 
        X = data['XC']
        Y = data['EM'] 
        
        # Preprocessing steps 
        X = X.to_list()
        X = np.ma.getdata(X)
        X = X[:,0,:,:]
        
        Y = Y.to_list()
        Y = np.array(Y)
        Y = np.where(Y > threshold, 1, 0) 
        
        # # Convert to float32  
        X = X.astype('float32')
        Y = Y.astype('float32')
        
        # Log normalize X values
        # X = np.log(X)
        # Normalize data 
        X = (X - np.min(X))/(np.max(X) - np.min(X))
        
        print('Float Conversion Completed')
        # Resize to make it 256,256 (2^x)
        X_resized = []
        Y_resized = []
        for x,y in zip(X,Y):
            X_resized.append(np.resize(x, (256,256)))
            Y_resized.append(np.resize(y, (256,256)))
        
        print('resize completed')
        del X, Y
        
        # Build a history of 5 prior days: 
        X_new = []
        for i in range(5, len(X_resized)):
            X_new.append(np.array(X_resized[i-5:i]).reshape(256,256,5))

        y_new = Y_resized[5:len(Y_resized)]
        
        del X_resized, Y_resized
        
        # Split data 
        # test_size = 0.2
        test_split = int(len(X_new)*test_size)
        
        # #Split the data into train and test 
        # x_train , x_rem, y_train, y_rem = train_test_split(X, y, test_size = test_size*2, random_state = random_state) 
        # x_test , x_val, y_test, y_val = train_test_split(x_rem, y_rem, test_size = 0.5, random_state = random_state)
        
        # # Expand doimensions of the data    
        
        # x_train = np.expand_dims(x_train, axis = -1)
        # x_val = np.expand_dims(x_val, axis = -1)
        # x_test = np.expand_dims(x_test, axis = -1)
        
        
        # Autocorrelation the data (Split change)
        x_train = X_new[:-(2*test_split)]
        y_train = y_new[:-(2*test_split)]
        
        x_val = X_new[-(2*test_split):-(test_split)]
        y_val = y_new[-(2*test_split):-(test_split)]
        
        x_test = X_new[-(test_split):]
        y_test = y_new[-(test_split):]
        
        print('Split Data Completed')
        
        print('Creating Tensorflow Dataset....')
        #Create tensorflow dataset with buffer size 1024 and batch size of batchsize (default is 8)  
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        #train_dataset = train_dataset.map(lambda x, y: tf.expand_dims(x, -1), y)
        
        
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        #validation_dataset = validation_dataset.map(lambda x, y: tf.expand_dims(x, -1), y)
        
        
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        #test_dataset = test_dataset.map(lambda x, y: tf.expand_dims(x, -1), y)
        
        
        # save TF records after splitting
        train_dataset.save(data_path + 'train.tfrecords')
        validation_dataset.save(data_path + 'validation.tfrecords')
        test_dataset.save(data_path + 'test.tfrecords')  
    
    if not testing:
        # Shuffle Dataset and batch it
        train_dataset = train_dataset.shuffle(buffer_size = buffer_size).batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=1024)
        validation_dataset = validation_dataset.shuffle(buffer_size = buffer_size).batch(batch_size)
        test_dataset = test_dataset.shuffle(buffer_size = buffer_size).batch(batch_size)
        
        #Repeat the dataset if needed
        if repeat:
            train_dataset = train_dataset.repeat()

        # train_dataset = train_dataset.map(lambda x, y: tf.expand_dims(x, -1), y)
        # validation_dataset = validation_dataset.map(lambda x, y: tf.expand_dims(x, -1), y)
        # test_dataset = test_dataset.map(lambda x, y: tf.expand_dims(x, -1), y)
        
        return train_dataset, validation_dataset, test_dataset 