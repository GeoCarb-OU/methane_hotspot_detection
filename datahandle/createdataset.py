# This file is used to create dataset from all the .nc4 available at disposal

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

#print("LIBRARY IMPORTS COMPLETED")
## Go through all the data 

#for name in sorted(glob.glob('/ourdisk/hpc/geocarb/vishnupk/WRF/*')):
#	ds = nc.Dataset(name)

#print("ABLE TO READ ALL THE DATA")
## Create arrays for data gathering


def check_print():
    print("Module has been loaded")

def data_creation(INPATH = '/ourdisk/hpc/geocarb/vishnupk/WRF/*', OUTPATH = '/scratch/vishnupk/xiao_data_v4.pkl'):
    XC = []
    EM = []
    P = []
    PB = []
    PSFC = []

    print(OUTPATH)
    print(type(OUTPATH))

    for name in sorted(glob.glob(INPATH)): 
        print(name)#/Users/vishnu/Documents/GeoCarb/Data.nosync/GeoCarb Data
        ds = nc.Dataset(name)
        XC.append(ds['CH4_ANT'][0,:,:,:] + ds['CH4_BIO'][0,:,:,:] + ds['CH4_BCK'][0,:,:,:] + ds['CH4_BBU'][0,:,:,:])
        EM.append(ds['E_CH4'][0,0,:,:])
        P.append(ds['P'][0,:,:,:])
        PB.append(ds['PB'][0,:,:,:])
        PSFC.append(ds['PSFC'][0,:,:])
        #CH4.append(ds['CH4'])
        lat = np.array(ds['XLAT'])
        lon = np.array(ds['XLONG'])
        #print(name)
    print(len(XC))

    print("READ ALL THE DATA")

    # Find the total pressure 
    TP =[]
    # simply add P and PB
    for i in range(len(P)):
        TP.append(P[i] + PB[i])


    delta = []

    for i in trange(len(XC)):
        temp = np.zeros_like(TP[0])
        
        temp[0,:,:] = TP[i][0,:,:]
        
        for j in range(1,47):
            temp[j,:,:] = TP[i][j,:,:] - TP[i][j-1, :,:]
            
        delta.append(temp)

    PWF = []

    for i in range(len(XC)):
        temp2 = delta[i]/delta[i][0,:,:]
        
        PWF.append(temp2)

    XCH4 = []

    for i in range(len(XC)):
        temp1 = sum(XC[i]*PWF[i])

        XCH4.append(temp1)

    print("CREATING DATA FILE")

    df = pd.DataFrame(columns= ['Surface Pressure', 'XCH4', 'Emissions'])


    for i in range(len(XC)):
        df = df.append({'Surface Pressure': PSFC, 'XCH4': XCH4[i], 'Emissions': EM[i]}, ignore_index = True)

    df.to_pickle(OUTPATH)

    print("COMPLETED")