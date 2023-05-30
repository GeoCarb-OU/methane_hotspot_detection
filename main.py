# This is the main package file. All the other resources are called from this file. 

# ALL required imports 
from datahandle import *

try:
	createdataset.data_creation(INPATH,OUTPATH)
	print("datafile creation succeded")
except:
	print("datafile creation failed") 


