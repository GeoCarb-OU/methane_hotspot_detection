# This is the main package file. All the other resources are called from this file. 

# ALL required imports 
from datahandle import createdataset
import sys
import inputhandler
INPATH = str("")
OUTPATH = str("")

#createdataset.check_print()
INPATH, OUTPATH = inputhandler.inoutpath(sys.argv) 
createdataset.data_creation(INPATH, OUTPATH) 
print(OUTPATH)
print(type(OUTPATH))