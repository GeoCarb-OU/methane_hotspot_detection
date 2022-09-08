# This is the main package file. All the other resources are called from this file. 

# ALL required imports 
from datahandle import *
import sys
import inputhandler

createdataset.data_creation(inputhandler.inoutpath(sys.argv)) 
