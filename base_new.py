'''
GeoCarb Methane Emisison Estimation with UNET 

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
Editor: Vishnu Kadiyala (vishnupk@ou.edu) 

Methane hotspot emission Estimation problem on the GeoCarb WRF data. 

'''

import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K



# You need to provide this yourself
from unet.model import *
from datahandle.data_loader import *
import matplotlib.pyplot as plt


#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

#################################################################


def create_parser():
    '''
    create argument parser
    '''

    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='UNET', fromfile_prefix_chars='@')
    
    return parser

def exp_type_to_hyperparameters(args):
    '''
    Translate the exp_type into a hyperparameter set

    :param args: ArgumentParser
    :return: Hyperparameter set (in dictionary form)
    '''
    if args.exp_type is None:
        # ADD MORE ARGUMENTS TO TEST DIFFERENT CONFIG OPTIONS
        p = {'rotation': range(1)}
    else:
        assert False, "Unrecognized exp_type"

    return p

######################################################################### 
def check_args(args):
    assert None not in [args.data_dir, args.exp_type], "Missing required argument"

def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    '''
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp_index
    if(index is None):
        return ""
    
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)

def generate_fname(args, params_str):
    
    return "string_with_parameters"

def build_regularization_kernel(lambda_l1, lambda_l2):
    '''
    This part is used to update kernel with regularization if regularization exists
    so that we can add the regularization to the model 
    
    I checked if kernel = None, then no regularization exists and we can add the model without regularization
    This had no issues with the model. Hence, proceeded with the model building.
    '''
    # Check if regularization exists, then update kernel with regularization
    #if either lambda_l2 or lambda_l1 exists, then uodate kernel with regularization:
    if (lambda_l2 is not None) or (lambda_l1 is not None):
        
        #if both lambda_l2 and lambda_l1 exist, then update kernel with l1_l2 regularization:
        if (lambda_l2 is not None) and (lambda_l1 is not None):
            kernel = tf.keras.regularizers.l1_l2(lambda_l1, lambda_l2)
        else:
            #if only lambda_l1 exists, then update kernel with l1 regularization:
            if lambda_l1 is not None:
                kernel = tf.keras.regularizers.l1(lambda_l1)
            
            #if only lambda_l2 exists, then update kernel with l2 regularization:
            if lambda_l2 is not None:
                kernel = tf.keras.regularizers.l2(lambda_l2)
    #set kernel to None if no regularization exists
    else:
        kernel = None
    
    
    return kernel 


def execute_exp(args = None):
    
        '''
    Perform the training and evaluation for a single model
    
    :param args: Argparse arguments
    '''
    # Create args if it has not been loaded already
    
    if args is None:
        parser = create_parser()
        args = parser.parse_args([])
        
    # print(args.exp_index)
    
    # Override arguments if we are using exp_index
    args_str = augment_args(args)

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch*multi_gpus   
    
    
    return None

def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser

    '''
    
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d"%ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s"%(i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s"%(len(indices),','.join(str(x) for x in indices)))
    
    
    
    
#################################################################
if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
    # WandB initialization
    wandb.init(project="MethaneHotspotDet", entity="ai2es", config=args, name="unet_deep", reinit=False)

    
    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU') 
    n_physical_devices = len(physical_devices)
    print(physical_devices)
    if(n_physical_devices > 0):
        py3nvml.grab_gpus(num_gpus=n_physical_devices, gpu_select=range(n_physical_devices))
        # for device in physical_devices:
            # tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_physical_devices)
    else:
        print('NO GPU')


    if(args.check):
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment

        # Set number of threads, if it is specified
        if args.cpus_per_task is not None:
            tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
            tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

        execute_exp(args, multi_gpus=n_physical_devices)
