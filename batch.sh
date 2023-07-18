#!/bin/bash
#SBATCH --partition=geocarb
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=15000
#SBATCH --output=results/unet/run1_%04a_stdout.txt
#SBATCH --error=results/unet/run1_%04a_stderr.txt
#SBATCH --time=05:00:00
#SBATCH --job-name=unet_deep
#SBATCH --mail-user=vishnupk@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/vishnupk/geocarb/methane_hotspot_detection
##SBATCH --array=0
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# 2023
python base.py @args.txt