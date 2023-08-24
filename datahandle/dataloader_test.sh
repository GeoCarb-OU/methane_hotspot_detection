#!/bin/bash
#SBATCH --partition=geocarb
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=50000
#SBATCH --output=results/unet/datatest_%j_stdout.txt
#SBATCH --error=results/unet/datatest_%j_stderr.txt
#SBATCH --time=24:00:00
#SBATCH --job-name=testloader
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
python check_dataloader.py