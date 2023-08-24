#!/bin/bash
#SBATCH --partition=geocarb
#SBATCH --cpus-per-task=20
# memory in MB
#SBATCH --mem=32000
#SBATCH --output=results/unet/test_%j_stdout.txt
#SBATCH --error=results/unet/test_%j_stderr.txt
#SBATCH --time=1:00:00
#SBATCH --job-name=unet(gpu)
#SBATCH --mail-user=vishnupk@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/vishnupk/geocarb/methane_hotspot_detection
##SBATCH --array=0
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf_2023_07

# 2023
python base.py @args.txt --job_id $SLURM_JOB_ID