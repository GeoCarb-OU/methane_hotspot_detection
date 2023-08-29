#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=15000
#SBATCH --output=results/unet/wwb_%j_stdout.txt
#SBATCH --error=results/unet/wwb_%j_stderr.txt
#SBATCH --time=10:00:00
#SBATCH --job-name=unet(gpu)
#SBATCH --mail-user=vishnupk@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/vishnupk/geocarb/methane_hotspot_detection
#SBATCH --array=0-7
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate tensorflow

# 2023
python base.py @args.txt --job_id $SLURM_JOB_ID --exp_index $SLURM_ARRAY_TASK_ID