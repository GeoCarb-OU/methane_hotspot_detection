#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=15000
#SBATCH --output=results/srnn/run2/hw5_%04a_stdout.txt
#SBATCH --error=results/srnn/run2/hw5_%04a_stderr.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=MHP_detect
#SBATCH --mail-user=vishnupk@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/vishnupk/deep_learning_practice_oscer/hw5
#SBATCH --array=0-4
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# 2023
python base.py @oscer.txt @exp.txt @simpleRNN.txt 