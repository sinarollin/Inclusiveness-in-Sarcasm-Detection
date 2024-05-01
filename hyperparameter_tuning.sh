#!/bin/bash -l
#SBATCH --chdir=/scratch/izar/roellin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559


python hyperparam_tuning.py