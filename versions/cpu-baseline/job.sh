#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1-1
#SBATCH -c 32
#SBATCH -t 15
#SBATCH --export=NONE
#SBATCH -o log/slurm.%j.out
#SBATCH -e log/slurm.%j.err
##SBATCH --gres=gpu:1          

module purge

module load rapidsai/21.06

python likelihood_func1.py
