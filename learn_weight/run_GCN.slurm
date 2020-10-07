#!/bin/bash 
#SBATCH -J GCN
#SBATCH --gres=gpu:1
#SBATCH -t 600:00
#SBATCH -o GCN.out
#SBATCH -e GCN.err
conda activate hzhenv
python3 train.py 
