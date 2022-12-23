#!/bin/bash
#
#SBATCH --job-name=annotate
#SBATCH --ntasks=8
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080:1
###SBATCH --nodelist=lisnode3
###SBATCH --constraint=cuda75|cuda80|cuda86
#SBATCH --output=out/annotate_gpu.out

source activate cf_timing
python -u annotate.py

