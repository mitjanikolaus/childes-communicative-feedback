#!/bin/bash
#
#SBATCH --job-name=annotate
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080:1
#SBATCH --nodelist=lisnode3
###SBATCH --constraint=cuda75|cuda80|cuda86
#SBATCH --output=out/annotate_grammaticality.out

source activate cf_timing
python -u annotate_grammaticality.py --utterances-file ~/data/communicative_feedback/utterances_annotated.csv --grammaticality-annotation-models "cointegrated/roberta-large-cola-krishna2020" # "yevheniimaslov/deberta-v3-large-cola"

