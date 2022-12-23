#!/bin/bash
#
#SBATCH --job-name=annotate
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080:1
###SBATCH --nodelist=lisnode3
###SBATCH --constraint=cuda75|cuda80|cuda86
#SBATCH --output=out/annotate_grammaticality_eval.out

source activate cf_timing
python -u annotate_grammaticality.py --utterances-file ./data/manual_annotation/grammaticality_manually_annotated.csv
#python annotate_grammaticality.py --utterances-file ~/data/communicative_feedback/utterances_for_annotation.csv --grammaticality-annotation-models "cointegrated/roberta-large-cola-krishna2020"
