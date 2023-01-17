import argparse
import os

import pandas as pd

from utils import UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_CLEAN_FILE
from tqdm import tqdm
tqdm.pandas()

# corpora that include more than 1% errors:
CORPORA_INCLUDED = ["Bernstein", "Braunwald", "MPI-EVA-Manchester", "Providence", "Thomas", "Hall", "Gelman", "Kuczaj", "EllisWeismer", "Clark", "Post", "Bates", "Lara"]
# corpora that include all error types:
# CORPORA_INCLUDED = ["Braunwald", "EllisWeismer", "Hall", "Lara", "MPI-EVA-Manchester", "Providence", "Thomas"]


def prepare(args, sample_equal_pos_neg=False):
    utterances = pd.read_csv(args.utterances_file, index_col=0, dtype={"error": object})

    if sample_equal_pos_neg:
        utterances = utterances[utterances.corpus.isin(CORPORA_INCLUDED) | (utterances.is_grammatical == False)]

        utterances.dropna(subset=["is_grammatical"], inplace=True)
        utterances.is_grammatical = utterances.is_grammatical.astype(bool)
        utts_ungrammatical = utterances[~utterances.is_grammatical]
        utts_grammatical = utterances[utterances.is_grammatical].sample(len(utts_ungrammatical), random_state=1)
        utterances = pd.concat([utts_grammatical, utts_ungrammatical])

    else:
        # Take only grammatical utterances from target corpora
        utterances = utterances[utterances.corpus.isin(CORPORA_INCLUDED)]

    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = prepare(args)

    os.makedirs(os.path.dirname(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_CLEAN_FILE), exist_ok=True)
    utterances.to_csv(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_CLEAN_FILE)
