import argparse
import os

import pandas as pd

from utils import UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, ERR_UNKNOWN, FILE_FINE_TUNING_CHILDES_ERRORS
from tqdm import tqdm
tqdm.pandas()


CORPORA_INCLUDED = ["Bernstein", "Braunwald", "MPI-EVA-Manchester", "Providence", "Thomas"]


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, dtype={"error": object})

    # Take all utterances from target corpora and only ungrammatical ones from other corpora
    utterances = utterances[utterances.corpus.isin(CORPORA_INCLUDED) | ~utterances.is_grammatical]

    print(f"removing {len(utterances[utterances.labels == ERR_UNKNOWN])} rows with unknown labels")
    utterances = utterances[utterances.labels != ERR_UNKNOWN]

    utts_ungrammatical = utterances[~utterances.is_grammatical]
    utts_grammatical = utterances[utterances.is_grammatical].sample(len(utts_ungrammatical), random_state=1)
    utterances = pd.concat([utts_grammatical, utts_ungrammatical])

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

    os.makedirs(os.path.dirname(FILE_FINE_TUNING_CHILDES_ERRORS), exist_ok=True)
    utterances.to_csv(FILE_FINE_TUNING_CHILDES_ERRORS)
