import argparse
import os

import pandas as pd

from utils import (
    UTTERANCES_WITH_SPEECH_ACTS_FILE, SPEAKER_CODE_CHILD, get_num_unique_words,
)

ANNOTATED_UTTERANCES_FILE = os.path.expanduser(
    "results/grammaticality/effect_of_context/utterances_for_annotation_compare.csv"
)


def compare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0)

    utterances.dropna(subset=["is_grammatical", "is_grammatical_no_prev"], inplace=True)

    utts_changed = utterances[utterances.is_grammatical != utterances.is_grammatical_no_prev]
    percentage_changed = 100 * len(utts_changed) / len(utterances)
    print(f"percentage_changed: {percentage_changed}%")

    utts_unknown = utterances[utterances.is_grammatical_no_prev == "?"]
    percentage_unknown = 100 * len(utts_unknown) / len(utterances)
    print(f"percentage_unknown before: {percentage_unknown}%")

    utts_unknown_after = utterances[utterances.is_grammatical == "?"]
    percentage_unknown_after = 100 * len(utts_unknown_after) / len(utterances)
    print(f"percentage_unknown after: {percentage_unknown_after}%")

    change_0_to_1 = utterances[(utterances.is_grammatical_no_prev == "0") & (utterances.is_grammatical == "1")]
    percentage_0_to_1 = 100 * len(change_0_to_1) / len(utterances)
    print(f"percentage_0_to_1: {percentage_0_to_1}%")

    change_1_to_0 = utterances[(utterances.is_grammatical_no_prev == "1") & (utterances.is_grammatical == "0")]
    percentage_1_to_0 = 100 * len(change_1_to_0) / len(utterances)
    print(f"percentage_1_to_0: {percentage_1_to_0}%")


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=ANNOTATED_UTTERANCES_FILE,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    compare(args)