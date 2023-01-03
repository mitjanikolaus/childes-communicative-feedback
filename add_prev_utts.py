import argparse
import os

import pandas as pd

from utils import (
    UTTERANCES_WITH_SPEECH_ACTS_FILE, add_prev_utts,
    UTTERANCES_WITH_PREV_UTTS_FILE, ANNOTATED_UTTERANCES_FILE, add_following_utts,
)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=ANNOTATED_UTTERANCES_FILE,
    )
    argparser.add_argument(
        "--num-utts",
        type=int,
    )
    argparser.add_argument(
        "--num-following-utts",
        type=int,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = pd.read_csv(args.utterances_file, index_col=0)

    print("Adding previous utterances..")
    if args.num_utts:
        utterances = add_prev_utts(utterances, num_utts=args.num_utts)

    if args.num_following_utts:
        utterances = add_following_utts(utterances, num_utts=args.num_following_utts)

    os.makedirs(os.path.dirname(UTTERANCES_WITH_PREV_UTTS_FILE), exist_ok=True)
    utterances.to_csv(UTTERANCES_WITH_PREV_UTTS_FILE)
