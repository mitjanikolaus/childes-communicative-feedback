import argparse
import os

import pandas as pd

from utils import (
    UTTERANCES_WITH_SPEECH_ACTS_FILE, SPEAKER_CODE_CHILD, get_num_unique_words, add_prev_utts,
)

TO_ANNOTATE_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_for_annotation.csv"
)


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0)

    print("Adding previous utterances..")
    utterances = add_prev_utts(utterances)

    if args.annotated_utterances_file:
        annotated_utts = pd.read_csv(args.annotated_utterances_file, index_col=0)
        annotated_utts = annotated_utts[["is_grammatical", "categories", "note"]]

        utterances = utterances.merge(annotated_utts, how="left", left_index=True, right_index=True)
        utterances.dropna(subset=["is_grammatical"], inplace=True)

    utterances.dropna(subset=["prev_transcript_clean"], inplace=True)

    utts_to_annotate = utterances[utterances.speaker_code == SPEAKER_CODE_CHILD]

    num_unique_words = get_num_unique_words(utts_to_annotate.transcript_clean)
    utts_to_annotate = utts_to_annotate[(num_unique_words > 1)]
    utts_to_annotate = utts_to_annotate[utts_to_annotate.is_speech_related & utts_to_annotate.is_intelligible]

    if args.max_utts:
        utts_to_annotate = utts_to_annotate.sample(args.max_utts, random_state=1)

    return utts_to_annotate


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=UTTERANCES_WITH_SPEECH_ACTS_FILE,
    )
    argparser.add_argument(
        "--annotated-utterances-file",
        type=str,
        required=False,
    )
    argparser.add_argument(
        "--max-utts",
        type=int,
        default=None,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = prepare(args)

    os.makedirs(os.path.dirname(TO_ANNOTATE_UTTERANCES_FILE), exist_ok=True)
    utterances.to_csv(TO_ANNOTATE_UTTERANCES_FILE)
