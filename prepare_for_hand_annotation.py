import argparse
import os

import pandas as pd

from annotate import DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED, DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE
from utils import (
    str2bool, SPEAKER_CODE_CHILD, get_num_unique_words,
)

TO_ANNOTATE_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_for_annotation.csv"
)


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0)

    utterances = utterances[utterances.speaker_code == SPEAKER_CODE_CHILD]

    num_unique_words = get_num_unique_words(utterances.transcript_clean)
    utts_to_annotate = utterances[(num_unique_words > 1)]
    utts_to_annotate = utts_to_annotate[utts_to_annotate.is_speech_related & utts_to_annotate.is_intelligible]

    utts_to_annotate = utts_to_annotate.sample(1000, random_state=1)

    return utts_to_annotate


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        required=True,
    )
    argparser.add_argument(
        "--label-partially-speech-related",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
        help="Label for partially speech-related utterances: Set to True to count as speech-related, False to count as "
             "not speech-related or None to exclude these utterances from the analysis",
    )
    argparser.add_argument(
        "--label-partially-intelligible",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
        help="Label for partially intelligible utterances: Set to True to count as intelligible, False to count as unintelligible or None to exclude these utterances from the analysis",
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    annotated_utts = prepare(args)

    os.makedirs(os.path.dirname(TO_ANNOTATE_UTTERANCES_FILE), exist_ok=True)
    annotated_utts.to_csv(TO_ANNOTATE_UTTERANCES_FILE)
