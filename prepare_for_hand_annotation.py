import argparse
import os
import re

import numpy as np
import pandas as pd

from annotate import is_intelligible, is_speech_related, clean_preprocessed_utterance, \
    DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED, DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE
from utils import (
    str2bool,
)

TO_ANNOTATE_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_for_annotation.csv"
)


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0)

    utterances = utterances[utterances.speaker_code == "CHI"]

    print("Annotating speech-relatedness..")
    utterances = utterances.assign(
        is_speech_related=utterances.transcript_raw.apply(
            is_speech_related,
            label_partially_speech_related=args.label_partially_speech_related,
        )
    )
    utterances.is_speech_related = utterances.is_speech_related.astype("boolean")

    print("Annotating intelligibility..")
    utterances = utterances.assign(
        is_intelligible=utterances.transcript_raw.apply(
            is_intelligible,
            label_partially_intelligible=args.label_partially_intelligible,
        )
    )

    print("Cleaning utterances..")
    utterances = utterances.assign(
        utt_clean=utterances.transcript_raw.apply(
            clean_preprocessed_utterance
        )
    )

    num_words = np.array([len(re.split('\s|\'', utt)) for utt in utterances.utt_clean.values])
    utts_to_annotate = utterances[(num_words > 1)]
    utts_to_annotate = utts_to_annotate[utts_to_annotate.is_speech_related & utts_to_annotate.is_intelligible]

    # utts_to_annotate = utts_to_annotate.sample(1000, random_state=1)

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

    annotated_utts = prepare(args)

    os.makedirs(os.path.dirname(TO_ANNOTATE_UTTERANCES_FILE), exist_ok=True)
    annotated_utts.to_csv(TO_ANNOTATE_UTTERANCES_FILE)
