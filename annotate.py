import argparse
import os
import pandas as pd
import numpy as np

from utils import (
    remove_punctuation,
    str2bool,
)
from preprocess import (
    PREPROCESSED_UTTERANCES_FILE,
)
from utils import (
    remove_nonspeech_events,
    CODE_UNINTELLIGIBLE,
)

ANNOTATED_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_annotated.csv"
)


DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED = True


def is_empty(utterance):
    utterance = remove_punctuation(utterance)
    return utterance == ""


def is_speech_related(
    utterance,
    label_partially_speech_related=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
    label_unintelligible=None,
    label_empty_utterance=None,
):
    """Label utterances as speech or non-speech."""
    utterance_without_punctuation = remove_punctuation(utterance)
    if utterance_without_punctuation == "":
        # By returning None, we can filter out these cases later
        return label_empty_utterance

    utt_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)

    utt_without_nonspeech = utt_without_nonspeech.strip()
    if utt_without_nonspeech == "":
        return False

    # We exclude completely unintelligible utterances (we don't know whether it's speech-related or not)
    is_completely_unintelligible = True
    for word in utt_without_nonspeech.split(" "):
        if word != CODE_UNINTELLIGIBLE and word != "":
            is_completely_unintelligible = False
            break
    if is_completely_unintelligible:
        # By returning None, we can filter out these cases later
        return label_unintelligible

    is_partly_speech_related = len(utt_without_nonspeech) != len(
        utterance_without_punctuation
    )
    if is_partly_speech_related:
        return label_partially_speech_related

    return True


def get_response_latency(row):
    if np.isnan(row["start_time_next"]) or np.isnan(row["end_time"]):
        return None

    return row["start_time_next"] - row["end_time"]


def annotate(args):
    utterances = pd.read_csv(PREPROCESSED_UTTERANCES_FILE, index_col=None)

    utterances = utterances.assign(
        is_speech_related=utterances.transcript_raw.apply(
            is_speech_related,
            label_partially_speech_related=args.label_partially_speech_related,
        )
    )

    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--label-partially-speech-related",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
        help="Label for partially speech-related utterances: Set to True to count as speech-related, False to count as "
        "not speech-related or None to exclude these utterances from the analysis",
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    annotated_utts = annotate(args)

    os.makedirs(os.path.dirname(ANNOTATED_UTTERANCES_FILE), exist_ok=True)
    annotated_utts.to_csv(ANNOTATED_UTTERANCES_FILE, index=False)
