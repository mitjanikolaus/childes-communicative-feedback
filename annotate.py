import argparse
import os
from ast import literal_eval

import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from utils import (
    remove_punctuation,
    str2bool,
    remove_babbling,
    ANNOTATED_UTTERANCES_FILE,
    split_into_words, PREPROCESSED_UTTERANCES_FILE,
    remove_superfluous_annotations,
)
from utils import (
    remove_nonspeech_events,
    IS_UNINTELLIGIBLE,
)

DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED = True

DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE = False

# Speech acts that relate to nonverbal/external events
SPEECH_ACTS_NONVERBAL_EVENTS = [
    "CR",  # Criticize or point out error in nonverbal act.
    "PM",  # Praise for motor acts i.e for nonverbal behavior.
    "WD",  # Warn of danger.
    "DS",  # Disapprove scold protest disruptive behavior.
    "AB",  # Approve of appropriate behavior.
    "TO",  # Mark transfer of object to hearer
    "ET",  # Express enthusiasm for hearer's performance.
    "ED",  # Exclaim in disapproval.
]


def is_speech_related(
        utterance,
        label_partially_speech_related=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
        label_unintelligible=pd.NA,
):
    """Label utterances as speech or non-speech."""
    utterance_without_punctuation = remove_punctuation(utterance)
    utt_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)

    utt_without_nonspeech = utt_without_nonspeech.strip()
    if utt_without_nonspeech == "":
        return False

    # We exclude completely unintelligible utterances (we don't know whether it's speech-related or not)
    is_completely_unintelligible = True
    for word in split_into_words(utt_without_nonspeech, remove_commas=True, remove_trailing_punctuation=False):
        if not IS_UNINTELLIGIBLE(word) and word != "":
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


def is_intelligible(
        utterance,
        label_partially_intelligible=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
        label_empty_utterance=False,
):
    utterance_without_punctuation = remove_punctuation(utterance)
    utterance_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)
    utterance_without_nonspeech = utterance_without_nonspeech.strip()
    if utterance_without_nonspeech == "":
        return label_empty_utterance

    utt_without_babbling = remove_babbling(utterance_without_nonspeech)

    utt_without_babbling = utt_without_babbling.strip()
    if utt_without_babbling == "":
        return False

    is_partly_intelligible = len(utt_without_babbling) != len(
        utterance_without_nonspeech
    )
    if is_partly_intelligible:
        return label_partially_intelligible

    return True


def annotate(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, converters={"pos": literal_eval, "tokens": literal_eval}, dtype={"error": object})
    utterances.dropna(subset=["transcript_raw"], inplace=True)
    raw_transcripts = utterances["transcript_raw"].apply(
        remove_superfluous_annotations
    )

    print("Annotating speech-relatedness..")
    utterances["is_speech_related"] = raw_transcripts.progress_apply(
        is_speech_related,
        label_partially_speech_related=args.label_partially_speech_related,
    )
    utterances.is_speech_related = utterances.is_speech_related.astype("boolean")

    print("Annotating intelligibility..")
    utterances["is_intelligible"] = raw_transcripts.progress_apply(
        is_intelligible,
        label_partially_intelligible=args.label_partially_intelligible,
    )

    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        default=PREPROCESSED_UTTERANCES_FILE,
        type=str,
        help="Path to utterances annotated with speech acts",
    )
    argparser.add_argument(
        "--out",
        default=ANNOTATED_UTTERANCES_FILE,
        type=str,
        help="Path to store output file",
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
    if not args.out.endswith(".csv"):
        raise ValueError("Out file should have .csv ending!")

    annotated_utts = annotate(args)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    annotated_utts.to_csv(args.out)
