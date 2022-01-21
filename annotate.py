import argparse
import os

import pandas as pd

from utils import (
    remove_punctuation,
    str2bool,
    remove_babbling,
    get_paralinguistic_event,
    paralinguistic_event_is_external,
    get_all_paralinguistic_events, ANNOTATED_UTTERANCES_FILE, UTTERANCES_WITH_SPEECH_ACTS_FILE, is_nan,
    SPEECH_ACTS_NO_FUNCTION,
)
from utils import (
    remove_nonspeech_events,
    CODE_UNINTELLIGIBLE,
)

DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED = True

DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE = True

# Speech acts that relate to nonverbal/external events
SPEECH_ACTS_NONVERBAL_EVENTS = [
    "CR",   # Criticize or point out error in nonverbal act.
    "PM",   # Praise for motor acts i.e for nonverbal behavior.
    "WD",   # Warn of danger.
    "DS",   # Disapprove scold protest disruptive behavior.
    "AB",   # Approve of appropriate behavior.
    "TO",   # Mark transfer of object to hearer
    "ET",   # Express enthusiasm for hearer's performance.
    "ED",   # Exclaim in disapproval.
]


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


def is_intelligible(
    utterance,
    label_partially_intelligible=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
    label_empty_utterance=None,
):
    utterance_without_punctuation = remove_punctuation(utterance)
    if utterance_without_punctuation == "":
        # By returning None, we can filter out these cases later
        return label_empty_utterance

    utterance_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)
    utterance_without_nonspeech = utterance_without_nonspeech.strip()
    if utterance_without_nonspeech == "":
        # By returning None, we can filter out these cases later
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


def speech_act_is_intelligible(speech_act):
    return speech_act not in SPEECH_ACTS_NO_FUNCTION


def has_multiple_events(utterance):
    return len(get_all_paralinguistic_events(utterance)) > 1


def is_external_event(utterance):
    utterance = remove_punctuation(utterance)

    event = get_paralinguistic_event(utterance)
    if event and paralinguistic_event_is_external(event) and utterance == event:
        return True

    return False


def annotate(args):
    utterances = pd.read_pickle(UTTERANCES_WITH_SPEECH_ACTS_FILE)

    utterances.dropna(
        subset=("transcript_raw",),
        inplace=True,
    )

    utterances = utterances[~utterances.transcript_raw.apply(has_multiple_events)]
    utterances = utterances[~utterances.transcript_raw.apply(is_external_event)]

    utterances = utterances.assign(
        is_speech_related=utterances.transcript_raw.apply(
            is_speech_related,
            label_partially_speech_related=args.label_partially_speech_related,
        )
    )

    if args.rule_based_intelligibility:
        utterances = utterances.assign(
            is_intelligible=utterances.transcript_raw.apply(
                is_intelligible,
                label_partially_intelligible=args.label_partially_intelligible,
            )
        )
    else:
        utterances = utterances.assign(
            is_intelligible=utterances.speech_act.apply(
                speech_act_is_intelligible,
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
    argparser.add_argument(
        "--label-partially-intelligible",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
        help="Label for partially intelligible utterances: Set to True to count as intelligible, False to count as unintelligible or None to exclude these utterances from the analysis",
    )
    argparser.add_argument(
        "--rule-based-intelligibility",
        action="store_true",
        help="Use rule-based approach to annotate intelligibility"
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    annotated_utts = annotate(args)

    os.makedirs(os.path.dirname(ANNOTATED_UTTERANCES_FILE), exist_ok=True)
    annotated_utts.to_pickle(ANNOTATED_UTTERANCES_FILE)
