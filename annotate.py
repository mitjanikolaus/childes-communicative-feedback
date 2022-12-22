import argparse
import os
from ast import literal_eval
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()

from utils import (
    remove_punctuation,
    str2bool,
    remove_babbling,
    ANNOTATED_UTTERANCES_FILE,
    split_into_words, PREPROCESSED_UTTERANCES_FILE, SPEAKER_CODE_CHILD, SPEAKER_CODES_CAREGIVER,
)
from utils import (
    remove_nonspeech_events,
    CODE_UNINTELLIGIBLE,
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


def add_prev_utts_for_transcript(utterances_transcript):
    utts_speech_related = utterances_transcript[utterances_transcript.is_speech_related.isin([pd.NA, True])]

    def add_prev_utt(utterance):
        if utterance.name in utts_speech_related.index:
            row_number = np.where(utts_speech_related.index.values == utterance.name)[0][0]
            if row_number > 0:
                prev_utt = utts_speech_related.loc[utts_speech_related.index[:row_number][-1]]
                return prev_utt.transcript_clean

        return pd.NA

    def add_prev_utt_speaker_code(utterance):
        if utterance.name in utts_speech_related.index:
            row_number = np.where(utts_speech_related.index.values == utterance.name)[0][0]
            if row_number > 0:
                prev_utt = utts_speech_related.loc[utts_speech_related.index[:row_number][-1]]
                return prev_utt.speaker_code

        return pd.NA

    utterances_transcript["prev_transcript_clean"] = utterances_transcript.apply(
        add_prev_utt,
        axis=1
    )
    utterances_transcript["prev_speaker_code"] = utterances_transcript.apply(
        add_prev_utt_speaker_code,
        axis=1
    )

    return utterances_transcript


def add_prev_utts(utterances):
    # Single-process version for debugging:
    # results = [add_prev_utts_for_transcript(utts_transcript)
    #     for utts_transcript in tqdm([group for _, group in utterances.groupby("transcript_file")])]
    utterances_grouped = [[group] for _, group in utterances.groupby("transcript_file")]
    with Pool(processes=8) as pool:
        results = pool.starmap(
            add_prev_utts_for_transcript,
            tqdm(utterances_grouped, total=len(utterances_grouped)),
        )

    utterances = pd.concat(results, verify_integrity=True)

    return utterances


def annotate(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, converters={"pos": literal_eval, "tokens": literal_eval})
    utterances.dropna(subset=["transcript_raw"], inplace=True)

    print("Annotating speech-relatedness..")
    utterances["is_speech_related"] = utterances.transcript_raw.progress_apply(
        is_speech_related,
        label_partially_speech_related=args.label_partially_speech_related,
    )
    utterances.is_speech_related = utterances.is_speech_related.astype("boolean")

    print("Annotating intelligibility..")
    utterances["is_intelligible"] = utterances.transcript_raw.progress_apply(
        is_intelligible,
        label_partially_intelligible=args.label_partially_intelligible,
    )

    print("Adding previous utterances..")
    utterances = add_prev_utts(utterances)

    print(f"prev utt speaker code {SPEAKER_CODE_CHILD}: {len(utterances[utterances.prev_speaker_code == SPEAKER_CODE_CHILD])/len(utterances)}")
    print(f"prev utt speaker code caregiver: {len(utterances[utterances.prev_speaker_code.isin(SPEAKER_CODES_CAREGIVER)])/len(utterances)}")

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
