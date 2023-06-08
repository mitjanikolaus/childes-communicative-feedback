import argparse
import os

import pandas as pd

from utils import (
    SPEAKER_CODE_CHILD,
    SPEAKER_CODES_CAREGIVER, ANNOTATED_UTTERANCES_FILE, filter_for_min_num_words, PROJECT_ROOT_DIR, split_into_words
)
import random

from tqdm import tqdm
tqdm.pandas()


TOKEN_CHILD = SPEAKER_CODE_CHILD
TOKEN_CAREGIVER = "CAR"
TOKEN_OTHER = "OTH"

MIN_NUM_WORDS = 1

# The caregivers of children in these corpora are using slang (e.g., "you was" or "she don't") and are therefore excluded
# We are unfortunately only studying mainstream US/UK English
EXCLUDED_CORPORA = ["Wells", "MPI-EVA-Manchester", "Post", "HSLLD", "Bohannon", "Brown", "Hall", "Brent", "Gleason", "Morisset", "Belfast"]

TRANSCRIPT_FILES_EXCLUDED = ["Braunwald/020128.cha", "MPI-EVA-Manchester/Fraser/030100b.cha", "Providence/Alex/021025.cha"]

MIN_AGE = 24
MAX_AGE = 60
NUM_UTTS_TO_ANNOTATE_PER_FILE = 200


def filter_for_child_caregiver_conversations(utterances):
    """Filter out transcripts which include other interlocutors"""

    child_car_utts = utterances[utterances.speaker_code.isin([TOKEN_CHILD, TOKEN_CAREGIVER])]
    child_car_utts_per_transcript = child_car_utts.groupby("transcript_file").size()
    utts_per_transcript = utterances.groupby("transcript_file").size()

    transcripts_child_car = utts_per_transcript[
        utts_per_transcript == child_car_utts_per_transcript
    ].index

    return utterances[
        utterances.transcript_file.isin(transcripts_child_car)
    ].copy()


def speaker_code_to_special_token(code):
    if code == SPEAKER_CODE_CHILD:
        return TOKEN_CHILD
    elif code in SPEAKER_CODES_CAREGIVER:
        return TOKEN_CAREGIVER
    else:
        return TOKEN_OTHER


def get_utts_to_annotate(utterances):
    utts_to_annotate = utterances[(utterances.speaker_code == SPEAKER_CODE_CHILD)]
    utts_to_annotate = filter_for_min_num_words(utts_to_annotate, MIN_NUM_WORDS)
    return utts_to_annotate


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, dtype={"error": object})

    utterances = utterances[utterances.is_speech_related == True]
    utterances = filter_for_min_num_words(utterances, MIN_NUM_WORDS).copy()

    utterances = utterances[(MIN_AGE <= utterances.age) & (utterances.age <= MAX_AGE)]

    utterances["speaker_code"] = utterances["speaker_code"].apply(speaker_code_to_special_token)

    # Remove already annotated transcripts
    utterances = utterances[~utterances.transcript_file.isin([TRANSCRIPT_FILES_EXCLUDED])]

    utterances = utterances[~utterances.corpus.isin(EXCLUDED_CORPORA)]

    utterances = filter_for_child_caregiver_conversations(utterances)

    # Shuffle transcripts
    groups = [utt for _, utt in utterances.groupby('transcript_file')]
    random.seed(1)
    random.shuffle(groups)
    utterances = pd.concat(groups).reset_index(drop=True)

    base_path = PROJECT_ROOT_DIR+"/data/manual_annotation/new"
    os.makedirs(base_path, exist_ok=True)

    file_idx = 0
    start_idx = 0
    end_idx = 0
    num_utts_to_annotate = 0

    utterances["num_words"] = utterances.transcript_clean.apply(
        lambda x: len(split_into_words(x, split_on_apostrophe=False, remove_commas=True,
                                       remove_trailing_punctuation=True)))

    while end_idx < len(utterances):
        if (utterances.iloc[end_idx].speaker_code == SPEAKER_CODE_CHILD) & (utterances.iloc[end_idx].num_words > 1):
            num_utts_to_annotate += 1

        if num_utts_to_annotate >= NUM_UTTS_TO_ANNOTATE_PER_FILE:
            utterances_selection = utterances.iloc[start_idx:end_idx].copy()
            utterances_selection["is_grammatical"] = ""
            utterances_selection["note"] = ""
            utterances_selection.loc[(utterances_selection.speaker_code == SPEAKER_CODE_CHILD) & (utterances_selection.num_words > 1), "is_grammatical"] = "TODO"

            utterances_selection = utterances_selection[["transcript_file", "speaker_code", "transcript_clean", "is_grammatical", "note"]]

            utterances_selection.to_csv(os.path.join(base_path, f"{file_idx}.csv"))
            num_utts_to_annotate = 0
            file_idx += 1
            start_idx = end_idx

        end_idx += 1


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

    prepare(args)
