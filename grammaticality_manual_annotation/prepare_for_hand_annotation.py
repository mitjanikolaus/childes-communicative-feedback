import argparse
import os

import pandas as pd

from utils import (
    SPEAKER_CODE_CHILD,
    SPEAKER_CODES_CAREGIVER, ANNOTATED_UTTERANCES_FILE, filter_for_min_num_words, PROJECT_ROOT_DIR
)
import random

from tqdm import tqdm
tqdm.pandas()


TOKEN_CHILD = "CHI"
TOKEN_CAREGIVER = "CAR"
TOKEN_OTHER = "OTH"

MIN_NUM_WORDS = 1

# The caregivers of these children are using slang (e.g., "you was" or "she don't") and are therefore excluded
# We are unfortunately only studying mainstream US English
EXCLUDED_CHILDREN = ["Brent_Jaylen", "Brent_Tyrese", "Brent_Vas", "Brent_Vas_Coleman", "Brent_Xavier"]

TRANSCRIPT_FILES_EXCLUDED = ["Braunwald/020128.cha", "MPI-EVA-Manchester/Fraser/030100b.cha", "Providence/Alex/021025.cha"]

MIN_AGE = 24
MAX_AGE = 60
NUM_UTTS_TO_ANNOTATE_PER_FILE = 200


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

    utterances = utterances[~utterances.child_name.isin(EXCLUDED_CHILDREN)]

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
    num_child_utts = 0
    while end_idx < len(utterances):
        if utterances.iloc[end_idx].speaker_code == SPEAKER_CODE_CHILD:
            num_child_utts += 1

        if num_child_utts >= NUM_UTTS_TO_ANNOTATE_PER_FILE:
            utterances_selection = utterances.iloc[start_idx:end_idx].copy()
            utterances_selection["is_grammatical"] = ""
            utterances_selection["labels"] = ""
            utterances_selection["note"] = ""
            utterances_selection.loc[utterances_selection.speaker_code == SPEAKER_CODE_CHILD, "is_grammatical"] = "TODO"

            utterances_selection = utterances_selection[["transcript_file", "speaker_code", "transcript_clean", "is_grammatical", "labels", "note"]]

            utterances_selection.to_csv(os.path.join(base_path, f"{file_idx}.csv"))
            num_child_utts = 0
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
