import argparse
import os

import pandas as pd

from cf_analyses.analysis_grammaticality import filter_corpora
from cf_analyses.analysis_intelligibility import filter_utts_for_num_words
from cf_analyses.cr_ack_annotations import annotate_crs_and_acks
from utils import (
    MICRO_CONVERSATIONS_WITHOUT_NON_SPEECH_FILE, PROJECT_ROOT_DIR, PREPROCESSED_UTTERANCES_FILE,
    SPEAKER_CODES_CAREGIVER,
)

DEFAULT_LM_MIN_AGE = 10
DEFAULT_LM_MAX_AGE = 60

DEFAULT_FB_MIN_AGE = 10
DEFAULT_FB_MAX_AGE = 60

MIN_NUM_WORDS = 2

CORPORA_EXCLUDED = []

CORPORA_INCLUDED = []

# The caregivers of these children are using slang (e.g., "you was" or "she don't") and are therefore excluded
# We are unfortunately only studying mainstream US English
EXCLUDED_CHILDREN = ["Brent_Jaylen", "Brent_Tyrese", "Brent_Vas", "Brent_Vas_Coleman", "Brent_Xavier"]

RESULTS_DIR = PROJECT_ROOT_DIR + "/results/rl_data/"


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=PREPROCESSED_UTTERANCES_FILE,
    )
    argparser.add_argument(
        "--conversations-file",
        type=str,
        default=MICRO_CONVERSATIONS_WITHOUT_NON_SPEECH_FILE,
    )
    argparser.add_argument(
        "--min-age-lm-data",
        type=int,
        default=DEFAULT_LM_MIN_AGE,
    )
    argparser.add_argument(
        "--max-age-lm-data",
        type=int,
        default=DEFAULT_LM_MAX_AGE,
    )
    argparser.add_argument(
        "--min-age-fb-data",
        type=int,
        default=DEFAULT_FB_MIN_AGE,
    )
    argparser.add_argument(
        "--max-age-fb-data",
        type=int,
        default=DEFAULT_FB_MAX_AGE,
    )

    args = argparser.parse_args()

    return args


def create_fb_data():
    conversations = pd.read_csv(args.conversations_file, index_col=0, dtype={"error": object, "labels": object})
    # Filter by age
    conversations = conversations[
        (args.min_age_fb_data <= conversations.age) & (conversations.age <= args.max_age_fb_data)
        ]

    print("Excluding children: ", EXCLUDED_CHILDREN)
    conversations = conversations[~conversations.child_name.isin(EXCLUDED_CHILDREN)]

    conversations = filter_corpora(conversations, CORPORA_INCLUDED, CORPORA_EXCLUDED)

    conversations.dropna(
        subset=(
            "utt_is_grammatical",
            "utt_is_intelligible",
            "response_is_speech_related",
        ),
        inplace=True,
    )
    conversations["utt_is_grammatical"] = conversations.utt_is_grammatical.astype(bool)
    conversations["follow_up_is_grammatical"] = conversations.follow_up_is_grammatical.astype(bool)

    conversations = conversations[conversations.utt_is_intelligible].copy()

    # Filtering out dummy responses (cases in which the child continues to talk)
    conversations = conversations[conversations.response_is_speech_related].copy()

    conversations = filter_utts_for_num_words(conversations, min_num_words=MIN_NUM_WORDS)

    annotate_crs_and_acks(conversations)

    print("Number of CRs: ", len(conversations[conversations.response_is_clarification_request]))
    print("Number of speech act CRs: ", len(conversations[conversations.response_is_clarification_request_speech_act]))
    print("Number of repetition CRs: ", len(conversations[conversations.response_is_repetition_clarification_request]))

    print("Number of Acks: ", len(conversations[conversations.response_is_acknowledgement]))
    print("Number of keyword Acks: ", len(conversations[conversations.response_is_keyword_acknowledgement]))
    print("Number of repetition Acks: ", len(conversations[conversations.response_is_repetition_acknowledgement]))

    conversations = conversations[
        ["transcript_file", "utt_transcript_clean", "response_transcript_clean", "response_is_clarification_request", "response_is_acknowledgement"]]
    conversations.to_csv(RESULTS_DIR + "conversations.csv", index=False)


def create_lm_data():
    utterances = pd.read_csv(args.utterances_file, index_col=0, dtype={"error": object, "labels": object})
    # Filter by age
    utterances = utterances[
        (args.min_age_lm_data <= utterances.age) & (utterances.age <= args.max_age_lm_data)
        ]

    print("Excluding children: ", EXCLUDED_CHILDREN)
    utterances = utterances[~utterances.child_name.isin(EXCLUDED_CHILDREN)]

    utterances = filter_corpora(utterances, CORPORA_INCLUDED, CORPORA_EXCLUDED)

    utterances = utterances[utterances.speaker_code.isin(SPEAKER_CODES_CAREGIVER)]

    utterances = filter_utts_for_num_words(utterances, min_num_words=1, utt_column_name="transcript_clean")

    print("Number of caregiver utterances: ", len(utterances))

    utterances = utterances[
        ["utterance_id", "transcript_clean", "transcript_file"]]
    utterances.to_csv(RESULTS_DIR + "caregiver_utterances.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    create_lm_data()
    create_fb_data()

