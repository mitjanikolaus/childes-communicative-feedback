import argparse
import itertools
import math
import os

import pandas as pd

from multiprocessing import Pool

from tqdm import tqdm

from utils import (
    str2bool, SPEAKER_CODE_CHILD, UTTERANCES_WITH_SPEECH_ACTS_FILE, SPEECH_ACT_NO_FUNCTION,
    SPEAKER_CODES_CAREGIVER, MICRO_CONVERSATIONS_FILE,
)

DEFAULT_RESPONSE_THRESHOLD = 1000

# 10 seconds
DEFAULT_MAX_NEG_RESPONSE_LATENCY = -10 * 1000  # ms

# 1 minute
DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP = 1 * 60 * 1000  # ms


DUMMY_RESPONSE = {
    "transcript_clean": "",
    "transcript_raw": "",
    "start_time": math.inf,
    "end_time": math.inf,
    "is_speech_related": False,
    "is_intelligible": False,
    "speech_act": SPEECH_ACT_NO_FUNCTION,
}

KEEP_KEYS = [
    "transcript_clean",
    "transcript_raw",
    "start_time",
    "end_time",
    "is_speech_related",
    "is_intelligible",
    "speech_act",
]


def update_dict_with_prefix(dict, prefix, keys_to_be_updated=KEEP_KEYS):
    dict_updated = {key: value for key, value in dict.items() if key not in KEEP_KEYS}
    dict_updated.update({prefix + key: dict[key] for key in keys_to_be_updated})
    return dict_updated


def get_dict_with_prefix(series, prefix, keep_keys=KEEP_KEYS):
    return {prefix + key: series[key] for key in keep_keys}


def get_micro_conversations_for_transcript(utterances_transcript, response_latency, add_prev_utterance):
    micro_convs = []
    utterances_child = utterances_transcript[
        utterances_transcript.speaker_code == SPEAKER_CODE_CHILD
    ]

    utts_child_before_caregiver_utt = utterances_child[
        utterances_child.speaker_code_next.isin(SPEAKER_CODES_CAREGIVER)
    ]
    for candidate_id in utts_child_before_caregiver_utt.index.values:
        if candidate_id + 1 not in utterances_transcript.index.values:
            continue
        if add_prev_utterance and candidate_id - 1 not in utterances_transcript.index.values:
            continue

        following_utts = utterances_transcript.loc[candidate_id + 1:]
        following_utts_child = following_utts[
            following_utts.speaker_code == SPEAKER_CODE_CHILD
        ]
        if len(following_utts_child) > 0:
            conversation = utterances_transcript.loc[candidate_id].to_dict()
            conversation["index"] = candidate_id
            conversation = update_dict_with_prefix(conversation, "utt_")

            response = get_dict_with_prefix(
                utterances_transcript.loc[candidate_id + 1], "response_"
            )
            conversation.update(response)

            follow_up = get_dict_with_prefix(following_utts_child.iloc[0], "follow_up_")
            conversation.update(follow_up)

            if add_prev_utterance:
                prev = get_dict_with_prefix(
                    utterances_transcript.loc[candidate_id - 1], "prev_"
                )
                conversation.update(prev)

            micro_convs.append(conversation)

    utts_child_no_response = utterances_child[
        (utterances_child.speaker_code_next == SPEAKER_CODE_CHILD)
        & (
            utterances_child.start_time_next - utterances_child.end_time
            >= response_latency
        )
    ]
    for candidate_id in utts_child_no_response.index.values:
        if add_prev_utterance and candidate_id - 1 not in utterances_transcript.index.values:
            continue

        following_utts = utterances_transcript.loc[candidate_id + 1:]
        following_utts_non_child = following_utts[
            following_utts.speaker_code != SPEAKER_CODE_CHILD
        ]
        if len(following_utts_non_child) > 0 and (
            following_utts_non_child.iloc[0].speaker_code not in SPEAKER_CODES_CAREGIVER
        ):
            # Child is not speaking to its caregiver, ignore this turn
            continue

        following_utts_child = following_utts[
            following_utts.speaker_code == SPEAKER_CODE_CHILD
        ]
        if len(following_utts_child) > 0:
            conversation = utterances_transcript.loc[candidate_id].to_dict()
            conversation["index"] = candidate_id
            conversation = update_dict_with_prefix(conversation, "utt_")

            response = get_dict_with_prefix(DUMMY_RESPONSE, "response_")
            conversation.update(response)

            follow_up = get_dict_with_prefix(following_utts_child.iloc[0], "follow_up_")
            conversation.update(follow_up)

            if add_prev_utterance:
                prev = get_dict_with_prefix(
                    utterances_transcript.loc[candidate_id - 1], "prev_"
                )
                conversation.update(prev)

            micro_convs.append(conversation)

    return micro_convs


def get_micro_conversations(utterances, response_latency, max_response_latency_follow_up, max_neg_response_latency,
                            use_is_grammatical=False, add_prev_utterance=False):
    if use_is_grammatical:
        KEEP_KEYS.append("is_grammatical")
        DUMMY_RESPONSE["is_grammatical"] = False

    print("Creating micro conversations from transcripts..")
    utterances_grouped = [group for _, group in utterances.groupby("transcript_file")]
    process_args = [(utts_transcript, response_latency, add_prev_utterance) for utts_transcript in utterances_grouped]

    # Single-process version for debugging:
    # results = [get_micro_conversations_for_transcript(utts_transcript, response_latency, add_prev_utterance)
    #     for utts_transcript in tqdm(utterances_grouped)]
    with Pool(processes=8) as pool:
        results = pool.starmap(
            get_micro_conversations_for_transcript,
            tqdm(process_args, total=len(process_args)),
        )

    conversations = pd.DataFrame(list(itertools.chain(*results))).set_index("index")

    conversations["response_latency"] = (
        conversations["response_start_time"] - conversations["utt_end_time"]
    )
    conversations["response_latency_follow_up"] = (
        conversations["follow_up_start_time"] - conversations["utt_end_time"]
    )

    # disregard conversations with follow up too far in the future
    conversations = conversations[
        (
            conversations.response_latency_follow_up
            <= max_response_latency_follow_up
        )
    ]

    # Disregard conversations with too long negative pauses
    conversations = conversations[
        (conversations.response_latency > max_neg_response_latency)
    ]

    return conversations


def extract(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0)

    utterances.drop(["tokens", "pos", "gra"], axis=1, inplace=True)

    conversations = get_micro_conversations(utterances, DEFAULT_RESPONSE_THRESHOLD,
                                            DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP,
                                            DEFAULT_MAX_NEG_RESPONSE_LATENCY,
                                            use_is_grammatical=False,
                                            add_prev_utterance=True)

    return conversations


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=UTTERANCES_WITH_SPEECH_ACTS_FILE,
    )
    argparser.add_argument(
        "--out",
        default=MICRO_CONVERSATIONS_FILE,
        type=str,
        help="Path to store output file",
    )
    argparser.add_argument(
        "--add-prev-utterance",
        type=str2bool,
        const=True,
        nargs="?",
        default=False,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    conversations = extract(args)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    conversations.to_csv(args.out)
