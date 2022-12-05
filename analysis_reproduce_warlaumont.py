import argparse
import itertools
import math
import os
from ast import literal_eval
from collections import Counter
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from scipy.stats import ttest_1samp
from tqdm import tqdm

from utils import (
    age_bin,
    str2bool,
    SPEECH_ACT_NO_FUNCTION,
    filter_transcripts_based_on_num_child_utts,
    SPEAKER_CODE_CHILD,
    SPEAKER_CODES_CAREGIVER, UTTERANCES_WITH_SPEECH_ACTS_FILE,
)

DEFAULT_RESPONSE_THRESHOLD = 1000

# 10 seconds
DEFAULT_MAX_NEG_RESPONSE_LATENCY = -10 * 1000  # ms

# 1 minute
DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP = 1 * 60 * 1000  # ms

DEFAULT_COUNT_ONLY_SPEECH_RELATED_RESPONSES = True

DEFAULT_MIN_RATIO_NONSPEECH = 0.0

DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT = 1

# Ages aligned to study of Warlaumont et al. or to our study (minimum 10 months)
DEFAULT_MIN_AGE = 10
# DEFAULT_MIN_AGE = 8
DEFAULT_MAX_AGE = 48

AGE_BIN_NUM_MONTHS = 6

# Providence: Some non-speech vocalizations such as laughter are incorrectly transcribed as 'yyy', and the timing
# information is of very poor quality
DEFAULT_EXCLUDED_CORPORA = ["Providence"]


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--corpora",
        nargs="+",
        type=str,
        required=False,
        help="Corpora to analyze. If not given, corpora are selected based on a response time variance threshold.",
    )
    argparser.add_argument(
        "--excluded-corpora",
        nargs="+",
        type=str,
        default=DEFAULT_EXCLUDED_CORPORA,
        help="Corpora to exclude from analysis",
    )
    argparser.add_argument(
        "--min-age",
        type=int,
        default=DEFAULT_MIN_AGE,
    )
    argparser.add_argument(
        "--max-age",
        type=int,
        default=DEFAULT_MAX_AGE,
    )
    argparser.add_argument(
        "--min-child-utts-per-transcript",
        type=int,
        default=DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT,
    )
    argparser.add_argument(
        "--min-ratio-nonspeech",
        type=int,
        default=DEFAULT_MIN_RATIO_NONSPEECH,
    )

    argparser.add_argument(
        "--response-latency",
        type=int,
        default=DEFAULT_RESPONSE_THRESHOLD,
        help="Response latency in milliseconds",
    )
    argparser.add_argument(
        "--max-neg-response-latency",
        type=int,
        default=DEFAULT_MAX_NEG_RESPONSE_LATENCY,
        help="Maximum negative response latency in milliseconds",
    )
    argparser.add_argument(
        "--max-response-latency-follow-up",
        type=int,
        default=DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP,
        help="Maximum response latency for the child follow-up in milliseconds",
    )
    argparser.add_argument(
        "--count-only-speech_related_responses",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_COUNT_ONLY_SPEECH_RELATED_RESPONSES,
    )

    args = argparser.parse_args()

    return args


def has_response(
    row,
    response_latency,
    count_only_speech_related_responses=False,
    count_only_intelligible_responses=False,
):
    if row["response_latency"] <= response_latency:
        if count_only_speech_related_responses:
            return row["response_is_speech_related"]
        elif count_only_intelligible_responses:
            return row["response_is_intelligible"]
        else:
            return True
    return False


DUMMY_RESPONSE = {
    "transcript_raw": "",
    "transcript_clean": "",
    "start_time": math.inf,
    "end_time": math.inf,
    "is_speech_related": False,
    "is_intelligible": False,
    "speech_act": SPEECH_ACT_NO_FUNCTION,
}

KEEP_KEYS = [
    "transcript_raw",
    "transcript_clean",
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


def get_micro_conversations_for_transcript(utterances_transcript, args):
    conversations = []
    utterances_child = utterances_transcript[
        utterances_transcript.speaker_code == SPEAKER_CODE_CHILD
    ]

    utts_child_before_caregiver_utt = utterances_child[
        utterances_child.speaker_code_next.isin(SPEAKER_CODES_CAREGIVER)
    ]
    for candidate_id in utts_child_before_caregiver_utt.index.values:
        if candidate_id + 1 not in utterances_transcript.index.values:
            continue

        following_utts = utterances_transcript.loc[candidate_id + 1 :]
        following_utts_child = following_utts[
            following_utts.speaker_code == SPEAKER_CODE_CHILD
        ]
        if len(following_utts_child) > 0:
            conversation = utterances_transcript.loc[candidate_id].to_dict()
            conversation = update_dict_with_prefix(conversation, "utt_")

            response = get_dict_with_prefix(
                utterances_transcript.loc[candidate_id + 1], "response_"
            )
            conversation.update(response)

            follow_up = get_dict_with_prefix(following_utts_child.iloc[0], "follow_up_")
            conversation.update(follow_up)
            conversations.append(conversation)

    utts_child_no_response = utterances_child[
        (utterances_child.speaker_code_next == SPEAKER_CODE_CHILD)
        & (
            utterances_child.start_time_next - utterances_child.end_time
            >= args.response_latency
        )
    ]
    for candidate_id in utts_child_no_response.index.values:
        following_utts = utterances_transcript.loc[candidate_id + 1 :]
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
            conversation = update_dict_with_prefix(conversation, "utt_")

            response = get_dict_with_prefix(DUMMY_RESPONSE, "response_")
            conversation.update(response)

            follow_up = get_dict_with_prefix(following_utts_child.iloc[0], "follow_up_")
            conversation.update(follow_up)

            conversations.append(conversation)

    return conversations


def get_micro_conversations(utterances, args, use_is_grammatical=False):
    if use_is_grammatical:
        KEEP_KEYS.append("is_grammatical")
        DUMMY_RESPONSE["is_grammatical"] = False

    print("Creating micro conversations from transcripts..")
    utterances_grouped = [group for _, group in utterances.groupby("transcript_file")]
    process_args = [(utts_transcript, args) for utts_transcript in utterances_grouped]

    # Single-process version for debugging:
    # results = [get_micro_conversations_for_transcript(utts_transcript, args)
    #     for utts_transcript in utterances_grouped]
    with Pool(processes=8) as pool:
        results = pool.starmap(
            get_micro_conversations_for_transcript,
            tqdm(process_args, total=len(process_args)),
        )

    conversations = pd.DataFrame(list(itertools.chain(*results)))

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
            <= args.max_response_latency_follow_up
        )
    ]

    # Disregard conversations with too long negative pauses
    conversations = conversations[
        (conversations.response_latency > args.max_neg_response_latency)
    ]

    return conversations


def perform_analysis_speech_relatedness(utterances, args):
    conversations = get_micro_conversations(utterances, args)

    conversations.to_csv("results/conversations_raw.csv", index=False)

    conversations.dropna(
        subset=(
            "response_latency",
            "response_latency_follow_up",
            "utt_is_speech_related",
            "response_is_speech_related",
            "follow_up_is_speech_related",
        ),
        inplace=True,
    )
    conversations.utt_is_speech_related = conversations.utt_is_speech_related.astype(bool)
    conversations.response_is_speech_related = conversations.response_is_speech_related.astype(bool)
    conversations.follow_up_is_speech_related = conversations.follow_up_is_speech_related.astype(bool)

    conversations = conversations.assign(
        has_response=conversations.apply(
            has_response,
            axis=1,
            response_latency=args.response_latency,
            count_only_speech_related_responses=args.count_only_speech_related_responses,
        )
    )

    counter_non_speech = Counter(
        conversations[
            conversations.utt_is_speech_related == False
        ].utt_transcript_raw.values
    )
    print("Most common non-speech related sounds: ")
    print(counter_non_speech.most_common(20))

    # Filter for corpora that actually annotate non-speech-related sounds
    good_corpora = []
    print("Ratios nonspeech/speech for each corpus:")
    for corpus in conversations.corpus.unique():
        d_corpus = conversations[conversations.corpus == corpus]
        ratio = len(d_corpus[d_corpus.utt_is_speech_related == False]) / len(d_corpus)
        if ratio > args.min_ratio_nonspeech:
            good_corpora.append(corpus)
        print(f"{corpus}: {ratio:.5f}")
    print("Filtered corpora: ", good_corpora)

    conversations = conversations[conversations.corpus.isin(good_corpora)]

    conversations = filter_transcripts_based_on_num_child_utts(
        conversations, args.min_child_utts_per_transcript
    )

    conversations["age"] = conversations.age.apply(
        age_bin,
        min_age=args.min_age,
        max_age=args.max_age,
        num_months=AGE_BIN_NUM_MONTHS,
    )

    results_dir = "results/reproduce_warlaumont/"
    os.makedirs(results_dir, exist_ok=True)

    conversations.to_csv(results_dir + "conversations.csv", index=False)

    ###
    # Analyses
    ###
    print(f"\nFound {len(conversations)} micro-conversations")
    print(f"Number of corpora in the analysis: {len(conversations.corpus.unique())}")
    print(
        f"Number of children in the analysis: {len(conversations.child_name.unique())}"
    )
    print(
        f"Number of transcripts in the analysis: {len(conversations.transcript_file.unique())}"
    )

    perform_per_transcript_analyses(conversations)

    make_plots(conversations, results_dir)

    plt.show()

    return conversations


def perform_per_transcript_analyses(conversations):
    prop_responses_to_speech_related = (
        conversations[conversations.utt_is_speech_related]
        .groupby("transcript_file")
        .agg({"has_response": "mean"})
    )
    prop_responses_to_non_speech_related = (
        conversations[conversations.utt_is_speech_related == False]
        .groupby("transcript_file")
        .agg({"has_response": "mean"})
    )
    contingency_caregiver_timing = (
        prop_responses_to_speech_related - prop_responses_to_non_speech_related
    )
    contingency_caregiver_timing = contingency_caregiver_timing.dropna()
    mean = contingency_caregiver_timing.has_response.mean()
    standard_error = contingency_caregiver_timing.has_response.sem()
    p_value = ttest_1samp(contingency_caregiver_timing.values, popmean=0, alternative="greater")[1]
    print(
        f"contingency_caregiver_timing: {mean:.4f} SE: {standard_error:.4f} p-value:{p_value}"
    )

    prop_follow_up_speech_related_if_response_to_speech_related = (
        conversations[conversations.utt_is_speech_related & conversations.has_response]
        .groupby("transcript_file")
        .agg({"follow_up_is_speech_related": "mean"})
    )
    prop_follow_up_speech_related_if_no_response_to_speech_related = (
        conversations[
            conversations.utt_is_speech_related & (conversations.has_response == False)
        ]
        .groupby("transcript_file")
        .agg({"follow_up_is_speech_related": "mean"})
    )
    contingency_children_pos_case = (
        prop_follow_up_speech_related_if_response_to_speech_related
        - prop_follow_up_speech_related_if_no_response_to_speech_related
    )
    contingency_children_pos_case = contingency_children_pos_case.dropna()
    mean = contingency_children_pos_case.follow_up_is_speech_related.mean()
    standard_error = contingency_children_pos_case.follow_up_is_speech_related.sem()
    p_value = ttest_1samp(contingency_children_pos_case.values, popmean=0, alternative="greater")[1]
    print(
        f"contingency_children_pos_case: {mean:.4f} SE: {standard_error:.4f} p-value:{p_value}"
    )


def make_plots(conversations, results_dir):
    proportion_speech_related_per_transcript = conversations.groupby(
        "transcript_file"
    ).agg({"utt_is_speech_related": "mean", "age": "min"})
    plt.figure(figsize=(6, 3))
    axis = sns.regplot(
        data=proportion_speech_related_per_transcript,
        x="age",
        y="utt_is_speech_related",
        marker=".",
        logistic=True,
    )
    axis.set(xlabel="age (months)", ylabel="prop_speech_related")
    axis.set_xticks(
        np.arange(
            conversations.age.min(),
            conversations.age.max() + 1,
            step=AGE_BIN_NUM_MONTHS,
        )
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proportion_speech_related.png"), dpi=300)

    conversations_duplicated = conversations.copy()
    conversations_duplicated["age"] = math.inf
    conversations_with_avg_age = conversations.append(conversations_duplicated, ignore_index=True)

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_avg_age,
        x="age",
        y="has_response",
        hue="utt_is_speech_related",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("non-speech-related")
    legend.texts[1].set_text("speech-related")
    sns.move_legend(axis, "lower right")
    axis.set(xlabel="age (months)", ylabel="prop_caregiver_response")
    axis.set_xticklabels(sorted(conversations_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "contingency_caregivers.png"), dpi=300
    )

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_avg_age[conversations_with_avg_age.utt_is_speech_related],
        x="age",
        y="follow_up_is_speech_related",
        hue="has_response",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("no response")
    legend.texts[1].set_text("response")
    sns.move_legend(axis, "lower right")
    axis.set(xlabel="age (months)", ylabel="prop_follow_up_is_speech_related")
    axis.set_xticklabels(sorted(conversations_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "contingency_children_pos_case.png"), dpi=300)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = pd.read_csv(UTTERANCES_WITH_SPEECH_ACTS_FILE, index_col=0, converters={"pos": literal_eval, "tokens": literal_eval})

    print("Excluding corpora: ", args.excluded_corpora)
    utterances = utterances[~utterances.corpus.isin(args.excluded_corpora)]

    if args.corpora:
        print("Including only corpora: ", args.corpora)
        utterances = utterances[utterances.corpus.isin(args.corpora)]

    # Filter by age
    utterances = utterances[
        (args.min_age <= utterances.age) & (utterances.age <= args.max_age)
    ]

    min_age = utterances.age.min()
    max_age = utterances.age.max()
    mean_age = utterances.age.mean()
    print(
        f"Mean of child age in analysis: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )

    conversations = perform_analysis_speech_relatedness(utterances, args)
