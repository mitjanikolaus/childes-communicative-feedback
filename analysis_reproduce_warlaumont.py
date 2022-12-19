import argparse
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from scipy.stats import ttest_1samp

from extract_micro_conversations import DEFAULT_RESPONSE_THRESHOLD
from utils import (
    age_bin,
    str2bool,
    filter_transcripts_based_on_num_child_utts,
    MICRO_CONVERSATIONS_FILE,
)

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


def perform_analysis_speech_relatedness(conversations, args):
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
            response_latency=DEFAULT_RESPONSE_THRESHOLD,
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

    conversations.to_csv(results_dir + "conversations.csv")

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
    conversations_with_avg_age = pd.concat([conversations, conversations_duplicated], ignore_index=True)

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

    conversations = pd.read_csv(MICRO_CONVERSATIONS_FILE, index_col=0)

    print("Excluding corpora: ", args.excluded_corpora)
    conversations = conversations[~conversations.corpus.isin(args.excluded_corpora)]

    if args.corpora:
        print("Including only corpora: ", args.corpora)
        utterances = conversations[conversations.corpus.isin(args.corpora)]

    # Filter by age
    conversations = conversations[
        (args.min_age <= conversations.age) & (conversations.age <= args.max_age)
    ]

    min_age = conversations.age.min()
    max_age = conversations.age.max()
    mean_age = conversations.age.mean()
    print(
        f"Mean of child age in analysis: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )

    perform_analysis_speech_relatedness(conversations, args)
