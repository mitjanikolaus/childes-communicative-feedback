import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ztest

from analysis_reproduce_warlaumont import (
    str2bool,
    get_micro_conversations,
    has_response,
)
from preprocess import (
    CANDIDATE_CORPORA,
)
from utils import (
    age_bin,
    filter_corpora_based_on_response_latency_length,
    ANNOTATED_UTTERANCES_FILE,
    filter_transcripts_based_on_num_child_utts,
)

DEFAULT_RESPONSE_THRESHOLD = 1000

DEFAULT_MIN_AGE = 10  # age of first words
DEFAULT_MAX_AGE = 48

AGE_BIN_NUM_MONTHS = 6

# Set to -1 to skip filtering
DEFAULT_RESPONSE_LATENCY_MAX_STANDARD_DEVIATIONS_OFF = -1

DEFAULT_COUNT_ONLY_INTELLIGIBLE_RESPONSES = True

DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT = 1

# 10 seconds
DEFAULT_MAX_NEG_RESPONSE_LATENCY = -10 * 1000  # ms

# 1 minute
DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP = 1 * 60 * 1000

# Forrester: Does not annotate non-word sounds starting with & (phonological fragment), these are treated as words and
# should be excluded when annotating intelligibility based on rules.
# Providence: Some non-speech vocalizations such as laughter are incorrectly transcribed as 'yyy', and the timing
# information is of very poor quality
DEFAULT_EXCLUDED_CORPORA = ["Forrester", "Providence"]

# currently not used to exclude corpora, just stored for reference:
CORPORA_NOT_LONGITUDINAL = ["Gleason", "Rollins", "Edinburgh"]

SPEECH_ACTS_CLARIFICATION_REQUEST = [
    "EQ",  # Eliciting question (e.g. hmm?).
    "RR",  # Request to repeat utterance.
]


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--corpora",
        nargs="+",
        type=str,
        required=False,
        choices=CANDIDATE_CORPORA,
        help="Corpora to analyze. If not given, corpora are selected based on a response time variance threshold.",
    )
    argparser.add_argument(
        "--excluded-corpora",
        nargs="+",
        type=str,
        choices=CANDIDATE_CORPORA,
        default=DEFAULT_EXCLUDED_CORPORA,
        help="Corpora to exclude from analysis",
    )
    argparser.add_argument(
        "--response-latency",
        type=int,
        default=DEFAULT_RESPONSE_THRESHOLD,
        help="Response latency in milliseconds",
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
        "--response-latency-max-standard-deviations-off",
        type=int,
        default=DEFAULT_RESPONSE_LATENCY_MAX_STANDARD_DEVIATIONS_OFF,
        help="Number of standard deviations that the mean response latency of a corpus can be off the reference mean",
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
        "--count-only-intelligible_responses",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_COUNT_ONLY_INTELLIGIBLE_RESPONSES,
    )

    args = argparser.parse_args()

    return args


CAREGIVER_NAMES = {
    "dad",
    "daddy",
    "dada",
    "mom",
    "mum",
    "mommy",
    "mummy",
    "mama",
    "mamma",
}


def response_is_clarification_request(micro_conv):
    if micro_conv["has_response"]:
        if micro_conv["response_speech_act"] in SPEECH_ACTS_CLARIFICATION_REQUEST:
            words = set(
                [w.lower() for w in micro_conv["utt_transcript_raw"].split(" ")][:-1]
            )
            # If the initial utterance is just a call for attention, the response is not a clarification request.
            if len(words) <= 1 and len(words & CAREGIVER_NAMES) > 0:
                return False
            else:
                return True
    return False


def pos_feedback(
    row,
):
    """Positive feedback is counted if there is a response, and it's not a clarification request"""
    if not row["has_response"] or row["response_is_clarification_request"]:
        return False

    return True


def melt_is_intelligible_variable(conversations):
    conversations_melted = conversations.copy()
    conversations_melted["utterance_is_intelligible"] = conversations_melted[
        "utt_is_intelligible"
    ]
    del conversations_melted["utt_is_intelligible"]
    conversations_melted = pd.melt(
        conversations_melted.reset_index(),
        id_vars=[
            "index",
            "response_is_clarification_request",
            "child_name",
            "age",
            "transcript_file",
            "has_response",
            "pos_feedback",
        ],
        value_vars=["utterance_is_intelligible", "follow_up_is_intelligible"],
        var_name="is_follow_up",
        value_name="is_intelligible",
    )
    conversations_melted["is_follow_up"] = conversations_melted["is_follow_up"].apply(
        lambda x: x == "follow_up_is_intelligible"
    )
    conversations_melted["conversation_id"] = conversations_melted["index"]
    del conversations_melted["index"]
    return conversations_melted


def perform_analysis(utterances, args):
    # Discard non-speech, but keep uncertain (xxx, labelled as NA)
    utterances = utterances[utterances.is_speech_related != False]

    conversations = get_micro_conversations(utterances, args)

    conversations.dropna(
        subset=("response_latency", "response_latency_follow_up"),
        inplace=True,
    )

    conversations = filter_corpora_based_on_response_latency_length(
        conversations,
        args.response_latency_max_standard_deviations_off,
        args.min_age,
        args.max_age,
        args.max_response_latency_follow_up,
    )

    conversations = conversations.assign(
        has_response=conversations.apply(
            has_response,
            axis=1,
            response_latency=args.response_latency,
            count_only_intelligible_responses=args.count_only_intelligible_responses,
        )
    )
    conversations.dropna(
        subset=("has_response",),
        inplace=True,
    )

    conversations = conversations.assign(
        response_is_clarification_request=conversations.apply(
            response_is_clarification_request, axis=1
        )
    )

    conversations = conversations.assign(
        pos_feedback=conversations.apply(
            pos_feedback,
            axis=1,
        )
    )
    conversations.dropna(
        subset=("pos_feedback",),
        inplace=True,
    )

    conversations = filter_transcripts_based_on_num_child_utts(
        conversations, args.min_child_utts_per_transcript
    )

    conversations["age"] = conversations.age.apply(
        age_bin,
        min_age=args.min_age,
        max_age=args.max_age,
        num_months=AGE_BIN_NUM_MONTHS,
    )

    results_dir = "results/intelligibility/"
    os.makedirs(results_dir, exist_ok=True)

    conversations.to_csv(results_dir + "conversations.csv", index=False)
    conversations = pd.read_csv(results_dir + "conversations.csv")

    # Melt is_intellgible variable for CR analyses
    conversations_melted = melt_is_intelligible_variable(conversations)
    conversations_melted.to_csv(results_dir + "conversations_melted.csv", index=False)
    conversations_melted = pd.read_csv(results_dir + "conversations_melted.csv")

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

    counter_cr = Counter(
        conversations[
            conversations.response_is_clarification_request
        ].response_transcript_raw.values
    )
    print("Most common clarification requests: ")
    print(counter_cr.most_common(20))

    counter_cr = Counter(
        conversations[
            conversations.utt_is_intelligible == False
        ].utt_transcript_raw.values
    )
    print("Most common unintelligible utterances: ")
    print(counter_cr.most_common(20))

    perform_per_transcript_analyses(conversations)

    make_plots(conversations, conversations_melted, results_dir)

    plt.show()

    return conversations


def perform_per_transcript_analyses(conversations):
    print("Per-transcript analysis: ")

    prop_responses_to_intelligible = (
        conversations[conversations.utt_is_intelligible]
        .groupby("transcript_file")
        .agg({"has_response": "mean"})
    )
    prop_responses_to_unintelligible = (
        conversations[~conversations.utt_is_intelligible]
        .groupby("transcript_file")
        .agg({"has_response": "mean"})
    )
    contingency_caregiver_timing = (
        prop_responses_to_intelligible - prop_responses_to_unintelligible
    )
    contingency_caregiver_timing = contingency_caregiver_timing.dropna().values
    p_value = ztest(contingency_caregiver_timing, value=0.0, alternative="larger")[1]
    print(
        f"contingency_caregiver_timing: {contingency_caregiver_timing.mean():.4f} +-{contingency_caregiver_timing.std():.4f} p-value:{p_value}"
    )

    convs_with_response = conversations[conversations.has_response]
    prop_responses_to_intelligible = (
        convs_with_response[convs_with_response.utt_is_intelligible]
        .groupby("transcript_file")
        .agg({"response_is_clarification_request": "mean"})
    )
    prop_responses_to_unintelligible = (
        convs_with_response[convs_with_response.utt_is_intelligible == False]
        .groupby("transcript_file")
        .agg({"response_is_clarification_request": "mean"})
    )
    contingency_caregiver_clarification_requests = (
        prop_responses_to_unintelligible - prop_responses_to_intelligible
    )
    contingency_caregiver_clarification_requests = (
        contingency_caregiver_clarification_requests.dropna().values
    )
    p_value = ztest(
        contingency_caregiver_clarification_requests, value=0.0, alternative="larger"
    )[1]
    print(
        f"contingency_caregiver_clarification_requests: {contingency_caregiver_clarification_requests.mean():.4f} +-{contingency_caregiver_clarification_requests.std():.4f} p-value:{p_value}"
    )

    prop_follow_up_intelligible_if_response_to_intelligible = (
        conversations[conversations.has_response & conversations.utt_is_intelligible]
        .groupby("transcript_file")
        .agg({"follow_up_is_intelligible": "mean"})
    )
    prop_follow_up_intelligible_if_no_response_to_intelligible = (
        conversations[
            (conversations.has_response == False) & conversations.utt_is_intelligible
        ]
        .groupby("transcript_file")
        .agg({"follow_up_is_intelligible": "mean"})
    )
    contingency_children = (
        prop_follow_up_intelligible_if_response_to_intelligible
        - prop_follow_up_intelligible_if_no_response_to_intelligible
    )
    contingency_children = contingency_children.dropna().values
    p_value = ztest(contingency_children, value=0.0, alternative="larger")[1]
    print(
        f"contingency_children_pos_case: {contingency_children.mean():.4f} +-{contingency_children.std():.4f} p-value:{p_value}"
    )


def make_plots(conversations, conversations_melted, results_dir):
    proportion_intelligible_per_transcript = conversations.groupby(
        "transcript_file"
    ).agg({"utt_is_intelligible": "mean", "age": "min"})
    plt.figure(figsize=(6, 3))
    axis = sns.regplot(
        data=proportion_intelligible_per_transcript,
        x="age",
        y="utt_is_intelligible",
        logistic=True,
        marker=".",
    )
    plt.tight_layout()
    axis.set(xlabel="age (months)", ylabel="proportion intelligible")
    axis.set_xticks(
        np.arange(
            conversations.age.min(),
            conversations.age.max() + 1,
            step=AGE_BIN_NUM_MONTHS,
        )
    )
    plt.savefig(os.path.join(results_dir, "proportion_intelligible.png"), dpi=300)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(4, 1), sharey="all")
    axis = sns.barplot(
        data=conversations,
        ax=axes[0],
        x="age",
        y="has_response",
        hue="utt_is_intelligible",
        linewidth=1,
        edgecolor="w",
    )
    axis2 = sns.barplot(
        data=conversations,
        ax=axes[1],
        x=[''] * len(conversations),
        y="has_response",
        hue="utt_is_intelligible",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="all data")
    legend = axis.legend()
    legend.texts[0].set_text("unintelligible utterance")
    legend.texts[1].set_text("intelligible utterance")
    sns.move_legend(axis, "lower right")
    axis.set(xlabel="age (months)", ylabel="proportion of time-contingent\nresponses")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(os.path.join(results_dir, "cf_quality_timing.png"), dpi=300)

    conversations_with_response = conversations[conversations.has_response]
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(4, 1), sharey="all")
    axis = sns.barplot(
        data=conversations_with_response,
        ax=axes[0],
        x="age",
        y="response_is_clarification_request",
        hue="utt_is_intelligible",
        linewidth=1,
        edgecolor="w",
    )
    axis2 = sns.barplot(
        data=conversations_with_response,
        ax=axes[1],
        x=[''] * len(conversations_with_response),
        y="response_is_clarification_request",
        hue="utt_is_intelligible",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="all data")
    legend = axis.legend()
    legend.texts[0].set_text("unintelligible utterance")
    legend.texts[1].set_text("intelligible utterance")
    sns.move_legend(axis, "upper right")
    axis.set(xlabel="age (months)", ylabel="proportion of\nclarification request responses")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(os.path.join(results_dir, "cf_quality_clarification_request.png"), dpi=300)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(4, 1), sharey="all")
    axis = sns.barplot(
        data=conversations[conversations.utt_is_intelligible],
        ax=axes[0],
        x="age",
        y="follow_up_is_intelligible",
        hue="has_response",
        linewidth=1,
        edgecolor="w",
    )
    axis2 = sns.barplot(
        data=conversations[conversations.utt_is_intelligible],
        ax=axes[1],
        x=[''] * len(conversations[conversations.utt_is_intelligible]),
        y="follow_up_is_intelligible",
        hue="has_response",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="all data")
    legend = axis.legend()
    legend.texts[0].set_text("no response")
    legend.texts[1].set_text("response")
    sns.move_legend(axis, "lower right")
    axis.set(xlabel="age (months)", ylabel="proportion of\nintelligible follow-ups")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(os.path.join(results_dir, "cf_effect_pos_feedback_on_intelligible_timing.png"), dpi=300)

    conversations_melted_with_response = conversations_melted[
        conversations_melted.has_response
    ]
    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_melted_with_response,
        x="response_is_clarification_request",
        y="is_intelligible",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="proportion intelligible")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "cf_effect_clarification_request_control.png"),
        dpi=300,
    )

    conversations_melted_cr = conversations_melted_with_response[
        conversations_melted_with_response.response_is_clarification_request
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(4, 1), sharey="all")
    axis = sns.barplot(
        data=conversations_melted_cr,
        ax=axes[0],
        x="age",
        y="is_intelligible",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
    )
    axis2 = sns.barplot(
        data=conversations_melted_cr,
        ax=axes[1],
        x=[''] * len(conversations_melted_cr),
        y="is_intelligible",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="all data")
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    sns.move_legend(axis, "upper left")
    axis.set(xlabel="age (months)", ylabel="proportion intelligible")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(os.path.join(results_dir, "cf_effect_clarification_request.png"), dpi=300)


if __name__ == "__main__":
    args = parse_args()

    print(args)

    utterances = pd.read_pickle(ANNOTATED_UTTERANCES_FILE)

    print("Excluding corpora: ", args.excluded_corpora)
    utterances = utterances[~utterances.corpus.isin(args.excluded_corpora)]

    if args.corpora:
        print("Including only corpora: ", args.corpora)
        utterances = utterances[utterances.corpus.isin(args.corpora)]

    # Filter by age
    utterances = utterances[
        (args.min_age <= utterances.age) & (utterances.age <= args.max_age)
    ]

    conversations = perform_analysis(utterances, args)
