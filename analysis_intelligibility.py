import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from analysis_reproduce_warlaumont import (
    perform_warlaumont_analysis,
    str2bool,
    get_micro_conversations,
    has_response,
)
from annotate import ANNOTATED_UTTERANCES_FILE
from preprocess import (
    CANDIDATE_CORPORA,
)
from utils import (
    age_bin,
    filter_corpora_based_on_response_latency_length,
)

DEFAULT_RESPONSE_THRESHOLD = 1000

DEFAULT_MIN_AGE = 10  # age of first words
DEFAULT_MAX_AGE = 48

AGE_BIN_NUM_MONTHS = 6

DEFAULT_RESPONSE_LATENCY_MAX_STANDARD_DEVIATIONS_OFF = 1

DEFAULT_COUNT_ONLY_INTELLIGIBLE_RESPONSES = True

DEFAULT_MIN_TRANSCRIPT_LENGTH = 0

# 1 second
DEFAULT_MAX_NEG_RESPONSE_LATENCY = -1 * 1000  # ms

# 1 minute
DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP = 1 * 60 * 1000  # ms

# Forrester: Does not annotate non-word sounds starting with & (phonological fragment), these are treated as words
DEFAULT_EXCLUDED_CORPORA = ["Forrester"]

# currently not used to exclude corpora, just stored for reference:
CORPORA_NOT_LONGITUDINAL = ["Gleason", "Rollins", "Edinburgh"]


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
        "--min-transcript-length",
        type=int,
        default=DEFAULT_MIN_TRANSCRIPT_LENGTH,
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


def caregiver_response_contingent_on_intelligibility(row):
    return ((row["is_intelligible"] == True) & (row["has_response"] == True)) | (
        (row["is_intelligible"] == False) & (row["has_response"] == False)
    )


def perform_contingency_analysis_intelligibility(conversations):
    # caregiver contingency:
    n_responses_to_intelligible = len(
        conversations[conversations.is_intelligible & conversations.has_response]
    )
    n_intelligible = len(conversations[conversations.is_intelligible])

    n_responses_to_unintelligible = len(
        conversations[
            (conversations.is_intelligible == False) & conversations.has_response
        ]
    )
    n_unintelligible = len(conversations[conversations.is_intelligible == False])

    if n_intelligible > 0 and n_unintelligible > 0:
        contingency_caregiver = (n_responses_to_intelligible / n_intelligible) - (
            n_responses_to_unintelligible / n_unintelligible
        )
    else:
        contingency_caregiver = np.nan

    # Contingency of child vocalization on previous adult response (positive case):
    n_follow_up_intelligible_if_response_to_intelligible = len(
        conversations[
            conversations.follow_up_intelligible
            & conversations.is_intelligible
            & conversations.has_response
        ]
    )

    n_follow_up_intelligible_if_no_response_to_intelligible = len(
        conversations[
            conversations.follow_up_intelligible
            & conversations.is_intelligible
            & (conversations.has_response == False)
        ]
    )
    n_no_responses_to_intelligible = len(
        conversations[
            conversations.is_intelligible & (conversations.has_response == False)
        ]
    )

    if n_responses_to_intelligible > 0 and n_no_responses_to_intelligible > 0:
        ratio_follow_up_intelligible_if_response_to_intelligible = (
            n_follow_up_intelligible_if_response_to_intelligible
            / n_responses_to_intelligible
        )
        ratio_follow_up_intelligible_if_no_response_to_intelligible = (
            n_follow_up_intelligible_if_no_response_to_intelligible
            / n_no_responses_to_intelligible
        )
        contingency_children_pos_case = (
            ratio_follow_up_intelligible_if_response_to_intelligible
            - ratio_follow_up_intelligible_if_no_response_to_intelligible
        )
    else:
        contingency_children_pos_case = np.nan

    proportion_intelligible = n_intelligible / (n_intelligible + n_unintelligible)

    return (
        contingency_caregiver,
        contingency_children_pos_case,
        proportion_intelligible,
    )


def perform_analysis_intelligibility(utterances, args):
    conversations = get_micro_conversations(utterances, args)

    conversations.dropna(
        subset=("is_intelligible", "response_is_intelligible", "follow_up_intelligible"),
        inplace=True,
    )

    conversations = filter_corpora_based_on_response_latency_length(
        conversations,
        args.response_latency_max_standard_deviations_off,
    )

    conversations = conversations.assign(
        has_response=conversations.apply(
            has_response,
            axis=1,
            response_latency=args.response_latency,
            max_neg_response_latency=args.max_neg_response_latency,
            count_only_intelligible_responses=args.count_only_intelligible_responses,
        )
    )

    conversations.dropna(
        subset=("has_response",),
        inplace=True,
    )

    # Get the number of children in all corpora:
    num_children = len(conversations.child_name.unique())
    print(f"Number of children in the analysis: {num_children}")

    # Label caregiver responses as contingent on child utterance or not
    conversations = conversations.assign(
        caregiver_response_contingent=conversations[
            ["is_intelligible", "has_response"]
        ].apply(caregiver_response_contingent_on_intelligibility, axis=1)
    )

    results_analysis = perform_warlaumont_analysis(
        conversations,
        args,
        perform_contingency_analysis_intelligibility,
        "proportion_intelligible",
    )
    results_dir = "results/intelligibility/"
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    sns.scatterplot(data=results_analysis, x="age", y="proportion_intelligible")

    conversations["age"] = conversations.age.apply(
        age_bin, min_age=args.min_age, max_age=args.max_age, num_months=AGE_BIN_NUM_MONTHS
    )

    plt.figure()
    plt.title("Caregiver contingency")
    sns.barplot(
        data=conversations,
        x="is_intelligible",
        y="has_response",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "contingency_caregivers.png"))

    plt.figure(figsize=(6, 3))
    plt.title("Caregiver contingency - per age group")
    axis = sns.barplot(
        data=conversations,
        x="age",
        y="has_response",
        hue="is_intelligible",
        ci=None,
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_caregiver_response")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "contingency_caregivers_per_age.png"))

    plt.figure()
    plt.title("Child contingency")
    sns.barplot(
        data=conversations[conversations.is_intelligible == True],
        x="has_response",
        y="follow_up_intelligible",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "contingency_children.png"))

    plt.figure(figsize=(6, 3))
    plt.title("Child contingency - per age group")
    axis = sns.barplot(
        data=conversations[conversations.is_intelligible == True],
        x="age",
        y="follow_up_intelligible",
        hue="has_response",
        ci=None,
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_follow_up_intelligible")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "contingency_children_per_age.png"))

    conversations.to_csv(results_dir + "conversations.csv", index=False)

    plt.show()

    return conversations


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

    min_age = utterances.age.min()
    max_age = utterances.age.max()
    mean_age = utterances.age.mean()
    print(
        f"Mean of child age in analysis: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )

    conversations = perform_analysis_intelligibility(utterances, args)
