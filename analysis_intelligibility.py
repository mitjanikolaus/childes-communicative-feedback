import argparse
import math
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


from analysis_reproduce_warlaumont import (
    perform_average_and_per_transcript_analysis,
    str2bool,
    get_micro_conversations,
    has_response,
)
from preprocess import (
    CANDIDATE_CORPORA,
)
from utils import (
    age_bin,
    filter_corpora_based_on_response_latency_length, ANNOTATED_UTTERANCES_FILE, SPEECH_ACTS_NO_FUNCTION,
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

# 10 seconds
DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP = 10 * 1000  # ms

# Forrester: Does not annotate non-word sounds starting with & (phonological fragment), these are treated as words
DEFAULT_EXCLUDED_CORPORA = ["Forrester"]

# currently not used to exclude corpora, just stored for reference:
CORPORA_NOT_LONGITUDINAL = ["Gleason", "Rollins", "Edinburgh"]

SPEECH_ACTS_CLARIFICATION_REQUEST = [
    "EQ",   # Eliciting question (e.g. hmm?).
    "RR",   # Request to repeat utterance.
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


def pos_feedback(
    row,
):
    """Positive feedback is counted if there is a response, and it's not a clarification request"""
    if not row["has_response"] or row["response_is_clarification_request"]:
        return False

    return True


def perform_analysis(utterances, args):
    conversations = get_micro_conversations(utterances, args)

    conversations.dropna(
        subset=("response_latency", "response_latency_follow_up"),
        inplace=True,
    )

    # Drop all non-speech related (but keep dummy responses!)
    conversations = conversations[
        conversations.is_speech_related &
        (conversations.response_is_speech_related | (conversations.response_start_time == math.inf)) &
        conversations.follow_up_is_speech_related
    ]

    conversations = filter_corpora_based_on_response_latency_length(
        conversations,
        args.response_latency_max_standard_deviations_off,
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
        response_is_clarification_request=conversations.response_speech_act.apply(
            lambda sa: sa in SPEECH_ACTS_CLARIFICATION_REQUEST)
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


    # Get the number of children in all corpora:
    num_children = len(conversations.child_name.unique())
    print(f"Number of children in the analysis: {num_children}")

    conversations["age"] = conversations.age.apply(
        age_bin, min_age=args.min_age, max_age=args.max_age, num_months=AGE_BIN_NUM_MONTHS
    )

    results_dir = "results/intelligibility/"
    os.makedirs(results_dir, exist_ok=True)

    conversations.to_csv(results_dir + "conversations.csv", index=False)
    conversations = pd.read_csv(results_dir + "conversations.csv")

    # Melt is_intellgible variable for CR analyses
    conversations_melted = conversations.copy()
    conversations_melted["utterance_is_intelligible"] = conversations_melted["is_intelligible"]
    del conversations_melted["is_intelligible"]
    conversations_melted = pd.melt(conversations_melted, id_vars=["response_is_clarification_request", "child_name", "age", "has_response", "pos_feedback"],
                                   value_vars=['utterance_is_intelligible', 'follow_up_is_intelligible'], var_name='is_follow_up',
                                   value_name='is_intelligible')
    conversations_melted["is_follow_up"] = conversations_melted["is_follow_up"].apply(lambda x: x == "follow_up_is_intelligible")
    conversations_melted.to_csv(results_dir + "conversations_melted.csv", index=False)
    conversations_melted = pd.read_csv(results_dir + "conversations_melted.csv")

    perform_average_and_per_transcript_analysis(
        conversations,
        args,
        perform_intelligibility_analysis,
    )

    make_plots(conversations, conversations_melted, results_dir)

    plt.show()

    return conversations


def make_plots(conversations, conversations_melted, results_dir):
    proportion_intelligible_per_transcript = conversations.groupby("transcript_file").agg({"is_intelligible": "mean", "age": "mean"})
    plt.figure(figsize=(12, 6))
    sns.regplot(
        data=proportion_intelligible_per_transcript,
        x="age",
        y="is_intelligible",
        marker=".",
        logx=True,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proportion_intelligible.png"))


    plt.figure(figsize=(12, 6))
    plt.title("Caregiver timing contingency")
    axis = sns.barplot(
        data=conversations,
        x="age",
        y="has_response",
        hue="is_intelligible",
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_response")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cf_quality_timing.png"))

    conversations_with_response = conversations[conversations.has_response]

    plt.figure(figsize=(12, 6))
    plt.title("Caregiver clarification request contingency")
    axis = sns.barplot(
        data=conversations_with_response,
        x="age",
        y="response_is_clarification_request",
        hue="is_intelligible",
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_clarification_request")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cf_quality_clarification_request.png"))

    plt.figure(figsize=(12, 6))
    plt.title("Caregiver contingency: timing + clarification requests")
    axis = sns.barplot(
        data=conversations,
        x="age",
        y="pos_feedback",
        hue="is_intelligible",
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_pos_feedback")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cf_quality_all.png"))

    plt.figure(figsize=(12, 6))
    plt.title("Child contingency: effect of positive feedback")
    axis = sns.barplot(
        data=conversations,
        x="age",
        y="follow_up_is_intelligible",
        hue="has_response",
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_follow_up_is_intelligible")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cf_effect_pos_feedback_timing.png"))

    plt.figure(figsize=(12, 6))
    plt.title("Child contingency: effect of clarification requests")
    axis = sns.barplot(
        data=conversations_melted[conversations_melted.response_is_clarification_request],
        x="age",
        y="is_intelligible",
        hue="is_follow_up",
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_is_intelligible")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cf_effect_neg_feedback_clarification_request.png"))

    plt.figure(figsize=(12, 6))
    plt.title("Child contingency: effect of clarification requests: control")
    axis = sns.barplot(
        data=conversations_melted[~conversations_melted.response_is_clarification_request],
        x="age",
        y="is_intelligible",
        hue="is_follow_up",
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_is_intelligible")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cf_effect_neg_feedback_clarification_request_control_condition.png"))

    # plt.figure(figsize=(12, 6))
    # plt.title("Child contingency: effect of pauses")
    # axis = sns.barplot(
    #     data=conversations_melted[~conversations_melted.has_response],
    #     x="age",
    #     y="is_intelligible",
    #     hue="is_follow_up",
    # )
    # sns.move_legend(axis, "lower right")
    # axis.set(ylabel="prob_is_intelligible")
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, "cf_effect_neg_feedback_pauses.png"))
    #
    # plt.figure(figsize=(12, 6))
    # plt.title("Child contingency: effect of pauses: control")
    # axis = sns.barplot(
    #     data=conversations_melted[~conversations_melted.has_response],
    #     x="age",
    #     y="is_intelligible",
    #     hue="is_follow_up",
    # )
    # sns.move_legend(axis, "lower right")
    # axis.set(ylabel="prob_is_intelligible")
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, "cf_effect_neg_feedback_pauses_control_case.png"))


def perform_intelligibility_analysis(conversations):
    # caregiver contingency (all):
    n_pos_feedback_to_intelligible = len(
        conversations[conversations.is_intelligible & conversations.pos_feedback]
    )
    n_intelligible = len(conversations[conversations.is_intelligible])

    n_pos_feedback_to_unintelligible = len(
        conversations[
            (conversations.is_intelligible == False) & conversations.pos_feedback
        ]
    )
    n_not_intelligible = len(conversations[conversations.is_intelligible == False])

    if n_intelligible > 0 and n_not_intelligible > 0:
        contingency_caregiver_all = (n_pos_feedback_to_intelligible / n_intelligible) - (
            n_pos_feedback_to_unintelligible / n_not_intelligible
        )
    else:
        contingency_caregiver_all = np.nan

    # caregiver contingency (timing):
    n_response_to_intelligible = len(
        conversations[conversations.is_intelligible & conversations.has_response]
    )
    n_intelligible = len(conversations[conversations.is_intelligible])

    n_response_to_unintelligible = len(
        conversations[
            (conversations.is_intelligible == False) & conversations.has_response
        ]
    )
    n_not_intelligible = len(conversations[conversations.is_intelligible == False])

    if n_intelligible > 0 and n_not_intelligible > 0:
        contingency_caregiver_timing = (n_response_to_intelligible / n_intelligible) - (
                n_response_to_unintelligible / n_not_intelligible
        )
    else:
        contingency_caregiver_timing = np.nan

    # Contingency of child vocalization on previous adult response (positive feedback: timing):
    n_follow_up_is_intelligible_if_response_to_intelligible = len(
        conversations[
            conversations.follow_up_is_intelligible
            & conversations.has_response
        ]
    )
    n_response_to_intelligible = len(
        conversations[
            (conversations.has_response)
        ]
    )

    n_follow_up_is_intelligible_if_no_response_to_intelligible = len(
        conversations[
            conversations.follow_up_is_intelligible
            & (conversations.has_response == False)
        ]
    )
    n_no_response_to_intelligible = len(
        conversations[
            (conversations.has_response == False)
        ]
    )

    if n_response_to_intelligible and n_no_response_to_intelligible:
        ratio_follow_up_is_intelligible_if_response = (
            n_follow_up_is_intelligible_if_response_to_intelligible
            / n_response_to_intelligible
        )
        ratio_follow_up_is_intelligible_if_no_response = (
            n_follow_up_is_intelligible_if_no_response_to_intelligible
            / n_no_response_to_intelligible
        )
        contingency_children_pos_case = (
            ratio_follow_up_is_intelligible_if_response
            - ratio_follow_up_is_intelligible_if_no_response
        )
    else:
        contingency_children_pos_case = np.nan

    # TODO: filtering out cases with no response
    convs_with_response = conversations[conversations.has_response]

    # Caregiver contingency (speech acts)
    n_pos_feedback_to_intelligible = len(
        convs_with_response[convs_with_response.is_intelligible & ~convs_with_response.response_speech_act.isin(SPEECH_ACTS_CLARIFICATION_REQUEST)]
    )
    n_intelligible = len(convs_with_response[convs_with_response.is_intelligible])

    n_pos_feedback_to_unintelligible = len(
        convs_with_response[
            (convs_with_response.is_intelligible == False) & ~convs_with_response.response_speech_act.isin(SPEECH_ACTS_CLARIFICATION_REQUEST)
        ]
    )
    n_not_intelligible = len(convs_with_response[convs_with_response.is_intelligible == False])

    if n_intelligible > 0 and n_not_intelligible > 0:
        contingency_caregiver_speech_acts = (n_pos_feedback_to_intelligible / n_intelligible) - (
                n_pos_feedback_to_unintelligible / n_not_intelligible
        )
    else:
        contingency_caregiver_speech_acts = np.nan


    # Negative feedback case:
    n_neg_feedback_to_unintelligible = len(
        convs_with_response[
            (convs_with_response.is_intelligible == False)
            & (convs_with_response.response_speech_act.isin(SPEECH_ACTS_CLARIFICATION_REQUEST))
        ]
    )
    n_neg_feedback = len(convs_with_response[convs_with_response.response_speech_act.isin(SPEECH_ACTS_CLARIFICATION_REQUEST)])

    n_follow_up_unintelligible_if_neg_feedback_to_unintelligible = len(
        convs_with_response[
            (convs_with_response.is_intelligible == False)
            & (convs_with_response.response_speech_act.isin(SPEECH_ACTS_CLARIFICATION_REQUEST))
            & (convs_with_response.follow_up_is_intelligible == False)
        ]
    )

    if n_neg_feedback:
        ratio_before_feedback = (n_neg_feedback_to_unintelligible / n_neg_feedback)
        # print("Ratio before: ", ratio_before_feedback)
        ratio_after_feedback = (n_follow_up_unintelligible_if_neg_feedback_to_unintelligible / n_neg_feedback)
        # print("Ratio after: ", ratio_after_feedback)
        contingency_children_neg_case = ratio_before_feedback - ratio_after_feedback
    else:
        contingency_children_neg_case = np.nan

    # Negative feedback control case:
    n_pos_feedback_to_unintelligible = len(
        convs_with_response[
            (convs_with_response.is_intelligible == False)
            & (~convs_with_response.response_speech_act.isin(SPEECH_ACTS_CLARIFICATION_REQUEST))
        ]
    )
    n_pos_feedback = len(convs_with_response[~convs_with_response.response_speech_act.isin(SPEECH_ACTS_CLARIFICATION_REQUEST)])

    n_follow_up_unintelligible_if_pos_feedback_to_unintelligible = len(
        convs_with_response[
            (convs_with_response.is_intelligible == False)
            & (~convs_with_response.response_speech_act.isin(SPEECH_ACTS_CLARIFICATION_REQUEST))
            & (convs_with_response.follow_up_is_intelligible == False)
        ]
    )

    if n_pos_feedback:
        ratio_before_feedback = (n_pos_feedback_to_unintelligible / n_pos_feedback)
        ratio_after_feedback = (n_follow_up_unintelligible_if_pos_feedback_to_unintelligible / n_pos_feedback)
        contingency_children_neg_case_control = ratio_before_feedback - ratio_after_feedback
    else:
        contingency_children_neg_case_control = np.nan

    return {
        "age": conversations.age.mean(),
        "contingency_caregiver_timing": contingency_caregiver_timing,
        "contingency_caregiver_speech_acts": contingency_caregiver_speech_acts,
        "contingency_caregiver_all": contingency_caregiver_all,
        "positive_feedback_timing": contingency_children_pos_case,
        "contingency_children_neg_case": contingency_children_neg_case,
        "contingency_children_neg_case_control": contingency_children_neg_case_control,
    }


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
        (args.min_age - AGE_BIN_NUM_MONTHS/2 <= utterances.age) & (utterances.age <= args.max_age + AGE_BIN_NUM_MONTHS/2)
    ]

    min_age = utterances.age.min()
    max_age = utterances.age.max()
    mean_age = utterances.age.mean()
    print(
        f"Mean of child age in analysis: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )

    conversations = perform_analysis(utterances, args)
