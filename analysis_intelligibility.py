import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ztest

from analysis_reproduce_warlaumont import (
    perform_warlaumont_analysis,
    perform_glm_analysis,
    perform_analyses,
    str2bool,
)
from utils import (
    filter_corpora_based_on_response_latency_length,
    get_path_of_utterances_file,
)
from search_child_utterances_and_responses import (
    CANDIDATE_CORPORA,
    DEFAULT_RESPONSE_THRESHOLD,
)
from utils import (
    remove_babbling,
    EMPTY_UTTERANCE,
    clean_utterance,
    remove_nonspeech_events,
    remove_whitespace,
)

# TODO: define age range
DEFAULT_MIN_AGE = 10  # age of first words?
DEFAULT_MAX_AGE = 60

# Number of standard deviations that the mean response latency of a corpus can be off the reference mean
DEFAULT_RESPONSE_LATENCY_MAX_STANDARD_DEVIATIONS_OFF = 1

# Label for partially intelligible utterances
# Set to True to count as intelligible, False to count as unintelligible or None to exclude these utterances from
# the analysis
DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE = None

DEFAULT_COUNT_ONLY_INTELLIGIBLE_RESPONSES = False

DEFAULT_MIN_TRANSCRIPT_LENGTH = 0

# TODO check that pause is not too long (neg): what is a reasonable value?
# 1 second
DEFAULT_MAX_NEG_RESPONSE_LATENCY = -1 * 1000  # ms


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
        "--count-only-intelligible_responses",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_COUNT_ONLY_INTELLIGIBLE_RESPONSES,
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


def caregiver_response_contingent_on_intelligibility(row):
    return (
        (row["utt_child_intelligible"] == True) & (row["caregiver_response"] == True)
    ) | (
        (row["utt_child_intelligible"] == False) & (row["caregiver_response"] == False)
    )


def perform_contingency_analysis_intelligibility(utterances):
    # caregiver contingency:
    n_responses_to_intelligible = len(
        utterances[utterances.utt_child_intelligible & utterances.caregiver_response]
    )
    n_intelligible = len(utterances[utterances.utt_child_intelligible])

    n_responses_to_unintelligible = len(
        utterances[
            (utterances.utt_child_intelligible == False) & utterances.caregiver_response
        ]
    )
    n_unintelligible = len(utterances[utterances.utt_child_intelligible == False])

    if n_intelligible > 0 and n_unintelligible > 0:
        contingency_caregiver = (n_responses_to_intelligible / n_intelligible) - (
            n_responses_to_unintelligible / n_unintelligible
        )
    else:
        contingency_caregiver = np.nan

    # Contingency of child vocalization on previous adult response (positive case):
    n_follow_up_intelligible_if_response_to_intelligible = len(
        utterances[
            utterances.follow_up_intelligible
            & utterances.utt_child_intelligible
            & utterances.caregiver_response
        ]
    )

    n_follow_up_intelligible_if_no_response_to_intelligible = len(
        utterances[
            utterances.follow_up_intelligible
            & utterances.utt_child_intelligible
            & (utterances.caregiver_response == False)
        ]
    )
    n_no_responses_to_intelligible = len(
        utterances[
            utterances.utt_child_intelligible & (utterances.caregiver_response == False)
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

    # Contingency of child vocalization on previous adult response (negative case):
    n_follow_up_intelligible_if_no_response_to_unintelligible = len(
        utterances[
            utterances.follow_up_intelligible
            & (utterances.utt_child_intelligible == False)
            & (utterances.caregiver_response == False)
        ]
    )
    n_no_responses_to_unintelligible = len(
        utterances[
            (utterances.utt_child_intelligible == False)
            & (utterances.caregiver_response == False)
        ]
    )

    n_follow_up_intelligible_if_response_to_unintelligible = len(
        utterances[
            utterances.follow_up_intelligible
            & (utterances.utt_child_intelligible == False)
            & utterances.caregiver_response
        ]
    )
    n_responses_to_unintelligible = len(
        utterances[
            (utterances.utt_child_intelligible == False) & utterances.caregiver_response
        ]
    )

    if n_no_responses_to_unintelligible > 0 and n_responses_to_unintelligible > 0:

        ratio_follow_up_intelligible_if_no_response_to_unintelligible = (
            n_follow_up_intelligible_if_no_response_to_unintelligible
            / n_no_responses_to_unintelligible
        )
        ratio_follow_up_intelligible_if_response_to_unintelligible = (
            n_follow_up_intelligible_if_response_to_unintelligible
            / n_responses_to_unintelligible
        )
        contingency_children_neg_case = (
            ratio_follow_up_intelligible_if_no_response_to_unintelligible
            - ratio_follow_up_intelligible_if_response_to_unintelligible
        )
    else:
        contingency_children_neg_case = np.nan

    # if (
    #     not contingency_children_pos_case == np.nan
    #     and not contingency_children_neg_case == np.nan
    # ):
    #     ratio_contingent_follow_ups = (
    #         n_follow_up_intelligible_if_response_to_intelligible
    #         + n_follow_up_intelligible_if_no_response_to_unintelligible
    #     ) / (n_responses_to_intelligible + n_no_responses_to_unintelligible)
    #     ratio_incontingent_follow_ups = (
    #         n_follow_up_intelligible_if_no_response_to_intelligible
    #         + n_follow_up_intelligible_if_response_to_unintelligible
    #     ) / (n_no_responses_to_intelligible + n_responses_to_unintelligible)
    #
    #     child_contingency_both_cases = (
    #         ratio_contingent_follow_ups - ratio_incontingent_follow_ups
    #     )
    #     print(f"Child contingency (both cases): {child_contingency_both_cases:.4f}")
    #     child_contingency_both_cases_same_weighting = np.mean(
    #         [contingency_children_pos_case, contingency_children_neg_case]
    #     )
    #
    #     print(
    #         f"Child contingency (both cases, same weighting of positive and negative cases): "
    #         f"{child_contingency_both_cases_same_weighting:.4f}"
    #     )
    #
    # else:
    #     child_contingency_both_cases = np.nan
    #     child_contingency_both_cases_same_weighting = np.nan

    return (
        contingency_caregiver,
        contingency_children_pos_case,
        contingency_children_neg_case,
    )


def perform_analysis_intelligibility(utterances, args):
    # Clean utterances
    utterances["utt_child"] = utterances.utt_child.apply(clean_utterance)
    utterances["utt_car"] = utterances.utt_car.apply(clean_utterance)
    utterances["utt_child_follow_up"] = utterances.utt_child_follow_up.apply(
        clean_utterance
    )

    # Remove nonspeech events
    utterances["utt_child"] = utterances.utt_child.apply(remove_nonspeech_events)
    utterances["utt_car"] = utterances.utt_car.apply(remove_nonspeech_events)
    utterances["utt_child_follow_up"] = utterances.utt_child_follow_up.apply(
        remove_nonspeech_events
    )

    # Drop empty children's utterances (these are non-speech related)
    utterances = utterances[
        (
            (utterances.utt_child != EMPTY_UTTERANCE)
            & (utterances.utt_child_follow_up != EMPTY_UTTERANCE)
        )
    ]

    # Label utterances as intelligible or unintelligible
    def is_intelligible(
        utterance, label_partially_intelligible=args.label_partially_intelligible
    ):
        utt_without_babbling = remove_babbling(utterance)

        utt_without_babbling = remove_whitespace(utt_without_babbling)
        if utt_without_babbling == EMPTY_UTTERANCE:
            return False

        is_partly_intelligible = len(utt_without_babbling) != len(utterance)
        if is_partly_intelligible:
            return label_partially_intelligible

        return True

    utterances = utterances.assign(
        utt_child_intelligible=utterances.utt_child.apply(is_intelligible)
    )
    utterances = utterances.assign(
        follow_up_intelligible=utterances.utt_child_follow_up.apply(is_intelligible)
    )

    # Label caregiver responses as present or not
    def caregiver_intelligible_response(row):
        return (row["response_latency"] <= args.response_latency) & (
            (not args.count_only_intelligible_responses)
            | is_intelligible(row["utt_car"], label_partially_intelligible=True)
        )

    utterances = utterances.assign(
        caregiver_response=utterances.apply(caregiver_intelligible_response, axis=1)
    )

    # Remove NaNs
    utterances = utterances.dropna(
        subset=("utt_child_intelligible", "follow_up_intelligible")
    )

    # Label caregiver responses as contingent on child utterance or not
    utterances = utterances.assign(
        caregiver_response_contingent=utterances[
            ["utt_child_intelligible", "caregiver_response"]
        ].apply(caregiver_response_contingent_on_intelligibility, axis=1)
    )

    perform_warlaumont_analysis(
        utterances, perform_contingency_analysis_intelligibility
    )

    perform_glm_analysis(utterances, "utt_child_intelligible", "follow_up_intelligible")

    sns.barplot(
        data=utterances,
        x="utt_child_intelligible",
        y="follow_up_intelligible",
        hue="caregiver_response_contingent",
    )
    plt.show()

    return utterances


if __name__ == "__main__":
    args = parse_args()

    utterances = perform_analyses(args, perform_analysis_intelligibility)
