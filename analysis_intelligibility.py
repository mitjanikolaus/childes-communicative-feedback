import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import filter_corpora_based_on_response_latency_length, get_path_of_utterances_file
from search_child_utterances_and_responses import CANDIDATE_CORPORA, DEFAULT_RESPONSE_THRESHOLD
from utils import (
    remove_babbling,
    EMPTY_UTTERANCE,
    clean_utterance,
    remove_nonspeech_events,
    remove_whitespace,
)

# TODO: define age range
MIN_AGE = 10  # age of first words?
MAX_AGE = 60


# Number of standard deviations that the mean response latency of a corpus can be off the reference mean
RESPONSE_LATENCY_STANDARD_DEVIATIONS_OFF = 1

# Label for partially intelligible utterances
# Set to True to count as intelligible, False to count as unintelligible or None to exclude these utterances from
# the analysis
LABEL_PARTIALLY_INTELLIGIBLE = None

# TODO check that pause is not too long (neg): what is a reasonable value?
# 1 second
MAX_NEG_RESPONSE_LATENCY = -1 * 1000  # ms

COUNT_ONLY_INTELLIGIBLE_RESPONSES = False

# Forrester: Does not annotate non-word sounds starting with & (phonological fragment), these are treated as words
EXCLUDED_CORPORA = ["Forrester"]

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
        "--response-latency",
        type=int,
        default=DEFAULT_RESPONSE_THRESHOLD,
        help="Response latency in milliseconds",
    )
    args = argparser.parse_args()

    return args


def is_intelligible(
    utterance, label_partially_intelligible=LABEL_PARTIALLY_INTELLIGIBLE
):
    utt_without_babbling = remove_babbling(utterance)

    utt_without_babbling = remove_whitespace(utt_without_babbling)
    if utt_without_babbling == EMPTY_UTTERANCE:
        return False

    is_partly_intelligible = len(utt_without_babbling) != len(utterance)
    if is_partly_intelligible:
        return label_partially_intelligible

    return True


def caregiver_response_contingent_on_intelligibility(row):
    return (
        (row["utt_child_intelligible"] == True) & (row["caregiver_response"] == True)
    ) | (
        (row["utt_child_intelligible"] == False) & (row["caregiver_response"] == False)
    )


def perform_analysis_intelligibility(utterances, response_latency):
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
    utterances = utterances.assign(
        utt_child_intelligible=utterances.utt_child.apply(is_intelligible)
    )
    utterances = utterances.assign(
        follow_up_intelligible=utterances.utt_child_follow_up.apply(is_intelligible)
    )

    # Label caregiver responses as present or not
    def caregiver_intelligible_response(row):
        return (row["response_latency"] <= response_latency) & (
                (not COUNT_ONLY_INTELLIGIBLE_RESPONSES)
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

    print(f"\nFound {len(utterances)} turns")
    if len(utterances) > 0:
        # Caregiver contingency:
        n_responses_intelligible = len(
            utterances[
                utterances.utt_child_intelligible & utterances.caregiver_response
            ]
        )
        n_intelligible = len(utterances[utterances.utt_child_intelligible])

        n_responses_unintelligible = len(
            utterances[
                (utterances.utt_child_intelligible == False)
                & utterances.caregiver_response
            ]
        )
        n_unintelligible = len(utterances[utterances.utt_child_intelligible == False])

        contingency_caregiver = (n_responses_intelligible / n_intelligible) - (
            n_responses_unintelligible / n_unintelligible
        )
        print(f"Caregiver contingency: {contingency_caregiver:.4f}")

        # Contingency of child vocalization on previous adult response (positive case):
        n_follow_up_intelligible_if_response_to_intelligible = len(
            utterances[
                utterances.follow_up_intelligible
                & utterances.utt_child_intelligible
                & utterances.caregiver_response
            ]
        )
        n_responses_to_intelligible = len(
            utterances[
                utterances.utt_child_intelligible & utterances.caregiver_response
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
                utterances.utt_child_intelligible
                & (utterances.caregiver_response == False)
            ]
        )

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
                (utterances.utt_child_intelligible == False)
                & utterances.caregiver_response
            ]
        )

        if (
            (n_no_responses_to_unintelligible > 0)
            and (n_responses_to_unintelligible > 0)
            and (n_responses_to_intelligible > 0)
            and (n_no_responses_to_intelligible > 0)
        ):
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

            print(
                f"Child contingency (positive case): {contingency_children_pos_case:.4f}"
            )

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

            print(
                f"Child contingency (negative case): {contingency_children_neg_case:.4f}"
            )

            ratio_contingent_follow_ups = (
                n_follow_up_intelligible_if_response_to_intelligible
                + n_follow_up_intelligible_if_no_response_to_unintelligible
            ) / (n_responses_to_intelligible + n_no_responses_to_unintelligible)
            ratio_incontingent_follow_ups = (
                n_follow_up_intelligible_if_no_response_to_intelligible
                + n_follow_up_intelligible_if_response_to_unintelligible
            ) / (n_no_responses_to_intelligible + n_responses_to_unintelligible)

            child_contingency_both_cases = (
                ratio_contingent_follow_ups - ratio_incontingent_follow_ups
            )
            print(f"Child contingency (both cases): {child_contingency_both_cases:.4f}")
            child_contingency_both_cases_same_weighting = np.mean(
                [contingency_children_pos_case, contingency_children_neg_case]
            )

            print(
                f"Child contingency (both cases, same weighting of positive and negative cases): "
                f"{child_contingency_both_cases_same_weighting:.4f}"
            )

        # Statsmodels prefers 1 and 0 over True and False:
        utterances.replace({False: 0, True: 1}, inplace=True)

        mod = smf.glm(
            "caregiver_response ~ utt_child_intelligible",
            family=sm.families.Binomial(),
            data=utterances,
        ).fit()
        print(mod.summary())

        mod = smf.glm(
            "follow_up_intelligible ~ caregiver_response_contingent",
            family=sm.families.Binomial(),
            data=utterances,
        ).fit()
        print(mod.summary())

        print("GLM - positive case")
        mod = smf.glm(
            "follow_up_intelligible ~ caregiver_response_contingent",
            family=sm.families.Binomial(),
            data=utterances[utterances.utt_child_intelligible == True],
        ).fit()
        print(mod.summary())

        print("GLM - negative case")
        mod = smf.glm(
            "follow_up_intelligible ~ caregiver_response_contingent",
            family=sm.families.Binomial(),
            data=utterances[utterances.utt_child_intelligible == False],
        ).fit()
        print(mod.summary())

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

    utterances = pd.read_csv(get_path_of_utterances_file(args.response_latency), index_col=None)

    # Fill in empty strings for dummy caregiver responses
    utterances.utt_car.fillna("", inplace=True)

    # Remove preprocessed_utterances with too long negative pauses
    utterances = utterances[(utterances.response_latency >= MAX_NEG_RESPONSE_LATENCY)]

    if not args.corpora:
        print(f"No corpora given, selecting based on average response latency")
        args.corpora = filter_corpora_based_on_response_latency_length(
            CANDIDATE_CORPORA,
            utterances,
            MIN_AGE,
            MAX_AGE,
            RESPONSE_LATENCY_STANDARD_DEVIATIONS_OFF,
        )

    print("Excluding corpora: ", EXCLUDED_CORPORA)
    utterances = utterances[~utterances.corpus.isin(EXCLUDED_CORPORA)]

    print(f"Corpora included in analysis: {args.corpora}")
    # Filter by corpora
    utterances = utterances[utterances.corpus.isin(args.corpora)]

    # Filter by age
    utterances = utterances[(MIN_AGE <= utterances.age) & (utterances.age <= MAX_AGE)]

    min_age = utterances.age.min()
    max_age = utterances.age.max()
    mean_age = utterances.age.mean()
    print(
        f"Mean of child age in analysis: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )
    mean_latency = utterances[
        utterances.response_latency < math.inf
    ].response_latency.mean()
    std_mean_latency = utterances[
        utterances.response_latency < math.inf
    ].response_latency.std()
    print(
        f"Mean of response latency in analysis: {mean_latency:.1f} +/- {std_mean_latency:.1f}"
    )

    utterances = perform_analysis_intelligibility(utterances, args.response_latency)
