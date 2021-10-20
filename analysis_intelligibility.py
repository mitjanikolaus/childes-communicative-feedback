import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import filter_corpora_based_on_response_latency_length
from search_adjacent_utterances import CANDIDATE_CORPORA
from utils import (
    remove_babbling,
    EMPTY_UTTERANCE,
    clean_utterance,
    remove_nonspeech_events,
    PATH_ADJACENT_UTTERANCES,
    remove_whitespace,
)

# TODO: define age range
MIN_AGE = 10    # age of first words?
MAX_AGE = 48

# 1s response threshold
RESPONSE_THRESHOLD = 1000  # ms

# Number of standard deviations that the mean response latency of a corpus can be off the reference mean
RESPONSE_LATENCY_STANDARD_DEVIATIONS_OFF = 1

# Label for partially intelligible utterances
# Set to True to count as intelligible, False to count as unintelligible or None to exclude these utterances from
# the analysis
LABEL_PARTIALLY_INTELLIGIBLE = None

# TODO check that pause is not too long (neg): what is a reasonable value?
# 1 second
MAX_NEG_RESPONSE_LATENCY = -1 * 1000  # ms

# Forrester: Does not annotate non-word sounds starting with & (phonological fragment), these are treated as words
EXCLUDED_CORPORA = ["Forrester"]

# currently not used to exclude corpora, just stored for reference:
CORPORA_NOT_LONGITUDINAL = ["Gleason", "Rollins"]


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


def caregiver_intelligible_response(row):
    return row["response_latency"] <= RESPONSE_THRESHOLD


def caregiver_response_contingent_on_intelligibility(row):
    return (
        (row["utt_child_intelligible"] == True) & (row["caregiver_response"] == True)
    ) | (
        (row["utt_child_intelligible"] == False) & (row["caregiver_response"] == False)
    )


def perform_analysis_intelligibility(adj_utterances):
    # Clean utterances
    adj_utterances["utt_child"] = adj_utterances.utt_child.apply(clean_utterance)
    adj_utterances["utt_car"] = adj_utterances.utt_car.apply(clean_utterance)
    adj_utterances["utt_child_follow_up"] = adj_utterances.utt_child_follow_up.apply(
        clean_utterance
    )

    # Remove nonspeech events
    adj_utterances["utt_child"] = adj_utterances.utt_child.apply(
        remove_nonspeech_events
    )
    adj_utterances["utt_car"] = adj_utterances.utt_car.apply(remove_nonspeech_events)
    adj_utterances["utt_child_follow_up"] = adj_utterances.utt_child_follow_up.apply(
        remove_nonspeech_events
    )

    # Drop empty utterances (these are non-speech related)
    adj_utterances = adj_utterances[
        (
            (adj_utterances.utt_child != EMPTY_UTTERANCE)
            & (adj_utterances.utt_car != EMPTY_UTTERANCE)
            & (adj_utterances.utt_child_follow_up != EMPTY_UTTERANCE)
        )
    ]

    # Label utterances as intelligible or unintelligible
    adj_utterances = adj_utterances.assign(
        utt_child_intelligible=adj_utterances.utt_child.apply(is_intelligible)
    )
    adj_utterances = adj_utterances.assign(
        follow_up_intelligible=adj_utterances.utt_child_follow_up.apply(is_intelligible)
    )

    # Label caregiver responses as present or not
    adj_utterances = adj_utterances.assign(
        caregiver_response=adj_utterances.apply(caregiver_intelligible_response, axis=1)
    )

    # Remove NaNs
    adj_utterances = adj_utterances.dropna(
        subset=("utt_child_intelligible", "follow_up_intelligible")
    )

    # Label caregiver responses as contingent on child utterance or not
    adj_utterances = adj_utterances.assign(
        caregiver_response_contingent=adj_utterances[
            ["utt_child_intelligible", "caregiver_response"]
        ].apply(caregiver_response_contingent_on_intelligibility, axis=1)
    )

    print(
        f"\nFound {len(adj_utterances)} turns"
    )
    if len(adj_utterances) > 0:
        # Caregiver contingency:
        n_responses_intelligible = len(
            adj_utterances[
                adj_utterances.utt_child_intelligible
                & adj_utterances.caregiver_response
            ]
        )
        n_intelligible = len(
            adj_utterances[adj_utterances.utt_child_intelligible]
        )

        n_responses_unintelligible = len(
            adj_utterances[
                (adj_utterances.utt_child_intelligible == False)
                & adj_utterances.caregiver_response
            ]
        )
        n_unintelligible = len(
            adj_utterances[adj_utterances.utt_child_intelligible == False]
        )

        contingency_caregiver = (n_responses_intelligible / n_intelligible) - (
            n_responses_unintelligible / n_unintelligible
        )
        print(f"Caregiver contingency: {contingency_caregiver:.4f}")

        # Contingency of child vocalization on previous adult response (positive case):
        n_follow_up_intelligible_if_response_to_intelligible = len(
            adj_utterances[
                adj_utterances.follow_up_intelligible
                & adj_utterances.utt_child_intelligible
                & adj_utterances.caregiver_response
            ]
        )
        n_responses_to_intelligible = len(
            adj_utterances[
                adj_utterances.utt_child_intelligible
                & adj_utterances.caregiver_response
            ]
        )

        n_follow_up_intelligible_if_no_response_to_intelligible = len(
            adj_utterances[
                adj_utterances.follow_up_intelligible
                & adj_utterances.utt_child_intelligible
                & (adj_utterances.caregiver_response == False)
            ]
        )
        n_no_responses_to_intelligible = len(
            adj_utterances[
                adj_utterances.utt_child_intelligible
                & (adj_utterances.caregiver_response == False)
            ]
        )

        # Contingency of child vocalization on previous adult response (negative case):
        n_follow_up_intelligible_if_no_response_to_unintelligible = len(
            adj_utterances[
                adj_utterances.follow_up_intelligible
                & (adj_utterances.utt_child_intelligible == False)
                & (adj_utterances.caregiver_response == False)
            ]
        )
        n_no_responses_to_unintelligible = len(
            adj_utterances[
                (adj_utterances.utt_child_intelligible == False)
                & (adj_utterances.caregiver_response == False)
            ]
        )

        n_follow_up_intelligible_if_response_to_unintelligible = len(
            adj_utterances[
                adj_utterances.follow_up_intelligible
                & (adj_utterances.utt_child_intelligible == False)
                & adj_utterances.caregiver_response
            ]
        )
        n_responses_to_unintelligible = len(
            adj_utterances[
                (adj_utterances.utt_child_intelligible == False)
                & adj_utterances.caregiver_response
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
            print(
                f"Child contingency (both cases): {child_contingency_both_cases:.4f}"
            )
            child_contingency_both_cases_same_weighting = np.mean(
                [contingency_children_pos_case, contingency_children_neg_case]
            )

            print(
                f"Child contingency (both cases, same weighting of positive and negative cases): "
                f"{child_contingency_both_cases_same_weighting:.4f}"
            )

        # Statsmodels prefers 1 and 0 over True and False:
        adj_utterances.replace({False: 0, True: 1}, inplace=True)

        mod = smf.glm(
            "caregiver_response ~ utt_child_intelligible",
            family=sm.families.Binomial(),
            data=adj_utterances,
        ).fit()
        print(mod.summary())

        mod = smf.glm(
            "follow_up_intelligible ~ caregiver_response_contingent",
            family=sm.families.Binomial(),
            data=adj_utterances,
        ).fit()
        print(mod.summary())

        mod = smf.glm(
            "follow_up_intelligible ~ utt_child_intelligible * caregiver_response_contingent",
            family=sm.families.Binomial(),
            data=adj_utterances,
        ).fit()
        print(mod.summary())

        sns.barplot(
            data=adj_utterances,
            x="utt_child_intelligible",
            y="follow_up_intelligible",
            hue="caregiver_response_contingent",
        )
        plt.show()


if __name__ == "__main__":
    args = parse_args()

    adjacent_utterances = pd.read_csv(PATH_ADJACENT_UTTERANCES, index_col=None)

    # Remove adjacent_utterances with too long negative pauses
    adjacent_utterances = adjacent_utterances[
        (adjacent_utterances.response_latency >= MAX_NEG_RESPONSE_LATENCY)
    ]

    if not args.corpora:
        print(f"No corpora given, selecting based on average response time")
        args.corpora = filter_corpora_based_on_response_latency_length(
            CANDIDATE_CORPORA,
            adjacent_utterances,
            RESPONSE_LATENCY_STANDARD_DEVIATIONS_OFF,
        )

    print("Excluding corpora: ", EXCLUDED_CORPORA)
    adjacent_utterances = adjacent_utterances[
        ~adjacent_utterances.corpus.isin(EXCLUDED_CORPORA)
    ]

    print(f"Corpora included in analysis: {args.corpora}")
    # Filter by corpora
    adjacent_utterances = adjacent_utterances[
        adjacent_utterances.corpus.isin(args.corpora)
    ]

    # Filter by age
    adjacent_utterances = adjacent_utterances[
        (MIN_AGE <= adjacent_utterances.age) & (adjacent_utterances.age <= MAX_AGE)
    ]

    min_age = adjacent_utterances.age.min()
    max_age = adjacent_utterances.age.max()
    mean_age = adjacent_utterances.age.mean()
    print(
        f"Mean of child age in analysis: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )
    mean_latency = adjacent_utterances.response_latency.mean()
    std_mean_latency = adjacent_utterances.response_latency.std()
    print(
        f"Mean of response latency in analysis: {mean_latency:.1f} +/- {std_mean_latency:.1f}"
    )

    perform_analysis_intelligibility(adjacent_utterances.copy())
