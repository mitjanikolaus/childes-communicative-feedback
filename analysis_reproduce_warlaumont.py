import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import filter_corpora_based_on_response_latency_length
from search_child_utterances_and_responses import CANDIDATE_CORPORA, RESPONSE_THRESHOLD
from utils import (
    EMPTY_UTTERANCE,
    clean_utterance,
    remove_nonspeech_events,
    PATH_UTTERANCES_RESPONSES,
    CODE_UNINTELLIGIBLE,
    remove_whitespace,
)


# Number of standard deviations that the mean response latency of a corpus can be off the reference mean
RESPONSE_LATENCY_STANDARD_DEVIATIONS_OFF = 1

# Label for partially speech-related utterances
# Set to True to count as speech-related, False to count as not speech-related or None to exclude these utterances from
# the analysis
LABEL_PARTIALLY_SPEECH_RELATED = True

# TODO check that pause is not too long (neg): what is a reasonable value?
# 1 second
MAX_NEG_RESPONSE_LATENCY = -1 * 1000  # ms

MIN_RATIO_NONSPEECH = 0.000001

# Ages aligned to study of Warlaumont et al.
MIN_AGE = 8
MAX_AGE = 48


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


def caregiver_speech_related_response(row):
    return (
        row["response_latency"] <= RESPONSE_THRESHOLD
    )  # & is_speech_related(row["utt_car"], label_partially_speech_related=True)


def is_speech_related(
    utterance, label_partially_speech_related=LABEL_PARTIALLY_SPEECH_RELATED
):
    utt_without_nonspeech = remove_nonspeech_events(utterance)

    utt_without_nonspeech = remove_whitespace(utt_without_nonspeech)
    if utt_without_nonspeech == EMPTY_UTTERANCE:
        return False

    # We exclude completely unintelligible utterances (we don't know whether it's speech-related or not)
    if utterance == CODE_UNINTELLIGIBLE:
        return None

    is_partly_speech_related = len(utt_without_nonspeech) != len(utterance)
    if is_partly_speech_related:
        return label_partially_speech_related

    return True


def caregiver_response_contingent_on_speech_relatedness(row):
    return (
        (row["utt_child_speech_related"] == True) & (row["caregiver_response"] == True)
    ) | (
        (row["utt_child_speech_related"] == False)
        & (row["caregiver_response"] == False)
    )


def perform_analysis_speech_relatedness(utterances):
    # Clean utterances
    utterances["utt_child"] = utterances.utt_child.apply(clean_utterance)
    utterances["utt_car"] = utterances.utt_car.apply(clean_utterance)
    utterances["utt_child_follow_up"] = utterances.utt_child_follow_up.apply(
        clean_utterance
    )

    # Drop empty utterances
    utterances = utterances[
        (
            (utterances.utt_child != EMPTY_UTTERANCE)
            & (utterances.utt_car != EMPTY_UTTERANCE)
            & (utterances.utt_child_follow_up != EMPTY_UTTERANCE)
        )
    ]

    # Label utterances as speech or non-speech
    utterances = utterances.assign(
        utt_child_speech_related=utterances.utt_child.apply(is_speech_related)
    )
    utterances = utterances.assign(
        follow_up_speech_related=utterances.utt_child_follow_up.apply(is_speech_related)
    )

    # Label caregiver responses as present or not
    utterances = utterances.assign(
        caregiver_response=utterances.apply(caregiver_speech_related_response, axis=1)
    )

    # Remove NaNs
    utterances = utterances.dropna(
        subset=("utt_child_speech_related", "follow_up_speech_related")
    )

    # counter_non_speech = Counter(utterances[utterances.utt_child_speech_related == False].utt_child.values)
    # print("Most common non-speech related sounds: ")
    # print(counter_non_speech.most_common())

    # Filter for corpora that actually annotate non-speech-related sounds
    good_corpora = []
    print("Ratios nonspeech/speech for each corpus:")
    for corpus in utterances.corpus.unique():
        d_corpus = utterances[utterances.corpus == corpus]
        ratio = len(d_corpus[d_corpus.utt_child_speech_related == False]) / len(
            d_corpus[d_corpus.utt_child_speech_related == True]
        )
        if ratio > MIN_RATIO_NONSPEECH:
            good_corpora.append(corpus)
        print(f"{corpus}: {ratio:.5f}")
    print("Filtered corpora: ", good_corpora)

    utterances = utterances[utterances.corpus.isin(good_corpora)]

    # Get the number of children in all corpora:
    num_children = len(utterances.child_name.unique())
    print(f"Number of children in the analysis: {num_children}")

    # Label caregiver responses as contingent on child utterance or not
    utterances = utterances.assign(
        caregiver_response_contingent=utterances[
            ["utt_child_speech_related", "caregiver_response"]
        ].apply(caregiver_response_contingent_on_speech_relatedness, axis=1)
    )

    n_responses_to_speech = len(
        utterances[utterances.utt_child_speech_related & utterances.caregiver_response]
    )
    n_speech = len(utterances[utterances.utt_child_speech_related])

    n_responses_to_non_speech = len(
        utterances[
            (utterances.utt_child_speech_related == False)
            & utterances.caregiver_response
        ]
    )
    n_non_speech = len(utterances[utterances.utt_child_speech_related == False])

    contingency_caregiver = (n_responses_to_speech / n_speech) - (
        n_responses_to_non_speech / n_non_speech
    )
    print(f"Caregiver contingency: {contingency_caregiver:.4f}")

    # Contingency of child vocalization on previous adult response (positive case):
    n_follow_up_speech_related_if_response_to_speech_related = len(
        utterances[
            utterances.follow_up_speech_related
            & utterances.utt_child_speech_related
            & utterances.caregiver_response
        ]
    )
    n_responses_to_speech_related = len(
        utterances[utterances.utt_child_speech_related & utterances.caregiver_response]
    )

    n_follow_up_speech_related_if_no_response_to_speech_related = len(
        utterances[
            utterances.follow_up_speech_related
            & utterances.utt_child_speech_related
            & (utterances.caregiver_response == False)
        ]
    )
    n_no_responses_to_speech_related = len(
        utterances[
            utterances.utt_child_speech_related
            & (utterances.caregiver_response == False)
        ]
    )

    ratio_follow_up_speech_related_if_response_to_speech_related = (
        n_follow_up_speech_related_if_response_to_speech_related
        / n_responses_to_speech_related
    )
    ratio_follow_up_speech_related_if_no_response_to_speech_related = (
        n_follow_up_speech_related_if_no_response_to_speech_related
        / n_no_responses_to_speech_related
    )
    contingency_children_pos_case = (
        ratio_follow_up_speech_related_if_response_to_speech_related
        - ratio_follow_up_speech_related_if_no_response_to_speech_related
    )
    print(f"Child contingency (positive case): {contingency_children_pos_case:.4f}")

    # Statsmodels prefers 1 and 0 over True and False:
    utterances.replace({False: 0, True: 1}, inplace=True)

    mod = smf.glm(
        "caregiver_response ~ utt_child_speech_related",
        family=sm.families.Binomial(),
        data=utterances,
    ).fit()
    print(mod.summary())

    mod = smf.glm(
        "follow_up_speech_related ~ caregiver_response_contingent",
        family=sm.families.Binomial(),
        data=utterances[utterances.utt_child_speech_related == True],
    ).fit()
    print(mod.summary())


if __name__ == "__main__":
    args = parse_args()

    utterances = pd.read_csv(PATH_UTTERANCES_RESPONSES, index_col=None)

    # Fill in empty strings for dummy caregiver responses
    utterances.utt_car.fillna("", inplace=True)

    # Remove utterances with too long negative pauses
    utterances = utterances[(utterances.response_latency >= MAX_NEG_RESPONSE_LATENCY)]

    if not args.corpora:
        print(f"No corpora given, selecting based on average response time")
        args.corpora = filter_corpora_based_on_response_latency_length(
            CANDIDATE_CORPORA,
            utterances,
            MIN_AGE,
            MAX_AGE,
            RESPONSE_LATENCY_STANDARD_DEVIATIONS_OFF,
        )

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

    perform_analysis_speech_relatedness(utterances.copy())
