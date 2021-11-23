import argparse
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ztest

from utils import age_bin
from utils import (
    filter_corpora_based_on_response_latency_length,
    get_path_of_utterances_file, get_binomial_test_data,
)
from search_child_utterances_and_responses import (
    CANDIDATE_CORPORA,
    DEFAULT_RESPONSE_THRESHOLD,
)
from utils import (
    EMPTY_UTTERANCE,
    clean_utterance,
    remove_nonspeech_events,
    CODE_UNINTELLIGIBLE,
    remove_whitespace,
)

DEFAULT_RESPONSE_LATENCY_MAX_STANDARD_DEVIATIONS_OFF = 1

DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED = True

DEFAULT_COUNT_ONLY_SPEECH_RELATED_RESPONSES = True

DEFAULT_MIN_RATIO_NONSPEECH = 0.0

DEFAULT_MIN_TRANSCRIPT_LENGTH = 0

# Ages aligned to study of Warlaumont et al.
DEFAULT_MIN_AGE = 8
DEFAULT_MAX_AGE = 48

# TODO check that pause is not too long (neg): what is a reasonable value?
# 1 second
DEFAULT_MAX_NEG_RESPONSE_LATENCY = -1 * 1000  # ms

# 10 seconds
DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP = 10 * 1000  # ms


DEFAULT_EXCLUDED_CORPORA = []


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    elif v.lower() in ("none", "nan"):
        return None
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
        "--min-ratio-nonspeech",
        type=int,
        default=DEFAULT_MIN_RATIO_NONSPEECH,
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
        "--count-only-speech_related_responses",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_COUNT_ONLY_SPEECH_RELATED_RESPONSES,
    )
    argparser.add_argument(
        "--label-partially-speech-related",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
        help="Label for partially speech-related utterances: Set to True to count as speech-related, False to count as not speech-related or None to exclude these utterances from the analysis",
    )

    args = argparser.parse_args()

    return args


def caregiver_response_contingent_on_speech_relatedness(row):
    return (
        (row["utt_child_speech_related"] == True) & (row["caregiver_response"] == True)
    ) | (
        (row["utt_child_speech_related"] == False)
        & (row["caregiver_response"] == False)
    )


def perform_warlaumont_analysis(utterances, args, analysis_function, label_positive_valence):
    print(f"\nFound {len(utterances)} turns")
    print("Overall analysis: ")
    (
        contingency_caregiver,
        contingency_children_pos_case,
        contingency_children_neg_case,
        _
    ) = analysis_function(utterances)
    print(f"Caregiver contingency: {contingency_caregiver:.4f}")
    print(f"Child contingency (positive case): {contingency_children_pos_case:.4f}")
    # print(f"Child contingency (negative case): {contingency_children_neg_case:.4f}")

    print("Per-transcript analysis: ")
    results = []

    for transcript in utterances.transcript_file.unique():
        utts_transcript = utterances[utterances.transcript_file == transcript]
        if len(utts_transcript) > args.min_transcript_length:
            (
                contingency_caregiver,
                contingency_children_pos_case,
                contingency_children_neg_case,
                proportion_positive_valence,
            ) = analysis_function(utts_transcript)
            results.append({"age": utts_transcript.age.values[0], "contingency_caregiver": contingency_caregiver, "contingency_children_pos_case": contingency_children_pos_case, "contingency_children_neg_case": contingency_children_neg_case, label_positive_valence: proportion_positive_valence})
    results = pd.DataFrame(results)

    p_value = ztest(results.contingency_caregiver.dropna(), value=0.0, alternative="larger")[1]
    print(
        f"Caregiver contingency: {results.contingency_caregiver.dropna().mean():.4f} +-{results.contingency_caregiver.dropna().std():.4f} p-value:{p_value}"
    )
    p_value = ztest(results.contingency_children_pos_case.dropna(), value=0.0, alternative="larger")[1]
    print(
        f"Child contingency (positive case): {results.contingency_children_pos_case.dropna().mean():.4f} +-{results.contingency_children_pos_case.dropna().std():.4f} p-value:{p_value}"
    )
    # p_value = ztest(results.contingency_children_neg_case.dropna(), value=0.0, alternative="larger")[1]
    # print(
    #     f"Child contingency (negative case): {results.contingency_children_neg_case.dropna().mean():.4f} +-{results.contingency_children_neg_case.dropna().std():.4f} p-value:{p_value}"
    # )

    return results


def perform_contingency_analysis_speech_relatedness(utterances):
    # caregiver contingency
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

    if n_non_speech > 0 and n_speech > 0:
        contingency_caregiver = (n_responses_to_speech / n_speech) - (
            n_responses_to_non_speech / n_non_speech
        )
    else:
        contingency_caregiver = np.nan

    # Contingency of child vocalization on previous adult response (positive case):
    n_follow_up_speech_related_if_response_to_speech_related = len(
        utterances[
            utterances.follow_up_speech_related
            & utterances.utt_child_speech_related
            & utterances.caregiver_response
        ]
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

    if n_responses_to_speech > 0 and n_no_responses_to_speech_related > 0:
        ratio_follow_up_speech_related_if_response_to_speech_related = (
            n_follow_up_speech_related_if_response_to_speech_related
            / n_responses_to_speech
        )
        ratio_follow_up_speech_related_if_no_response_to_speech_related = (
            n_follow_up_speech_related_if_no_response_to_speech_related
            / n_no_responses_to_speech_related
        )
        contingency_children_pos_case = (
            ratio_follow_up_speech_related_if_response_to_speech_related
            - ratio_follow_up_speech_related_if_no_response_to_speech_related
        )
    else:
        contingency_children_pos_case = np.nan

    # Contingency of child vocalization on previous adult response (negative case):
    n_follow_up_speech_related_if_no_response_to_non_speech_related = len(
        utterances[
            utterances.follow_up_speech_related
            & (utterances.utt_child_speech_related == False)
            & (utterances.caregiver_response == False)
        ]
    )
    n_no_responses_to_non_speech_related = len(
        utterances[
            (utterances.utt_child_speech_related == False)
            & (utterances.caregiver_response == False)
        ]
    )

    n_follow_up_speech_related_if_response_to_non_speech_related = len(
        utterances[
            utterances.follow_up_speech_related
            & (utterances.utt_child_speech_related == False)
            & utterances.caregiver_response
        ]
    )
    n_responses_to_non_speech_related = len(
        utterances[
            (utterances.utt_child_speech_related == False)
            & utterances.caregiver_response
        ]
    )

    if (
        n_no_responses_to_non_speech_related > 0
        and n_responses_to_non_speech_related > 0
    ):
        ratio_follow_up_speech_related_if_no_response_to_non_speech_related = (
            n_follow_up_speech_related_if_no_response_to_non_speech_related
            / n_no_responses_to_non_speech_related
        )
        ratio_follow_up_speech_related_if_response_to_non_speech_related = (
            n_follow_up_speech_related_if_response_to_non_speech_related
            / n_responses_to_non_speech_related
        )
        contingency_children_neg_case = (
            ratio_follow_up_speech_related_if_no_response_to_non_speech_related
            - ratio_follow_up_speech_related_if_response_to_non_speech_related
        )
    else:
        contingency_children_neg_case = np.nan

    proportion_speech_related = n_speech / (n_speech + n_non_speech)

    return (
        contingency_caregiver,
        contingency_children_pos_case,
        contingency_children_neg_case,
        proportion_speech_related,
    )


def perform_analysis_speech_relatedness(utterances, args):
    # Clean utterances
    utterances["utt_child"] = utterances.utt_child.apply(clean_utterance)
    utterances["utt_car"] = utterances.utt_car.apply(clean_utterance)
    utterances["utt_child_follow_up"] = utterances.utt_child_follow_up.apply(
        clean_utterance
    )

    # Drop empty children's utterances
    utterances = utterances[
        (
            (utterances.utt_child != EMPTY_UTTERANCE)
            & (utterances.utt_child_follow_up != EMPTY_UTTERANCE)
        )
    ]

    # Label utterances as speech or non-speech
    def is_speech_related(
        utterance,
        label_partially_speech_related=args.label_partially_speech_related,
        label_unintelligible=None,
    ):
        utt_without_nonspeech = remove_nonspeech_events(utterance)

        utt_without_nonspeech = remove_whitespace(utt_without_nonspeech)
        if utt_without_nonspeech == EMPTY_UTTERANCE:
            return False

        # We exclude completely unintelligible utterances (we don't know whether it's speech-related or not)
        is_completely_unintelligible = True
        for word in utt_without_nonspeech.split(" "):
            if word != CODE_UNINTELLIGIBLE and word != EMPTY_UTTERANCE:
                is_completely_unintelligible = False
                break
        if is_completely_unintelligible:
            return label_unintelligible

        is_partly_speech_related = len(utt_without_nonspeech) != len(utterance)
        if is_partly_speech_related:
            return label_partially_speech_related

        return True

    utterances = utterances.assign(
        utt_child_speech_related=utterances.utt_child.apply(is_speech_related)
    )
    utterances = utterances.assign(
        follow_up_speech_related=utterances.utt_child_follow_up.apply(is_speech_related)
    )

    # Label caregiver responses as present or not
    def caregiver_speech_related_response(row):
        return (row["response_latency"] <= args.response_latency) & (
            (not args.count_only_speech_related_responses)
            | is_speech_related(
                row["utt_car"],
                label_partially_speech_related=True,
                label_unintelligible=True,
            )
        )

    utterances = utterances.assign(
        caregiver_response=utterances.apply(caregiver_speech_related_response, axis=1)
    )

    # Remove NaNs
    utterances = utterances.dropna(
        subset=("utt_child_speech_related", "follow_up_speech_related")
    )

    counter_non_speech = Counter(utterances[utterances.utt_child_speech_related == False].utt_child.values)
    print("Most common non-speech related sounds: ")
    print(counter_non_speech.most_common())

    # Filter for corpora that actually annotate non-speech-related sounds
    good_corpora = []
    print("Ratios nonspeech/speech for each corpus:")
    for corpus in utterances.corpus.unique():
        d_corpus = utterances[utterances.corpus == corpus]
        ratio = len(d_corpus[d_corpus.utt_child_speech_related == False]) / len(
            d_corpus[d_corpus.utt_child_speech_related == True]
        )
        if ratio > args.min_ratio_nonspeech:
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

    results_analysis = perform_warlaumont_analysis(
        utterances, args, perform_contingency_analysis_speech_relatedness, "proportion_speech_related"
    )
    results_dir = "results/reproduce_warlaumont/"
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    sns.scatterplot(data=results_analysis, x="age", y="contingency_caregiver")
    plt.savefig(os.path.join(results_dir, "dev_contingency_caregivers.png"))

    plt.figure()
    sns.scatterplot(data=results_analysis, x="age", y="contingency_children_pos_case")
    plt.savefig(os.path.join(results_dir, "dev_contingency_children.png"))

    # plt.figure()
    # sns.scatterplot(data=results_analysis, x="age", y="contingency_children_neg_case")

    plt.figure()
    sns.regplot(data=results_analysis, x="age", y="proportion_speech_related", marker=".", ci=None)
    plt.savefig(os.path.join(results_dir, "dev_proportion_speech_related.png"))

    plt.figure()
    plt.title("Caregiver contingency")
    sns.barplot(
        data=utterances,
        x="utt_child_speech_related",
        y="caregiver_response",
    )
    plt.savefig(os.path.join(results_dir, "contingency_caregivers.png"))

    utterances["age"] = utterances.age.map(age_bin)

    plt.figure()
    plt.title("Caregiver contingency - per age group")
    sns.barplot(
        data=utterances,
        x="age",
        y="caregiver_response",
        hue="utt_child_speech_related"
    )
    plt.savefig(os.path.join(results_dir, "contingency_caregivers_per_age.png"))

    plt.figure()
    plt.title("Child contingency")
    sns.barplot(
        data=utterances[utterances.utt_child_speech_related == True],
        x="caregiver_response",
        y="follow_up_speech_related",
    )
    plt.savefig(os.path.join(results_dir, "contingency_children.png"))

    plt.figure()
    plt.title("Child contingency - per age group")
    sns.barplot(
        data=utterances[utterances.utt_child_speech_related == True],
        x="age",
        y="follow_up_speech_related",
        hue="caregiver_response"
    )
    plt.savefig(os.path.join(results_dir, "contingency_children_per_age.png"))

    plt.show()

    return utterances


def perform_analyses(args, analysis_function):
    utterances = pd.read_csv(
        get_path_of_utterances_file(args.response_latency), index_col=None
    )

    # Fill in empty strings for dummy caregiver responses
    utterances.utt_car.fillna("", inplace=True)

    # Remove utterances with too long negative pauses
    utterances = utterances[
        (utterances.response_latency >= args.max_neg_response_latency)
    ]

    # Remove utterances with follow up too far in the future
    utterances = utterances[
        (utterances.response_latency_follow_up <= args.max_response_latency_follow_up)
    ]

    if not args.corpora:
        print(f"No corpora given, selecting based on average response latency")
        args.corpora = filter_corpora_based_on_response_latency_length(
            CANDIDATE_CORPORA,
            utterances,
            args.min_age,
            args.max_age,
            args.response_latency_max_standard_deviations_off,
        )

    print(args)

    print("Excluding corpora: ", args.excluded_corpora)
    utterances = utterances[~utterances.corpus.isin(args.excluded_corpora)]

    print(f"Corpora included in analysis: {args.corpora}")

    # Filter by corpora
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
    # Compute mean latency over corpora (to compare to meta-analysis)
    mean_latency = utterances[utterances.response_latency < math.inf].groupby("corpus")["response_latency"].mean().mean()
    std_mean_latency = utterances[utterances.response_latency < math.inf].groupby("corpus")["response_latency"].mean().std()
    print(
        f"Mean of response latency in analysis: {mean_latency:.1f} +/- {std_mean_latency:.1f}"
    )

    return analysis_function(utterances, args)


if __name__ == "__main__":
    args = parse_args()

    utterances = perform_analyses(args, perform_analysis_speech_relatedness)
