import argparse
import math
import os
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from analysis_intelligibility import response_is_clarification_request, melt_variable, \
    DEFAULT_COUNT_ONLY_INTELLIGIBLE_RESPONSES, response_is_acknowledgement
from analysis_reproduce_warlaumont import get_micro_conversations, has_response
from utils import (
    age_bin,
    str2bool,
    filter_transcripts_based_on_num_child_utts, SPEAKER_CODE_CHILD, get_num_words, UTTERANCES_WITH_SPEECH_ACTS_FILE,
)

DEFAULT_COUNT_ONLY_SPEECH_RELATED_RESPONSES = True

DEFAULT_MIN_RATIO_NONSPEECH = 0.0

DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT = 1

# Ages aligned to study of Warlaumont et al. or to our study (minimum 10 months)
DEFAULT_MIN_AGE = 10
#TODO:
DEFAULT_MAX_AGE = 60

AGE_BIN_NUM_MONTHS = 6

# Forrester: Does not annotate non-word sounds starting with & (phonological fragment), these are treated as words and
# should be excluded when annotating intelligibility based on rules.
# Providence: Some non-speech vocalizations such as laughter are incorrectly transcribed as 'yyy', and the timing
# information is of very poor quality
DEFAULT_EXCLUDED_CORPORA = ["Providence", "Forrester"]

# The caregivers of these children are using slang (e.g., "you was" or "she don't") and are therefore excluded
# We are unfortunately only studying mainstream US English
EXCLUDED_CHILDREN = ["Brent_Jaylen", "Brent_Tyrese", "Brent_Vas", "Brent_Vas_Coleman", "Brent_Xavier"]

# GRAMMATICALITY_COLUMN = "is_grammatical_cointegrated_roberta-large-cola-krishna2020"
GRAMMATICALITY_COLUMN = "is_grammatical"

RESULTS_DIR = "results/grammaticality/"


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=UTTERANCES_WITH_SPEECH_ACTS_FILE,
    )
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

    argparser.add_argument(
        "--count-only-intelligible_responses",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_COUNT_ONLY_INTELLIGIBLE_RESPONSES,
    )

    args = argparser.parse_args()

    return args


def plot_num_words_vs_grammaticality(utterances):
    utterances["num_words"] = get_num_words(utterances.transcript_clean)

    plt.figure(figsize=(15, 10))
    sns.histplot(data=utterances[utterances.speaker_code == SPEAKER_CODE_CHILD], hue="is_grammatical", x="num_words",
                 multiple="dodge")
    plt.show()


def plot_grammaticality_development(utterances):
    # TODO: testing: later we should only drop from micro-conversations!
    utterances.dropna(subset=["is_grammatical"], inplace=True)
    utterances["is_grammatical"] = utterances.is_grammatical.astype(bool)
    utterances = utterances[utterances.is_intelligible & utterances.is_speech_related]

    # TODO: filter?
    utterances = filter_transcripts_based_on_num_child_utts(
        utterances, 10
    )

    proportion_grammatical_per_transcript_chi = utterances[utterances.speaker_code==SPEAKER_CODE_CHILD].groupby(
        "transcript_file"
    ).agg({"is_grammatical": "mean", "age": "mean"})
    sns.regplot(
        data=proportion_grammatical_per_transcript_chi,
        x="age",
        y="is_grammatical",
        marker=".",
        logistic=True,
        line_kws={"color": sns.color_palette("tab10")[0]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[0]},
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "proportion_grammatical_children.png"), dpi=300)

    proportion_grammatical_per_transcript_adu = utterances[utterances.speaker_code != SPEAKER_CODE_CHILD].groupby(
        "transcript_file"
    ).agg({"is_grammatical": "mean", "age": "mean"})
    sns.regplot(
        data=proportion_grammatical_per_transcript_adu,
        x="age",
        y="is_grammatical",
        marker=".",
        logistic=True,
        line_kws={"color": sns.color_palette("tab10")[1]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[1]},
    )
    plt.tight_layout()
    plt.legend(labels=["children", "adults"])
    plt.savefig(os.path.join(RESULTS_DIR, "proportion_grammatical_children_adults.png"), dpi=300)
    plt.show()


def perform_analysis_grammaticality(utterances, args):
    # Discard non-speech, but keep uncertain (xxx, labelled as NA)
    utterances = utterances[utterances.is_speech_related != False]

    conversations = get_micro_conversations(utterances, args, use_is_grammatical=True)

    conversations.dropna(
        subset=(
            "response_latency",
            "response_latency_follow_up",
            "utt_is_grammatical",
            "follow_up_is_grammatical",
            "response_is_speech_related",
        ),
        inplace=True,
    )
    conversations = conversations[conversations.utt_is_speech_related &
                                  conversations.response_is_speech_related &
                                  conversations.follow_up_is_speech_related &
                                  conversations.utt_is_intelligible &
                                  conversations.follow_up_is_intelligible]

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
        response_is_acknowledgement=conversations.apply(
            response_is_acknowledgement, axis=1
        )
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

    conversations.to_csv(RESULTS_DIR + "conversations.csv")
    conversations = pd.read_csv(RESULTS_DIR + "conversations.csv", index_col=0)

    # Melt is_grammatical variable for CR analyses
    conversations_melted = melt_variable(conversations, "is_grammatical")
    conversations_melted.to_csv(RESULTS_DIR + "conversations_melted.csv")
    conversations_melted = pd.read_csv(RESULTS_DIR + "conversations_melted.csv", index_col=0)


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

    # perform_per_transcript_analyses(conversations)

    make_plots(conversations, conversations_melted)

    return conversations


def make_plots(conversations, conversations_melted):
    proportion_grammatical_per_transcript = conversations.groupby(
        "transcript_file"
    ).agg({"utt_is_grammatical": "mean", "age": "min"})
    plt.figure(figsize=(6, 3))
    axis = sns.regplot(
        data=proportion_grammatical_per_transcript,
        x="age",
        y="utt_is_grammatical",
        logistic=True,
        marker=".",
    )
    plt.tight_layout()
    axis.set(xlabel="age (months)", ylabel="prop_grammatical")
    axis.set_xticks(
        np.arange(
            conversations.age.min(),
            conversations.age.max() + 1,
            step=AGE_BIN_NUM_MONTHS,
        )
    )
    plt.savefig(os.path.join(RESULTS_DIR, "proportion_grammatical.png"), dpi=300)

    # Duplicate all entries and set age to infinity to get summary bars over all age groups
    conversations_duplicated = conversations.copy()
    conversations_duplicated["age"] = math.inf
    conversations_with_avg_age = conversations.append(conversations_duplicated, ignore_index=True)

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_avg_age,
        x="age",
        y="has_response",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    sns.move_legend(axis, "lower right")
    axis.set(xlabel="age (months)", ylabel="prop_has_response")
    axis.set_xticklabels(sorted(conversations_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cf_quality_timing.png"), dpi=300)

    conversations_with_response = conversations_with_avg_age[conversations_with_avg_age.has_response]
    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_response,
        x="age",
        y="response_is_clarification_request",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    sns.move_legend(axis, "upper left")
    axis.set(xlabel="age (months)", ylabel="prop_clarification_request")
    axis.set_xticklabels(sorted(conversations_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_quality_clarification_request.png"), dpi=300
    )

    conversations_with_response = conversations_with_avg_age[conversations_with_avg_age.has_response]
    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_response,
        x="age",
        y="response_is_acknowledgement",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    sns.move_legend(axis, "upper left")
    axis.set(xlabel="age (months)", ylabel="prop_acknowledgement")
    axis.set_xticklabels(sorted(conversations_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_quality_acknowledgements.png"), dpi=300
    )

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_avg_age[conversations_with_avg_age.utt_is_grammatical],
        x="age",
        y="follow_up_is_grammatical",
        hue="has_response",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("no response")
    legend.texts[1].set_text("response")
    sns.move_legend(axis, "lower right")
    axis.set(xlabel="age (months)", ylabel="prop_follow_up_is_grammatical")
    axis.set_xticklabels(sorted(conversations_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_effect_pos_feedback_on_grammatical_timing.png"),
        dpi=300,
    )

    conversations_melted_with_response = conversations_melted[
        conversations_melted.has_response
    ]
    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_melted_with_response,
        x="response_is_clarification_request",
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prop_is_grammatical")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_effect_clarification_request_control.png"),
        dpi=300,
    )

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_melted_with_response,
        x="response_is_acknowledgement",
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prop_is_grammatical")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_effect_acknowledgement_control.png"),
        dpi=300,
    )

    # Duplicate all entries and set age to infinity to get summary bars over all age groups
    conversations_melted_duplicated = conversations_melted.copy()
    conversations_melted_duplicated["age"] = math.inf
    conversations_melted_with_avg_age = conversations_melted.append(conversations_melted_duplicated, ignore_index=True)

    conversations_melted_cr_with_avg_age = conversations_melted_with_avg_age[
        conversations_melted_with_avg_age.response_is_clarification_request
    ]

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_melted_cr_with_avg_age,
        x="age",
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    sns.move_legend(axis, "upper left")
    axis.set(xlabel="age (months)", ylabel="prop_is_grammatical")
    axis.set_xticklabels(sorted(conversations_melted_cr_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_effect_clarification_request.png"), dpi=300
    )

    conversations_melted_ack_with_avg_age = conversations_melted_with_avg_age[
        conversations_melted_with_avg_age.response_is_acknowledgement
    ]

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_melted_ack_with_avg_age,
        x="age",
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    sns.move_legend(axis, "upper left")
    axis.set(xlabel="age (months)", ylabel="prop_is_grammatical")
    axis.set_xticklabels(sorted(conversations_melted_ack_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_effect_acknowledgement.png"), dpi=300
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    utterances = pd.read_csv(args.utterances_file, index_col=0)

    utterances["is_grammatical"] = utterances[GRAMMATICALITY_COLUMN]

    print("Excluding corpora: ", args.excluded_corpora)
    utterances = utterances[~utterances.corpus.isin(args.excluded_corpora)]

    if args.corpora:
        print("Including only corpora: ", args.corpora)
        utterances = utterances[utterances.corpus.isin(args.corpora)]

    print("Excluding children: ", EXCLUDED_CHILDREN)
    utterances = utterances[~utterances.child_name.isin(EXCLUDED_CHILDREN)]

    # Filter by age
    utterances = utterances[
        (args.min_age <= utterances.age) & (utterances.age <= args.max_age)
    ]

    conversations = perform_analysis_grammaticality(utterances, args)
