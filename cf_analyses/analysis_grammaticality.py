import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis_intelligibility import response_is_clarification_request, melt_variable, response_is_acknowledgement, \
    get_repetition_ratios, filter_utts_for_num_words, filter_follow_ups_for_num_words
from utils import (
    age_bin,
    SPEAKER_CODE_CHILD, get_num_words,
    MICRO_CONVERSATIONS_WITHOUT_NON_SPEECH_FILE, PROJECT_ROOT_DIR,
)

# Ages aligned to study of Warlaumont et al. or to our study (minimum 10 months)
DEFAULT_MIN_AGE = 10
DEFAULT_MAX_AGE = 60

AGE_BIN_NUM_MONTHS = 6

MIN_NUM_WORDS = 1

CORPORA_EXCLUDED = []
# TODO Bates? VanHouten?
# CORPORA_INCLUDED = ['Thomas', 'MPI-EVA-Manchester', 'Providence', 'Braunwald', 'Lara', 'EllisWeismer']
CORPORA_INCLUDED = ['Providence', 'VanHouten', 'Thomas', 'Braunwald', 'Lara', 'MPI-EVA-Manchester', 'Bates', 'EllisWeismer']


# The caregivers of these children are using slang (e.g., "you was" or "she don't") and are therefore excluded
# We are unfortunately only studying mainstream US English
EXCLUDED_CHILDREN = ["Brent_Jaylen", "Brent_Tyrese", "Brent_Vas", "Brent_Vas_Coleman", "Brent_Xavier"]

RESULTS_DIR = PROJECT_ROOT_DIR+"/results/grammaticality/"


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=MICRO_CONVERSATIONS_WITHOUT_NON_SPEECH_FILE,
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

    args = argparser.parse_args()

    return args


def plot_num_words_vs_grammaticality(utterances):
    utterances["num_words"] = get_num_words(utterances.transcript_clean)

    plt.figure(figsize=(15, 10))
    sns.histplot(data=utterances[utterances.speaker_code == SPEAKER_CODE_CHILD], hue="is_grammatical", x="num_words",
                 multiple="dodge")
    plt.show()


def perform_analysis_grammaticality(conversations, args):
    conversations.dropna(
        subset=(
            "utt_is_grammatical",
            "utt_is_intelligible",
            "response_is_speech_related",
        ),
        inplace=True,
    )
    conversations = conversations[conversations.utt_is_intelligible].copy()

    # Filtering out dummy responses (cases in which the child continues to talk
    conversations = conversations[conversations.response_is_speech_related].copy()
    conversations = filter_utts_for_num_words(conversations, min_num_words=MIN_NUM_WORDS)

    repetition_ratios = conversations.apply(get_repetition_ratios, axis=1)
    conversations["utt_repetition_ratio"] = repetition_ratios.apply(lambda ratios: ratios[0])
    conversations["resp_repetition_ratio"] = repetition_ratios.apply(lambda ratios: ratios[1])

    conversations["response_is_clarification_request"] = conversations.apply(response_is_clarification_request, axis=1)
    conversations["response_is_acknowledgement"] = conversations.apply(response_is_acknowledgement, axis=1)

    print("Number of CRs: ", len(conversations[conversations.response_is_clarification_request]))
    print("Number of Acks: ", len(conversations[conversations.response_is_acknowledgement]))

    conversations["age"] = conversations.age.apply(
        age_bin,
        min_age=args.min_age,
        max_age=args.max_age,
        num_months=AGE_BIN_NUM_MONTHS,
    )

    conversations.to_csv(RESULTS_DIR + "conversations.csv")
    conversations = pd.read_csv(RESULTS_DIR + "conversations.csv", index_col=0)

    # Melt is_grammatical variable for CR effect analyses
    conversations_good_follow_ups = conversations.dropna(
        subset=(
            "follow_up_is_grammatical",
            "follow_up_is_intelligible",
        ),
    )
    conversations_good_follow_ups = filter_follow_ups_for_num_words(conversations_good_follow_ups, min_num_words=MIN_NUM_WORDS)
    conversations_good_follow_ups = conversations_good_follow_ups[conversations_good_follow_ups.follow_up_is_intelligible]


    conversations_melted = melt_variable(conversations_good_follow_ups, "is_grammatical")
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
    # Duplicate all entries and set age to infinity to get summary bars over all age groups
    conversations_duplicated = conversations.copy()
    conversations_duplicated["age"] = math.inf
    conversations_with_avg_age = pd.concat([conversations, conversations_duplicated], ignore_index=True)

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_avg_age,
        x="age",
        y="response_is_clarification_request",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    sns.move_legend(axis, "lower left")
    axis.set(xlabel="age (months)", ylabel="prop_clarification_request")
    axis.set_xticklabels(sorted(conversations_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_quality_clarification_request.png"), dpi=300
    )

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_avg_age,
        x="age",
        y="response_is_acknowledgement",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    sns.move_legend(axis, "lower left")
    axis.set(xlabel="age (months)", ylabel="prop_acknowledgement")
    axis.set_xticklabels(sorted(conversations_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_quality_acknowledgements.png"), dpi=300
    )

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_melted,
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
        data=conversations_melted,
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
    conversations_melted_with_avg_age = pd.concat([conversations_melted, conversations_melted_duplicated], ignore_index=True)

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


def filter_corpora(conversations):
    print("Including corpora: ", CORPORA_INCLUDED)
    conversations = conversations[conversations.corpus.isin(CORPORA_INCLUDED)]

    print("Excluding corpora: ", CORPORA_EXCLUDED)
    conversations = conversations[~conversations.corpus.isin(CORPORA_EXCLUDED)]

    return conversations


if __name__ == "__main__":
    args = parse_args()
    print(args)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    conversations = pd.read_csv(args.utterances_file, index_col=0, dtype={"error": object, "utt_is_grammatical": object, "response_is_grammatical": object, "follow_up_is_grammatical": object, "labels": object})
    # Filter by age
    conversations = conversations[
        (args.min_age <= conversations.age) & (conversations.age <= args.max_age)
    ]

    print("Excluding children: ", EXCLUDED_CHILDREN)
    conversations = conversations[~conversations.child_name.isin(EXCLUDED_CHILDREN)]

    corpora = filter_corpora(conversations)

    perform_analysis_grammaticality(conversations, args)
