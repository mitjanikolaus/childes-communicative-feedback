import argparse
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ztest

from analysis_reproduce_warlaumont import (
    str2bool,
    has_response,
)
from extract_micro_conversations import DEFAULT_RESPONSE_THRESHOLD
from utils import (
    age_bin,
    filter_transcripts_based_on_num_child_utts, split_into_words,
    MICRO_CONVERSATIONS_WITHOUT_NON_SPEECH_FILE,
)

DEFAULT_MIN_AGE = 10  # age of first words
DEFAULT_MAX_AGE = 48

AGE_BIN_NUM_MONTHS = 6

DEFAULT_COUNT_ONLY_INTELLIGIBLE_RESPONSES = True

DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT = 1

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

RESULTS_DIR = "results/intelligibility/"


def parse_args():
    argparser = argparse.ArgumentParser()
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
        "--count-only-intelligible_responses",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_COUNT_ONLY_INTELLIGIBLE_RESPONSES,
    )

    args = argparser.parse_args()

    return args


CAREGIVER_NAMES = [
    "dad",
    "daddy",
    "dada",
    "mom",
    "mum",
    "mommy",
    "mummy",
    "mama",
    "mamma",
]


# TODO: add "okay let's do it"? | take care: might be question!
RESPONSES_ACKNOWLEDGEMENT_IF_ALONE = {"right", "sure", "okay", "alright", "all right", "yep", "yeah"}

RESPONSES_ACKNOWLEDGEMENT_CERTAIN = {"uhhuh", "uhuh", "uhhum", "mhm", "mm", "huh", "ummhm"}


def response_is_acknowledgement(micro_conv):
    response = micro_conv["response_transcript_clean"].lower()
    words = [word.lower() for word in split_into_words(response, split_on_apostrophe=True, remove_commas=True, remove_trailing_punctuation=True)]
    if len(set(words) & (RESPONSES_ACKNOWLEDGEMENT_CERTAIN | RESPONSES_ACKNOWLEDGEMENT_IF_ALONE)) == len(set(words)):
        # Consider sentences ending with full stop, but not exclamation marks or question marks, as they are changing
        # the function of the word (i.e. "okay?" or "huh?" are not acknowledgements)
        if len(response) > 0 and response[-1] == ".":
            return True
        else:
            return False
    elif words[0] in RESPONSES_ACKNOWLEDGEMENT_CERTAIN:
        return True

    return False


def response_is_clarification_request(micro_conv):
    if micro_conv["has_response"]:
        if micro_conv["response_speech_act"] in SPEECH_ACTS_CLARIFICATION_REQUEST:
            utt = micro_conv["utt_transcript_clean"]
            unique_words = set(split_into_words(utt, split_on_apostrophe=True, remove_commas=True, remove_trailing_punctuation=True))
            if len(unique_words) == 1 and unique_words.pop().lower() in CAREGIVER_NAMES:
                # If the initial utterance is just a call for attention, the response is not a clarification request.
                return False
            else:
                return True
    return False


# List of stopwords to be ignored for repetition calculation
STOPWORDS = {'s', 'the', 'a', 'to', 't', 'and', 'no', 'yeah', 'oh', 'is', 'in', 'on', 'yes', 'not',  'of', 'okay', 'right', 'with', 'for', 'up', 'some', 'just', 'at', 'because', 'so', 'but', 'out', 'if', 'mhm',  'off', 'about', 'too', 'over', 'ah', 'again', 'or', 'mm', 'as', 'huh', 'from', 'else', 'an', 'alright', 'ooh'}


def get_repetition_ratios(micro_conv):
    utt = micro_conv["utt_transcript_clean"].lower()
    words_utt = split_into_words(utt, split_on_apostrophe=True, remove_commas=True, remove_trailing_punctuation=True)
    words_utt = {word for word in words_utt if word not in STOPWORDS}

    response = micro_conv["response_transcript_clean"].lower()
    words_response = split_into_words(response, split_on_apostrophe=True, remove_commas=True, remove_trailing_punctuation=True)
    words_response = {word for word in words_response if word not in STOPWORDS}

    overlap = words_utt & words_response

    if len(words_utt) == 0:
        utt_rep_ratio = 0
    else:
        utt_rep_ratio = len(overlap) / len(words_utt)

    if len(words_response) == 0:
        resp_rep_ratio = 0
    else:
        resp_rep_ratio = len(overlap) / len(words_response)

    return [utt_rep_ratio, resp_rep_ratio]


def melt_variable(conversations, variable_suffix):
    value_var_names = ["utt_"+variable_suffix, "follow_up_"+variable_suffix]
    conversations_melted = conversations.copy()
    conversations_melted = pd.melt(
        conversations_melted.reset_index(),
        id_vars=[
            "index",
            "response_is_clarification_request",
            "response_is_acknowledgement",
            "child_name",
            "age",
            "transcript_file",
            "has_response",
        ],
        value_vars=value_var_names,
        var_name="is_follow_up",
        value_name=variable_suffix,
    )
    conversations_melted["is_follow_up"] = conversations_melted["is_follow_up"].apply(
        lambda x: x == value_var_names[1]
    )
    conversations_melted["conversation_id"] = conversations_melted["index"]
    del conversations_melted["index"]
    return conversations_melted


def perform_analysis(conversations, args):
    conversations.dropna(
        subset=("response_latency", "response_latency_follow_up"),
        inplace=True,
    )

    conversations = conversations.assign(
        has_response=conversations.apply(
            has_response,
            axis=1,
            response_latency=DEFAULT_RESPONSE_THRESHOLD,
            count_only_intelligible_responses=args.count_only_intelligible_responses,
        )
    )
    conversations.dropna(
        subset=("has_response",),
        inplace=True,
    )

    conversations.response_transcript_clean = conversations.response_transcript_clean.astype(str)

    repetition_ratios = conversations.apply(get_repetition_ratios, axis=1)
    conversations["utt_repetition_ratio"] = repetition_ratios.apply(lambda ratios: ratios[0])
    conversations["resp_repetition_ratio"] = repetition_ratios.apply(lambda ratios: ratios[1])

    conversations["response_is_clarification_request"] = conversations.apply(response_is_clarification_request, axis=1)
    conversations["response_is_acknowledgement"] = conversations.apply(response_is_acknowledgement, axis=1)

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

    # Melt is_intellgible variable for CR analyses
    conversations_melted = melt_variable(conversations, variable_suffix="is_intelligible")
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

    counter_cr = Counter(
        conversations[
            conversations.response_is_clarification_request
        ].response_transcript_raw.values
    )
    print("Most common clarification requests: ")
    print(counter_cr.most_common(20))

    counter_cr = Counter(
        conversations[
            conversations.response_is_acknowledgement
        ].response_transcript_raw.values
    )
    print("Most common acknowledgements: ")
    print(counter_cr.most_common(20))

    counter_unint = Counter(
        conversations[
            conversations.utt_is_intelligible == False
        ].utt_transcript_raw.values
    )
    print("Most common unintelligible utterances: ")
    print(counter_unint.most_common(20))

    perform_per_transcript_analyses(conversations)

    make_plots(conversations, conversations_melted)

    return conversations


def print_most_common_words(utterances):
    utterances["words"] = utterances.transcript_clean.apply(split_into_words, split_on_apostrophe=True, remove_commas=True, remove_trailing_punctuation=True)
    exploded = utterances[["words"]].explode("words")
    exploded["words"] = exploded["words"].str.lower()
    print(exploded["words"].value_counts()[:200].to_string())
    print(exploded["words"].value_counts()[:200].index.to_list())


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


def make_plots(conversations, conversations_melted):
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
    axis.set(xlabel="age (months)", ylabel="prop_intelligible")
    axis.set_xticks(
        np.arange(
            conversations.age.min(),
            conversations.age.max() + 1,
            step=AGE_BIN_NUM_MONTHS,
        )
    )
    plt.savefig(os.path.join(RESULTS_DIR, "proportion_intelligible.png"), dpi=300)

    # Duplicate all entries and set age to infinity to get summary bars over all age groups
    conversations_duplicated = conversations.copy()
    conversations_duplicated["age"] = math.inf
    conversations_with_avg_age = pd.concat([conversations, conversations_duplicated], ignore_index=True)

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_avg_age,
        x="age",
        y="has_response",
        hue="utt_is_intelligible",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("unintelligible")
    legend.texts[1].set_text("intelligible")
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
        hue="utt_is_intelligible",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("unintelligible")
    legend.texts[1].set_text("intelligible")
    sns.move_legend(axis, "upper left")
    axis.set(xlabel="age (months)", ylabel="prop_clarification_request")
    axis.set_xticklabels(sorted(conversations_with_response.age.unique()[:-1].astype(int)) + ["all"])
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
        hue="utt_is_intelligible",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("unintelligible")
    legend.texts[1].set_text("intelligible")
    sns.move_legend(axis, "upper left")
    axis.set(xlabel="age (months)", ylabel="prop_acknowledgement")
    axis.set_xticklabels(sorted(conversations_with_response.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_quality_acknowledgements.png"), dpi=300
    )

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_with_avg_age[conversations_with_avg_age.utt_is_intelligible],
        x="age",
        y="follow_up_is_intelligible",
        hue="has_response",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("no response")
    legend.texts[1].set_text("response")
    sns.move_legend(axis, "lower right")
    axis.set(xlabel="age (months)", ylabel="prop_follow_up_is_intelligible")
    axis.set_xticklabels(sorted(conversations_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_effect_pos_feedback_on_intelligible_timing.png"),
        dpi=300,
    )

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
    axis.set(ylabel="prop_is_intelligible")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_effect_clarification_request_control.png"),
        dpi=300,
    )

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_melted_with_response,
        x="response_is_acknowledgement",
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
    axis.set(ylabel="prop_is_intelligible")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_effect_acknowledgement_control.png"),
        dpi=300,
    )

    # Duplicate all entries and set age to infinity to get summary bars over all age groups
    conversations_melted_duplicated = conversations_melted.copy()
    conversations_melted_duplicated["age"] = math.inf
    conversations_melted_with_avg_age = pd.concat([conversations_melted,conversations_melted_duplicated], ignore_index=True)

    conversations_melted_cr_with_avg_age = conversations_melted_with_avg_age[
        conversations_melted_with_avg_age.response_is_clarification_request
    ]

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_melted_cr_with_avg_age,
        x="age",
        y="is_intelligible",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    sns.move_legend(axis, "upper left")
    axis.set(xlabel="age (months)", ylabel="prop_is_intelligible")
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
        y="is_intelligible",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    sns.move_legend(axis, "upper left")
    axis.set(xlabel="age (months)", ylabel="prop_is_intelligible")
    axis.set_xticklabels(sorted(conversations_melted_ack_with_avg_age.age.unique()[:-1].astype(int)) + ["all"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "cf_effect_acknowledgement.png"), dpi=300
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    conversations = pd.read_csv(MICRO_CONVERSATIONS_WITHOUT_NON_SPEECH_FILE, index_col=0)

    print("Excluding corpora: ", args.excluded_corpora)
    conversations = conversations[~conversations.corpus.isin(args.excluded_corpora)]

    if args.corpora:
        print("Including only corpora: ", args.corpora)
        conversations = conversations[conversations.corpus.isin(args.corpora)]

    # Filter by age
    conversations = conversations[
        (args.min_age <= conversations.age) & (conversations.age <= args.max_age)
    ]

    perform_analysis(conversations, args)
