import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from analysis_intelligibility import melt_variable, filter_utts_for_num_words, filter_follow_ups_for_num_words
from cr_ack_annotations import annotate_crs_and_acks
from grammaticality_data_preprocessing.analyze_childes_error_data import PALETTE_CATEGORICAL, HUE_ORDER, explode_labels
from utils import (
    age_bin,
    SPEAKER_CODE_CHILD, get_num_words,
    MICRO_CONVERSATIONS_WITHOUT_NON_SPEECH_FILE, PROJECT_ROOT_DIR,
)

DEFAULT_MIN_AGE = 10
DEFAULT_MAX_AGE = 60

AGE_BIN_NUM_MONTHS = 6

MIN_NUM_WORDS = 1

CORPORA_EXCLUDED = []

CORPORA_INCLUDED = ['Providence', 'Lara', 'EllisWeismer']
# CORPORA_INCLUDED = ['Thomas', 'MPI-EVA-Manchester', 'Providence', 'Braunwald', 'Lara', 'EllisWeismer', 'Bates']

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
    conversations["utt_is_grammatical"] = conversations.utt_is_grammatical.astype(bool)
    conversations["follow_up_is_grammatical"] = conversations.follow_up_is_grammatical.astype(bool)

    conversations = conversations[conversations.utt_is_intelligible].copy()

    # Filtering out dummy responses (cases in which the child continues to talk
    conversations = conversations[conversations.response_is_speech_related].copy()
    conversations = filter_utts_for_num_words(conversations, min_num_words=MIN_NUM_WORDS)

    annotate_crs_and_acks(conversations)

    num_utts_grammatical = len(conversations[conversations.utt_is_grammatical])
    num_utts_ungrammatical = len(conversations[~conversations.utt_is_grammatical])
    percentage_ungrammatical = 100 * num_utts_ungrammatical / len(conversations)
    print(f"Number of grammatical utterances: {num_utts_grammatical}")
    print(f"Number of ungrammatical utterances: {num_utts_ungrammatical} ({percentage_ungrammatical:.1f}%)")

    num_follow_ups_grammatical = len(conversations[conversations.follow_up_is_grammatical])
    num_follow_ups_ungrammatical = len(conversations[~conversations.follow_up_is_grammatical])
    percentage_ungrammatical = 100 * num_follow_ups_ungrammatical / len(conversations)
    print(f"Number of grammatical follow-ups: {num_follow_ups_grammatical}")
    print(f"Number of ungrammatical follow-ups: {num_follow_ups_ungrammatical} ({percentage_ungrammatical:.1f}%)")

    print("Number of CRs: ", len(conversations[conversations.response_is_clarification_request]))
    print("Number of speech act CRs: ", len(conversations[conversations.response_is_clarification_request_speech_act]))
    print("Number of repetition CRs: ", len(conversations[conversations.response_is_repetition_clarification_request]))

    print("Number of Acks: ", len(conversations[conversations.response_is_acknowledgement]))
    print("Number of keyword Acks: ", len(conversations[conversations.response_is_keyword_acknowledgement]))
    print("Number of repetition Acks: ", len(conversations[conversations.response_is_repetition_acknowledgement]))

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
    make_plots(conversations, conversations_melted, RESULTS_DIR)
    make_plots_error_types(conversations)


def make_plots_error_types(conversations):
    results_dir_error_types = PROJECT_ROOT_DIR + "/results/grammaticality/error_types/"
    os.makedirs(results_dir_error_types, exist_ok=True)

    convs = conversations[~conversations.utt_is_grammatical]
    convs = explode_labels(convs.copy())

    err_counts = convs["label"].value_counts(normalize=True).rename("Baseline").reset_index()

    err_counts_cr = convs[convs.response_is_clarification_request]["label"].value_counts(normalize=True).rename("CR").reset_index()
    merged = err_counts.merge(err_counts_cr)

    merged = merged.melt(id_vars="index", value_name="proportion", var_name="condition")
    fig, ax = plt.subplots(figsize=(7, 3))
    colors_with_little_alpha = [(p, q, r, .9) for p, q, r in PALETTE_CATEGORICAL]
    data_bs = merged[merged.condition == 'Baseline'].set_index("index").reindex(HUE_ORDER).reset_index()
    ax.bar(x='index', height='proportion', data=data_bs, width=-0.4, align='edge',
           color=colors_with_little_alpha, linewidth=1, edgecolor=".1")
    colors_with_alpha = [(p, q, r, .5) for p, q, r in PALETTE_CATEGORICAL]
    data_cr = merged[merged.condition == 'CR'].set_index("index").reindex(HUE_ORDER).reset_index()
    ax.bar(x='index', height='proportion', data=data_cr, width=0.4, align='edge',
           color=colors_with_alpha, linewidth=1, edgecolor=".1", hatch="///")
    plt.xticks("")
    plt.xlabel("Error type")
    plt.ylabel("Proportion")
    patches = [Patch(facecolor=c, label=l, edgecolor="black") for c, l in zip(PALETTE_CATEGORICAL, HUE_ORDER)]
    first_legend = ax.legend(handles=patches, ncol=2, fontsize=9)
    ax.add_artist(first_legend)
    patches_2 = [Patch(facecolor=PALETTE_CATEGORICAL[0], label="Baseline", edgecolor="black"),
                 Patch(facecolor=colors_with_alpha[0], label="After Clarification Request", hatch="///",
                       edgecolor="black")]
    ax.legend(handles=patches_2, bbox_to_anchor=(0.628, 0.47), fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir_error_types, "error_types.png"), dpi=300
    )

    err_counts_cr_sp = convs[convs.response_is_clarification_request_speech_act]["label"].value_counts(normalize=True).rename("CR_SP").reset_index()
    merged = err_counts.merge(err_counts_cr_sp, how="left")

    err_counts_cr_rep = convs[convs.response_is_repetition_clarification_request]["label"].value_counts(normalize=True).rename("CR_REP").reset_index()
    merged = merged.merge(err_counts_cr_rep, how="left")
    merged = merged.melt(id_vars="index", value_name="proportion", var_name="condition")
    plt.figure(figsize=(6, 3))
    sns.barplot(data=merged, x="index", y="proportion", hue="condition")
    plt.xticks(rotation=75)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir_error_types, "error_types_cr_type.png"), dpi=300
    )


def make_plots(conversations, conversations_melted, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nFound {len(conversations)} micro-conversations")
    print(f"Number of corpora in the analysis: {len(conversations.corpus.unique())}")
    print(
        f"Number of children in the analysis: {len(conversations.child_name.unique())}"
    )
    print(
        f"Number of transcripts in the analysis: {len(conversations.transcript_file.unique())}"
    )

    # Duplicate all entries and set age to infinity to get summary bars over all age groups
    conversations_duplicated = conversations.copy()
    conversations_duplicated["age"] = math.inf

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(4, 1), sharey="all")
    axis = sns.barplot(
        data=conversations,
        ax=axes[0],
        x="age",
        y="response_is_clarification_request",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    axis2 = sns.barplot(
        data=conversations,
        ax=axes[1],
        x=[''] * len(conversations),
        y="response_is_clarification_request",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="all data")
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="age (months)", ylabel="prop_clarification_request")
    # plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(
        os.path.join(results_dir, "cf_quality_clarification_request.png"), dpi=300
    )

    conversations_exploded_labels = explode_labels(conversations.copy())
    cr_ratio_grammatical = conversations[conversations.utt_is_grammatical].response_is_clarification_request.mean()
    conversations_exploded_ungrammatical = conversations_exploded_labels[~conversations.utt_is_grammatical]
    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_exploded_ungrammatical,
        x="label",
        y="response_is_clarification_request",
        order=HUE_ORDER,
        palette=PALETTE_CATEGORICAL,
        linewidth=1,
        edgecolor="w",
    )
    plt.axhline(y=cr_ratio_grammatical, color="black", linestyle="--")
    axis.set(xlabel="", ylabel="prop_clarification_request")
    xticklabels = [l.get_text().replace("_", "\n") for l in axis.get_xticklabels()]
    axis.set_xticklabels(xticklabels)
    plt.xticks(rotation=75, size=6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "cf_quality_clarification_request_by_error_type.png"), dpi=300
    )

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(1, 1), sharey="none")
    convs_ungrammatical_cr_rep = conversations[~conversations.response_is_clarification_request_speech_act]
    axis = sns.barplot(
        data=convs_ungrammatical_cr_rep,
        ax=axes[0],
        x=[''] * len(convs_ungrammatical_cr_rep),
        y="response_is_clarification_request",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    convs_ungrammatical_cr_sp = conversations[~conversations.response_is_repetition_clarification_request]
    axis2 = sns.barplot(
        data=convs_ungrammatical_cr_sp,
        ax=axes[1],
        x=[''] * len(convs_ungrammatical_cr_sp),
        y="response_is_clarification_request",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="")
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    axis.title.set_text('CR Repetition')
    axis2.title.set_text('CR Speech Act')
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="", ylabel="prop_response_is_clarification_request")
    # plt.ylim((0, 0.35))
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1)
    plt.savefig(
        os.path.join(results_dir, "cf_quality_clarification_request_by_cr_type.png"), dpi=300
    )

    conversations_child_names_fixed = conversations.copy()
    conversations_child_names_fixed.loc[conversations_child_names_fixed.child_name.str.startswith("EllisWeismer"), "child_name"] = "EllisWeismer"

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_child_names_fixed,
        x="child_name",
        y="response_is_clarification_request",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="child", ylabel="prop_clarification_request")
    xticklabels = [l.get_text().replace("_","\n") for l in axis.get_xticklabels()]
    axis.set_xticklabels(xticklabels)
    plt.xticks(rotation=75, size=6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "cf_quality_clarification_request_by_child.png"), dpi=300
    )

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(4, 1), sharey="all")
    axis = sns.barplot(
        data=conversations,
        ax=axes[0],
        x="age",
        y="response_is_acknowledgement",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    axis2 = sns.barplot(
        data=conversations,
        ax=axes[1],
        x=[''] * len(conversations),
        y="response_is_acknowledgement",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="all data")
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="age (months)", ylabel="prop_acknowledgement")
    # plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(
        os.path.join(results_dir, "cf_quality_acknowledgements.png"), dpi=300
    )

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_exploded_labels,
        x="label",
        y="response_is_acknowledgement",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="", ylabel="prop_acknowledgement")
    xticklabels = [l.get_text().replace("_", "\n") for l in axis.get_xticklabels()]
    axis.set_xticklabels(xticklabels)
    plt.xticks(rotation=75, size=6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "cf_quality_acknowledgement_by_error_type.png"), dpi=300
    )

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_child_names_fixed,
        x="child_name",
        y="response_is_acknowledgement",
        hue="utt_is_grammatical",
        linewidth=1,
        edgecolor="w",
    )
    legend = axis.legend()
    legend.texts[0].set_text("ungrammatical")
    legend.texts[1].set_text("grammatical")
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="child", ylabel="prop_acknowledgement")
    xticklabels = [l.get_text().replace("_","\n") for l in axis.get_xticklabels()]
    axis.set_xticklabels(xticklabels)
    plt.xticks(rotation=75, size=6)
    # plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "cf_quality_acknowledgements_by_child.png"), dpi=300
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
    # sns.move_legend(axis, "lower right")
    axis.set(ylabel="prop_is_grammatical")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "cf_effect_clarification_request_control.png"),
        dpi=300,
    )

    conversations_ungrammatical = conversations[~conversations.utt_is_grammatical]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(4, 1), sharey="all")
    axis = sns.barplot(
        data=conversations_ungrammatical,
        ax=axes[0],
        x="age",
        y="follow_up_is_grammatical",
        hue="response_is_clarification_request",
        linewidth=1,
        edgecolor="w",
    )
    axis2 = sns.barplot(
        data=conversations_ungrammatical,
        ax=axes[1],
        x=[''] * len(conversations_ungrammatical),
        y="follow_up_is_grammatical",
        hue="response_is_clarification_request",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="all data")
    legend = axis.legend()
    legend.texts[0].set_text("other response")
    legend.texts[1].set_text("clarification request")
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="age (months)", ylabel="prop_follow_up_is_grammatical")
    # plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(
        os.path.join(results_dir, "cf_effect_clarification_request_2.png"), dpi=300
    )

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(1, 1), sharey="all")
    convs_ungrammatical_cr_rep = conversations_ungrammatical[~conversations_ungrammatical.response_is_clarification_request_speech_act]
    axis = sns.barplot(
        data=convs_ungrammatical_cr_rep,
        ax=axes[0],
        x=[''] * len(convs_ungrammatical_cr_rep),
        y="follow_up_is_grammatical",
        hue="response_is_clarification_request",
        linewidth=1,
        edgecolor="w",
    )
    convs_ungrammatical_cr_sp = conversations_ungrammatical[~conversations_ungrammatical.response_is_repetition_clarification_request]
    axis2 = sns.barplot(
        data=convs_ungrammatical_cr_sp,
        ax=axes[1],
        x=[''] * len(convs_ungrammatical_cr_sp),
        y="follow_up_is_grammatical",
        hue="response_is_clarification_request",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="")
    legend = axis.legend()
    legend.texts[0].set_text("other response")
    legend.texts[1].set_text("clarification request")
    axis.title.set_text('CR Repetition')
    axis2.title.set_text('CR Speech Act')
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="", ylabel="prop_follow_up_is_grammatical")
    # plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(
        os.path.join(results_dir, "cf_effect_clarification_request_2_by_cr_type.png"), dpi=300
    )


    conversations_ungrammatical_by_error_type = explode_labels(conversations_ungrammatical.copy())

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_ungrammatical_by_error_type,
        x="label",
        y="follow_up_is_grammatical",
        hue="response_is_clarification_request",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("other response")
    legend.texts[1].set_text("clarification request")
    # sns.move_legend(axis, "lower right")
    axis.set(ylabel="prop_follow_up_is_grammatical")
    plt.xticks(rotation=75, size=6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "cf_effect_clarification_request_2_by_error_type.png"),
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
    # sns.move_legend(axis, "lower right")
    axis.set(ylabel="prop_is_grammatical")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "cf_effect_acknowledgement_control.png"),
        dpi=300,
    )

    # Duplicate all entries and set age to infinity to get summary bars over all age groups
    conversations_melted_duplicated = conversations_melted.copy()
    conversations_melted_duplicated["age"] = math.inf

    conversations_melted_cr = conversations_melted[
        conversations_melted.response_is_clarification_request
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(4, 1), sharey="all")
    axis = sns.barplot(
        data=conversations_melted_cr,
        ax=axes[0],
        x="age",
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
    )
    axis2 = sns.barplot(
        data=conversations_melted_cr,
        ax=axes[1],
        x=[''] * len(conversations_melted_cr),
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="all data")
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="age (months)", ylabel="prop_is_grammatical")
    # plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(
        os.path.join(results_dir, "cf_effect_clarification_request.png"), dpi=300
    )

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(1, 1), sharey="all")
    convs_ungrammatical_cr_rep = conversations_melted_cr[~conversations_melted_cr.response_is_clarification_request_speech_act]
    axis = sns.barplot(
        data=convs_ungrammatical_cr_rep,
        ax=axes[0],
        x=[''] * len(convs_ungrammatical_cr_rep),
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
    )
    convs_ungrammatical_cr_sp = conversations_melted_cr[~conversations_melted_cr.response_is_repetition_clarification_request]
    axis2 = sns.barplot(
        data=convs_ungrammatical_cr_sp,
        ax=axes[1],
        x=[''] * len(convs_ungrammatical_cr_sp),
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="")
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    axis.title.set_text('CR Repetition')
    axis2.title.set_text('CR Speech Act')
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="", ylabel="prop_follow_up_is_grammatical")
    # plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(
        os.path.join(results_dir, "cf_effect_clarification_request_by_cr_type.png"), dpi=300
    )

    conversations_melted_cr_child_names_fixed = conversations_melted_cr.copy()
    conversations_melted_cr_child_names_fixed.loc[conversations_melted_cr_child_names_fixed.child_name.str.startswith("EllisWeismer"), "child_name"] = "EllisWeismer"

    plt.figure(figsize=(6, 3))
    axis = sns.barplot(
        data=conversations_melted_cr_child_names_fixed,
        x="child_name",
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
        palette=sns.color_palette(),
    )
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    # sns.move_legend(axis, "upper left")
    axis.set(xlabel="child", ylabel="prop_is_grammatical")
    xticklabels = [l.get_text().replace("_","\n") for l in axis.get_xticklabels()]
    axis.set_xticklabels(xticklabels)
    plt.xticks(rotation=75, size=6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "cf_effect_clarification_request_by_child.png"), dpi=300
    )

    conversations_melted_ack = conversations_melted[
        conversations_melted.response_is_acknowledgement
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(4, 1), sharey="all")
    axis = sns.barplot(
        data=conversations_melted_ack,
        ax=axes[0],
        x="age",
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
    )
    axis2 = sns.barplot(
        data=conversations_melted_cr,
        ax=axes[1],
        x=[''] * len(conversations_melted_cr),
        y="is_grammatical",
        hue="is_follow_up",
        linewidth=1,
        edgecolor="w",
    )
    axis2.legend_.remove()
    axis2.set(ylabel="", xlabel="all data")
    legend = axis.legend()
    legend.texts[0].set_text("utterance")
    legend.texts[1].set_text("follow-up")
    # sns.move_legend(axis, "lower left")
    axis.set(xlabel="age (months)", ylabel="prop_is_grammatical")
    # plt.ylim((0, 0.35))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(
        os.path.join(results_dir, "cf_effect_acknowledgement.png"), dpi=300
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

    conversations = pd.read_csv(args.utterances_file, index_col=0, dtype={"error": object, "labels": object})
    # Filter by age
    conversations = conversations[
        (args.min_age <= conversations.age) & (conversations.age <= args.max_age)
    ]

    print("Excluding children: ", EXCLUDED_CHILDREN)
    conversations = conversations[~conversations.child_name.isin(EXCLUDED_CHILDREN)]

    conversations = filter_corpora(conversations)

    perform_analysis_grammaticality(conversations, args)
