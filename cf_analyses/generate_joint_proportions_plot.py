import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from cf_analyses.analysis_intelligibility import DEFAULT_EXCLUDED_CORPORA as DEFAULT_EXCLUDED_CORPORA_INTELLIGIBILITY
from cf_analyses.analysis_reproduce_warlaumont import DEFAULT_EXCLUDED_CORPORA as DEFAULT_EXCLUDED_CORPORA_SPEECH_RELATEDNESS
from cf_analyses.analysis_grammaticality import filter_corpora as filter_corpora_grammaticality
from cf_analyses.analysis_reproduce_warlaumont import AGE_BIN_NUM_MONTHS
from utils import filter_transcripts_based_on_num_child_utts, \
    UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, PROJECT_ROOT_DIR, SPEAKER_CODE_CHILD, filter_for_min_num_words

MIN_AGE = 10
MAX_AGE = 60

DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT = 100

# Do not filter for min num words because this will remove many unintelligible utterances which are also not grammatical
MIN_NUM_WORDS = 0


def make_proportion_plots(utterances_all, results_dir):
    plt.figure(figsize=(13, 7))

    utterances_all = utterances_all[
        (MIN_AGE <= utterances_all.age) & (utterances_all.age <= MAX_AGE)
    ].copy()

    utterances_all = utterances_all[utterances_all.speaker_code == SPEAKER_CODE_CHILD].copy()

    utterances = filter_transcripts_based_on_num_child_utts(
        utterances_all, DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT
    )

    utterances_filtered_speech_relatedness = utterances[~utterances.corpus.isin(DEFAULT_EXCLUDED_CORPORA_SPEECH_RELATEDNESS)]
    proportion_speech_like_per_transcript = utterances_filtered_speech_relatedness.groupby(
        "transcript_file"
    ).agg({"is_speech_related": "mean", "age": "mean"})
    axis = sns.regplot(
        data=proportion_speech_like_per_transcript,
        x="age",
        y="is_speech_related",
        marker=".",
        logistic=True,
        line_kws={"color": sns.color_palette("tab10")[0]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[0]},
        label="proportion_speech_like",
    )

    utterances.loc[utterances.is_speech_related == False, "is_intelligible"] = False
    utterances_filtered_intelligibility = utterances[~utterances.corpus.isin(DEFAULT_EXCLUDED_CORPORA_INTELLIGIBILITY)]
    proportion_intelligible_per_transcript = utterances_filtered_intelligibility.groupby(
        "transcript_file"
    ).agg({"is_intelligible": "mean", "age": "mean"})
    sns.regplot(
        data=proportion_intelligible_per_transcript,
        x="age",
        y="is_intelligible",
        logistic=True,
        marker=".",
        line_kws={"color": sns.color_palette("tab10")[1]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[1]},
        label="proportion_intelligible",
    )

    utterances.loc[~utterances.is_intelligible, "is_grammatical"] = False
    utterances_filtered_grammaticality = filter_corpora_grammaticality(utterances)

    utterances_filtered_grammaticality = filter_for_min_num_words(utterances_filtered_grammaticality, MIN_NUM_WORDS)

    proportion_grammatical_per_transcript = utterances_filtered_grammaticality.groupby(
        "transcript_file"
    ).agg({"is_grammatical": "mean", "age": "mean"})
    sns.regplot(
        data=proportion_grammatical_per_transcript,
        x="age",
        y="is_grammatical",
        logistic=True,
        marker=".",
        line_kws={"color": sns.color_palette("tab10")[2]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[2]},
        label="proportion_grammatical",
    )
    axis.set(xlabel="age (months)", ylabel="")

    axis.legend(loc="lower right")

    axis.set_xticks(np.arange(MIN_AGE, MAX_AGE + 1, step=AGE_BIN_NUM_MONTHS))
    plt.xlim((MIN_AGE-1, MAX_AGE+1))
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proportions.png"), dpi=300)

    # utterances_filtered_grammaticality = filter_corpora_grammaticality(utterances_all)
    # utts_grammaticality_analysis = utterances_filtered_grammaticality[utterances_filtered_grammaticality.is_intelligible]
    # utts_grammaticality_analysis = filter_for_min_num_utts(utts_grammaticality_analysis, min_num_words=1)
    #
    # plt.figure(figsize=(13, 7))
    # data = utts_grammaticality_analysis.groupby("age").agg({"is_grammatical": "mean", "utterance_id": "size"}).reset_index().rename(
    #     columns={"is_grammatical": "prop_grammatical", "utterance_id": "num_utterances"})
    # sns.barplot(data=data, x="age", y="prop_grammatical", label="proportion_grammatical", color="b")
    # plt.savefig(os.path.join(results_dir, "prop_grammatical.png"), dpi=300)
    #
    # plt.figure(figsize=(13, 7))
    # sns.barplot(data=data, x="age", y="num_utterances", label="num_utterances", color="r")
    # plt.savefig(os.path.join(results_dir, "num_utterances.png"), dpi=300)
    #
    # not_grammatical = utts_grammaticality_analysis[utts_grammaticality_analysis.is_grammatical == False]
    # not_grammatical = not_grammatical.groupby("age").agg({"utterance_id": "size"}).reset_index().rename(
    #     columns={"utterance_id": "num_errors"})
    # plt.figure(figsize=(13, 7))
    # sns.barplot(data=not_grammatical, x="age", y="num_errors", label="num_errors", color="g")
    # plt.savefig(os.path.join(results_dir, "num_errors.png"), dpi=300)

    plt.show()


if __name__ == "__main__":
    utterances = pd.read_csv(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, index_col=0, dtype={"error": object})

    results_dir = PROJECT_ROOT_DIR+"/results/"
    make_proportion_plots(utterances, results_dir)

    plt.show()
