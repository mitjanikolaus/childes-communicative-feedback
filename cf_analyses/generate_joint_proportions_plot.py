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
    UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, PROJECT_ROOT_DIR, SPEAKER_CODE_CHILD, split_into_words

MIN_AGE = 10
MAX_AGE = 60

DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT = 100

# Do not filter for min num words because this will remove many unintelligible utterances which are also not grammatical
MIN_NUM_WORDS = 0


def make_proportion_plots(utterances, results_dir):
    plt.figure(figsize=(15, 7))

    utterances = utterances[utterances.speaker_code == SPEAKER_CODE_CHILD]

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

    num_words = utterances_filtered_grammaticality.transcript_clean.apply(
        lambda x: len(split_into_words(x, split_on_apostrophe=False, remove_commas=True,
                                       remove_trailing_punctuation=True)))
    utterances_filtered_grammaticality = utterances_filtered_grammaticality[num_words >= MIN_NUM_WORDS]

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
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proportions.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    utterances = pd.read_csv(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, index_col=0, dtype={"error": object})

    utterances = utterances[
        (MIN_AGE <= utterances.age) & (utterances.age <= MAX_AGE)
    ]
    utterances = filter_transcripts_based_on_num_child_utts(
        utterances, DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT
    )

    results_dir = PROJECT_ROOT_DIR+"/results/"
    make_proportion_plots(utterances, results_dir)

    plt.show()
