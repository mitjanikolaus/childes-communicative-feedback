import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from cf_analyses.analysis_reproduce_warlaumont import AGE_BIN_NUM_MONTHS
from utils import MICRO_CONVERSATIONS_FILE, filter_transcripts_based_on_num_child_utts

MIN_AGE = 10
MAX_AGE = 60

DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT = 10


def make_proportion_plots(conversations, results_dir):
    plt.figure(figsize=(6, 4))

    proportion_speech_like_per_transcript = conversations.groupby(
        "transcript_file"
    ).agg({"utt_is_speech_related": "mean", "age": "mean"})
    axis = sns.regplot(
        data=proportion_speech_like_per_transcript,
        x="age",
        y="utt_is_speech_related",
        marker=".",
        logistic=True,
        line_kws={"color": sns.color_palette("tab10")[0]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[0]},
        label="proportion_speech_like",
    )

    proportion_intelligible_per_transcript = conversations.groupby(
        "transcript_file"
    ).agg({"utt_is_intelligible": "mean", "age": "mean"})
    sns.regplot(
        data=proportion_intelligible_per_transcript,
        x="age",
        y="utt_is_intelligible",
        logistic=True,
        marker=".",
        line_kws={"color": sns.color_palette("tab10")[1]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[1]},
        label="proportion_intelligible",
    )

    conversations["utt_is_grammatical"] = conversations.utt_is_grammatical.replace({pd.NA: False})
    conversations.loc[~conversations.utt_is_intelligible, "utt_is_grammatical"] = False
    proportion_grammatical_per_transcript = conversations.groupby(
        "transcript_file"
    ).agg({"utt_is_grammatical": "mean", "age": "mean"})
    sns.regplot(
        data=proportion_grammatical_per_transcript,
        x="age",
        y="utt_is_grammatical",
        logistic=True,
        marker=".",
        line_kws={"color": sns.color_palette("tab10")[2]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[2]},
        label="proportion_grammatical",
    )
    axis.set(xlabel="age (months)", ylabel="")

    axis.legend(loc="lower right")

    axis.set_xticks(np.arange(12, conversations.age.max() + 1, step=AGE_BIN_NUM_MONTHS))
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proportions.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    # TODO: run with data from all corpora
    conversations = pd.read_csv(MICRO_CONVERSATIONS_FILE)

    conversations = conversations[
        (MIN_AGE <= conversations.age) & (conversations.age <= MAX_AGE)
    ]
    conversations = filter_transcripts_based_on_num_child_utts(
        conversations, DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT
    )

    results_dir = "results/"
    make_proportion_plots(conversations, results_dir)

    plt.show()
