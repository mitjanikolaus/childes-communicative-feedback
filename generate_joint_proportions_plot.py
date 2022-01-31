import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from analysis_reproduce_warlaumont import AGE_BIN_NUM_MONTHS


def make_proportion_plots(conversations, results_dir):
    proportion_intelligible_per_transcript = conversations.groupby(
        "transcript_file"
    ).agg({"utt_is_intelligible": "mean", "age": "mean"})
    plt.figure(figsize=(6, 4))
    sns.regplot(
        data=proportion_intelligible_per_transcript,
        x="age",
        y="utt_is_intelligible",
        logistic=True,
        marker=".",
        line_kws={"color": sns.color_palette("tab10")[1]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[1]},
    )
    proportion_speech_related_per_transcript = conversations.groupby(
        "transcript_file"
    ).agg({"utt_is_speech_related": "mean", "age": "mean"})
    axis = sns.regplot(
        data=proportion_speech_related_per_transcript,
        x="age",
        y="utt_is_speech_related",
        marker=".",
        logistic=True,
        line_kws={"color": sns.color_palette("tab10")[0]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[0]},
    )
    axis.set(xlabel="age (months)", ylabel="")
    axis.legend(
        labels=["proportion_intelligible", "proportion_speech_related"],
        loc="lower right",
    )
    axis.set_xticks(np.arange(12, conversations.age.max() + 1, step=AGE_BIN_NUM_MONTHS))
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proportions.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    conversations = pd.read_csv("results/reproduce_warlaumont/conversations.csv")

    results_dir = "results/"
    make_proportion_plots(conversations, results_dir)

    plt.show()
