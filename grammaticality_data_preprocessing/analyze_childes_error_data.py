import os

import pandas as pd

from utils import SPEAKER_CODE_CHILD, UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, ERR_SUBJECT, ERR_PLURAL, \
    ERR_TENSE_ASPECT, ERR_PROGRESSIVE, ERR_OTHER, ERR_VERB, ERR_AUXILIARY, ERR_PREPOSITION, \
    ERR_SUBJECT, ERR_OBJECT, ERR_POSSESSIVE, ERR_SV_AGREEMENT, ERR_DETERMINER
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
tqdm.pandas()


RESULTS_DIR = "results/grammaticality_annotations_childes"

MIN_NUM_ERRORS = 100
ERROR_RATIO_THRESHOLD = 0.01

HUE_ORDER = [ERR_SUBJECT, ERR_VERB, ERR_OBJECT, ERR_DETERMINER, ERR_PREPOSITION, ERR_AUXILIARY, ERR_PROGRESSIVE, ERR_POSSESSIVE, ERR_SV_AGREEMENT, ERR_TENSE_ASPECT, ERR_PLURAL, ERR_OTHER]

PALETTE_CATEGORICAL = sns.color_palette() + [(0, 0, 0), (1, 1, 1)]


def explode_labels(utterances):
    utterances["label"] = utterances.labels.astype(str).apply(lambda x: x.split(", "))
    utterances.drop(columns="labels", inplace=True)
    return utterances.explode("label")


def plot_corpus_error_stats(utterances):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    utterances = utterances[utterances.speaker_code == SPEAKER_CODE_CHILD].copy()
    utterances.dropna(subset=["is_grammatical"], inplace=True)

    error_utts = utterances[utterances.is_grammatical == False].copy()

    print(f"Total utts: {len(utterances)} | Errors: {len(error_utts)}")
    num_errors = error_utts.corpus.value_counts()

    # Filter for min utts
    corpora_to_plot = num_errors[num_errors > MIN_NUM_ERRORS].index
    error_utts = error_utts[error_utts.corpus.isin(corpora_to_plot)].copy()
    utterances = utterances[utterances.corpus.isin(corpora_to_plot)].copy()

    num_errors = error_utts.corpus.value_counts()
    num_errors.plot(kind="bar")
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "error_counts_absolute.png"), dpi=300
    )

    num_utts_data = utterances.corpus.value_counts()
    print("Total number of utts per corpus:")
    print(num_utts_data.to_dict())

    print("Mean age for corpora:")
    print(utterances.groupby("corpus").agg({"age": "mean"}).to_dict()["age"])

    num_utts = num_utts_data.to_frame()
    num_utts = num_utts.rename(columns={"corpus": "num_utts"})

    joined = num_utts.join(num_errors)
    joined = joined.rename(columns={"corpus": "num_errors"})
    joined.fillna(0, inplace=True)
    joined["ratio"] = joined["num_errors"] / joined["num_utts"]
    joined.sort_values(by="ratio", inplace=True)
    joined.plot(y='ratio', kind="bar")
    plt.axhline(y=ERROR_RATIO_THRESHOLD, linestyle='--', color='r')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, "error_proportions.png"), dpi=300
    )
    print(f"Corpora with more than {ERROR_RATIO_THRESHOLD}% errors: {joined[joined.ratio > ERROR_RATIO_THRESHOLD].index.to_list()}")

    utts_exploded = explode_labels(error_utts)

    err_counts = utts_exploded.groupby(['corpus'])["label"].value_counts().rename("count").reset_index()

    err_counts["ratio"] = err_counts.apply(lambda row: row["count"] / num_utts_data[row.corpus], axis=1)
    sns.set_palette(PALETTE_CATEGORICAL)
    plt.figure(figsize=(12, 4))
    ax = sns.barplot(x="corpus", y="ratio", hue="label", data=err_counts, order=joined.index, hue_order=HUE_ORDER,
                     linewidth=1, edgecolor=".1")
    plt.ylabel("num errors per child utterance")
    plt.xlabel("")
    plt.legend(loc='upper left', ncol=2, fontsize=12)
    xticklabels = ["MPI-EVA-\nManchester" if l.get_text() == "MPI-EVA-Manchester" else l.get_text() for l in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels)
    plt.xticks(rotation=75, size=7)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(
        os.path.join(RESULTS_DIR, "error_proportions_by_label.png"), dpi=300
    )


def analyze():
    utterances = pd.read_csv(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, index_col=0, dtype={"error": object})

    plot_corpus_error_stats(utterances)
    plt.show()


if __name__ == "__main__":
    analyze()