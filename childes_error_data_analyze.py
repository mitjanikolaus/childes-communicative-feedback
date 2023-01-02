import os

import pandas as pd

from annotate_grammaticality import plot_error_type_stats
from utils import ANNOTATED_UTTERANCES_FILE, ERR_UNKNOWN, SPEAKER_CODE_CHILD
import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()

CHILDES_ERRORS_DATA_FILE = os.path.expanduser(
    "data/utterances_errors.csv"
)


def plot_corpus_error_stats(error_utterances, all_utterances, drop_unknown=True):
    utts = error_utterances.dropna(subset=["is_grammatical", "labels"]).copy()
    utts["label"] = utts.labels.astype(str).apply(lambda x: x.split(", "))
    utts.drop(columns="labels", inplace=True)
    utts = utts.explode("label")
    if drop_unknown:
        print(f"removing {len(utts[utts.label == ERR_UNKNOWN])} rows with unknown errors")
        utts = utts[utts.label != ERR_UNKNOWN]

    utts.corpus.value_counts().plot(kind="barh")
    plt.subplots_adjust(left=0.2, right=0.99)
    plt.show()

    num_errors = error_utterances.corpus.value_counts()

    num_utts = all_utterances[all_utterances.speaker_code == SPEAKER_CODE_CHILD].corpus.value_counts()

    num_utts = num_utts.to_frame()
    num_utts = num_utts.rename(columns={"corpus": "num_utts"})
    joined = num_utts.join(num_errors)
    joined = joined.rename(columns={"corpus": "num_errors"})
    joined.fillna(0, inplace=True)
    joined["ratio"] = joined["num_errors"] / joined["num_utts"]
    joined.plot(y='ratio', kind="bar")
    plt.savefig("out.png")
    plt.show()


def analyze():
    error_utterances = pd.read_csv(CHILDES_ERRORS_DATA_FILE, index_col=0, dtype={"error": object})
    all_utterances = pd.read_csv(ANNOTATED_UTTERANCES_FILE, index_col=0, dtype={"error": object})

    plot_corpus_error_stats(error_utterances, all_utterances)
    plot_error_type_stats(error_utterances)
    # plt.show()


if __name__ == "__main__":
    analyze()