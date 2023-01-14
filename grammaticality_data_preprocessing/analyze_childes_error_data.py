import os

import pandas as pd

from utils import ANNOTATED_UTTERANCES_FILE, ERR_UNKNOWN, SPEAKER_CODE_CHILD, UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_CLEAN_FILE
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
tqdm.pandas()


COLORS_PLOT_CATEGORICAL = [
"#000000",
"#FFFF00",
"#1CE6FF",
"#FF34FF",
"#FF4A46",
"#008941",
"#006FA6",
"#A30059",
"#FFDBE5",
"#7A4900",
"#0000A6",
"#63FFAC",
"#B79762",
"#004D43",
"#8FB0FF",
"#997D87",
"#5A0007",
"#809693",
"#FEFFE6",
"#1B4400",
"#4FC601",
"#3B5DFF",
"#4A3B53",
"#FF2F80",
"#61615A",
"#BA0900",
"#6B7900",
"#00C2A0",
"#FFAA92",
"#FF90C9",
"#B903AA",
"#D16100",
"#DDEFFF",
"#000035",
"#7B4F4B",
"#A1C299",
"#300018",
"#0AA6D8",
"#013349",
"#00846F",
"#372101",
"#FFB500",
"#C2FFED",
"#A079BF",
"#CC0744",
"#C0B9B2",
"#C2FF99",
"#001E09",
"#00489C",
"#6F0062",
"#0CBD66",
"#EEC3FF",
"#456D75",
"#B77B68",
"#7A87A1",
"#788D66",
"#885578",
"#FAD09F",
"#FF8A9A",
"#D157A0",
"#BEC459",
"#456648",
"#0086ED",
"#886F4C",
"#34362D",
"#B4A8BD",
"#00A6AA",
"#452C2C",
"#636375",
"#A3C8C9",
"#FF913F",
"#938A81",
"#575329",
]


def plot_corpus_error_stats(error_utterances):
    utts = error_utterances.dropna(subset=["is_grammatical", "labels"]).copy()
    utts["label"] = utts.labels.astype(str).apply(lambda x: x.split(", "))
    utts.drop(columns="labels", inplace=True)
    utts = utts.explode("label")

    utts.corpus.value_counts().plot(kind="barh")
    plt.subplots_adjust(left=0.2, right=0.99)

    num_errors = utts.corpus.value_counts()

    all_utterances = pd.read_csv(ANNOTATED_UTTERANCES_FILE, index_col=0, dtype={"error": object})
    num_utts_data = all_utterances[all_utterances.speaker_code == SPEAKER_CODE_CHILD].corpus.value_counts()

    num_utts = num_utts_data.to_frame()
    num_utts = num_utts.rename(columns={"corpus": "num_utts"})

    joined = num_utts.join(num_errors)
    joined = joined.rename(columns={"corpus": "num_errors"})
    joined.fillna(0, inplace=True)
    joined["ratio"] = joined["num_errors"] / joined["num_utts"]
    joined.plot(y='ratio', kind="bar")

    print("Total number of utts per corpus:")
    print(num_utts_data.to_dict())

    err_counts = utts.groupby(['corpus'])["label"].value_counts().rename("count").reset_index()

    def divide_by_num(row):
        row["ratio"] = row["count"] / num_utts_data[row.corpus]
        return row

    err_counts = err_counts.apply(divide_by_num, axis=1)
    sns.set_palette(COLORS_PLOT_CATEGORICAL)

    plt.figure()
    ax = sns.barplot(x="corpus", y="ratio", hue="label", data=err_counts)
    plt.ylim((0, 0.03))
    plt.xlabel("num errors per child utterance")
    plt.legend(loc='upper right')
    plt.setp(ax.get_xticklabels(), rotation=90)


def analyze():
    utterances = pd.read_csv(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_CLEAN_FILE, index_col=0, dtype={"error": object})

    error_utterances = utterances[~utterances.is_grammatical]

    print(f"Total utts: {len(utterances)} | Errors: {len(error_utterances)}")

    plot_corpus_error_stats(error_utterances)
    plt.show()


if __name__ == "__main__":
    analyze()