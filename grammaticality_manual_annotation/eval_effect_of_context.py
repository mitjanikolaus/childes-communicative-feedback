import argparse
import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from utils import PROJECT_ROOT_DIR

ANNOTATED_UTTERANCES_FILE = PROJECT_ROOT_DIR+"/data/manual_annotation/grammaticality_eval_effect_of_context.csv"


def compare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0)

    utterances.dropna(subset=["is_grammatical", "is_grammatical_no_prev"], inplace=True)

    utts_changed = utterances[utterances.is_grammatical != utterances.is_grammatical_no_prev]
    percentage_changed = 100 * len(utts_changed) / len(utterances)
    print(f"percentage_changed: {percentage_changed}%")

    utts_unknown = utterances[utterances.is_grammatical_no_prev == "?"]
    percentage_unknown = 100 * len(utts_unknown) / len(utterances)
    print(f"percentage_unknown before: {percentage_unknown}%")

    utts_unknown_after = utterances[utterances.is_grammatical == "?"]
    percentage_unknown_after = 100 * len(utts_unknown_after) / len(utterances)
    print(f"percentage_unknown after: {percentage_unknown_after}%")

    change_0_to_1 = utterances[(utterances.is_grammatical_no_prev == "0") & (utterances.is_grammatical == "1")]
    percentage_0_to_1 = 100 * len(change_0_to_1) / len(utterances)
    print(f"percentage_0_to_1: {percentage_0_to_1}%")

    change_1_to_0 = utterances[(utterances.is_grammatical_no_prev == "1") & (utterances.is_grammatical == "0")]
    percentage_1_to_0 = 100 * len(change_1_to_0) / len(utterances)
    print(f"percentage_1_to_0: {percentage_1_to_0}%")

    not_grammatical = utterances[utterances.is_grammatical == "0"]
    percentage_not_grammatical = 100 * len(not_grammatical) / len(utterances)
    print(f"percentage_not_grammatical: {percentage_not_grammatical}%")

    counts_1 = pd.DataFrame(utterances["is_grammatical_no_prev"].value_counts())
    counts_1.rename(columns={"is_grammatical_no_prev": "count"}, inplace=True)
    counts_1["context"] = False

    counts_2 = pd.DataFrame(utterances["is_grammatical"].value_counts())
    counts_2["context"] = True
    counts_2.rename(columns={"is_grammatical": "count"}, inplace=True)


    counts = pd.concat([counts_1, counts_2], ignore_index=False)
    counts.reset_index(names="is_grammatical", inplace=True)
    counts.replace({"0": False, "1": True}, inplace=True)

    sns.barplot(data=counts, x="is_grammatical", y="count", hue="context")

    plt.tight_layout()
    plt.savefig("results/grammaticality/effect_of_context/effect_of_context.png", dpi=300)
    plt.show()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=ANNOTATED_UTTERANCES_FILE,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    compare(args)