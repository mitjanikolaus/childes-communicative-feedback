import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import (
    filter_transcripts_based_on_num_child_utts, SPEAKER_CODE_CHILD, UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_CLEAN_FILE
)

DEFAULT_COUNT_ONLY_SPEECH_RELATED_RESPONSES = True

DEFAULT_MIN_RATIO_NONSPEECH = 0.0

DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT = 1

# Ages aligned to study of Warlaumont et al. or to our study (minimum 10 months)
DEFAULT_MIN_AGE = 10
#TODO:
DEFAULT_MAX_AGE = 60

AGE_BIN_NUM_MONTHS = 6

# Forrester: Does not annotate non-word sounds starting with & (phonological fragment), these are treated as words and
# should be excluded when annotating intelligibility based on rules.
# Providence: Some non-speech vocalizations such as laughter are incorrectly transcribed as 'yyy', and the timing
# information is of very poor quality
DEFAULT_EXCLUDED_CORPORA = ["Providence", "Forrester"]

# The caregivers of these children are using slang (e.g., "you was" or "she don't") and are therefore excluded
# We are unfortunately only studying mainstream US English
EXCLUDED_CHILDREN = ["Brent_Jaylen", "Brent_Tyrese", "Brent_Vas", "Brent_Vas_Coleman", "Brent_Xavier"]

# GRAMMATICALITY_COLUMN = "is_grammatical_lightning_logs_version_1178162_checkpoints_last.ckpt"
GRAMMATICALITY_COLUMN = "is_grammatical"

RESULTS_DIR = "results/grammaticality/"


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_CLEAN_FILE,
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
        "--min-child-utts-per-transcript",
        type=int,
        default=DEFAULT_MIN_CHILD_UTTS_PER_TRANSCRIPT,
    )

    args = argparser.parse_args()

    return args


def plot_grammaticality_development(utterances):
    utterances.dropna(subset=["is_grammatical"], inplace=True)
    utterances["is_grammatical"] = utterances.is_grammatical.astype(bool)
    utterances = utterances[utterances.is_intelligible & utterances.is_speech_related]

    utterances = filter_transcripts_based_on_num_child_utts(
        utterances, args.min_child_utts_per_transcript
    )

    proportion_grammatical_per_transcript_chi = utterances[utterances.speaker_code==SPEAKER_CODE_CHILD].groupby(
        "transcript_file"
    ).agg({"is_grammatical": "mean", "age": "mean"})
    sns.regplot(
        data=proportion_grammatical_per_transcript_chi,
        x="age",
        y="is_grammatical",
        marker=".",
        logistic=True,
        line_kws={"color": sns.color_palette("tab10")[0]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[0]},
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "proportion_grammatical_children.png"), dpi=300)

    proportion_grammatical_per_transcript_adu = utterances[utterances.speaker_code != SPEAKER_CODE_CHILD].groupby(
        "transcript_file"
    ).agg({"is_grammatical": "mean", "age": "mean"})
    sns.regplot(
        data=proportion_grammatical_per_transcript_adu,
        x="age",
        y="is_grammatical",
        marker=".",
        logistic=True,
        line_kws={"color": sns.color_palette("tab10")[1]},
        scatter_kws={"alpha": 0.2, "s": 20, "color": sns.color_palette("tab10")[1]},
    )
    plt.tight_layout()
    plt.legend(labels=["children", "adults"])
    plt.savefig(os.path.join(RESULTS_DIR, "proportion_grammatical_children_adults.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    utterances = pd.read_csv(args.utterances_file, index_col=0)

    utterances["is_grammatical"] = utterances[GRAMMATICALITY_COLUMN]

    print("Excluding corpora: ", args.excluded_corpora)
    utterances = utterances[~utterances.corpus.isin(args.excluded_corpora)]

    if args.corpora:
        print("Including only corpora: ", args.corpora)
        utterances = utterances[utterances.corpus.isin(args.corpora)]

    print("Excluding children: ", EXCLUDED_CHILDREN)
    utterances = utterances[~utterances.child_name.isin(EXCLUDED_CHILDREN)]

    # Filter by age
    utterances = utterances[
        (args.min_age <= utterances.age) & (utterances.age <= args.max_age)
    ]

    plot_grammaticality_development(utterances)

