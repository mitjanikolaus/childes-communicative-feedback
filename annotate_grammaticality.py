import argparse
import os

import pandas as pd
import torch

from annotate import annotate_grammaticality

OUT_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_annotated_model_results.csv"
)

DEFAULT_MODELS_GRAMMATICALITY_ANNOTATION = [
    "cointegrated/roberta-large-cola-krishna2020",
    "textattack/bert-base-uncased-CoLA",
    "yevheniimaslov/deberta-v3-large-cola",
    "cointegrated/roberta-large-cola-krishna2020",
    "textattack/distilbert-base-cased-CoLA",
    "ModelTC/bert-base-uncased-cola",
    "WillHeld/roberta-base-cola",
    "Aktsvigun/electra-large-cola"
]
BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"


def annotate(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0)

    utterances["is_grammatical_m"] = utterances.is_grammatical_m.astype(bool)

    for model_name in args.grammaticality_annotation_models:
        print(f"Annotating grammaticality with {model_name}..")
        column_name = "is_grammatical_" + model_name.replace('/', '_')
        utterances[column_name] = annotate_grammaticality(utterances.utt_clean.values, model_name)
        utterances[column_name] = utterances[column_name].astype(bool)
        acc = (utterances[column_name] == utterances.is_grammatical_m).mean()
        print(f"Accuracy: {acc:.3f}")

    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        required=True,
    )
    argparser.add_argument(
        "--grammaticality-annotation-models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS_GRAMMATICALITY_ANNOTATION,
    )
    argparser.add_argument(
        "--out-file",
        type=str,
        default=OUT_UTTERANCES_FILE
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    annotated_utts = annotate(args)

    annotated_utts.to_pickle(args.out_file)
    annotated_utts.to_csv(args.out_file.replace(".p", ".csv"))
