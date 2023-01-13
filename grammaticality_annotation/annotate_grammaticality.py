import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerFast, AutoConfig

import matplotlib

from grammaticality_annotation.data import tokenize
from grammaticality_annotation.pretrain_lstm import TOKENIZER_PATH, TOKEN_PAD, TOKEN_EOS, TOKEN_UNK, TOKEN_SEP, \
    LSTMSequenceClassification
from utils import get_num_unique_words, ERR_UNKNOWN

if os.environ["DISPLAY"] != ":0":
    matplotlib.use("Agg")

DEFAULT_MODEL_GRAMMATICALITY_ANNOTATION = "cointegrated/roberta-large-cola-krishna2020"
MODELS_ACCEPTABILITY_JUDGMENTS_INVERTED = ["cointegrated/roberta-large-cola-krishna2020"]
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"


OUT_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_annotated_grammaticality.csv"
)

DEFAULT_MODELS_GRAMMATICALITY_ANNOTATION = [
    "cointegrated/roberta-large-cola-krishna2020",
    "textattack/bert-base-uncased-CoLA",
    "yevheniimaslov/deberta-v3-large-cola",
    "textattack/distilbert-base-cased-CoLA",
    "ModelTC/bert-base-uncased-cola",
    "WillHeld/roberta-base-cola",
    "Aktsvigun/electra-large-cola"
]
BATCH_SIZE = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_LABELS = 2
MAX_SEQ_LENGTH = 40


def annotate_grammaticality(utterances, model_name, label_empty_utterance=pd.NA,
                            label_one_word_utterance=pd.NA, label_empty_prev_utterance=pd.NA):
    if os.path.isfile(model_name):
        if "pretrain_lstm" in model_name:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
            tokenizer.add_special_tokens(
                {'pad_token': TOKEN_PAD, 'eos_token': TOKEN_EOS, 'unk_token': TOKEN_UNK, 'sep_token': TOKEN_SEP})
            model = LSTMSequenceClassification.load_from_checkpoint(model_name, num_labels=NUM_LABELS).to(device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name_or_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    grammaticalities = np.zeros_like(utterances.transcript_clean, dtype=bool).astype(object)  # cast to object to allow for NA
    num_unique_words = get_num_unique_words(utterances.transcript_clean)
    grammaticalities[(num_unique_words == 0)] = label_empty_utterance
    grammaticalities[(num_unique_words == 1)] = label_one_word_utterance
    grammaticalities[utterances.prev_transcript_clean.isna()] = label_empty_prev_utterance

    utts_to_annotate = utterances[num_unique_words > 1]
    utts_to_annotate = utts_to_annotate[~utts_to_annotate.prev_transcript_clean.isna()]

    dataset = Dataset.from_pandas(utts_to_annotate)

    def tokenize_batch(batch):
        return tokenize(batch, tokenizer, MAX_SEQ_LENGTH)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tokenize_batch)

    annotated_grammaticalities = []
    for tokenized in tqdm(dataloader):
        predicted_class_ids = model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask).logits.argmax(dim=-1)
        batch_grammaticalities = predicted_class_ids.bool()
        if model_name in MODELS_ACCEPTABILITY_JUDGMENTS_INVERTED:
            batch_grammaticalities = ~batch_grammaticalities
        batch_grammaticalities = batch_grammaticalities.cpu().numpy().astype(object)

        annotated_grammaticalities.extend(batch_grammaticalities.tolist())

    grammaticalities[(num_unique_words > 1) & (~utts_to_annotate.prev_transcript_clean.isna())] = annotated_grammaticalities

    return grammaticalities


def plot_error_type_stats(utterances, drop_unknown=True):
    if "is_grammatical" in utterances.columns:
        utts = utterances.dropna(subset=["is_grammatical", "labels"]).copy()
        utts["label"] = utts.labels.astype(str).apply(lambda x: x.split(", "))
        utts.drop(columns="labels", inplace=True)
        utts = utts.explode("label")
        if drop_unknown:
            print(f"removing {len(utts[utts.label == ERR_UNKNOWN])} rows with unknown errors")
            utts = utts[utts.label != ERR_UNKNOWN]
        utts.label.value_counts().plot(kind="barh")
        plt.subplots_adjust(left=0.2, right=0.99)


def plot_errors(utterances):
    if "is_grammatical" in utterances.columns:
        plt.figure()
        utts = utterances.dropna(subset=["is_grammatical", "labels"]).copy()
        utts["label"] = utts.labels.astype(str).apply(lambda x: x.replace("?", "").split(", "))
        utts.drop(columns="labels", inplace=True)
        keep_columns = [column for column in utts.columns if "_is_correct" in column or column == "label"]
        utts = utts[keep_columns]
        utts = utts.explode("label")
        utts_grouped = utts.groupby("label").mean()
        utts_melted = utts_grouped.reset_index().melt("label", var_name='cols', value_name='vals')
        sns.barplot(data=utts_melted, x="label", y="vals", hue="cols")
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.25, top=0.99)
        plt.legend(loc='lower left', fontsize='5')


def column_name_model_grammaticality(model_name):
    return "is_grammatical_" + model_name.replace('/', '_')


def column_name_model_correct(model_name):
    return model_name.replace('/', '_') + "_is_correct"


def annotate(utterances):
    for model_name in args.grammaticality_annotation_models:
        column_name = column_name_model_grammaticality(model_name)
        if column_name in utterances.columns:
            print(f"Annotation for {model_name} already done. Skipping.")
            continue
        print(f"Annotating grammaticality with {model_name}..")
        utterances[column_name] = annotate_grammaticality(utterances, model_name)

    if "is_grammatical" in utterances.columns:
        utterances.dropna(subset=["is_grammatical"], inplace=True)
        print(f"Accuracy scores for {len(utterances)} samples:")
        utterances["is_grammatical"] = utterances.is_grammatical.astype(bool)

        results = []
        for model_name in args.grammaticality_annotation_models:
            column_name = column_name_model_grammaticality(model_name)
            column_name_correct = column_name_model_correct(model_name)

            utterances[column_name] = utterances[column_name].astype(bool)
            utterances[column_name_correct] = utterances[column_name] == utterances.is_grammatical
            acc = utterances[column_name_correct].mean()
            utt_pos = utterances[utterances.is_grammatical]
            acc_pos = utt_pos[column_name_correct].mean()
            utt_neg = utterances[~utterances.is_grammatical]
            acc_neg = utt_neg[column_name_correct].mean()

            results.append({"model": model_name, "accuracy": acc, "accuracy (pos)": acc_pos, "accuracy (neg)": acc_neg})

        results = pd.DataFrame(results)
        pd.set_option('display.precision', 2)
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.max_rows', len(args.grammaticality_annotation_models))
        pd.set_option('display.width', 1000)
        print(results)

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
        "--out",
        type=str,
        default=OUT_UTTERANCES_FILE
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.out.endswith(".csv"):
        raise ValueError("Out file should have .csv ending!")

    utterances = pd.read_csv(args.utterances_file, index_col=0)
    plot_error_type_stats(utterances)

    annotated_utts = annotate(utterances)
    plot_errors(annotated_utts)

    annotated_utts.to_csv(args.out)

    plt.show()
