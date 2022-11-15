import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_GRAMMATICALITY_ANNOTATION = "cointegrated/roberta-large-cola-krishna2020"
MODELS_ACCEPTABILITY_JUDGMENTS_INVERTED = ["cointegrated/roberta-large-cola-krishna2020"]
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"


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
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_num_words(utt_gra_tags):
    return len([tag for tag in utt_gra_tags if tag is not None and tag["rel"] != "PUNCT"])


def annotate_grammaticality(clean_utterances, model_name, label_empty_utterance=pd.NA,
                            label_one_word_utterance=pd.NA):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    grammaticalities = np.zeros_like(clean_utterances, dtype=bool).astype(object)  # cast to object to allow for NA
    num_words = torch.tensor([len(re.split('\s|\'', utt)) for utt in clean_utterances])
    grammaticalities[(num_words == 0)] = label_empty_utterance
    grammaticalities[(num_words == 1)] = label_one_word_utterance

    utts_to_annotate = clean_utterances[(num_words > 1)]

    batches = [utts_to_annotate[x:x + BATCH_SIZE] for x in range(0, len(utts_to_annotate), BATCH_SIZE)]

    annotated_grammaticalities = []
    for batch in tqdm(batches):
        tokenized = tokenizer(list(batch), padding=True, return_tensors="pt").to(device)

        predicted_class_ids = model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask).logits.argmax(dim=-1)
        batch_grammaticalities = predicted_class_ids.bool()
        if model_name in MODELS_ACCEPTABILITY_JUDGMENTS_INVERTED:
            batch_grammaticalities = ~batch_grammaticalities
        batch_grammaticalities = batch_grammaticalities.cpu().numpy().astype(object)

        annotated_grammaticalities.extend(batch_grammaticalities.tolist())

    grammaticalities[(num_words > 1)] = annotated_grammaticalities

    return grammaticalities


def annotate(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0)

    for model_name in args.grammaticality_annotation_models:
        print(f"Annotating grammaticality with {model_name}..")
        column_name = "is_grammatical_" + model_name.replace('/', '_')
        utterances[column_name] = annotate_grammaticality(utterances.utt_clean.values, model_name)

    if "is_grammatical_m" in utterances.columns:
        utterances = utterances.dropna(subset=["is_grammatical_m"])
        print(f"Accuracy scores for {len(utterances)} samples:")
        utterances["is_grammatical_m"] = utterances.is_grammatical_m.astype(bool)

        results = []
        for model_name in args.grammaticality_annotation_models:
            column_name = "is_grammatical_" + model_name.replace('/', '_')
            utterances[column_name] = utterances[column_name].astype(bool)
            acc = (utterances[column_name] == utterances.is_grammatical_m).mean()
            utt_pos = utterances[utterances.is_grammatical_m]
            acc_pos = (utt_pos[column_name] == utt_pos.is_grammatical_m).mean()
            utt_neg = utterances[~utterances.is_grammatical_m]
            acc_neg = (utt_neg[column_name] == utt_neg.is_grammatical_m).mean()

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
