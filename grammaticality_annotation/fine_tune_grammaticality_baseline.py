import argparse
import itertools
from collections import Counter
import numpy as np

import nltk
from pytorch_lightning import Trainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from transformers import (
    PreTrainedTokenizerFast,
)
from sklearn.svm import SVC, LinearSVC

from grammaticality_annotation.data import create_dataset_dict, TEXT_FIELDS
from grammaticality_annotation.pretrain_lstm import TOKENIZER_PATH, TOKEN_PAD, TOKEN_EOS, TOKEN_UNK, TOKEN_SEP, LSTMSequenceClassification

RANDOM_STATE = 1


def tokenize(datapoint, tokenizer):
    text = tokenizer.sep_token.join([datapoint[TEXT_FIELDS[0]], datapoint[TEXT_FIELDS[1]]])
    encoded = tokenizer.encode(text)
    datapoint["encoded"] = encoded
    return datapoint


def create_features(datapoint, vocab_unigrams, vocab_bigrams, vocab_trigrams):
    unigrams = Counter([u for u in datapoint["encoded"] if u in vocab_unigrams])
    bigrams = Counter(nltk.ngrams(unigrams, 2))
    trigrams = Counter(nltk.ngrams(unigrams, 3))

    # TODO: counts or one-hot encoding?
    feat_unigrams = [unigrams[u] for u in vocab_unigrams]
    feat_bigrams = [bigrams[b] for b in vocab_bigrams]
    feat_trigrams = [trigrams[t] for t in vocab_trigrams]
    datapoint["features"] = feat_unigrams + feat_bigrams + feat_trigrams

    return datapoint


def main(args):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    tokenizer.add_special_tokens(
        {'pad_token': TOKEN_PAD, 'eos_token': TOKEN_EOS, 'unk_token': TOKEN_UNK, 'sep_token': TOKEN_SEP})

    datasets = create_dataset_dict(args.train_datasets, args.additional_val_datasets, args.val_split_proportion)

    datasets = datasets.map(tokenize, fn_kwargs={"tokenizer": tokenizer})

    unigrams = itertools.chain(*datasets["train"]["encoded"])
    bigrams = nltk.ngrams(itertools.chain(*datasets["train"]["encoded"]), 2)
    trigrams = nltk.ngrams(itertools.chain(*datasets["train"]["encoded"]), 3)

    unigrams = [u for u, c in Counter(unigrams).most_common(100)]
    bigrams = [b for b, c in Counter(bigrams).most_common(100)]
    trigrams = [t for t, c in Counter(trigrams).most_common(100)]

    datasets = datasets.map(create_features, fn_kwargs={"vocab_unigrams": unigrams, "vocab_bigrams": bigrams, "vocab_trigrams": trigrams})

    data_train = datasets["train"]

    # clf = SVC(random_state=RANDOM_STATE, class_weight="balanced", verbose=True)
    # clf = LinearSVC(random_state=RANDOM_STATE, class_weight="balanced", verbose=True)
    clf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=True)

    clf.fit(data_train["features"], data_train["labels"])

    data_val = datasets["validation"]
    predictions = clf.predict(data_val["features"])

    labels_val = np.array(datasets["validation"]["labels"])
    mcc = matthews_corrcoef(labels_val, predictions)
    print("MCC: ", mcc)

    accuracy = np.mean(labels_val == predictions)
    print("Accuracy: ", accuracy)

    accuracy_pos = np.mean(predictions[labels_val == 1] == 1)
    print("Accuracy (pos): ", accuracy_pos)

    accuracy_neg = np.mean(predictions[labels_val == 0] == 0)
    print("Accuracy (neg): ", accuracy_neg)


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--train-datasets",
        type=str,
        nargs="+",
        default=["manual_annotations"],
    )
    argparser.add_argument(
        "--additional-val-datasets",
        type=str,
        nargs="+",
        default=[],
    )
    argparser.add_argument(
        "--val-split-proportion",
        type=float,
        default=0.5,
        help="Val split proportion (only for manually annotated data)"
    )
    argparser = Trainer.add_argparse_args(argparser)

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
