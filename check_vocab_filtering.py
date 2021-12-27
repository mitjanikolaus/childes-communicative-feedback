from collections import Counter

import pandas as pd

import nltk

from analysis_reproduce_warlaumont import (
    DEFAULT_MIN_AGE,
    DEFAULT_MAX_AGE,
    DEFAULT_RESPONSE_LATENCY_MAX_STANDARD_DEVIATIONS_OFF,
)
from preprocess import CANDIDATE_CORPORA, DEFAULT_RESPONSE_THRESHOLD
from utils import (
    is_babbling,
    VOCAB,
    CODE_PHONOLGICAL_CONSISTENT_FORM,
    filter_corpora_based_on_response_latency_length,
    clean_utterance,
    remove_nonspeech_events,
    EMPTY_UTTERANCE, get_path_of_utterances_file,
)


def check_vocab(adj_utterances):
    # Clean utterances
    adj_utterances["utt_child"] = adj_utterances.utt_child.apply(clean_utterance)
    adj_utterances["utt_car"] = adj_utterances.utt_car.apply(clean_utterance)
    adj_utterances["utt_child_follow_up"] = adj_utterances.utt_child_follow_up.apply(
        clean_utterance
    )

    # Remove nonspeech events
    adj_utterances["utt_child"] = adj_utterances.utt_child.apply(
        remove_nonspeech_events
    )
    adj_utterances["utt_car"] = adj_utterances.utt_car.apply(remove_nonspeech_events)
    adj_utterances["utt_child_follow_up"] = adj_utterances.utt_child_follow_up.apply(
        remove_nonspeech_events
    )

    missing = []

    stemmer = nltk.stem.PorterStemmer()
    for utt in (
        list(adj_utterances.utt_child.values)
        + list(adj_utterances.utt_car.values)
        + list(adj_utterances.utt_child_follow_up.values)
    ):
        utt = remove_nonspeech_events(utt)
        words = utt.split(" ")
        for word in words:
            word = word.lower()
            stem = stemmer.stem(word)
            if (
                stem not in VOCAB
                and word not in VOCAB
                and word.replace(CODE_PHONOLGICAL_CONSISTENT_FORM, "") not in VOCAB
                and not is_babbling(word)
                and not word == EMPTY_UTTERANCE
            ):
                # if word.endswith(CODE_PHONOLGICAL_CONSISTENT_FORM) or word.endswith(CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION):
                # missing.append(word.replace(CODE_PHONOLGICAL_CONSISTENT_FORM, ""))
                missing.append(word)

    print(Counter(missing).most_common())
    for word, freq in Counter(missing).most_common():
        if freq > 5:
            print(word)


if __name__ == "__main__":
    utterances = pd.read_csv(get_path_of_utterances_file(DEFAULT_RESPONSE_THRESHOLD), index_col=None)

    # Fill in empty strings for dummy caregiver responses
    utterances.utt_car.fillna("", inplace=True)

    corpora = filter_corpora_based_on_response_latency_length(
        CANDIDATE_CORPORA,
        utterances,
        DEFAULT_MIN_AGE,
        DEFAULT_MAX_AGE,
        DEFAULT_RESPONSE_LATENCY_MAX_STANDARD_DEVIATIONS_OFF,
    )

    print(f"Corpora included in analysis: {corpora}")

    # Filter corpora
    utterances = utterances[utterances.corpus.isin(corpora)]

    check_vocab(utterances.copy())
