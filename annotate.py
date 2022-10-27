import argparse
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import (
    remove_punctuation,
    str2bool,
    remove_babbling,
    ANNOTATED_UTTERANCES_FILE,
    UTTERANCES_WITH_SPEECH_ACTS_FILE, is_empty,
)
from utils import (
    remove_nonspeech_events,
    CODE_UNINTELLIGIBLE,
)

DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED = True

DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE = False

MODEL_GRAMMATICALITY_ANNOTATION = "cointegrated/roberta-large-cola-krishna2020"

# Speech acts that relate to nonverbal/external events
SPEECH_ACTS_NONVERBAL_EVENTS = [
    "CR",  # Criticize or point out error in nonverbal act.
    "PM",  # Praise for motor acts i.e for nonverbal behavior.
    "WD",  # Warn of danger.
    "DS",  # Disapprove scold protest disruptive behavior.
    "AB",  # Approve of appropriate behavior.
    "TO",  # Mark transfer of object to hearer
    "ET",  # Express enthusiasm for hearer's performance.
    "ED",  # Exclaim in disapproval.
]


def is_speech_related(
    utterance,
    label_partially_speech_related=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
    label_unintelligible=None,
):
    """Label utterances as speech or non-speech."""
    utterance_without_punctuation = remove_punctuation(utterance)
    utt_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)

    utt_without_nonspeech = utt_without_nonspeech.strip()
    if utt_without_nonspeech == "":
        return False

    # We exclude completely unintelligible utterances (we don't know whether it's speech-related or not)
    is_completely_unintelligible = True
    for word in utt_without_nonspeech.split(" "):
        if word != CODE_UNINTELLIGIBLE and word != "":
            is_completely_unintelligible = False
            break
    if is_completely_unintelligible:
        # By returning None, we can filter out these cases later
        return label_unintelligible

    is_partly_speech_related = len(utt_without_nonspeech) != len(
        utterance_without_punctuation
    )
    if is_partly_speech_related:
        return label_partially_speech_related

    return True


def is_intelligible(
    utterance,
    label_partially_intelligible=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
    label_empty_utterance=False,
):
    utterance_without_punctuation = remove_punctuation(utterance)
    utterance_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)
    utterance_without_nonspeech = utterance_without_nonspeech.strip()
    if utterance_without_nonspeech == "":
        return label_empty_utterance

    utt_without_babbling = remove_babbling(utterance_without_nonspeech)

    utt_without_babbling = utt_without_babbling.strip()
    if utt_without_babbling == "":
        return False

    is_partly_intelligible = len(utt_without_babbling) != len(
        utterance_without_nonspeech
    )
    if is_partly_intelligible:
        return label_partially_intelligible

    return True


def is_grammatical(utt_clean, tokenizer, model, label_empty_utterance=None, label_one_word_utterance=None):
    if is_empty(utt_clean):
        return label_empty_utterance

    tokenized = tokenizer(utt_clean)
    num_words = max(tokenized.encodings[0].word_ids[1:-1])
    if num_words == 1:
        return label_one_word_utterance

    input_ids = torch.tensor(tokenized.input_ids).unsqueeze(0)
    attention_mask = torch.tensor(tokenized.attention_mask).unsqueeze(0)

    model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax().item()
    predicted_class_id = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax().item()

    return not bool(predicted_class_id)


def clean_utterance(utterance):
    final_punctuation = None
    while len(utterance) > 0 and utterance[-1] in [".", "!", "?"]:
        final_punctuation = utterance[-1]
        utterance = utterance[:-1]

    # Remove commas at beginning of utterance
    while len(utterance) > 0 and utterance[0] == ",":
        utterance = utterance[1:]

    utt_clean = " ".join(utterance)

    # Remove underscores
    utt_clean = utt_clean.replace("_", " ")

    # Remove spacing before comma
    utt_clean = utt_clean.replace(" , ", ", ")

    # Transform to lower case:
    utt_clean = utt_clean.lower()

    if final_punctuation:
        utt_clean += final_punctuation
    return utt_clean


def annotate(args):
    utterances = pd.read_pickle(UTTERANCES_WITH_SPEECH_ACTS_FILE)

    print("Annotating speech-relatedness..")
    utterances = utterances.assign(
        is_speech_related=utterances.transcript_raw.apply(
            is_speech_related,
            label_partially_speech_related=args.label_partially_speech_related,
        )
    )

    print("Annotating intelligibility..")
    utterances = utterances.assign(
        is_intelligible=utterances.transcript_raw.apply(
            is_intelligible,
            label_partially_intelligible=args.label_partially_intelligible,
        )
    )

    print("Cleaning utterances..")
    utterances = utterances.assign(
        utt_clean=utterances.tokens.apply(
            clean_utterance
        )
    )

    print("Annotating grammaticality..")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_GRAMMATICALITY_ANNOTATION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_GRAMMATICALITY_ANNOTATION)
    utterances = utterances.assign(
        is_grammatical=utterances.utt_clean.apply(
            is_grammatical,
            tokenizer=tokenizer,
            model=model,
        )
    )

    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--label-partially-speech-related",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
        help="Label for partially speech-related utterances: Set to True to count as speech-related, False to count as "
        "not speech-related or None to exclude these utterances from the analysis",
    )
    argparser.add_argument(
        "--label-partially-intelligible",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
        help="Label for partially intelligible utterances: Set to True to count as intelligible, False to count as unintelligible or None to exclude these utterances from the analysis",
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    annotated_utts = annotate(args)

    os.makedirs(os.path.dirname(ANNOTATED_UTTERANCES_FILE), exist_ok=True)
    annotated_utts.to_pickle(ANNOTATED_UTTERANCES_FILE)
