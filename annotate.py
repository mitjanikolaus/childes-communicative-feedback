import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import (
    remove_punctuation,
    str2bool,
    remove_babbling,
    ANNOTATED_UTTERANCES_FILE,
    UTTERANCES_WITH_SPEECH_ACTS_FILE, remove_events_and_non_parseable_words,
)
from utils import (
    remove_nonspeech_events,
    CODE_UNINTELLIGIBLE,
)

DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED = True

DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE = False

DEFAULT_MODEL_GRAMMATICALITY_ANNOTATION = "cointegrated/roberta-large-cola-krishna2020"
MODELS_ACCEPTABILITY_JUDGMENTS_INVERTED = ["cointegrated/roberta-large-cola-krishna2020"]
BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        label_unintelligible=pd.NA,
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


def get_num_words(utt_gra_tags):
    return len([tag for tag in utt_gra_tags if tag is not None and tag["rel"] != "PUNCT"])


def annotate_grammaticality(clean_utterances, gra_tags, tokenizer, model, label_empty_utterance=pd.NA,
                            label_one_word_utterance=pd.NA, label_noun_phrase_utterance=pd.NA):
    grammaticalities = np.zeros_like(clean_utterances, dtype=bool).astype(object)  # cast to object to allow for NA
    num_words = torch.tensor([len(re.split('\s|\'', utt)) for utt in clean_utterances])
    grammaticalities[(num_words == 0)] = label_empty_utterance
    grammaticalities[(num_words == 1)] = label_one_word_utterance

    utts_to_annoatate = clean_utterances[(num_words > 1)]

    batches = [utts_to_annoatate[x:x + BATCH_SIZE] for x in range(0, len(utts_to_annoatate), BATCH_SIZE)]

    annotated_grammaticalities = []
    for batch in tqdm(batches):
        tokenized = tokenizer(list(batch), padding=True).to(device)

        input_ids = torch.tensor(tokenized.input_ids)
        attention_mask = torch.tensor(tokenized.attention_mask)

        predicted_class_ids = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1)
        batch_grammaticalities = predicted_class_ids.bool()
        if model in MODELS_ACCEPTABILITY_JUDGMENTS_INVERTED:
            batch_grammaticalities = ~batch_grammaticalities
        batch_grammaticalities = np.array(batch_grammaticalities).astype(object)

        annotated_grammaticalities.extend(batch_grammaticalities.tolist())

    grammaticalities[(num_words > 1)] = annotated_grammaticalities

    return grammaticalities


def clean_preprocessed_utterance(utterance):
    final_punctuation = None
    while len(utterance) > 0 and utterance[-1] in [".", "!", "?"]:
        final_punctuation = utterance[-1]
        utterance = utterance[:-1]

    utt_clean = remove_events_and_non_parseable_words(utterance)

    # Remove underscores
    utt_clean = utt_clean.replace("_", " ")

    # Remove spacing before commas and double commas
    utt_clean = utt_clean.replace(" ,", ",")
    utt_clean = utt_clean.replace(",,", ",")

    # Transform to lower case and strip:
    utt_clean = utt_clean.lower().strip()
    utt_clean = utt_clean.replace("  ", " ")

    # Remove remaining commas at beginning and end of utterance
    while len(utt_clean) > 0 and utt_clean[0] == ",":
        utt_clean = utt_clean[1:].strip()
    while len(utt_clean) > 0 and utt_clean[-1] == ",":
        utt_clean = utt_clean[:-1].strip()

    if final_punctuation:
        utt_clean += final_punctuation
    else:
        utt_clean += "."

    return utt_clean


def annotate(args):
    utterances = pd.read_pickle(UTTERANCES_WITH_SPEECH_ACTS_FILE)

    # TODO remove:
    utterances = utterances[utterances.speaker_code == "CHI"]

    print("Annotating speech-relatedness..")
    utterances = utterances.assign(
        is_speech_related=utterances.transcript_raw.apply(
            is_speech_related,
            label_partially_speech_related=args.label_partially_speech_related,
        )
    )
    utterances.is_speech_related = utterances.is_speech_related.astype("boolean")

    print("Annotating intelligibility..")
    utterances = utterances.assign(
        is_intelligible=utterances.transcript_raw.apply(
            is_intelligible,
            label_partially_intelligible=args.label_partially_intelligible,
        )
    )

    print("Cleaning utterances..")
    utterances = utterances.assign(
        utt_clean=utterances.transcript_raw.apply(
            clean_preprocessed_utterance
        )
    )

    # num_words = np.array([len(re.split('\s|\'', utt)) for utt in utterances.utt_clean.values])
    # utts_to_annoatate = utterances[(num_words > 1)]
    # utterances = utts_to_annoatate.sample(1000, random_state=1)

    #
    print("Annotating grammaticality..")
    tokenizer = AutoTokenizer.from_pretrained(args.grammaticality_annotation_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.grammaticality_annotation_model).to(device)
    utterances["is_grammatical"] = annotate_grammaticality(utterances.utt_clean.values, utterances.gra.values,
                                                           tokenizer, model)
    utterances.is_grammatical = utterances.is_grammatical.astype("boolean")

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
    argparser.add_argument(
        "--grammaticality-annotation-model",
        type=str,
        default=DEFAULT_MODEL_GRAMMATICALITY_ANNOTATION,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    annotated_utts = annotate(args)

    file_path_with_model = f"{ANNOTATED_UTTERANCES_FILE.split('.p')[0]}_{args.grammaticality_annotation_model.replace('/', '_')}.p"
    os.makedirs(os.path.dirname(ANNOTATED_UTTERANCES_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(file_path_with_model), exist_ok=True)

    annotated_utts.to_pickle(ANNOTATED_UTTERANCES_FILE)
    annotated_utts.to_pickle(file_path_with_model)
    annotated_utts.to_csv(file_path_with_model.replace(".p", ".csv"))
