import argparse
import os
from ast import literal_eval

import pandas as pd

from utils import UTTERANCES_WITH_SPEECH_ACTS_FILE

OMISSIONS_DATA_FILE = os.path.expanduser(
    "data/utterances_omission.csv"
)


def has_omission(tokens):
    for token in tokens:
        if token.startswith("0"):
            return True
    return False


def get_omission_type(row):
    errors = []
    for token, gra in zip(row["tokens"], row["gra"]):
        if token.startswith("0"):
            rel = gra["rel"]

            if rel in ['SUBJ']:
                errors.append("subject")
            elif rel in ["OBJ", "OBJ2"]:
                errors.append("object")
            elif rel in ["ROOT", "CSUBJ", "COBJ", "CPRED", "CPOBJ", "POBJ", "SRL", "PRED", "XJCT", "CJCT", "CMOD", "XMOD", "COMP"]:
                errors.append("verb")
            elif rel in ["DET"]:
                errors.append("determiner")
            elif rel in ["JCT", "NJCT"]: # Adjunct
                errors.append("preposition")
            elif rel in ["INF"]: # "to" for infintive verbs
                errors.append("infinitive")
            elif rel in ["AUX"]:
                errors.append("auxiliary")
            elif rel in ["LP", "PUNCT"]:
                print(f"{rel}: {row['tokens']}")
                errors.append("???")
            elif rel in ["INCROOT", "OM", "NEG", "LINK", "CONJ", "QUANT", "PQ", "ENUM", "MOD", "COORD", "COM", "DATE", "NAME", "POSTMOD"]:
                errors.append("other")
            else:
                print(rel)

                # if word in ["det", "a", "an", "the"]:
                #     errors.append("determiner")
                # elif word in ["is", "am", "are", "was", "v", "be", "looks", "said", "let", "wanna", "think", "come", "look", "brought", "take", "know", "go", "going", "put", "like", "say", "play", "got", "want", "been", "remember", "hear", "says", "make", "see", "need", "went", "goes", "give", "get"]:
                #     errors.append("verb")
                # elif word in ["will", "do", "does", "can", "has", "had", "did", "could", "may", "gonna", "should", "shall", "would"]:
                #     errors.append("auxiliary")
                # elif word in ["it", "where", "he", "they", "ya", "we", "pro", "you"]:
                #     errors.append("subject")
                # elif word in ["truck", "here"]:
                #     errors.append("object")
                # elif word in ["to", "of", "at", "off", "up", "in", "on", "from", "as", "for", "about", "with"]:
                #     errors.append("preposition")
                # elif word in ["isn't"]:
                #     errors.append("verb")
                #     errors.append("other")
                # elif word in ["let's"]:
                #     errors.append("verb")
                #     errors.append("object")
                # elif word in ["we'll", "what's", "that's", "it's", "there's"]:
                #     errors.append("subject")
                #     errors.append("verb")
                # elif word in ["not", "their", "well", "out", "if", "his", "ones", "how", "oh", "lot", "all", "us", "house", "there", "more", "ton", "class", "brush", "enough", "and", "alright", "which", "little", "her", "chair", "shirt", "apron", "zero", "x"]:
                #     errors.append("other")
                # elif word in {'donald', 'someone', 'she', 'any', 'just', 'who', 'then', 'one', 'what', 'many', 'went', 'birthday', 'have', 'baby', 'were', 'down', 'or', 'you', 'your', 'n', 'day', 'thing', 'when', 'i', 'me', 'as', 'this', 'for', 'room', 'babies', 'him', 'okay', 'with', 'about', 'wrist', 'give', 'pies', 'so', 'bear', 'time', 'why', 'e', 'good', 'no', 'my', 'right', 'some', 'get', 'pro', 'that', 'them'}:
                #     errors.append("other")
                # else:
                #     print(f"Warning: Word not covered: {word}")

    if len(errors) > 1:
        print(errors)
        return str(", ".join(errors))[1:-1]
    else:
        return errors[0]


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, converters={"pos": literal_eval, "tokens": literal_eval, "gra": literal_eval})
    utterances["is_grammatical"] = ~utterances.tokens.apply(has_omission)
    utterances = utterances[~utterances.is_grammatical]
    utterances["labels"] = utterances.apply(get_omission_type, axis=1)

    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=UTTERANCES_WITH_SPEECH_ACTS_FILE,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = prepare(args)

    os.makedirs(os.path.dirname(OMISSIONS_DATA_FILE), exist_ok=True)
    utterances.to_csv(OMISSIONS_DATA_FILE)
