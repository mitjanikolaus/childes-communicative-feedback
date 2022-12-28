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
            word = token[1:]
            if rel in ['SUBJ']:
                errors.append("subject")
            elif rel in ["OBJ", "OBJ2"]:
                errors.append("object")
            elif rel in ["ROOT", "CSUBJ", "COBJ", "CPRED", "CPOBJ", "POBJ", "SRL", "PRED", "XJCT", "CJCT", "CMOD", "XMOD", "COMP"]:
                errors.append("verb")
            elif rel in ["DET"]:
                errors.append("determiner")
            elif rel in ["JCT", "NJCT"]:
                errors.append("preposition")
            elif rel in ["INF"]:    # "to" for infintive verbs
                errors.append("verb")
            elif rel in ["AUX"]:
                errors.append("auxiliary")
            elif rel in ["LP", "PUNCT", "INCROOT", "OM", "NEG", "LINK", "CONJ", "QUANT", "PQ", "ENUM", "MOD", "COORD", "COM", "DATE", "NAME", "POSTMOD"]:
                # Fallback: check whether we can guess the category by looking at the actual omitted word
                if word in ["det", "a", "an", "the"]:
                    errors.append("determiner")
                elif word in ["is", "am", "are", "were", "was", "v", "be", "want", "like", "see", "know", "need", "think", "come", "put", "said", "says", "play", "look", "make"]:
                    errors.append("verb")
                elif word in ["will", "had", "do", "does", "did", "have", "has", "can", "may", "would", "could"]:
                    errors.append("auxiliary")
                elif word in ["to", "of", "at", "off", "up", "in", "on", "from", "as", "for", "about", "with"]:
                    errors.append("preposition")
                elif word in ["i", "pro", "you", "it", "they", "he", "we"]:
                    errors.append("subject")
                else:
                    errors.append("other")
            else:
                print(f"Warning: Rel not covered: {rel}")

    if len(errors) > 1:
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
