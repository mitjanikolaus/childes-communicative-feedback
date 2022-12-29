import argparse
import os
import re
from ast import literal_eval

import pandas as pd

from utils import PREPROCESSED_UTTERANCES_FILE

CHILDES_ERRORS_DATA_FILE = os.path.expanduser(
    "data/utterances_errors.csv"
)


def get_errors(row):
    errors = get_omission_errors(row)
    errors += get_other_error(row)

    # if len(errors) == 0 and "[*" in row["transcript_raw"] and "[:" not in row["transcript_raw"]:
    #     print(row["transcript_raw"])
    #     print(row["transcript_file"])
        # matches = re.findall(r"\[\*[^]]*]", utt)

    if len(errors) == 0:
        return None
    elif len(errors) == 1:
        return errors[0]
    else:
        return str(", ".join(errors))[1:-1]


def get_other_error(row):
    utt = row["transcript_raw"]
    matches = re.finditer(r"\[: [^]]*]", utt)

    errors = []
    for match in matches:
        word = match.group()[2:-1].strip()
        prev_word = utt[:match.start()].strip().split(" ")[-1]
        if prev_word == word:
            continue
        elif prev_word == word[:-2] and word[-2:] == "'s":
            errors.append("possessive")
        elif (prev_word, word) in [('boy', 'boys')]:
            errors.append("possessive")
        elif (prev_word, word) in [('mans', 'men'), ('mans@c', 'men'), ('mens', 'men'), ('womans', 'women'), ('firemens', 'firemen'), ('firemans', 'firemen'),  ('foremans', 'foremen'), ('mouses', 'mice'), ('sheeps', 'sheep'), ('foots', 'feet'), ('foots', 'foot'), ('knifes', 'knives'), ('‹barefeeted', 'barefoot')]:
            errors.append("plural")
        elif (prev_word, word) in [('putted', 'put'), ('falled', 'fell'), ('maked', 'made'), ('dugged', 'dug'), ('waked', 'woke'), ('bended', 'bent'), ('flyed', 'flew'), ('felled', 'fell'), ('runned', 'ran'), ('stucked', 'stuck'), ('sleeped', 'slept'), ('shutted', 'shut'), ('babysitted', 'babysat'), ('broked', 'broken'), ('sawded', 'sawed'), ('readed', 'read'), ('sitted', 'sat'), ('eated', 'ate'), ('haved', 'had'), ('flied', 'flew'), ('growed', 'grew'), ('swimmed', 'swam'), ('flieded', 'flied'), ('getted', 'got'), ('ated', 'ate'), ('sayed', 'said'), ('throwed', 'threw'), ('forgat', 'forgot'), ('losed', 'lost'), ('bringed', 'brought'), ('goed', 'went'), ('broked', 'broke'), ('drawed', 'drew'), ('comed', 'came'), ('leapt', 'leaped'), ('drinked', 'drank'), ('et', 'ate'), ('sleept', 'slept'), ('gat', 'got'), ('leant', 'leaned'), ('broke', 'broken'), ('breaked', 'broke'), ('doed', 'did'), ('maded', 'made'), ('fixeded', 'fixed'), ('sawed', 'saw'), ('camed', 'came'), ('catched', 'caught'), ('seed', 'saw'), ('telled', 'told'), ('written', 'wrote'), ('flewed', 'flew'), ('goed', 'went like'), ('goed', 'said'), ('bited', 'bit'), ('sended', 'sent'), ('taked', 'took'), ('hided', 'hid'), ('hitted', 'hit'), ('brokened', 'broken'), ('swammed', 'swam'), ('stunged', 'stung'), ('goned', 'gone'), ('builded', 'built'), ('sanged', 'sang'), ('blewed', 'blew'), ('cwied', 'cried'), ('founded', 'found'), ('dranked', 'drank'), ('slided', 'slid'), ('bented', 'bent')]:
            errors.append("past")
        elif (prev_word, word) in [('a', 'an'), ('a', 'the')]:
            errors.append("determiner")
        elif (prev_word, word) in [('coughes', 'coughs')]:
            errors.append("verb")
        elif (prev_word, word) in [('is', 'are'), ('gots', 'got'), ("here's", 'here are')]:
            errors.append("sv_agreement")
        elif (prev_word, word) in [('a', 'of'), ('at', 'to'), ('want', 'want to')]:
            errors.append("preposition")
        elif (prev_word, word) in [('him', 'he'), ('my', 'I')]:
            errors.append("subject")
        elif (prev_word, word) in [('theirselves', 'themselves'), ('themself', 'themselves'), ('hisself', 'himself')]:
            errors.append("other")

    return errors


def get_omission_errors(row):
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

    return errors


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, converters={"pos": literal_eval, "tokens": literal_eval, "gra": literal_eval})

    utterances["labels"] = utterances.apply(get_errors, axis=1)

    utterances["is_grammatical"] = ~utterances.labels.isna()
    utterances = utterances[~utterances.is_grammatical]
    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=PREPROCESSED_UTTERANCES_FILE,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = prepare(args)

    os.makedirs(os.path.dirname(CHILDES_ERRORS_DATA_FILE), exist_ok=True)
    utterances.to_csv(CHILDES_ERRORS_DATA_FILE)
