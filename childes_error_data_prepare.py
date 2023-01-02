import argparse
import os
import re
from ast import literal_eval

import pandas as pd

from utils import categorize_error, ANNOTATED_UTTERANCES_FILE, ERR_VERB, ERR_AUXILIARY, ERR_PREPOSITION, \
    ERR_SUBJECT, ERR_OBJECT, ERR_POSSESSIVE, ERR_SV_AGREEMENT, ERR_DETERMINER, ERR_UNKNOWN, \
    remove_superfluous_annotations, add_prev_utts
from tqdm import tqdm
tqdm.pandas()


CHILDES_ERRORS_DATA_FILE = os.path.expanduser(
    "data/utterances_errors.csv"
)


def get_errors(row):
    errors_tier = get_errors_marked_on_tier(row)
    errors_omission = get_omission_errors(row)
    errors_colon = get_errors_marked_with_colon(row)

    errors = set(errors_tier + errors_omission + errors_colon)

    if len(errors) == 0:
        return pd.NA
    elif len(errors) == 1:
        return errors.pop()
    else:
        return str(", ".join(errors))


def get_errors_marked_on_tier(row):
    error_tier = row["error"]

    if pd.isna(error_tier):
        return []

    errors = error_tier.split(";")
    out_errors = []
    for error in errors:
        error = error.replace(".", "")
        error = re.sub("<\s*\S+\s*>", "", error)
        error = error.strip()

        if error in ["[?]", "?", ""]:
            out_errors.append(ERR_UNKNOWN)
        elif error == "self correction":
            continue
        elif error in ["break in syllable", "breaks in syllable"]:
            continue  # disfluency
        elif error in ["$WR", "$MWR"]:
            continue  # word repetition
        elif re.fullmatch("0[\S+\s*]+=[\s*\S+]+", error):
            # omission error with full word
            word = re.search("=\s*\S+", error)[0][1:].strip()
            if " " in word:
                word = word.split(" ")[0]
            out_errors.extend(guess_omission_error_types(word, row))
        elif "=" not in error and "0" in error:
            word = error.replace("0", "")
            out_errors.extend(guess_omission_error_types(word, row))
        else:
            if re.fullmatch("\S+\s*0[\S+\s*]+=[\s*\S+]+", error):
                # omission error with partial word
                word = re.search("\S+\s*0[\S+\s*]+=", error)[0][:-1]
                word_error = word.split("0")[0].strip().lower()
                word_corrected = word.replace("0", "").strip().lower()
            elif re.fullmatch("\S+\s*=\s*\S+", error):
                word_error = error.split("=")[0].strip().lower()
                word_corrected = error.split("=")[1].replace("0", "").strip().lower()
            elif "=" in error:
                word_error = error.split("=")[0].replace("0", "").strip().lower()
                word_corrected = error.split("=")[1].replace("0", "").strip().lower()
            else:
                # print(f"uncategorized error: {error} | {row['transcript_raw']}")
                continue

            errs = categorize_error(word_error, word_corrected, row)
            out_errors.extend(errs)

    return out_errors


def get_errors_marked_with_colon(row):
    utt = row["transcript_raw"]
    matches = re.finditer(r"\[: [^]]*]", utt)

    errors = []
    for match in matches:
        word_corrected = match[0][2:-1].strip().lower()
        word_corrected = remove_superfluous_annotations(word_corrected)
        word_error = utt[:match.start()].strip()
        word_error = remove_superfluous_annotations(word_error)
        word_error = word_error.split(" ")[-1].lower()
        if word_error == word_corrected:
            continue
        else:
            errors.extend(categorize_error(word_error, word_corrected, row))

    return errors


def guess_omission_error_types(word, utt):
    word = word.lower()
    if word in ["what's", "that's", "i'm", "i've", "he's", "it's", "there's"]:
        errors = [ERR_SUBJECT, ERR_VERB]
        return errors
    elif word in ["det", "a", "an", "the"]:
        error = ERR_DETERMINER
    elif word in ["is", "am", "are", "were", "was", "v", "be", "want", "like", "see", "know", "need", "think", "come",
                  "put", "said", "says", "play", "look", "make", "let's", "go", "ran", "got", "came", "hear", "get",
                  "brought", "going"]:
        error = ERR_VERB
    elif word in ["will", "had", "do", "does", "did", "have", "has", "hav", "can", "may", "would", "could", "shall", "'ve",
                  "'ll"]:
        error = ERR_AUXILIARY
    elif word in ["to", "of", "at", "off", "up", "in", "on", "from", "as", "for", "about", "with"]:
        error = ERR_PREPOSITION
    elif word in ["i", "pro", "you", "it", "they", "she", "he", "we"]:
        error = ERR_SUBJECT
    elif word in ["them", "her", "me", "myself", "him"]:
        error = ERR_OBJECT
    elif word in ["'s", "0's", "my", "his"]:
        error = ERR_POSSESSIVE
    elif word in ["s"] and "says" in utt["tokens"]:
        error = ERR_SV_AGREEMENT
    else:
        error = ERR_UNKNOWN
    return [error]

RELS_VERB = ["ROOT", "CSUBJ", "COBJ", "CPRED", "CPOBJ", "POBJ", "SRL", "PRED", "XJCT", "CJCT", "CMOD", "XMOD", "COMP"]

def get_omission_errors(row):
    errors = []
    for token, gra in zip(row["tokens"], row["gra"]):
        if token.startswith("0"):
            rel = gra["rel"]
            word = token[1:]
            if rel in ['SUBJ']:
                errors.append(ERR_SUBJECT)
            elif rel in ["OBJ", "OBJ2"]:
                errors.append(ERR_OBJECT)
            elif rel in RELS_VERB:
                errors.append(ERR_VERB)
            elif rel in ["DET"]:
                errors.append(ERR_DETERMINER)
            elif rel in ["JCT", "NJCT"]:
                errors.append(ERR_PREPOSITION)
            elif rel in ["INF"]:    # "to" for infintive verbs
                errors.append(ERR_VERB)
            elif rel in ["AUX"]:
                errors.append(ERR_AUXILIARY)
            else:
                # Fallback: check whether we can guess the category by looking at the actual omitted word
                errs = guess_omission_error_types(word, row)
                if errs == [ERR_UNKNOWN]:
                    errs = []
                    all_rels = set([gra["rel"] for gra in row["gra"]])
                    if "SUBJ" not in all_rels:
                        errs.append(ERR_SUBJECT)
                    if len(set(RELS_VERB) & all_rels) == 0:
                        errs.append(ERR_VERB)
                    if len(errs) == 0:
                        errs = [ERR_UNKNOWN]
                errors.extend(errs)

    return errors


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, converters={"pos": literal_eval, "tokens": literal_eval, "gra": literal_eval}, dtype={"error": object})

    utterances["labels"] = utterances.apply(get_errors, axis=1)

    utterances["is_grammatical"] = utterances.labels.isna()

    print("Adding previous utterances..")
    utterances = add_prev_utts(utterances)

    utterances = utterances[~utterances.is_grammatical]
    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=ANNOTATED_UTTERANCES_FILE,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = prepare(args)

    os.makedirs(os.path.dirname(CHILDES_ERRORS_DATA_FILE), exist_ok=True)
    utterances.to_csv(CHILDES_ERRORS_DATA_FILE)
