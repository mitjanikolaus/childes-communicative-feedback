import argparse
import os
import re
from ast import literal_eval

import pandas as pd
import matplotlib.pyplot as plt

from grammaticality_data_preprocessing.analyze_childes_error_data import plot_corpus_error_stats
from utils import categorize_error, ERR_VERB, ERR_AUXILIARY, ERR_PREPOSITION, \
    ERR_SUBJECT, ERR_OBJECT, ERR_POSSESSIVE, ERR_SV_AGREEMENT, ERR_DETERMINER, ERR_UNKNOWN, \
    remove_superfluous_annotations, \
    ERR_PRESENT_PROGRESSIVE, ERR_PAST, ERR_PLURAL, UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, \
    ERR_OTHER, UTTERANCES_WITH_PREV_UTTS_FILE
from tqdm import tqdm
tqdm.pandas()


def get_error_labels(row):
    errors_tier = get_errors_marked_on_tier(row)
    errors_omission = get_omission_errors(row)
    errors_colon = get_errors_marked_with_colon(row)
    errors = set(errors_tier + errors_omission + errors_colon)

    if len(errors) == 0 and "[*" in row["transcript_raw"]:
        errors_star = get_errors_marked_with_star(row)
        errors = set(errors_star)

    if len(errors) > 1:
        if ERR_UNKNOWN in errors:
            errors.remove(ERR_UNKNOWN)

    if len(errors) == 0:
        return pd.NA
    elif len(errors) == 1:
        return errors.pop()
    else:
        return str(", ".join(errors))


def get_errors_marked_with_star(row):
    utt = row["transcript_raw"]
    if "[:" in utt:
        # Error with colons have already been processed
        return []
    if "[*]" in utt:
        err = get_error_from_whole_utt(row)
        return [err]
    else:
        errors = []
        matches = re.finditer("\[\* [^]]*]", utt)
        for m in matches:
            word = m[0][2:-1].strip()
            if word.startswith("0"):
                before = utt[0: m.span()[0]].strip()
                prev_word = before.split(" ")[-1]
                errs = guess_omission_error_types(word.replace("0", ""), row, prev_word)
                errors.extend(errs)
            elif "0" in word:
                word_error = word.split("0")[0].strip().lower()
                word_corrected = word.replace("0", "").strip().lower()
                errs = categorize_error(word_error, word_corrected, row)
                errors.extend(errs)
            elif word == "pos":
                errors.append(ERR_SUBJECT)
            elif word in ["m:=ed", "+ed", "-ed", "m:ed", "m"]:
                errors.append(ERR_PAST)
            elif word in["m:=s"]:
                errors.append(ERR_PLURAL)
            else:
                errors.append(ERR_UNKNOWN)

        return errors


def get_error_from_whole_utt(row):
    utt = row["transcript_raw"].lower()
    prev_word = utt.split("[*]")[0].strip().split(" ")[-1]
    following_word = utt.split("[*]")[1].strip().split(" ")[0]

    if prev_word in ["here", "there"] and following_word in ["are", "go"]:
        return ERR_SUBJECT

    for t in ["what are them [*]", "do [*] like this"]:
        if t in utt:
            return ERR_OBJECT

    if prev_word in ["do"] and following_word in ["again"]:
        return ERR_OBJECT

    if prev_word in ["a"] and following_word[0] in ["a", "e", "i", "o", "u"]:
        return ERR_DETERMINER

    if prev_word in ["this"] and following_word in ["beginning"]:
        return ERR_DETERMINER

    if prev_word in ["want"] and following_word in ["do", "play", "go", "put"]:
        return ERR_PREPOSITION

    for t in ["i done [*] it", "finish [*]", "fall [*] over"]:
        if t in utt:
            return ERR_PAST

    if prev_word in ["throwed", "falled", "telled"]:
        return ERR_PAST

    for t in ["that go [*]", "this go [*]", "lion go [*]", "i weren't [*]", "what's [*] these", "go [*] there", "what happen [*]", "he do [*]", "he want [*]"]:
        if t in utt:
            return ERR_SV_AGREEMENT

    for t in ["what [*] that say", "i [*] done it", "where [*] that go", "where [*] this go", "what [*] he", "what [*] she"]:
        if t in utt:
            return ERR_AUXILIARY

    if prev_word in ["i", "you", "he", "she", "we", "they"] and following_word in ["done", "go", "do", "finished", "show", "get", "finish", "found", "broken", "put", "be", "got", "make", "like", "gone", "read"]:
        return ERR_AUXILIARY

    if re.search("\[\*] \S*[\s\S]+ing", utt):
        return ERR_PRESENT_PROGRESSIVE

    if prev_word in ["what", "it", "i", "you", "he", "it", "that", "who", "where", "mummy", "they", "there", "this"] and following_word in \
            ["you", "here", "broken", "better", "mine", "that", "there", "these", "this", "not", "dada", "daddy", "jacob", "pilchard", "the", "my", "his", "her", "our", "your", "not", "no", "what", "all", "lots", "a", "dizzy"]:
        return ERR_VERB
    for t in ["here it [*]", "what [?] [*] that", "where [*] dada", "who's [*] a girl", "<a@p this> [*]"]:
        if t in utt:
            return ERR_VERB

    return ERR_UNKNOWN


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


VERBS = ["is", "am", "are", "were", "was", "v", "be", "want", "like", "see", "know", "need", "think", "come",
          "put", "said", "says", "play", "look", "make", "let", "let's", "go", "ran", "got", "came", "hear", "get",
          "brought", "going", "gonna", "grunts", "fussing", "fusses", "whines", "squeals", "yells", "singing",
         "turn", "whining", "sighs", "laughs", "saw", "'re", "'m", "eat", "re"]

AUXILIARIES = ["will", "had", "do", "does", "did", "have", "has", "hav", "can", "may", "would", "could", "shall", "'ve",
                  "'ll"]

PREPOSITIONS = ["prep", "to", "of", "at", "off", "up", "in", "on", "from", "as", "for", "about", "with", "out"]

DETERMINERS = ["det", "a", "an", "the", "one", "some", "all", "more", "any", "many"]

SUBJECTS = ["i", "pro", "you", "it", "they", "she", "he", "we", "what", "that", "there", "where", "when", "who", "how", "why"]

OBJECTS = ["them", "her", "me", "myself", "him"]

WORDS_OTHER = ["and", "if", "not", "no", "or", "because"]


def guess_omission_error_types(word, utt, prev_word=None):
    word = word.lower().replace(":", "")
    if word in DETERMINERS or word == "n" and prev_word == "a":
        error = ERR_DETERMINER
    elif word in VERBS:
        error = ERR_VERB
    elif word in AUXILIARIES:
        error = ERR_AUXILIARY
    elif word in PREPOSITIONS:
        error = ERR_PREPOSITION
    elif word in SUBJECTS:
        error = ERR_SUBJECT
    elif word in OBJECTS:
        error = ERR_OBJECT
    elif word in ["'s", "0's", "my", "his"]:
        error = ERR_POSSESSIVE
    elif word in ["ing"]:
        error = ERR_PRESENT_PROGRESSIVE
    elif word in ["ed", "en", "ne", "n", "ten", "ped"]:
        error = ERR_PAST
    elif (word in ["es", "es'nt"] or word in ["s"] and prev_word in VERBS):
        error = ERR_SV_AGREEMENT
    elif word in ["s"]:
        error = ERR_PLURAL
    elif word in WORDS_OTHER:
        error = ERR_OTHER
    elif len(word.split("'")) > 0 and word.split("'")[0] in SUBJECTS and word.split("'")[1] in ["s", "m", "ve"]:
        errors = [ERR_SUBJECT, ERR_VERB]
        return errors
    elif word in ["zero", "x", "etc"]:
        error = ERR_UNKNOWN
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
            elif rel in ["INF"]:    # "to" for infinitive verbs
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

    utterances["labels"] = utterances.apply(get_error_labels, axis=1)

    def is_grammatical(label):
        if pd.isna(label):
            return True
        if label == ERR_UNKNOWN:
            return pd.NA
        else:
            return False

    utterances["is_grammatical"] = utterances.labels.apply(is_grammatical).astype(object)

    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=UTTERANCES_WITH_PREV_UTTS_FILE,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = prepare(args)

    os.makedirs(os.path.dirname(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE), exist_ok=True)
    utterances.to_csv(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE)

    plot_corpus_error_stats(utterances)
    plt.show()
