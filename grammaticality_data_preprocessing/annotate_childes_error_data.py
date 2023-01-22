import argparse
import itertools
import os
import re
from ast import literal_eval

import pandas as pd
import matplotlib.pyplot as plt

from grammaticality_data_preprocessing.analyze_childes_error_data import plot_corpus_error_stats
from utils import categorize_error, ERR_VERB, ERR_AUXILIARY, ERR_PREPOSITION, \
    ERR_SUBJECT, ERR_OBJECT, ERR_POSSESSIVE, ERR_SV_AGREEMENT, ERR_DETERMINER, ERR_UNKNOWN, \
    remove_superfluous_annotations, \
    ERR_TENSE_ASPECT, ERR_PLURAL, UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, \
    ERR_OTHER, UTTERANCES_WITH_SPEECH_ACTS_FILE, split_into_words, WORDS_WRONG_TENSE_ASPECT_INFLECTION, \
    WORDS_WRONG_PLURAL_INFLECTION, clean_utterance
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
    if "[*]" in utt and not re.findall("\[\* [^]]*]", utt):
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
            elif word in ["pos", "i"]:
                errors.append(ERR_SUBJECT)
            elif word in ["m:=ed", "m:+ed", "+ed", "-ed", "m:ed", "m", "m:m", "m:", "forgot", "fell"]:
                errors.append(ERR_TENSE_ASPECT)
            elif word in ["+is"]:
                errors.append(ERR_VERB)
            elif word in ["m:+s", "m:0s", "her"]:
                errors.append(ERR_POSSESSIVE)
            elif word in["m:=s", "-s", "+s"]:
                errors.append(ERR_PLURAL)
            else:
                errors.append(ERR_UNKNOWN)

        return errors

NOUNS_THIRD_PERSON = ["daddy", "that", "this", "lion", "what", "he", "she", "it"]
VERBS_NOT_THIRD_PERSON = ["go", "happen", "do", "want", "can", "come", "like", "have"]
NOUN_VERB_ILLEGAL = [" ".join(t) for t in itertools.product(NOUNS_THIRD_PERSON, VERBS_NOT_THIRD_PERSON)]


def get_error_from_whole_utt(row):
    utt = row["transcript_raw"].lower()
    before = clean_utterance(utt.split("[*]")[0].strip())
    split_previous = split_into_words(before, split_on_apostrophe=False, remove_commas=True, remove_trailing_punctuation=True)
    prev_word = split_previous[-1] if split_previous else ""
    after = clean_utterance(utt.split("[*]")[1].strip())
    split_following = split_into_words(after, split_on_apostrophe=False, remove_commas=True, remove_trailing_punctuation=True)
    following_word = split_following[0] if split_following else ""

    if prev_word in ["here", "there", "me"] and following_word in ["are", "go", "do"]:
        return ERR_SUBJECT

    for t in ["what are them [*]", "do [*] like this", "can me [*]"]:
        if t in utt:
            return ERR_OBJECT

    if prev_word in ["a"] and following_word and following_word[0] in ["a", "e", "i", "o", "u"]:
        return ERR_DETERMINER

    if prev_word in ["this"] and following_word in ["beginning"]:
        return ERR_DETERMINER

    if prev_word in AUXILIARIES and following_word in VERBS:
        return ERR_PREPOSITION

    if prev_word in ["in", "on", "next", "one"] and following_word in ["the", "my"]:
        return ERR_PREPOSITION

    if prev_word in ["it"] and following_word in ["me"]:
        return ERR_PREPOSITION

    for t in NOUN_VERB_ILLEGAL + ["i weren't [*]", "they goes [*]", "is [*] there", "what's [*] these", "go [*] there", "there's [*] two", "don't [*] fit"]:
        if t in utt:
            return ERR_SV_AGREEMENT

    for t in ["what [*] that say", "i [*] done it", "i done [*] it", "i not [*]", "not eat it [*]", "where [*] that go", "where [*] this go", "what [*] he", "what [*] she", "i [*] gonna", "is [*] that go", "why [*] you"]:
        if t in utt:
            return ERR_AUXILIARY

    if prev_word in ["i", "you", "he", "she", "we", "they"] and following_word in ["done", "go", "do", "finished", "show", "get", "finish", "found", "broken", "put", "be", "got", "make", "like", "gone", "read", "just", "already"]:
        return ERR_AUXILIARY

    for t in ["two bed [*]", "two box [*]", "more ball [*]", "a scissors"]:
        if t in utt:
            return ERR_PLURAL

    if re.search("\[\*] \S*[\s\S]+ing", utt):
        return ERR_TENSE_ASPECT

    if prev_word in VERBS + AUXILIARIES and following_word in ["again", "in", "a", "here"]:
        return ERR_OBJECT

    if prev_word in VERBS + AUXILIARIES and following_word in ["cheese", "barbie", "elephant", "orange", "green", "yellow",
                                                               "banana", "tissue", "small", "sandwich", "sun", "bottle",
                                                               "horse", "crayon", "puzzle", "noisy", "tent", "fairy", "cup"]:
        return ERR_DETERMINER

    if prev_word in SUBJECTS + ["one"] and following_word in SUBJECTS + OBJECTS + DETERMINERS + ['purple', 'for', 'here', 'broken', 'mine', 'these', 'not', 'better', 'pilchard', 'my', 'his', 'our', 'my', 'your', 'not', 'no', 'lots', 'dizzy']:
        return ERR_VERB

    for t in ["here it [*]", "what [?] [*] that", "where [*] dada", "who's [*] a girl", "<a@p this> [*]"]:
        if t in utt:
            return ERR_VERB

    if prev_word in ["much", "many"]:
        return ERR_OTHER

    words = set(split_into_words(row["transcript_clean"].lower(), split_on_apostrophe=False, remove_commas=True, remove_trailing_punctuation=True))
    if len(words & WORDS_WRONG_TENSE_ASPECT_INFLECTION) > 0:
        return ERR_TENSE_ASPECT
    if len (words & WORDS_WRONG_PLURAL_INFLECTION) > 0:
        return ERR_PLURAL

    # if row["gra"]:
    #     all_rels = set([gra["rel"] for gra in row["gra"]])
        # if len(set(RELS_VERB) & all_rels) == 0:
        #     # print(f"Verb error: {utt} ") #({all_rels})
        #     return ERR_VERB
        # if len(set(RELS_SUBJECT) & all_rels) == 0:
        #     print(f"Subject error: {utt} ({all_rels})") #
        #     return ERR_SUBJECT

    return ERR_OTHER


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
            continue
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
            errs = categorize_error(word_error, word_corrected, row)
            errors.extend(errs)

    return errors


VERBS = ["is", "am", "are", "were", "was", "v", "be", "want", "like", "see", "know", "need", "think", "come",
          "put", "said", "says", "play", "look", "make", "let", "let's", "go", "ran", "got", "came", "hear", "get",
          "brought", "going", "gonna", "grunts", "fussing", "fusses", "whines", "squeals", "yells", "singing",
         "turn", "whining", "sighs", "laughs", "saw", "'re", "'m", "eat", "re", "do", "shake", "pick", "cut", "use",
         "sit", "smell", "drink", "have", "help", "speak", "tip", "stick", "take", "flush", "try", "hold", "dress",
         "carry", "watch", "read", "color", "find", "listen", "open", "draw", "sleep", "press", "stay", "tie", "leave",
         "count", "wait"]

AUXILIARIES = ["will", "had", "do", "does", "did", "have", "has", "hav", "can", "may", "would", "could", "shall", "'ve",
                  "'ll", "want"]

PREPOSITIONS = ["prep", "to", "of", "at", "off", "up", "in", "on", "from", "as", "for", "about", "with", "out"]

DETERMINERS = ["det", "a", "an", "the", "one", "some", "all", "more", "any", "many"]

SUBJECTS = ['you', 'that', 'how', 'why', 'when', 'i', 'she', 'he', 'there', 'who', 'where', 'mummy', 'dada', 'daddy', 'jacob', 'it', 'this', 'they', 'pro', 'what', 'we']

OBJECTS = ["them", "her", "me", "myself", "him"]

WORDS_OTHER = ["and", "if", "not", "no", "or", "because"]


def guess_omission_error_types(word, utt, prev_word=None):
    word = word.lower().replace(":", "")
    if word in DETERMINERS or word == "n" and prev_word == "a":
        error = ERR_DETERMINER
    elif word in AUXILIARIES:
        error = ERR_AUXILIARY
    elif word in VERBS:
        error = ERR_VERB
    elif word in PREPOSITIONS:
        error = ERR_PREPOSITION
    elif word in SUBJECTS:
        error = ERR_SUBJECT
    elif word in OBJECTS:
        error = ERR_OBJECT
    elif word in ["'s"] and prev_word in SUBJECTS:
        error = ERR_VERB
    elif word in ["'s", "0's", "my", "his", "your"]:
        error = ERR_POSSESSIVE
    elif word in ["ing"]:
        error = ERR_TENSE_ASPECT
    elif word in ["ed", "en", "ne", "n", "ten", "ped"]:
        error = ERR_TENSE_ASPECT
    elif word in ["es", "es'nt"] or word in ["s"] and prev_word in VERBS:
        error = ERR_SV_AGREEMENT
    elif word in ["s"]:
        error = ERR_PLURAL
    elif word in WORDS_OTHER:
        error = ERR_OTHER
    elif len(word.split("'")) > 0 and word.split("'")[0] in SUBJECTS and word.split("'")[1] in ["s", "m", "ve"]:
        errors = [ERR_SUBJECT, ERR_VERB]
        return errors
    elif word in ["zero", "x", "etc"]:
        error = ERR_OTHER
    elif word in ["."]:
        return []
    else:
        error = ERR_OTHER
    return [error]


RELS_SUBJECT = ["SUBJ"]
RELS_VERB = ["ROOT", "CSUBJ", "COBJ", "CPRED", "CPOBJ", "POBJ", "SRL", "PRED", "XJCT", "CJCT", "CMOD", "XMOD", "COMP"]


def get_omission_errors(row):
    errors = []
    for token, gra in zip(row["tokens"], row["gra"]):
        if token.startswith("0"):
            word = token[1:]
            if not gra or "rel" not in gra.keys():
                errs = guess_omission_error_types(word, row)
                errors.extend(errs)
            else:
                rel = gra["rel"]
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
                    errors.append(ERR_PREPOSITION)
                elif rel in ["AUX"]:
                    errors.append(ERR_AUXILIARY)
                else:
                    # Fallback: check whether we can guess the category by looking at the actual omitted word
                    errs = guess_omission_error_types(word, row)
                    errors.extend(errs)

    return errors


def annotate(utterances):
    utterances["labels"] = utterances.progress_apply(get_error_labels, axis=1)

    print("Num unknown errors: ", len(utterances[utterances["labels"] == ERR_UNKNOWN]))

    def is_grammatical(labels):
        if pd.isna(labels):
            return True
        elif labels == ERR_UNKNOWN:
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
        default=UTTERANCES_WITH_SPEECH_ACTS_FILE,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = pd.read_csv(args.utterances_file, index_col=0, converters={"pos": literal_eval, "tokens": literal_eval, "gra": literal_eval}, dtype={"error": object})
    utterances = annotate(utterances)

    os.makedirs(os.path.dirname(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE), exist_ok=True)
    utterances.to_csv(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE)

    plot_corpus_error_stats(utterances)
    plt.show()
