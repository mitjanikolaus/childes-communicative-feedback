import argparse
import itertools
import os
import re

import pandas as pd
import pylangacq
from tqdm import tqdm
tqdm.pandas()

from utils import (
    is_empty,
    POS_PUNCTUATION,
    PREPROCESSED_UTTERANCES_FILE,
    SPEAKER_CODE_CHILD,
    get_all_paralinguistic_events,
    remove_punctuation,
    get_paralinguistic_event,
    paralinguistic_event_is_external, clean_utterance, remove_timing_information, SPEAKER_CODES_CAREGIVER,
    replace_actually_said_words,
)

NAMES_PATH = "data/names.csv"
CHILDES_DATA_BASE_PATH = os.path.expanduser(f"~/data/CHILDES/")


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--corpora",
        nargs="+",
        type=str,
        required=True,
        help="Corpora to analyze.",
    )
    argparser.add_argument(
        "--require-timing-information",
        default=False,
        action="store_true",
    )
    argparser.add_argument(
        "--out",
        default=PREPROCESSED_UTTERANCES_FILE,
        type=str,
        help="Path to store output file",
    )
    args = argparser.parse_args()

    return args


def get_pos_tag(tag):
    if tag:
        return str(tag).lower()
    else:
        return None


def get_gra_tag(tag):
    if tag:
        return {"dep": tag.dep, "head": tag.head, "rel": tag.rel}
    else:
        return None


NAMES = pd.read_csv(NAMES_PATH, index_col=0).sort_values(by="percent", ascending=False)


def replace_untranscribed_names(utterances):
    """Use common English names to replace untranscribed names"""
    words_not_transcribed = []
    for utt in utterances.transcript_raw.values:
        words = utt.split(" ")
        words_not_transcribed.extend(list(itertools.chain(*[re.findall("[_().A-Za-z]+www", word) for word in words])))

    name_templates = sorted(set(words_not_transcribed))

    name_dict = dict()
    for name in name_templates:
        name_no_www = re.sub("(www[w]*)+_[W]*www[w]*|www[w]*", "", name)
        name_no_www = name_no_www.replace("(.)", "")
        first_letter = name_no_www[-1].upper()
        if len(name_no_www) == 2:
            first_letter = name_no_www[0].upper()
        candidates = NAMES[NAMES.name.str.startswith(first_letter)].name

        prefix = ""
        candidate_2 = None
        if len(name_no_www) > 2:
            prefix = name_no_www
            if "_" in name:
                prefix = name.split("_")[0]
            if first_letter == "_":
                if prefix in ["Mrs", "Miss", "Aunty", "Auntie", "Missus", "Jeannine", "Gina", "Eleanor",
                              "Granny"]:
                    candidates = NAMES[NAMES.sex == "girl"].name
                elif prefix in ["Uncle", "Mr"]:
                    candidates = NAMES[NAMES.sex == "boy"].name
                else:
                    candidates = NAMES.name
        if prefix[:-1] in ["Mummypig", "Uncle", "Auntie"]:
            prefix = prefix[:-1]
        if "www" in prefix:
            first_letter_2 = re.sub("(www[w]*)+_[W]*www[w]*|www[w]*", "", prefix)[-1].upper()
            candidates_2 = NAMES[NAMES.name.str.startswith(first_letter_2)].name
            candidate_2 = candidates_2.iloc[0]
        i = 0
        candidate = (prefix + " " + candidates.iloc[i]).strip()
        if candidate_2:
            candidate = candidate_2 + " " + candidates.iloc[i]
        while candidate in name_dict.values():
            i += 1
            candidate = (prefix + " " + candidates.iloc[i]).strip()
            if candidate_2:
                candidate = candidate_2 + " " + candidates.iloc[i]
        name_dict[name] = candidate

    def replace_names(utt):
        matches = re.findall("[_().A-Za-z]+www", utt)
        matches = sorted(matches, key=len, reverse=True)
        for match in matches:
            if match not in ["eeeowww", "miaoowww"]:
                new_name = name_dict[match]
                if "_" in match and not "www" in match.split("_")[0]:
                    new_name = match.split("_")[0] + " " + new_name
                utt = utt.replace(match, new_name)
        return utt

    utterances["transcript_raw"] = utterances.transcript_raw.apply(replace_names)
    return utterances


def add_error_codes_from_actually_said_words(row):
    if "actually says" in row["transcript_raw"]:
        utt_replaced, error_codes = replace_actually_said_words(row["transcript_raw"])
        row["transcript_clean"] = utt_replaced
        if pd.isna(row["error"]) or row["error"] == "":
            row["error"] = ";".join(error_codes)
        else:
            row["error"] += ";" + ";".join(error_codes)

    return row


def preprocess_utterances(corpus, transcripts, start_index, args):
    file_paths = transcripts.file_paths()

    ages = transcripts.ages(months=True)

    # Get target child names (prepend corpus name to make the names unique)
    child_names = [
        header["Participants"][SPEAKER_CODE_CHILD]["corpus"]
        + "_"
        + header["Participants"][SPEAKER_CODE_CHILD]["name"]
        if SPEAKER_CODE_CHILD in header["Participants"]
        else None
        for header in transcripts.headers()
    ]

    utts_by_file = transcripts.utterances(
        by_files=True,
    )

    all_utts = []

    for file, age, child_name, utts_transcript in tqdm(zip(
        file_paths, ages, child_names, utts_by_file
    ), total=len(file_paths)):
        # Filter out empty transcripts and transcripts without age or child information
        if len(utts_transcript) == 0:
            continue
        if "Interview" in file:
            # Interview transcripts do not contain child-caregiver interactions
            continue
        if age is None or age == 0:
            # Child age can sometimes be read from the file name
            if corpus in ["MPI-EVA-Manchester", "Bernstein", "Brent", "Braunwald", "Weist", "MacWhinney"]:
                age_info = os.path.basename(file).split(".cha")[0]
                age = int(age_info[0:2])*12 + int(age_info[2:4])
            elif corpus == "Rollins":
                age_info = os.path.basename(file).split(".cha")[0]
                age = int(age_info[2:4])
            elif corpus == "Tommerdahl" and os.path.basename(file) == "MEH2.cha":
                age = 39    # Missing value copied from "MEH2.cha"
            elif corpus == "Gleason" and file.endswith("Father/eddie.cha"):
                age = 52    # Missing value taken from https://childes.talkbank.org/access/Eng-NA/Gleason.html
            else:
                print("Missing age information: ", file)
                continue
        if child_name is None:
            # Child is not present in transcript, ignore
            continue

        file_path = file.replace(CHILDES_DATA_BASE_PATH, "")

        # Make a dataframe
        utts_transcript = pd.DataFrame(
            [
                {
                    "utterance_id": id,
                    "speaker_code": utt.participant,
                    "transcript_raw": utt.tiers[utt.participant],
                    "tokens": [
                        t.word.lower() for t in utt.tokens if t.word != "CLITIC"
                    ],
                    "pos": [
                        get_pos_tag(t.pos)
                        for t in utt.tokens
                        if t.pos not in POS_PUNCTUATION
                    ],
                    "gra": [
                        get_gra_tag(t.gra)
                        for t in utt.tokens
                    ],
                    "start_time": utt.time_marks[0] if utt.time_marks else pd.NA,
                    "end_time": utt.time_marks[1] if utt.time_marks else pd.NA,
                    "age": round(age),
                    "corpus": corpus,
                    "transcript_file": file_path,
                    "child_name": child_name,
                    "error": utt.tiers["%err"] if "%err" in utt.tiers.keys() else pd.NA
                }
                for id, utt in enumerate(utts_transcript)
            ]
        )

        if len(utts_transcript[utts_transcript.speaker_code == SPEAKER_CODE_CHILD]) == 0:
            # Child is not present in transcript, ignore
            continue
        if len(set(utts_transcript.speaker_code.unique()) & set(SPEAKER_CODES_CAREGIVER)) ==  0:
            # Caregiver is not present in transcript, ignore
            continue

        if args.require_timing_information:
            # Verify that we have at least timing information for some of the utterances
            if len(utts_transcript["start_time"].dropna()) == 0:
                continue

        utts_transcript.dropna(subset=["transcript_raw"], inplace=True)
        utts_transcript["transcript_raw"] = utts_transcript["transcript_raw"].apply(
            remove_timing_information
        )

        utts_transcript = utts_transcript[
            ~utts_transcript.transcript_raw.apply(has_multiple_events)
        ]
        utts_transcript = utts_transcript[
            ~utts_transcript.transcript_raw.apply(is_external_event)
        ]
        utts_transcript = utts_transcript[~utts_transcript.transcript_raw.apply(is_empty)]

        utts_transcript = utts_transcript[utts_transcript["transcript_raw"] != ""]
        utts_transcript.dropna(subset=["transcript_raw", "speaker_code"], inplace=True)

        all_utts.append(utts_transcript)

    if not all_utts:
        return start_index

    utterances = pd.concat(all_utts, ignore_index=True)

    utterances = replace_untranscribed_names(utterances)

    utterances["transcript_clean"] = utterances["transcript_raw"]
    utterances = utterances.apply(add_error_codes_from_actually_said_words, axis=1)
    utterances["transcript_clean"] = utterances.transcript_clean.apply(clean_utterance)

    utterances.index = pd.RangeIndex(start_index, start_index+len(utterances))
    if start_index == 0:
        utterances.to_csv(args.out)
    else:
        utterances.to_csv(args.out, mode="a", header=False)

    return start_index+len(utterances)


def has_multiple_events(utterance):
    return len(get_all_paralinguistic_events(utterance)) > 1


def is_external_event(utterance):
    utterance = remove_punctuation(utterance)

    event = get_paralinguistic_event(utterance)
    if event and paralinguistic_event_is_external(event) and utterance == event:
        return True

    return False


def preprocess_transcripts(args):
    all_utterances = []
    start_index = 0
    for corpus in args.corpora:
        print(f"Reading transcripts of {corpus} corpus.. ", end="")
        transcripts = pylangacq.read_chat(
            os.path.join(CHILDES_DATA_BASE_PATH, corpus),
        )
        print("done.")

        print(f"Preprocessing utterances of {corpus}.. ")
        start_index = preprocess_utterances(corpus, transcripts, start_index, args)

    return all_utterances


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if not args.out.endswith(".csv"):
        raise ValueError("Out file should have .csv ending!")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    preprocess_transcripts(args)
