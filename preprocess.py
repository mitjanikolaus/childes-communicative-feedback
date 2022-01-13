import argparse
import os

import pandas as pd
import pylangacq

from utils import clean_utterance, POS_PUNCTUATION, PREPROCESSED_UTTERANCES_FILE

SPEAKER_CODE_CHILD = "CHI"

SPEAKER_CODES_CAREGIVER = [
    "MOT",
    "FAT",
    "DAD",
    "MOM",
    "GRA",
    "GRF",
    "GRM",
    "GMO",
    "GFA",
    "CAR",
]
# Corpora that are conversational (have child AND caregiver transcripts), are English, and have timing information
CANDIDATE_CORPORA = [
    "Edinburgh",
    "VanHouten",
    "MPI-EVA-Manchester",
    "McMillan",
    "Rollins",
    "Gleason",
    "Forrester",
    "Braunwald",
    "Bloom",
    "McCune",
    "Tommerdahl",
    "Soderstrom",
    "Weist",
    "NewmanRatner",
    "Snow",
    "Thomas",
    "Peters",
    "MacWhinney",
    "Sachs",
    "Bernstein",
    "Brent",
    "Nelson",
    "Providence",
]


def parse_args():
    argparser = argparse.ArgumentParser()
    args = argparser.parse_args()

    return args


def get_pos_tag(tag):
    tag = str(tag).lower()
    return tag


def preprocess_utterances(corpus, transcripts):
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

    for file, age, child_name, utts_transcript in zip(
        file_paths, ages, child_names, utts_by_file
    ):
        # Filter out empty transcripts and transcripts without age or child information
        if len(utts_transcript) == 0:
            # print("Empty transcript: ", file)
            continue
        if age is None or age == 0:
            print("Missing age information: ", file)
            continue
        if child_name is None:
            print("Missing child name information: ", file)
            continue

        # Make a dataframe
        utts_transcript = pd.DataFrame(
            [
                {
                    "utterance_id": id,
                    "speaker_code": utt.participant,
                    "transcript_raw": utt.tiers[utt.participant],
                    "tokens": [t.word.lower() for t in utt.tokens if t.word != "CLITIC"],
                    "pos": [get_pos_tag(t.pos) for t in utt.tokens if t.pos not in POS_PUNCTUATION],
                    "start_time": utt.time_marks[0] if utt.time_marks else None,
                    "end_time": utt.time_marks[1] if utt.time_marks else None,
                    "age": round(age),
                    "corpus": corpus,
                    "transcript_file": file,
                    "child_name": child_name,
                }
                for id, utt in enumerate(utts_transcript)
            ]
        )

        if len(utts_transcript) == 0:
            continue

        # Verify that we have at least timing information for some of the utterances
        if len(utts_transcript["start_time"].dropna()) == 0:
            continue

        utts_transcript["transcript_raw"] = utts_transcript["transcript_raw"].apply(
            clean_utterance
        )

        utts_transcript = utts_transcript[utts_transcript["transcript_raw"] != ""]
        utts_transcript.dropna(subset=["transcript_raw", "speaker_code"], inplace=True)

        utts_transcript["speaker_code_next"] = utts_transcript.speaker_code.shift(-1)
        utts_transcript["start_time_next"] = utts_transcript.start_time.shift(-1)

        all_utts.append(utts_transcript)

    utterances = pd.concat(all_utts, ignore_index=True)

    return utterances


def preprocess_transcripts():
    all_utterances = []
    for corpus in CANDIDATE_CORPORA:
        print(f"Reading transcripts of {corpus} corpus.. ", end="")
        transcripts = pylangacq.read_chat(
            os.path.expanduser(f"~/data/CHILDES/{corpus}/"),
        )
        print("done.")

        print(f"Preprocessing utterances.. ", end="")
        utterances_corpus = preprocess_utterances(corpus, transcripts)
        print("done.")

        all_utterances.append(utterances_corpus)

    all_utterances = pd.concat(all_utterances, ignore_index=True)

    return all_utterances


if __name__ == "__main__":
    args = parse_args()

    preprocessed_utterances = preprocess_transcripts()

    os.makedirs(os.path.dirname(PREPROCESSED_UTTERANCES_FILE), exist_ok=True)
    preprocessed_utterances.to_pickle(PREPROCESSED_UTTERANCES_FILE)
