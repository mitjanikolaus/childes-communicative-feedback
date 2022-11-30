import argparse
import os

import pandas as pd
import pylangacq
from tqdm import tqdm

from utils import (
    remove_superfluous_annotations,
    is_empty,
    POS_PUNCTUATION,
    PREPROCESSED_UTTERANCES_FILE,
    SPEAKER_CODE_CHILD,
    get_all_paralinguistic_events,
    remove_punctuation,
    get_paralinguistic_event,
    paralinguistic_event_is_external, clean_utterance,
)


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


def preprocess_utterances(corpus, transcripts, args):
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
            # print("Empty transcript: ", file)
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

        if len(utts_transcript[utts_transcript.speaker_code == SPEAKER_CODE_CHILD]) == 0:
            # Child is not present in transcript, ignore
            continue

        if args.require_timing_information:
            # Verify that we have at least timing information for some of the utterances
            if len(utts_transcript["start_time"].dropna()) == 0:
                continue

        utts_transcript["transcript_raw"] = utts_transcript["transcript_raw"].apply(
            remove_superfluous_annotations
        )
        utts_transcript.dropna(subset=("transcript_raw",), inplace=True)

        utts_transcript = utts_transcript[
            ~utts_transcript.transcript_raw.apply(has_multiple_events)
        ]
        utts_transcript = utts_transcript[
            ~utts_transcript.transcript_raw.apply(is_external_event)
        ]
        utts_transcript = utts_transcript[~utts_transcript.transcript_raw.apply(is_empty)]

        utts_transcript["transcript_clean"] = utts_transcript.transcript_raw.apply(clean_utterance)

        utts_transcript = utts_transcript[utts_transcript["transcript_raw"] != ""]
        utts_transcript.dropna(subset=["transcript_raw", "speaker_code"], inplace=True)

        utts_transcript["speaker_code_next"] = utts_transcript.speaker_code.shift(-1)
        utts_transcript["start_time_next"] = utts_transcript.start_time.shift(-1)

        all_utts.append(utts_transcript)

    utterances = pd.concat(all_utts, ignore_index=True)

    return utterances


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
    for corpus in args.corpora:
        print(f"Reading transcripts of {corpus} corpus.. ", end="")
        transcripts = pylangacq.read_chat(
            os.path.expanduser(f"~/data/CHILDES/{corpus}/"),
        )
        print("done.")

        print(f"Preprocessing utterances of {corpus}.. ")
        utterances_corpus = preprocess_utterances(corpus, transcripts, args)

        all_utterances.append(utterances_corpus)

    all_utterances = pd.concat(all_utterances, ignore_index=True)

    return all_utterances


if __name__ == "__main__":
    args = parse_args()
    print(args)

    preprocessed_utterances = preprocess_transcripts(args)

    os.makedirs(os.path.dirname(PREPROCESSED_UTTERANCES_FILE), exist_ok=True)
    preprocessed_utterances.to_pickle(PREPROCESSED_UTTERANCES_FILE)
