import argparse
import os
import pandas as pd
import pylangacq

from utils import (
    PATH_ADJACENT_UTTERANCES,
)


SPEAKER_CODE_CHILD = "CHI"

SPEAKER_CODES_CAREGIVER = ["MOT", "FAT", "DAD", "MOM", "GRA", "GRF", "GRM", "CAR"]

CANDIDATE_CORPORA = [
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
    argparser.add_argument(
        "--corpora",
        nargs="+",
        type=str,
        required=False,
        choices=CANDIDATE_CORPORA,
        help="Corpora to analyze. If not given, corpora are selected based on a response time variance threshold.",
    )
    args = argparser.parse_args()

    return args


def find_child_caregiver_adjacent_utterances(corpus, transcripts):
    adjacent_utterances = []

    ages = transcripts.age(months=True)

    # Get target child names (prepend corpus name to make the names unique)
    target_child_names = {
        file: participants[SPEAKER_CODE_CHILD]["corpus"]
        + "_"
        + participants[SPEAKER_CODE_CHILD]["participant_name"]
        for file, participants in transcripts.participants().items()
        if SPEAKER_CODE_CHILD in participants
    }

    utts_child = transcripts.utterances(
        by_files=True,
        clean=False,
        time_marker=True,
        raise_error_on_missing_time_marker=False,
        phon=True,  # Setting phon to true to keep "xxx" and "yyy" in utterances
    )

    # Filter out empty transcripts and transcripts without age information
    utts_child = {
        file: utts
        for file, utts in utts_child.items()
        if (len(utts) > 0) and (ages[file] is not None) and (ages[file] != 0)
    }

    # Filter out transcripts without child information
    utts_child = {
        file: utts for file, utts in utts_child.items() if file in target_child_names
    }

    for file, utts in utts_child.items():
        age = ages[file]
        child_name = target_child_names[file]
        # print(f"Child: {child_name} Age: ", round(age))

        # Make a dataframe
        utts = pd.DataFrame(
            [
                {
                    "speaker_code": speaker,
                    "utt": utt,
                    "start_time": timing[0],
                    "end_time": timing[1],
                }
                for speaker, utt, timing in utts
            ]
        )
        utts["speaker_code_next"] = utts.speaker_code.shift(-1)
        utts["start_time_next"] = utts.start_time.shift(-1)

        # check that timing information is present
        utts_filtered = utts.dropna(subset=["end_time", "start_time_next"])

        # check for adjacency pairs Child-Caregiver
        utts_filtered = utts_filtered[
            (utts_filtered.speaker_code == SPEAKER_CODE_CHILD)
            & utts_filtered.speaker_code_next.isin(SPEAKER_CODES_CAREGIVER)
        ]

        for candidate_id in utts_filtered.index.values:
            utt1 = utts.loc[candidate_id]
            utt2 = utts.loc[candidate_id + 1 :].iloc[0]
            following_utts = utts.loc[utt2.name + 1 :]
            following_utts_child = following_utts[
                following_utts.speaker_code == SPEAKER_CODE_CHILD
            ]
            if len(following_utts_child) > 0:
                utt3 = following_utts_child.iloc[0]
                response_latency = round(utt2.start_time - utt1.end_time, 3)

                # if response_latency > RESPONSE_THRESHOLD:
                #     print(f"{utt1.speaker_code}: {utt1.utt}")
                #     print(f"Pause: {response_latency}")
                #     print(f"{utt2.speaker_code}: {utt2.utt}")
                #     print(f"{utt3.speaker_code}: {utt3.utt}\n")
                adjacent_utterances.append(
                    {
                        "response_latency": response_latency,
                        "age": round(age),
                        "corpus": corpus,
                        "transcript_file": file,
                        "child_name": child_name,
                        "utt_child": utt1.utt,
                        "utt_car": utt2.utt,
                        "utt_child_follow_up": utt3.utt,
                    }
                )

    adjacent_utterances = pd.DataFrame(adjacent_utterances)

    return adjacent_utterances


def preprocess_transcripts(corpora):
    adjacent_utterances = pd.DataFrame()
    for corpus in corpora:
        print(f"Reading transcripts of {corpus} corpus.. ", end="")
        transcripts = pylangacq.read_chat(
            os.path.expanduser(f"~/data/CHILDES/{corpus}/*.cha"),
            parse_morphology_information=False,
        )
        print("done.")
        adj_utterances_transcript = find_child_caregiver_adjacent_utterances(
            corpus, transcripts
        )

        adjacent_utterances = adjacent_utterances.append(
            adj_utterances_transcript, ignore_index=True
        )

    return adjacent_utterances


if __name__ == "__main__":
    args = parse_args()

    adjacent_utterances = preprocess_transcripts(CANDIDATE_CORPORA)
    adjacent_utterances.to_csv(PATH_ADJACENT_UTTERANCES, index=False)
