import argparse
import math
import os
import re

import pandas as pd
import pylangacq
import numpy as np

from utils import get_path_of_utterances_file, remove_whitespace

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

# 1s response threshold
DEFAULT_RESPONSE_THRESHOLD = 1000  # ms


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--response-latency",
        type=int,
        default=DEFAULT_RESPONSE_THRESHOLD,
        help="Response latency in milliseconds",
    )
    args = argparser.parse_args()

    return args


def find_child_utterances_and_responses(
    corpus, transcripts, response_latency_threshold
):
    utterances = []

    file_paths = transcripts.file_paths()

    ages = transcripts.ages(months=True)

    # Get target child names (prepend corpus name to make the names unique)
    child_names = [
        header["Participants"][SPEAKER_CODE_CHILD]["corpus"]
        + "_"
        + header["Participants"][SPEAKER_CODE_CHILD]["name"]
        if SPEAKER_CODE_CHILD in header["Participants"] else None
        for header in transcripts.headers()
    ]

    utts_by_file = transcripts.utterances(
        by_files=True,
    )

    for file, age, child_name, utts in zip(file_paths, ages, child_names, utts_by_file):
        # Filter out empty transcripts and transcripts without age or child information
        if len(utts) < 0:
            print("Empty transcript: ", file)
            continue
        if age is None or age == 0:
            print("Missing age information: ", file)
            continue
        if child_name == None:
            print("Missing child name information: ", file)
            continue

        # Make a dataframe
        utts = pd.DataFrame(
            [
                {
                    "speaker_code": utt.participant,
                    "utt": utt.tiers[utt.participant],
                    "start_time": utt.time_marks[0],
                    "end_time": utt.time_marks[1],
                }
                for utt in utts if utt.time_marks
            ]
        )
        if len(utts) == 0:
            continue

        utts["speaker_code_next"] = utts.speaker_code.shift(-1)
        utts["start_time_next"] = utts.start_time.shift(-1)

        # Find all child utterances
        utts_child = utts[
            utts.speaker_code == SPEAKER_CODE_CHILD
        ]

        # Filter for utterances that don't have a child follow-up within the response latency threshold
        utts_child_without_child_follow_up = utts_child[
            ~(
                (utts_child.speaker_code_next == SPEAKER_CODE_CHILD)
                & (
                    utts_child.start_time_next
                    < utts_child.end_time + response_latency_threshold
                )
            )
        ]

        for candidate_id in utts_child_without_child_follow_up.index.values:
            utt1 = utts.loc[candidate_id]
            if candidate_id + 1 not in utts.index:
                continue
            subsequent_utt = utts.loc[candidate_id + 1]
            if subsequent_utt.speaker_code in SPEAKER_CODES_CAREGIVER:
                utt2 = subsequent_utt
            elif subsequent_utt.speaker_code in SPEAKER_CODE_CHILD:
                latency = round(subsequent_utt["start_time"] - utt1["end_time"], 3)
                assert latency >= response_latency_threshold
                following_utts = utts.loc[subsequent_utt.name + 1 :]
                following_utts_non_child = following_utts[
                    following_utts.speaker_code != SPEAKER_CODE_CHILD
                ]
                if (
                    len(following_utts_non_child) > 0
                    and following_utts_non_child.iloc[0].speaker_code
                    in SPEAKER_CODES_CAREGIVER
                ):
                    # The child didn't receive a response within the threshold, and continues to talk
                    # We add a dummy caregiver response with infinite start_time in this case
                    utt2 = {
                        "utt": "",
                        "start_time": math.inf,
                        "end_time": math.inf,
                        "speaker_code": SPEAKER_CODES_CAREGIVER[-1],
                    }
                else:
                    # Child is not speaking to its caregiver, ignore this turn
                    continue
            else:
                # Child is not speaking to its caregiver, ignore this turn
                continue

            following_utts = utts.loc[utt1.name + 1 :]
            following_utts_child = following_utts[
                following_utts.speaker_code == SPEAKER_CODE_CHILD
            ]
            if len(following_utts_child) > 0:
                utt3 = following_utts_child.iloc[0]
                response_latency = round(utt2["start_time"] - utt1["end_time"], 3)
                if utt2["end_time"] == math.inf:
                    # If there was not caregiver response, we calculate the response latency starting from the first utt
                    response_latency_follow_up = round(utt3["start_time"] - utt1["end_time"], 3)
                else:
                    response_latency_follow_up = round(utt3["start_time"] - utt2["end_time"], 3)

                # Remove timing information from utterance
                utt_child = re.sub(r"[^]+?", "", utt1["utt"])
                utt_caregiver = re.sub(r"[^]+?", "", utt2["utt"])
                utt_child_follow_up = re.sub(r"[^]+?", "", utt3["utt"])

                # Remove punctuation and whitespace
                utt_child = remove_whitespace(re.sub(r"[\.!\?]+\s*$", "", utt_child))
                utt_caregiver = remove_whitespace(
                    re.sub(r"[\.!\?]+\s*$", "", utt_caregiver)
                )
                utt_child_follow_up = remove_whitespace(
                    re.sub(r"[\.!\?]+\s*$", "", utt_child_follow_up)
                )

                # Prepend previous utterance of the child if it was uttered right before
                if (candidate_id - 1) in utts.index:
                    previous_utt_child = utts.loc[candidate_id - 1]
                    if (
                        previous_utt_child.speaker_code == SPEAKER_CODE_CHILD
                        and not np.isnan(previous_utt_child.end_time)
                        and (
                            utt1.start_time - previous_utt_child.end_time
                            < response_latency_threshold
                        )
                    ):
                        previous_utt_child = re.sub(
                            r"[^]+?", "", previous_utt_child["utt"]
                        )
                        previous_utt_child = remove_whitespace(
                            re.sub(r"[\.!\?]+\s*$", "", previous_utt_child)
                        )

                        utt_child = previous_utt_child + " " + utt_child

                # Append subsequent utterance of the caregiver if it was uttered right after
                if utt2["start_time"] < math.inf and (utt2.name + 1) in utts.index:
                    subsequent_utt_caregiver = utts.loc[utt2.name + 1]
                    if (
                        subsequent_utt_caregiver.speaker_code in SPEAKER_CODES_CAREGIVER
                        and not np.isnan(subsequent_utt_caregiver.start_time)
                        and (
                            subsequent_utt_caregiver.start_time - utt2.end_time
                            < response_latency_threshold
                        )
                    ):
                        subsequent_utt_caregiver = re.sub(
                            r"[^]+?", "", subsequent_utt_caregiver["utt"]
                        )
                        subsequent_utt_caregiver = remove_whitespace(
                            re.sub(r"[\.!\?]+\s*$", "", subsequent_utt_caregiver)
                        )

                        utt_caregiver = utt_caregiver + " " + subsequent_utt_caregiver

                # Append subsequent utterance of the child follow-up if it was uttered right after
                if (utt3.name + 1) in utts.index:
                    subsequent_utt_child_follow_up = utts.loc[utt3.name + 1]
                    if (
                        subsequent_utt_child_follow_up.speaker_code
                        == SPEAKER_CODE_CHILD
                        and not np.isnan(subsequent_utt_child_follow_up.start_time)
                        and (
                            subsequent_utt_child_follow_up.start_time - utt3.end_time
                            < response_latency_threshold
                        )
                    ):
                        subsequent_utt_child_follow_up = re.sub(
                            r"[^]+?", "", subsequent_utt_child_follow_up["utt"]
                        )
                        subsequent_utt_child_follow_up = remove_whitespace(
                            re.sub(r"[\.!\?]+\s*$", "", subsequent_utt_child_follow_up)
                        )

                        utt_child_follow_up = (
                            utt_child_follow_up + " " + subsequent_utt_child_follow_up
                        )

                # if response_latency > response_latency_threshold:
                #     print(f"{utt1.speaker_code}: {utt1["utt"]}")
                #     print(f"pause: {response_latency}")
                #     print(f"{utt2['speaker_code']}: {utt2['utt']}")
                #     print(f"{utt3.speaker_code}: {utt3["utt"]}\n")
                utterances.append(
                    {
                        "response_latency": response_latency,
                        "response_latency_follow_up": response_latency_follow_up,
                        "age": round(age),
                        "corpus": corpus,
                        "transcript_file": file,
                        "child_name": child_name,
                        "utt_child": utt_child,
                        "utt_car": utt_caregiver,
                        "utt_child_follow_up": utt_child_follow_up,
                    }
                )

    utterances = pd.DataFrame(utterances)

    return utterances


def preprocess_transcripts(response_latency):
    all_utterances = pd.DataFrame()
    for corpus in CANDIDATE_CORPORA:
        print(f"Reading transcripts of {corpus} corpus.. ", end="")
        transcripts = pylangacq.read_chat(
            os.path.expanduser(f"~/data/CHILDES/{corpus}/"),
            clean=False,
            parse_morph_info=False,
        )
        print("done.")

        print(f"Searching for child utterances and responses.. ", end="")
        utterances_transcript = find_child_utterances_and_responses(
            corpus, transcripts, response_latency
        )
        print("done.")

        all_utterances = all_utterances.append(utterances_transcript, ignore_index=True)

    return all_utterances


if __name__ == "__main__":
    args = parse_args()

    preprocessed_utterances = preprocess_transcripts(args.response_latency)
    file_name = get_path_of_utterances_file(args.response_latency)
    preprocessed_utterances.to_csv(file_name, index=False)
