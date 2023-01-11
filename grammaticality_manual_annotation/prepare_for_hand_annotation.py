import argparse
import os

import pandas as pd

from utils import (
    SPEAKER_CODE_CHILD, get_num_unique_words,
    FILE_GRAMMATICALITY_ANNOTATIONS, SPEAKER_CODES_CAREGIVER
)

from tqdm import tqdm
tqdm.pandas()


TO_ANNOTATE_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_for_annotation.csv"
)

TOKEN_CHILD = "[CHI]"
TOKEN_CAREGIVER = "[CAR]"
TOKEN_OTHER = "[OTH]"


def speaker_code_to_special_token(code):
    if code == SPEAKER_CODE_CHILD:
        return TOKEN_CHILD
    elif code in SPEAKER_CODES_CAREGIVER:
        return TOKEN_CAREGIVER
    else:
        return TOKEN_OTHER


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, dtype={"error": object})

    utterances = utterances.iloc[args.num_utts_ignore:]

    # utterances.dropna(subset=["is_grammatical"], inplace=True)

    utterances = utterances[utterances.prev_speaker_code.isin(SPEAKER_CODES_CAREGIVER + [SPEAKER_CODE_CHILD])]

    utterances["utterance"] = utterances["speaker_code"].apply(speaker_code_to_special_token) + " " + utterances["transcript_clean"]
    utterances["previous_utterance"] = utterances["prev_speaker_code"].apply(speaker_code_to_special_token) + " " + utterances["prev_transcript_clean"]

    # utterances["is_grammatical"] = ""
    # utterances["labels"] = ""
    # utterances["note"] = ""

    utterances = utterances[utterances.speaker_code == SPEAKER_CODE_CHILD]

    num_unique_words = get_num_unique_words(utterances.transcript_clean)
    utterances = utterances[(num_unique_words > 1)]
    utterances = utterances[utterances.is_speech_related & utterances.is_intelligible]

    utterances = utterances[["previous_utterance", "utterance", "is_grammatical", "labels", "note"]]
    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=FILE_GRAMMATICALITY_ANNOTATIONS,
    )
    argparser.add_argument(
        "--num-utts-ignore",
        type=int,
        default=400,
        help="First x utts to ignore (already annotated)"
    )
    argparser.add_argument(
        "--num-utts",
        type=int,
        default=200,
        help="Number of utts to annotate"
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = prepare(args)

    os.makedirs(os.path.dirname(TO_ANNOTATE_UTTERANCES_FILE), exist_ok=True)
    utterances.to_csv(TO_ANNOTATE_UTTERANCES_FILE)
