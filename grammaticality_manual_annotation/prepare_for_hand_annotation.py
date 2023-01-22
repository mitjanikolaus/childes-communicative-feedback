import argparse
import os

import numpy as np
import pandas as pd

from utils import (
    SPEAKER_CODE_CHILD,
    SPEAKER_CODES_CAREGIVER, ANNOTATED_UTTERANCES_FILE, filter_for_min_num_utts, PROJECT_ROOT_DIR
)

from tqdm import tqdm
tqdm.pandas()


TOKEN_CHILD = "[CHI]"
TOKEN_CAREGIVER = "[CAR]"
TOKEN_OTHER = "[OTH]"


MIN_NUM_WORDS = 1
CORPORA_INCLUDED = ['Thomas', 'MPI-EVA-Manchester', 'Providence', 'Braunwald', 'Lara', 'EllisWeismer']


def speaker_code_to_special_token(code):
    if code == SPEAKER_CODE_CHILD:
        return TOKEN_CHILD
    elif code in SPEAKER_CODES_CAREGIVER:
        return TOKEN_CAREGIVER
    else:
        return TOKEN_OTHER


def get_utts_to_annotate(utterances):
    utts_to_annotate = utterances[(utterances.speaker_code == SPEAKER_CODE_CHILD)]
    utts_to_annotate = filter_for_min_num_utts(utts_to_annotate, MIN_NUM_WORDS)
    return utts_to_annotate


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, dtype={"error": object})

    utterances = utterances.iloc[args.num_utts_ignore:]

    base_path = PROJECT_ROOT_DIR+"/data/manual_annotation/transcripts"

    for corpus in CORPORA_INCLUDED:
        utterances_corpus = utterances[utterances.corpus == corpus].copy()
        transcripts = utterances_corpus.transcript_file.unique()
        np.random.seed(2)
        transcript = np.random.choice(transcripts)
        print("Filtering for only speech-like utterances")
        utts_transcript = filter_for_min_num_utts(
            utterances_corpus[(utterances_corpus.transcript_file == transcript) & (utterances_corpus.is_speech_related == True)],
            MIN_NUM_WORDS).copy()

        utts_to_annotate = get_utts_to_annotate(utts_transcript)

        while len(utts_to_annotate) < args.num_utts:
            transcript = np.random.choice(transcripts)
            utts_transcript = filter_for_min_num_utts(utterances_corpus[(utterances_corpus.transcript_file == transcript) & (utterances_corpus.is_speech_related == True)], MIN_NUM_WORDS).copy()
            utts_to_annotate = get_utts_to_annotate(utts_transcript)

        print("Transcript: ", transcript)
        utts_transcript["utterance"] = utts_transcript["speaker_code"].apply(speaker_code_to_special_token) + " " + utts_transcript["transcript_clean"]

        utts_transcript["is_grammatical"] = ""
        utts_transcript["labels"] = ""
        utts_transcript["note"] = ""

        index_max = utts_to_annotate.iloc[args.num_utts - 1].name
        utts_transcript = utts_transcript.loc[:index_max]

        utts_transcript.loc[utts_transcript.index.isin(utts_to_annotate.index), "is_grammatical"] = "TODO"

        print("Num utts to annotate: ", len(utts_transcript.loc[utts_transcript.index.isin(utts_to_annotate.index)]))
        print("Total num utts: ", len(utts_transcript))

        utts_transcript = utts_transcript[["utterance", "is_grammatical", "labels", "note"]]

        transcript_name = transcript.replace("/", "_")

        file_name = os.path.join(base_path, transcript_name.replace(".cha", ".csv"))
        utts_transcript.to_csv(file_name)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=ANNOTATED_UTTERANCES_FILE,
    )
    argparser.add_argument(
        "--num-utts-ignore",
        type=int,
        default=0,
        help="First x utts to ignore (already annotated)"
    )
    argparser.add_argument(
        "--num-utts",
        type=int,
        default=100,
        help="Number of utts to annotate"
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    prepare(args)
