import argparse
import os

import pandas as pd

from utils import (
    SPEAKER_CODE_CHILD, get_num_unique_words,
    UTTERANCES_WITH_PREV_UTTS_FILE
)

from tqdm import tqdm
tqdm.pandas()


TO_ANNOTATE_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_for_annotation.csv"
)


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0)

    if args.annotated_utterances_file:
        annotated_utts = pd.read_csv(args.annotated_utterances_file, index_col=0)

        utterances.dropna(subset=["prev_transcript_clean"], inplace=True)

        utterances["is_grammatical"] = pd.NA
        utterances["labels"] = pd.NA
        utterances["note"] = pd.NA

        def find_match(utt):
            same_transcript = utterances[utterances.transcript_file == utt.transcript_file]
            same_id = same_transcript[same_transcript.utterance_id == utt.utterance_id]
            if len(same_id) > 0:
                utterances.loc[same_id.iloc[0].name, "is_grammatical"] = utt.is_grammatical
                utterances.loc[same_id.iloc[0].name, "labels"] = utt.labels
                utterances.loc[same_id.iloc[0].name, "note"] = utt.note
                if not same_id.iloc[0].transcript_clean == utt.transcript_clean:
                    print()
                    print(same_id.iloc[0].transcript_raw)
                    print(same_id.iloc[0].transcript_clean)
                    print(utt.transcript_clean)
                    print()
            else:
                print(f"utt not found: id {utt.utterance_id} in {utt.transcript_file}")

        annotated_utts.progress_apply(find_match, axis=1)

        utterances.dropna(subset=["is_grammatical"], inplace=True)

        print(f"len before drop: {len(utterances)}")
        utterances.drop_duplicates(subset=["transcript_clean", "prev_transcript_clean"], inplace=True)
        print(f"len after drop: {len(utterances)}")

    utts_to_annotate = utterances[utterances.speaker_code == SPEAKER_CODE_CHILD]

    num_unique_words = get_num_unique_words(utts_to_annotate.transcript_clean)
    utts_to_annotate = utts_to_annotate[(num_unique_words > 1)]
    utts_to_annotate = utts_to_annotate[utts_to_annotate.is_speech_related & utts_to_annotate.is_intelligible]

    if args.max_utts:
        utts_to_annotate = utts_to_annotate.sample(args.max_utts, random_state=1)

    return utts_to_annotate


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=UTTERANCES_WITH_PREV_UTTS_FILE,
    )
    argparser.add_argument(
        "--annotated-utterances-file",
        type=str,
        required=False,
    )
    argparser.add_argument(
        "--max-utts",
        type=int,
        default=None,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances = prepare(args)

    os.makedirs(os.path.dirname(TO_ANNOTATE_UTTERANCES_FILE), exist_ok=True)
    utterances.to_csv(TO_ANNOTATE_UTTERANCES_FILE)
