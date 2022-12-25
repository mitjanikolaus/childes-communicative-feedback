import os
import re

import pandas as pd

from utils import (
    remove_superfluous_annotations,
    is_empty,
    get_all_paralinguistic_events,
    remove_punctuation,
    get_paralinguistic_event,
    paralinguistic_event_is_external, clean_utterance,
)

DATA_PATH = "data/hiller_fernandez_2016/data/annotated_data"
HILLER_FERNANDEZ_DATA_OUT_PATH = "data/hiller_fernandez_2016/hiller_fernandez_preprocessed.csv"

LABEL_TRANSFORMATION = {
    'other': "other",
    'synt:subj': "subject",
    'synt:obj': "object",
    'synt:verb': "verb",
    'umorph:auxverb': "auxiliary",
    'umorph:det': "determiner",
    'vmorph:regpast': "past",
    'vmorph:other': "other",
    'umorph:prep': "preposition",
    'vmorph:3rdpers': "sv_agreement",
    'nmorph:poss': "possessive",
    'vmorph:irrpast': "past",
    'umorph:presprogr': "present_progressive",
    'nmorph:regplural': "plural",
    'nmorph:irrplural': "plural",
    'umorph:other': "other"
}


def preprocess_utterances(file_path):
    child_name = file_path.split("/")[-2].lower()

    all_utts = []

    utt = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            if line == "\n":
                if utt:
                    if not "response_transcript_raw" in utt.keys():
                        print(utt)
                    all_utts.append(utt)
                utt = {}
            ident = line[:5]
            if ident == "*CHI:":
                utt["transcript_raw"] = line[5:]
            elif ident in ["*CAR:", "*MOT:", "*INV:", "*DAD:", "*FAT:", "*GRM:", "*GRF:", "*DAD:"]:
                utt["response_transcript_raw"] = line[5:]
            elif ident == "%cof:":
                if "$ERR" not in line:
                    # error not annotated, skip example
                    utt = None
                    continue
                if "$ERR = 0" in line:
                    utt["is_grammatical"] = True
                    utt["labels"] = None
                else:
                    utt["is_grammatical"] = False
                    errors = re.split("\$ERR = ", line)[1:]
                    errors = [re.split(";", error)[0].replace("\n", "") for error in errors]
                    errors = [LABEL_TRANSFORMATION[error] for error in errors]
                    utt["labels"] = ", ".join(errors)

    utts_transcript = pd.DataFrame(
        [
            {
                "utterance_id": id,
                "transcript_raw": utt["transcript_raw"],
                "response_transcript_raw": utt["response_transcript_raw"],
                "is_grammatical": utt["is_grammatical"],
                "labels": utt["labels"],
                "transcript_file": file_path,
                "child_name": child_name,
            }
            for id, utt in enumerate(all_utts)
        ]
    )

    utts_transcript["transcript_raw"] = utts_transcript["transcript_raw"].apply(
        remove_superfluous_annotations
    )
    utts_transcript["response_transcript_raw"] = utts_transcript["response_transcript_raw"].apply(
        remove_superfluous_annotations
    )

    utts_transcript["transcript_clean"] = utts_transcript.transcript_raw.apply(clean_utterance)
    utts_transcript["response_transcript_clean"] = utts_transcript.response_transcript_raw.apply(clean_utterance)

    return utts_transcript


def has_multiple_events(utterance):
    return len(get_all_paralinguistic_events(utterance)) > 1


def is_external_event(utterance):
    utterance = remove_punctuation(utterance)

    event = get_paralinguistic_event(utterance)
    if event and paralinguistic_event_is_external(event) and utterance == event:
        return True

    return False


def preprocess_transcripts():
    all_utterances = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".cex"):
                utterances_transcript = preprocess_utterances(os.path.join(root, file))
                all_utterances.append(utterances_transcript)

    all_utterances = pd.concat(all_utterances, ignore_index=True)

    return all_utterances


if __name__ == "__main__":
    preprocessed_utterances = preprocess_transcripts()

    os.makedirs(os.path.dirname(HILLER_FERNANDEZ_DATA_OUT_PATH), exist_ok=True)
    preprocessed_utterances.to_csv(HILLER_FERNANDEZ_DATA_OUT_PATH)
