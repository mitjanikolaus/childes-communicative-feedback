import os
import re

import pandas as pd

from utils import (
    remove_superfluous_annotations,
    clean_utterance, ERR_OTHER, ERR_SUBJECT, ERR_OBJECT, ERR_VERB, ERR_AUXILIARY, ERR_DETERMINER, ERR_TENSE_ASPECT,
    ERR_PREPOSITION, ERR_SV_AGREEMENT, ERR_POSSESSIVE, ERR_PLURAL, PROJECT_ROOT_DIR, ERR_PRESENT_PROGRESSIVE,
)

DATA_PATH = PROJECT_ROOT_DIR+"/data/hiller_fernandez_2016/data/annotated_data"
HILLER_FERNANDEZ_DATA_OUT_PATH = PROJECT_ROOT_DIR+"/data/hiller_fernandez_preprocessed.csv"

LABEL_TRANSFORMATION = {
    'other': ERR_OTHER,
    'synt:subj': ERR_SUBJECT,
    'synt:obj': ERR_OBJECT,
    'synt:verb': ERR_VERB,
    'umorph:auxverb': ERR_AUXILIARY,
    'umorph:det': ERR_DETERMINER,
    'vmorph:regpast': ERR_TENSE_ASPECT,
    'vmorph:other': ERR_OTHER,
    'umorph:prep': ERR_PREPOSITION,
    'vmorph:3rdpers': ERR_SV_AGREEMENT,
    'nmorph:poss': ERR_POSSESSIVE,
    'vmorph:irrpast': ERR_TENSE_ASPECT,
    'umorph:presprogr': ERR_PRESENT_PROGRESSIVE,
    'nmorph:regplural': ERR_PLURAL,
    'nmorph:irrplural': ERR_PLURAL,
    'umorph:other': ERR_OTHER
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

    utts_transcript["prev_transcript_clean"] = "."

    return utts_transcript


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
