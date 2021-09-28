import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import pylangacq
import seaborn as sns

SPEAKER_CODE_CHILD = "CHI"

# TODO: check grandparents?
SPEAKER_CODES_CAREGIVER = ["MOT", "FAT"]

# minimum distance for longitudinal study (in months)
MIN_DISTANCE_LONGITUDINAL = 3

# Unintelligible words with an unclear phonetic shape should be transcribed as
CODE_UNCLEAR = "xxx"

# Use the symbol yyy when you plan to code all material phonologically on a %pho line.
# (usually used when utterance cannot be matched to particular words)
CODE_PHONETIC = "yyy"

CODES_UNINTELLIGIBLE = [CODE_UNCLEAR, CODE_PHONETIC]

EMPTY_UTTERANCE = "."
CODE_BABBLING = "@b"
CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION = "@u"
CODE_PHONOLGICAL_CONSISTENT_FORM = "@p"

CODE_CHILD_INVENTED_FORM = "@c"
CODE_NEOLOGISM = "@n"

# AGE_BINS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
# AGE_BINS_WINDOW = 3

# Only one bin, use this setup if you do not want to control for age
AGE_BINS = [24]
AGE_BINS_WINDOW = 100

# 1s response threshold
RESPONSE_THRESHOLD = 1000  # ms

# TODO check that pause is not too long: what is a reasonable value?
# 10 seconds
MAX_NEG_PAUSE_LENGTH = -10 * 1000

DEFAULT_CORPORA = [
    "Bloom",
    "Braunwald",
    "Soderstrom",
    "Weist",
    "NewmanRatner",
    "Snow",
    "Thomas",
    "Peters",
    "MacWhinney",
    "Sachs",
    "McCune",
    "Bernstein",
    "Brent",
    "Nelson",
    "Tommerdahl",
    "Providence",
]


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--corpora",
        nargs="+",
        type=str,
        default=DEFAULT_CORPORA,
    )
    args = argparser.parse_args()

    return args


def find_feedback_occurrences(transcripts):
    feedback_occ = []

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
        clean=True,
        time_marker=True,
        raise_error_on_missing_time_marker=False,
    )

    # Filter out empty transcripts and transcripts without age information
    utts_child = {
        file: utts
        for file, utts in utts_child.items()
        if (len(utts) > 0) and (ages[file] is not None)
    }

    # Filter out transcripts without child information
    utts_child = {
        file: utts for file, utts in utts_child.items() if file in target_child_names
    }

    for file, utts in utts_child.items():
        print(file)
        age = ages[file]
        child_name = target_child_names[file]
        print(f"Child: {child_name} Age: ", round(age))

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
                pause_length = round(utt2.start_time - utt1.end_time, 3)
                if pause_length > 0:
                    print(f"{utt1.speaker_code}: {utt1.utt}")
                    print(f"{utt2.speaker_code}: {utt2.utt}")
                    print(f"Pause: {pause_length}\n")
                    print(f"{utt3.speaker_code}: {utt3.utt}")
                feedback_occ.append(
                    {
                        "length": pause_length,
                        "age": round(age),
                        "child_name": child_name,
                        "utt_child": utt1.utt,
                        "utt_car": utt2.utt,
                        "utt_child_follow_up": utt3.utt,
                    }
                )

    feedback_occ = pd.DataFrame(feedback_occ)

    return feedback_occ


def remove_babbling(utterance):
    words = utterance.split(" ")
    filtered_utterance = []
    for word in words:
        # TODO:  word.endswith(CODE_CHILD_INVENTED_FORM)?
        if not (
            word.endswith(CODE_BABBLING)
            or word.endswith(CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION)
            or word.endswith(CODE_PHONOLGICAL_CONSISTENT_FORM)
        ):
            filtered_utterance.append(word)

    return " ".join(filtered_utterance)


def preprocess_transcripts(corpora):
    feedback = pd.DataFrame()
    for corpus in corpora:
        print(f"Reading transcripts of {corpus} corpus.. ", end="")
        transcripts = pylangacq.read_chat(
            os.path.expanduser(f"~/data/CHILDES/{corpus}/*.cha"),
            parse_morphology_information=False,
        )
        print("done.")
        feedback_transcript = find_feedback_occurrences(transcripts)

        feedback = feedback.append(feedback_transcript, ignore_index=True)

    return feedback


if __name__ == "__main__":
    args = parse_args()

    file_name = os.path.expanduser(
        f"~/data/communicative_feedback/feedback{'_'.join(args.corpora)}.csv"
    )

    feedback = preprocess_transcripts(args.corpora)
    feedback.to_csv(file_name, index=False)

    feedback = pd.read_csv(file_name, index_col=None)

    # Remove feedback with too long negative pauses
    feedback = feedback[(feedback.length > MAX_NEG_PAUSE_LENGTH)]

    # Remove babbling
    feedback["utt_child"] = feedback.utt_child.apply(remove_babbling)
    feedback["utt_child_follow_up"] = feedback.utt_child_follow_up.apply(
        remove_babbling
    )

    feedback["intelligible"] = feedback.utt_child != EMPTY_UTTERANCE
    feedback["intelligible_follow_up"] = feedback.utt_child_follow_up != EMPTY_UTTERANCE

    for age in AGE_BINS:
        feedback_age = feedback[
            (feedback.age > age - AGE_BINS_WINDOW)
            & (feedback.age <= age + AGE_BINS_WINDOW)
        ]
        print(
            f"\nFound {len(feedback_age)} turns for age {age} (+/- {AGE_BINS_WINDOW} months)"
        )
        if len(feedback_age) > 0:
            mean_length_intelligible = feedback_age[
                feedback_age.intelligible
            ].length.mean()
            print(
                f"Mean pause length after intellgible utts: {mean_length_intelligible:.3f}ms"
            )
            mean_length_unintelligible = feedback_age[
                ~feedback_age.intelligible
            ].length.mean()
            print(
                f"Mean pause length after unintelligible utts: {mean_length_unintelligible:.3f}ms"
            )

            # Caregiver contingency:
            n_responses_clear = len(
                feedback_age[
                    feedback_age.intelligible
                    & (feedback_age.length <= RESPONSE_THRESHOLD)
                ]
            )
            n_clear = len(feedback_age[feedback_age.intelligible])

            n_responses_unclear = len(
                feedback_age[~feedback_age.intelligible & (feedback_age.length <= 1)]
            )
            n_unclear = len(feedback_age[~feedback_age.intelligible])

            contingency_caregiver = (n_responses_clear / n_clear) - (
                n_responses_unclear / n_unclear
            )
            print(f"Caregiver contingency: {contingency_caregiver:.4f}")

            # Contingency of child speech-related vocalization on previous adult response:
            n_intelligible_follow_up_if_response = len(
                feedback_age[
                    feedback_age.intelligible_follow_up
                    & feedback_age.intelligible
                    & (feedback_age.length <= 1)
                ]
            )
            n_responses = len(feedback_age[feedback_age.length <= 1])

            n_intelligible_follow_up_if_no_response = len(
                feedback_age[
                    feedback_age.intelligible_follow_up
                    & feedback_age.intelligible
                    & (feedback_age.length > 1)
                ]
            )
            n_no_responses = len(feedback_age[feedback_age.length > 1])

            if (n_responses > 0) and (n_no_responses > 0):
                contingency_children = (
                    n_intelligible_follow_up_if_response / n_responses
                ) / (n_intelligible_follow_up_if_no_response / n_no_responses)

                print(f"Child contingency: {contingency_children:.4f}")

            g = sns.FacetGrid(feedback_age, col="intelligible")
            g.map(sns.histplot, "length", bins=30, log_scale=(False, True))
            plt.show()
