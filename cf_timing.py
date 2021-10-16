import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import pylangacq
import seaborn as sns
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import (
    remove_babbling,
    EMPTY_UTTERANCE,
    clean_utterance,
    remove_nonspeech_events,
)


SPEAKER_CODE_CHILD = "CHI"

SPEAKER_CODES_CAREGIVER = ["MOT", "FAT", "DAD", "MOM", "GRA", "GRF", "GRM", "CAR"]

# Separate analysis for children of different age groups
# AGE_BINS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
# AGE_BINS_WINDOW = 3

# Only one bin, use this setup if you do not want to control for age
AGE_BINS = [24]
AGE_BINS_WINDOW = 100

# 1s response threshold
RESPONSE_THRESHOLD = 1000  # ms

# Label for partially intelligible utterances
# Set to True to count as intelligible, False to count as unintelligible or None to exclude these utterances from
# the analysis
LABEL_PARTIALLY_INTELLIGIBLE = None
# same as above for speech-relatedness
LABEL_PARTIALLY_SPEECH_RELATED = True

# TODO check that pause is not too long (neg): what is a reasonable value?
# 1 second
MAX_NEG_RESPONSE_LATENCY = -1 * 1000  # ms

MIN_RATIO_NONSPEECH = 0.001

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
        clean=True,
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
            clean=False,
        )
        print("done.")
        adj_utterances_transcript = find_child_caregiver_adjacent_utterances(
            corpus, transcripts
        )

        adjacent_utterances = adjacent_utterances.append(
            adj_utterances_transcript, ignore_index=True
        )

    return adjacent_utterances


def is_intelligible(
    utterance, label_partially_intelligible=LABEL_PARTIALLY_INTELLIGIBLE
):
    utt_without_babbling = remove_babbling(utterance)

    if utt_without_babbling == EMPTY_UTTERANCE:
        return False

    is_partly_intelligible = len(utt_without_babbling) != len(utterance)
    if is_partly_intelligible:
        return label_partially_intelligible

    return True


def caregiver_intelligible_response(row):
    return row["response_latency"] <= RESPONSE_THRESHOLD


def caregiver_speech_related_response(row):
    return (row["response_latency"] <= RESPONSE_THRESHOLD) #& is_speech_related(row["utt_car"], label_partially_speech_related=True)


def caregiver_response_contingent_on_intelligibility(row):
    return (
        (row["utt_child_intelligible"] == True) & (row["caregiver_response"] == True)
    ) | (
        (row["utt_child_intelligible"] == False) & (row["caregiver_response"] == False)
    )


def is_speech_related(
    utterance, label_partially_speech_related=LABEL_PARTIALLY_SPEECH_RELATED
):
    utt_without_nonspeech = remove_nonspeech_events(utterance)

    if utt_without_nonspeech == EMPTY_UTTERANCE:
        return False

    is_partly_speech_related = len(utt_without_nonspeech) != len(utterance)
    if is_partly_speech_related:
        return label_partially_speech_related

    return True


def caregiver_response_contingent_on_speech_relatedness(row):
    return (
        (row["utt_child_speech_related"] == True) & (row["caregiver_response"] == True)
    ) | (
        (row["utt_child_speech_related"] == False)
        & (row["caregiver_response"] == False)
    )


def filter_corpora_based_on_response_latency_length(corpora, adj_utterances):
    # Calculate mean and stddev of response latency using data from Nguyen, Versyp, Cox, Fusaroli (2021)
    latency_data = pd.read_csv("data/MA turn-taking.csv")

    # Use only non-clinical data:
    latency_data = latency_data[latency_data["clinical group"] == "Healthy"]

    mean_latency = latency_data.adult_response_latency.mean()
    std_mean_latency = latency_data.adult_response_latency.std()
    print(
        f"Mean of response latency in meta-analysis: {mean_latency:.1f} +/- {std_mean_latency:.1f}"
    )

    min_age = latency_data.mean_age_infants_months.min()
    max_age = latency_data.mean_age_infants_months.max()
    mean_age = latency_data.mean_age_infants_months.max()
    print(
        f"Mean of child age in meta-analysis: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )

    # Filter corpora to be in range of mean +/- 1 standard deviation
    filtered = []
    for corpus in corpora:
        mean = adj_utterances[
            adj_utterances.corpus == corpus
        ].response_latency.values.mean()
        if mean_latency - std_mean_latency < mean < mean_latency + std_mean_latency:
            filtered.append(corpus)

    return filtered


def perform_analysis_intelligibility(adj_utterances):
    # Clean utterances
    adj_utterances["utt_child"] = adj_utterances.utt_child.apply(clean_utterance)
    adj_utterances["utt_car"] = adj_utterances.utt_car.apply(clean_utterance)
    adj_utterances["utt_child_follow_up"] = adj_utterances.utt_child_follow_up.apply(
        clean_utterance
    )

    # Remove nonspeech events
    adj_utterances["utt_child"] = adj_utterances.utt_child.apply(
        remove_nonspeech_events
    )
    adj_utterances["utt_car"] = adj_utterances.utt_car.apply(remove_nonspeech_events)
    adj_utterances["utt_child_follow_up"] = adj_utterances.utt_child_follow_up.apply(
        remove_nonspeech_events
    )

    # Drop empty utterances (these are non-speech related)
    adj_utterances = adj_utterances[
        (
            (adj_utterances.utt_child != EMPTY_UTTERANCE)
            & (adj_utterances.utt_car != EMPTY_UTTERANCE)
            & (adj_utterances.utt_child_follow_up != EMPTY_UTTERANCE)
        )
    ]

    # Label utterances as intelligible or unintelligible
    adj_utterances = adj_utterances.assign(
        utt_child_intelligible=adj_utterances.utt_child.apply(is_intelligible)
    )
    adj_utterances = adj_utterances.assign(
        follow_up_intelligible=adj_utterances.utt_child_follow_up.apply(is_intelligible)
    )

    # Label caregiver responses as present or not
    adj_utterances = adj_utterances.assign(
        caregiver_response=adj_utterances.apply(caregiver_intelligible_response, axis=1)
    )

    # Remove NaNs
    adj_utterances = adj_utterances.dropna(
        subset=("utt_child_intelligible", "follow_up_intelligible")
    )

    # Label caregiver responses as contingent on child utterance or not
    adj_utterances = adj_utterances.assign(
        caregiver_response_contingent=adj_utterances[
            ["utt_child_intelligible", "caregiver_response"]
        ].apply(caregiver_response_contingent_on_intelligibility, axis=1)
    )

    for age in AGE_BINS:
        adj_utterances_age = adj_utterances[
            (adj_utterances.age > age - AGE_BINS_WINDOW)
            & (adj_utterances.age <= age + AGE_BINS_WINDOW)
        ]
        print(
            f"\nFound {len(adj_utterances_age)} turns for age {age} (+/- {AGE_BINS_WINDOW} months)"
        )
        if len(adj_utterances_age) > 0:
            # Caregiver contingency:
            n_responses_intelligible = len(
                adj_utterances_age[
                    adj_utterances_age.utt_child_intelligible
                    & adj_utterances_age.caregiver_response
                ]
            )
            n_intelligible = len(
                adj_utterances_age[adj_utterances_age.utt_child_intelligible]
            )

            n_responses_unintelligible = len(
                adj_utterances_age[
                    (adj_utterances_age.utt_child_intelligible == False)
                    & adj_utterances_age.caregiver_response
                ]
            )
            n_unintelligible = len(
                adj_utterances_age[adj_utterances_age.utt_child_intelligible == False]
            )

            contingency_caregiver = (n_responses_intelligible / n_intelligible) - (
                n_responses_unintelligible / n_unintelligible
            )
            print(f"Caregiver contingency: {contingency_caregiver:.4f}")

            # Contingency of child vocalization on previous adult response (positive case):
            n_follow_up_intelligible_if_response_to_intelligible = len(
                adj_utterances_age[
                    adj_utterances_age.follow_up_intelligible
                    & adj_utterances_age.utt_child_intelligible
                    & adj_utterances_age.caregiver_response
                ]
            )
            n_responses_to_intelligible = len(
                adj_utterances_age[
                    adj_utterances_age.utt_child_intelligible
                    & adj_utterances_age.caregiver_response
                ]
            )

            n_follow_up_intelligible_if_no_response_to_intelligible = len(
                adj_utterances_age[
                    adj_utterances_age.follow_up_intelligible
                    & adj_utterances_age.utt_child_intelligible
                    & (adj_utterances_age.caregiver_response == False)
                ]
            )
            n_no_responses_to_intelligible = len(
                adj_utterances_age[
                    adj_utterances_age.utt_child_intelligible
                    & (adj_utterances_age.caregiver_response == False)
                ]
            )

            # Contingency of child vocalization on previous adult response (negative case):
            n_follow_up_intelligible_if_no_response_to_unintelligible = len(
                adj_utterances_age[
                    adj_utterances_age.follow_up_intelligible
                    & (adj_utterances_age.utt_child_intelligible == False)
                    & (adj_utterances_age.caregiver_response == False)
                ]
            )
            n_no_responses_to_unintelligible = len(
                adj_utterances_age[
                    (adj_utterances_age.utt_child_intelligible == False)
                    & (adj_utterances_age.caregiver_response == False)
                ]
            )

            n_follow_up_intelligible_if_response_to_unintelligible = len(
                adj_utterances_age[
                    adj_utterances_age.follow_up_intelligible
                    & (adj_utterances_age.utt_child_intelligible == False)
                    & adj_utterances_age.caregiver_response
                ]
            )
            n_responses_to_unintelligible = len(
                adj_utterances_age[
                    (adj_utterances_age.utt_child_intelligible == False)
                    & adj_utterances_age.caregiver_response
                ]
            )

            if (
                (n_no_responses_to_unintelligible > 0)
                and (n_responses_to_unintelligible > 0)
                and (n_responses_to_intelligible > 0)
                and (n_no_responses_to_intelligible > 0)
            ):
                ratio_follow_up_intelligible_if_response_to_intelligible = (
                    n_follow_up_intelligible_if_response_to_intelligible
                    / n_responses_to_intelligible
                )
                ratio_follow_up_intelligible_if_no_response_to_intelligible = (
                    n_follow_up_intelligible_if_no_response_to_intelligible
                    / n_no_responses_to_intelligible
                )
                contingency_children_pos_case = (
                    ratio_follow_up_intelligible_if_response_to_intelligible
                    - ratio_follow_up_intelligible_if_no_response_to_intelligible
                )

                print(
                    f"Child contingency (positive case): {contingency_children_pos_case:.4f}"
                )

                ratio_follow_up_intelligible_if_no_response_to_unintelligible = (
                    n_follow_up_intelligible_if_no_response_to_unintelligible
                    / n_no_responses_to_unintelligible
                )
                ratio_follow_up_intelligible_if_response_to_unintelligible = (
                    n_follow_up_intelligible_if_response_to_unintelligible
                    / n_responses_to_unintelligible
                )
                contingency_children_neg_case = (
                    ratio_follow_up_intelligible_if_no_response_to_unintelligible
                    - ratio_follow_up_intelligible_if_response_to_unintelligible
                )

                print(
                    f"Child contingency (negative case): {contingency_children_neg_case:.4f}"
                )

                ratio_contingent_follow_ups = (
                    n_follow_up_intelligible_if_response_to_intelligible
                    + n_follow_up_intelligible_if_no_response_to_unintelligible
                ) / (n_responses_to_intelligible + n_no_responses_to_unintelligible)
                ratio_incontingent_follow_ups = (
                    n_follow_up_intelligible_if_no_response_to_intelligible
                    + n_follow_up_intelligible_if_response_to_unintelligible
                ) / (n_no_responses_to_intelligible + n_responses_to_unintelligible)

                child_contingency_both_cases = (
                    ratio_contingent_follow_ups - ratio_incontingent_follow_ups
                )
                print(
                    f"Child contingency (both cases): {child_contingency_both_cases:.4f}"
                )
                child_contingency_both_cases_same_weighting = np.mean(
                    [contingency_children_pos_case, contingency_children_neg_case]
                )

                print(
                    f"Child contingency (both cases, same weighting of positive and negative cases): "
                    f"{child_contingency_both_cases_same_weighting:.4f}"
                )

        # Statsmodels prefers 1 and 0 over True and False:
        adj_utterances.replace({False: 0, True: 1}, inplace=True)

        mod = smf.glm(
            "caregiver_response ~ utt_child_intelligible",
            family=sm.families.Binomial(),
            data=adj_utterances,
        ).fit()
        print(mod.summary())

        mod = smf.glm(
            "follow_up_intelligible ~ caregiver_response_contingent",
            family=sm.families.Binomial(),
            data=adj_utterances,
        ).fit()
        print(mod.summary())

        mod = smf.glm(
            "follow_up_intelligible ~ utt_child_intelligible * caregiver_response_contingent",
            family=sm.families.Binomial(),
            data=adj_utterances,
        ).fit()
        print(mod.summary())

        sns.barplot(
            data=adj_utterances,
            x="utt_child_intelligible",
            y="follow_up_intelligible",
            hue="caregiver_response_contingent",
        )
        plt.show()


def perform_analysis_speech_relatedness(adj_utterances):
    # Clean utterances
    adj_utterances["utt_child"] = adj_utterances.utt_child.apply(clean_utterance)
    adj_utterances["utt_car"] = adj_utterances.utt_car.apply(clean_utterance)
    adj_utterances["utt_child_follow_up"] = adj_utterances.utt_child_follow_up.apply(
        clean_utterance
    )

    # Drop empty utterances
    adj_utterances = adj_utterances[
        (
            (adj_utterances.utt_child != EMPTY_UTTERANCE)
            & (adj_utterances.utt_car != EMPTY_UTTERANCE)
            & (adj_utterances.utt_child_follow_up != EMPTY_UTTERANCE)
        )
    ]

    # Label utterances as speech or non-speech
    adj_utterances = adj_utterances.assign(
        utt_child_speech_related=adj_utterances.utt_child.apply(is_speech_related)
    )
    adj_utterances = adj_utterances.assign(
        follow_up_speech_related=adj_utterances.utt_child_follow_up.apply(
            is_speech_related
        )
    )

    # Label caregiver responses as present or not
    adj_utterances = adj_utterances.assign(
        caregiver_response=adj_utterances.apply(
            caregiver_speech_related_response, axis=1
        )
    )

    # Remove NaNs
    adj_utterances = adj_utterances.dropna(
        subset=("utt_child_speech_related", "follow_up_speech_related")
    )

    counter_non_speech = Counter(adj_utterances[adj_utterances.utt_child_speech_related == False].utt_child.values)
    print("Most common non-speech related sounds: ")
    print(counter_non_speech.most_common())

    # Filter for corpora that actually annotate non-speech-related sounds
    good_corpora = []
    for corpus in adj_utterances.corpus.unique():
        d_corpus = adj_utterances[adj_utterances.corpus == corpus]
        ratio = len(d_corpus[d_corpus.utt_child_speech_related == False]) / len(
            d_corpus[d_corpus.utt_child_speech_related == True]
        )
        if ratio > MIN_RATIO_NONSPEECH:
            good_corpora.append(corpus)
    print("Filtered corpora: ", good_corpora)
    adj_utterances = adj_utterances[adj_utterances.corpus.isin(good_corpora)]

    # Label caregiver responses as contingent on child utterance or not
    adj_utterances = adj_utterances.assign(
        caregiver_response_contingent=adj_utterances[
            ["utt_child_speech_related", "caregiver_response"]
        ].apply(caregiver_response_contingent_on_speech_relatedness, axis=1)
    )

    n_responses_to_speech = len(
        adj_utterances[
            adj_utterances.utt_child_speech_related & adj_utterances.caregiver_response
        ]
    )
    n_speech = len(adj_utterances[adj_utterances.utt_child_speech_related])

    n_responses_to_non_speech = len(
        adj_utterances[
            (adj_utterances.utt_child_speech_related == False)
            & adj_utterances.caregiver_response
        ]
    )
    n_non_speech = len(adj_utterances[adj_utterances.utt_child_speech_related == False])

    contingency_caregiver = (n_responses_to_speech / n_speech) - (
        n_responses_to_non_speech / n_non_speech
    )
    print(f"Caregiver contingency: {contingency_caregiver:.4f}")

    # Contingency of child vocalization on previous adult response (positive case):
    n_follow_up_speech_related_if_response_to_speech_related = len(
        adj_utterances[
            adj_utterances.follow_up_speech_related
            & adj_utterances.utt_child_speech_related
            & adj_utterances.caregiver_response
        ]
    )
    n_responses_to_speech_related = len(
        adj_utterances[
            adj_utterances.utt_child_speech_related & adj_utterances.caregiver_response
        ]
    )

    n_follow_up_speech_related_if_no_response_to_speech_related = len(
        adj_utterances[
            adj_utterances.follow_up_speech_related
            & adj_utterances.utt_child_speech_related
            & (adj_utterances.caregiver_response == False)
        ]
    )
    n_no_responses_to_speech_related = len(
        adj_utterances[
            adj_utterances.utt_child_speech_related
            & (adj_utterances.caregiver_response == False)
        ]
    )

    ratio_follow_up_speech_related_if_response_to_speech_related = (
        n_follow_up_speech_related_if_response_to_speech_related
        / n_responses_to_speech_related
    )
    ratio_follow_up_speech_related_if_no_response_to_speech_related = (
        n_follow_up_speech_related_if_no_response_to_speech_related
        / n_no_responses_to_speech_related
    )
    contingency_children_pos_case = (
        ratio_follow_up_speech_related_if_response_to_speech_related
        - ratio_follow_up_speech_related_if_no_response_to_speech_related
    )
    print(f"Child contingency (positive case): {contingency_children_pos_case:.4f}")

    # Statsmodels prefers 1 and 0 over True and False:
    adj_utterances.replace({False: 0, True: 1}, inplace=True)

    mod = smf.glm(
        "caregiver_response ~ utt_child_speech_related",
        family=sm.families.Binomial(),
        data=adj_utterances,
    ).fit()
    print(mod.summary())

    mod = smf.glm(
        "follow_up_speech_related ~ caregiver_response_contingent",
        family=sm.families.Binomial(),
        data=adj_utterances[adj_utterances.utt_child_speech_related == True],
    ).fit()
    print(mod.summary())


if __name__ == "__main__":
    args = parse_args()

    file_name = os.path.expanduser(
        f"~/data/communicative_feedback/chi_car_adjacent_utterances.csv"
    )

    adjacent_utterances = preprocess_transcripts(CANDIDATE_CORPORA)
    adjacent_utterances.to_csv(file_name, index=False)

    adjacent_utterances = pd.read_csv(file_name, index_col=None)

    # Remove adjacent_utterances with too long negative pauses
    adjacent_utterances = adjacent_utterances[
        (adjacent_utterances.response_latency >= MAX_NEG_RESPONSE_LATENCY)
    ]

    if not args.corpora:
        print(f"No corpora given, selecting based on average response time")
        args.corpora = filter_corpora_based_on_response_latency_length(
            CANDIDATE_CORPORA, adjacent_utterances
        )

    print(f"Corpora included in analysis: {args.corpora}")

    min_age = adjacent_utterances.age.min()
    max_age = adjacent_utterances.age.max()
    mean_age = adjacent_utterances.age.mean()
    print(
        f"Mean of child age selected corpora: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )

    # Filter corpora
    adjacent_utterances = adjacent_utterances[
        adjacent_utterances.corpus.isin(args.corpora)
    ]

    # perform_analysis_intelligibility(adjacent_utterances.copy())
    #
    perform_analysis_speech_relatedness(adjacent_utterances.copy())
