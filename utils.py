import argparse
import math
import re

import pandas as pd


def get_path_of_utterances_file(response_latency):
    return f"~/data/communicative_feedback/chi_utts_car_response_latency_{response_latency}.csv"


# codes that will be excluded from analysis
IS_UNTRANSCRIBED = lambda word: "www" in word
IS_INTERRUPTION = lambda word: word.startswith("+/")
IS_SELF_INTERRUPTION = lambda word: word == "+//"
IS_TRAILING_OFF = lambda word: word == "+..."
IS_TRAILING_OFF_2 = lambda word: word == "+.."
IS_EXCLUDED_WORD = lambda word: "@x:" in word
IS_PAUSE = lambda word: bool(re.match(r"\(\d*?\.*\d*?\)", word))
IS_OMITTED_WORD = lambda word: word == "0"
IS_SATELLITE_MARKER = lambda word: word == "‡"
IS_QUOTATION_MARKER = lambda word: word in ['+"/', '+"/.', '+"', '+".']
IS_UNKNOWN_CODE = lambda word: word == "zzz"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    elif v.lower() in ("none", "nan"):
        return None
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def is_excluded_code(word):
    if (
        IS_UNTRANSCRIBED(word)
        or IS_INTERRUPTION(word)
        or IS_SELF_INTERRUPTION(word)
        or IS_TRAILING_OFF(word)
        or IS_TRAILING_OFF_2(word)
        or IS_EXCLUDED_WORD(word)
        or IS_PAUSE(word)
        or IS_OMITTED_WORD(word)
        or IS_SATELLITE_MARKER(word)
        or IS_QUOTATION_MARKER(word)
        or IS_UNKNOWN_CODE(word)
    ):
        return True
    return False


def age_bin(age, min_age, max_age, num_months):
    return min(max_age, max(min_age, int((age + num_months / 2) / num_months) * num_months))


def is_simple_event(word):
    return word.startswith("&=")


def is_laughter(word):
    return word in ["haha", "hahaha", "hahahaha"]


def word_is_speech_related(word):
    if is_simple_event(word):
        return paralinguistic_event_is_speech_related(word)

    if is_laughter(word):
        return False

    return True


def get_paralinguistic_event(utterance):
    matches = re.findall(r"\[=! [^]]*]", utterance)
    if len(matches) < 1:
        return None
    if len(matches) == 1:
        event = matches[0]
        return event
    else:
        raise ValueError("Multiple paralinguistic events: ", utterance)


def get_all_paralinguistic_events(utterance):
    matches = re.finditer(r"\[=! [^]]*]", utterance)
    events = []
    for match in matches:
        pos = match.span()
        event = utterance[pos[0] : pos[1]]
        events.append(event)
    return events


def paralinguistic_event_is_intelligible(event):
    if "sing" in event or "sung" in event or "hum" in event or "whisper" in event:
        return True
    return False


def paralinguistic_event_is_speech_related(event):
    if (
        "babbl" in event
        or "hum" in event
        or ("sing" in event and not "fussing" in event and not "kissing" in event)
        or "sung" in event
        or "whisper" in event
        or "mumbl" in event
        or "mutter" in event
        or "voc" in event
    ):
        return True
    return False


def paralinguistic_event_is_external(event):
    if (
        paralinguistic_event_is_speech_related(event)
        or "laugh" in event
        or ("mak" in event and "noise" in event)
        or "cough" in event
        or "squeak" in event
        or "squeal" in event
        or "crie" in event
        or "moan" in event
        or "giggl" in event
        or "shout" in event
        or "snor" in event
        or "hiccup" in event
        or "cry" in event
        or "sigh" in event
        or "gasp" in event
        or "exhale" in event
        or "clap" in event
        or "gurgle" in event
        or "groan" in event
        or "yawn" in event
        or "murmum" in event
        or "whinge" in event
        or "whine" in event
        or "whining" in event
        or "whing" in event
        or "sneeze" in event
        or "roar" in event
        or "shriek" in event
        or "growl" in event
        or "grunt" in event
        or "chuckle" in event
        or "slurp" in event
        or "sob" in event
        or "raspberr" in event
        or "scream" in event
        or "whimper" in event
        or "burp" in event
        or "whimper" in event
        or "chant" in event
        or "whistl" in event
        or "yell" in event
        or "kiss" in event
        or "cheer" in event
        or "screech" in event
        or "blow" in event
        or "cooing" in event
        or "belch" in event
        or "squawk" in event
    ):
        return False
    return True


def remove_nonspeech_events(utterance):
    # Remove paralinguistic events
    event = get_paralinguistic_event(utterance)
    keep_event = None
    if event:
        utterance = utterance.replace(event, "")
        if paralinguistic_event_is_speech_related(event):
            keep_event = event
        else:
            # For cases like "mm [=! squeal]":
            words = utterance.strip().split(" ")
            if len(words) == 1 and words[0].lower() not in VOCAB and not is_babbling(words[0]):
                return ""

    words = utterance.strip().split(" ")
    cleaned_utterance = [
        word
        for word in words
        if word_is_speech_related(word) and not is_excluded_code(word)
    ]

    if keep_event:
        cleaned_utterance.append(keep_event)

    cleaned_utterance = " ".join(cleaned_utterance)
    return cleaned_utterance.strip()


def clean_utterance(utterance):
    """Remove all superfluous annotation information."""
    # Remove timing information:
    utterance = re.sub(r"[^]+?", "", utterance)
    # remove postcodes
    utterance = re.sub(r"\[\+[^]]*]", "", utterance)
    # remove precodes
    utterance = re.sub(r"\[-[^]]*]", "", utterance)
    # remove comments
    utterance = re.sub(r"\[%[^]]*]", "", utterance)
    # remove explanations:
    utterance = re.sub(r"\[= [^]]*]", "", utterance)
    # remove replacements:
    utterance = re.sub(r"\[:+ [^]]*]", "", utterance)
    # remove error codes:
    utterance = re.sub(r"\[\*[^]]*]", "", utterance)
    # remove repetition markers / collapses:
    utterance = re.sub(r"\[/[^]]*]", "", utterance)
    utterance = re.sub(r"\[x[^]]*]", "", utterance)
    # remove overlap markers
    utterance = re.sub(r"\[<\d*]", "", utterance)
    utterance = re.sub(r"\[>\d*]", "", utterance)
    # remove best guess markers
    utterance = re.sub(r"\[\?[^]]*]", "", utterance)
    # remove alternative transcriptions
    utterance = re.sub(r"\[=\? [^]]*]", "", utterance)
    # remove stress markers
    utterance = re.sub(r"\[!+]", "", utterance)
    # Remove "complex local events"
    utterance = re.sub(r"\[\^\S*]", "", utterance)
    # Remove arrows:
    utterance = re.sub(r"↓", "", utterance)
    utterance = re.sub(r"→", "", utterance)
    utterance = re.sub(r"↑", "", utterance)
    # Remove inhalations
    utterance = re.sub(r"∙", "", utterance)

    words = utterance.split(" ")
    cleaned_utterance = []
    for word in words:
        if not word == "" and not is_excluded_code(word):
            # remove other codes:
            word = re.sub(r"@z:\S*", "", word)
            # child invented forms, family forms, neologisms
            word = re.sub(r"@c", "", word)
            word = re.sub(r"@f", "", word)
            word = re.sub(r"@n", "", word)
            # onomatopeia
            word = re.sub(r"@o", "", word)
            # singing
            word = re.sub(r"@si", "", word)
            # word play
            word = re.sub(r"@wp", "", word)
            # dialect
            word = re.sub(r"@d", "", word)
            # single letters
            word = re.sub(r"@l", "", word)
            # multiple letters
            word = re.sub(r"@k", "", word)
            # test words
            word = re.sub(r"@t", "", word)
            # other language marker
            word = re.sub(r"@s\S*", "", word)
            # metalinguistic
            word = re.sub(r"@q", "", word)
            # remove brackets
            word = word.replace("(", "").replace(")", "")
            word = word.replace("<", "").replace(">", "")
            word = word.replace("‹", "").replace("›", "")
            word = word.replace("⌊", "").replace("⌋", "")
            word = word.replace("⌈", "").replace("⌉", "")
            word = word.replace("°", "")
            # compound words
            word = word.replace("_", " ")
            word = word.replace("+", " ")
            # remove lengthening
            word = re.sub(r":", "", word)
            # remove inter-syllable pauses
            word = re.sub(r"\^", "", word)
            # remove filled-pause prefix
            word = re.sub(r"&-", "", word)

            cleaned_utterance.append(word)

    cleaned_utterance = " ".join(cleaned_utterance)

    cleaned_utterance = cleaned_utterance.strip()
    return cleaned_utterance


def remove_punctuation(utterance):
    try:
        cleaned_utterance = re.sub(r"[,\"„”]", "", utterance)
        cleaned_utterance = re.sub(r"''", "", cleaned_utterance)
        cleaned_utterance = re.sub(r"[\.!\?]+\s*$", "", cleaned_utterance)
    except TypeError as e:
        print(utterance)
        raise e
    return cleaned_utterance.strip()


# Unintelligible words with an unclear phonetic shape should be transcribed as
CODE_UNINTELLIGIBLE = "xxx"

# Use the symbol yyy when you plan to code all material phonologically on a %pho line.
# (usually used when utterance cannot be matched to particular words)
CODE_PHONETIC = "yyy"

CODE_BABBLING = "@b"
CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION = "@u"
CODE_INTERJECTION = "@i"
CODE_PHONOLGICAL_CONSISTENT_FORM = "@p"
CODE_PHONOLOGICAL_FRAGMENT = "&"

OTHER_BABBLING = ["da", "ba", "baa", "babaa", "ababa", "bada", "dada", "gagaa", "gaga"]


VOCAB = set(
    pd.read_csv("data/childes_custom_vocab.csv", header=None, names=["word"]).word
)


def is_babbling(word):
    # Catching simple events (&=) first, because otherwise they could interpreted as phonological fragment (&)
    if is_simple_event(word):
        return not paralinguistic_event_is_intelligible(word)
    if (
        word.endswith(CODE_BABBLING)
        or word.endswith(CODE_INTERJECTION)
        or word.startswith(CODE_PHONOLOGICAL_FRAGMENT)
        or word == CODE_UNINTELLIGIBLE
        or word == CODE_PHONETIC
        or word.lower() in OTHER_BABBLING
        or (
            word.endswith(CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION)
            and word.lower().replace(CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION, "")
            not in VOCAB
        )
        or (
            word.endswith(CODE_PHONOLGICAL_CONSISTENT_FORM)
            and word.lower().replace(CODE_PHONOLGICAL_CONSISTENT_FORM, "") not in VOCAB
        )
    ):
        return True
    return False


def remove_babbling(utterance):
    # Remove any paralinguistic events
    event = get_paralinguistic_event(utterance)
    keep_event = None
    if event:
        utterance = utterance.replace(event, "")
        if paralinguistic_event_is_intelligible(event):
            keep_event = event
        else:
            # For cases like "mm [=! babbling]":
            words = utterance.strip().split(" ")
            if len(words) == 1 and words[0].lower() not in VOCAB:
                return ""

    words = utterance.strip().split(" ")
    filtered_utterance = [
        word for word in words if not (is_babbling(word) or is_excluded_code(word))
    ]

    if keep_event:
        filtered_utterance.append(event)

    filtered_utterance = " ".join(filtered_utterance)
    return filtered_utterance.strip()


def filter_corpora_based_on_response_latency_length(
    conversations, standard_deviations_off
):
    print(f"Filtering corpora based on average response latency")

    # Calculate mean and stddev of response latency using data from Nguyen, Versyp, Cox, Fusaroli (2021)
    latency_data = pd.read_csv("data/MA turn-taking.csv")

    # Use only non-clinical data:
    latency_data = latency_data[latency_data["clinical group"] == "Healthy"]

    # Use only data of the target age range:
    min_age = conversations.age.min()
    max_age = conversations.age.max()
    latency_data = latency_data[
        (latency_data.mean_age_infants_months >= min_age)
        & (latency_data.mean_age_infants_months <= max_age)
    ]

    mean_latency = latency_data.adult_response_latency.mean()
    std_mean_latency = latency_data.adult_response_latency.std()
    print(
        f"Mean of response latency in meta-analysis: {mean_latency:.1f} +/- {std_mean_latency:.1f}"
    )

    min_age = latency_data.mean_age_infants_months.min()
    max_age = latency_data.mean_age_infants_months.max()
    mean_age = latency_data.mean_age_infants_months.mean()
    print(
        f"Mean of child age in meta-analysis: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )

    # Filter corpora to be in range of mean +/- 1 standard deviation
    filtered = []
    print("Response latencies:")
    for corpus in conversations.corpus.unique():
        mean = conversations[
            (conversations.corpus == corpus)
            & (conversations.response_latency < math.inf)
        ].response_latency.values.mean()
        print(f"{corpus}: {mean:.1f}")
        if (
            mean_latency - standard_deviations_off * std_mean_latency
            < mean
            < mean_latency + standard_deviations_off * std_mean_latency
        ):
            filtered.append(corpus)

    print(f"Corpora included in analysis after filtering: {filtered}")
    conversations = conversations[conversations.corpus.isin(filtered)]
    return conversations


def get_binomial_test_data(column_name_1, column_name_2):
    data = []

    data.append({column_name_1: 0, column_name_2: 0})
    data.append({column_name_1: 0, column_name_2: 1})
    data.append({column_name_1: 1, column_name_2: 0})
    data.append({column_name_1: 1, column_name_2: 1})

    return pd.DataFrame(data)
