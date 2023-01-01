import argparse
import os
import re
from multiprocessing import Pool

import enchant
import numpy as np
import pandas as pd
from tqdm import tqdm

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

PREPROCESSED_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances.csv"
)

UTTERANCES_WITH_SPEECH_ACTS_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_with_speech_acts.csv"
)

ANNOTATED_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_annotated.csv"
)

MICRO_CONVERSATIONS_FILE = os.path.expanduser(
    "~/data/communicative_feedback/micro_conversations.csv"
)

MICRO_CONVERSATIONS_WITHOUT_NON_SPEECH_FILE = os.path.expanduser(
    "~/data/communicative_feedback/micro_conversations_without_non_speech.csv"
)

SPEECH_ACT_NO_FUNCTION = "YY"
SPEECH_ACTS_NO_FUNCTION = ["YY", "OO"]


POS_PUNCTUATION = [
    ".",
    "?",
    "...",
    "!",
    "+/",
    "+/?",
    "" "...?",
    ",",
    "-",
    '+"/.',
    "+...",
    "++/.",
    "+/.",
]

# codes that will be excluded from analysis
IS_UNTRANSCRIBED = lambda word: word.replace(".", "") == "www"
IS_INTERRUPTION = lambda word: word.startswith("+/")
IS_SELF_INTERRUPTION = lambda word: word == "+//"
IS_TRAILING_OFF = lambda word: word == "+..."
IS_TRAILING_OFF_2 = lambda word: word == "+.."
IS_EXCLUDED_WORD = lambda word: "@x" in word
IS_PAUSE = lambda word: bool(re.match(r"\(\d*?\.*\d*?\)", word))
IS_OMITTED_WORD = lambda word: word.startswith("0")
IS_SATELLITE_MARKER = lambda word: word == "‡"
IS_QUOTATION_MARKER = lambda word: word in ['+"/', '+"/.', '+"', '+".']
IS_UNKNOWN_CODE = lambda word: word == "zzz"


def is_nan(value):
    return value != value


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
    word = word.replace(",", "")

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
    return min(
        max_age, max(min_age, int((age + num_months / 2) / num_months) * num_months)
    )


def is_simple_event(word):
    return word.startswith("&=")


def utterance_is_laughter(utterance):
    return utterance in ["ha ha", "ha ha ha"]


def word_is_laughter(word):
    word = word.replace(",", "")

    return word in [
        "haha",
        "hahaha",
        "hahahaha",
        "hehehe",
        "heehee",
        "hehe",
        "hohoho",
        "hhh",
        "hah",
    ]


def word_is_parseable_speech(word, vocab_check):
    word = word.replace(",", "")
    if (
            is_simple_event(word)
            or word_is_laughter(word)
            or word.lower() in OTHER_NONSPEECH
            or is_excluded_code(word)
            or is_babbling(word, vocab_check)
            or word.endswith(CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION)
            or word.endswith(CODE_PHONOLOGICAL_CONSISTENT_FORM)
    ):
        return False

    return True


def word_is_speech_related(word):
    if is_simple_event(word):
        return paralinguistic_event_is_speech_related(word)

    if word_is_laughter(word):
        return False

    if word.lower() in OTHER_NONSPEECH:
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


def is_empty(utterance):
    utterance = remove_punctuation(utterance)
    return utterance == ""


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
    utterance = utterance.strip()
    keep_event = None
    if event:
        utterance = utterance.replace(event, "")
        if paralinguistic_event_is_speech_related(event):
            keep_event = event
        else:
            if utterance == "":
                return ""
            # For cases like "mm [=! squeal]":
            words = split_into_words(utterance, split_on_apostrophe=False, remove_commas=True, remove_trailing_punctuation=False)
            if (
                len(words) == 1
                and not is_word(words[0])
                and not is_babbling(words[0])
                and not word_is_speech_related(words[0])
            ):
                return ""

    if utterance_is_laughter(utterance):
        return ""

    words = split_into_words(utterance, split_on_apostrophe=False, remove_commas=True, remove_trailing_punctuation=False)
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
    utterance = remove_superfluous_annotations(utterance)

    final_punctuation = None
    while len(utterance) > 0 and utterance[-1] in [".", "!", "?"]:
        final_punctuation = utterance[-1]
        utterance = utterance[:-1]

    utt_clean = remove_events_and_non_parseable_words(utterance)
    utt_clean = replace_slang_forms(utt_clean)
    utt_clean = clean_disfluencies(utt_clean)

    # Remove underscores
    utt_clean = utt_clean.replace("_", " ")

    # Remove spacing before commas and double commas
    utt_clean = utt_clean.replace(" ,", ",")
    utt_clean = utt_clean.replace(",,", ",")
    utt_clean = re.sub("\s\.$", ".", utt_clean)
    utt_clean = utt_clean.replace(",.", ".")

    # Strip:
    utt_clean = utt_clean.strip()
    utt_clean = utt_clean.replace("   ", " ")
    utt_clean = utt_clean.replace("  ", " ")

    # Remove remaining commas at beginning and end of utterance
    while len(utt_clean) > 0 and utt_clean[0] == ",":
        utt_clean = utt_clean[1:].strip()
    while len(utt_clean) > 0 and utt_clean[-1] == ",":
        utt_clean = utt_clean[:-1].strip()

    if final_punctuation:
        utt_clean += final_punctuation
    else:
        utt_clean += "."

    return utt_clean


def remove_timing_information(utterance):
    # Remove timing information:
    utterance = re.sub(r"[^]+?", "", utterance)
    return utterance


def replace_actually_said_words(utterance):
    # Remove incomplete annotations
    utterance = utterance.replace("[= actually says]", "")

    # Recursively replace replacements
    match = re.search("(\[[\*|\?]]\s)*\[=[?|!]? actually says[^]]+](\s\[\*])*", utterance)
    error_tags = []
    while match:
        replacement = match[0].split("actually says")[-1].split("]")[0].strip()
        replacement = remove_superfluous_annotations(replacement)
        num_words = len(replacement.split(" "))
        before = utterance[0: match.span()[0]].strip()
        before = remove_superfluous_annotations(before)
        event = get_paralinguistic_event(before)
        if event:
            before = before.replace(event, "")
        before = before.split(" ")
        words_to_replace = " ".join(before[len(before) - num_words:])
        if replacement == "dunno" and words_to_replace == "know":
            num_words += 1
            words_to_replace = " ".join(before[len(before) - num_words:])
        if "'" in replacement and "'" not in words_to_replace:
            num_words += 1
            words_to_replace = " ".join(before[len(before) - num_words:])

        after = utterance[match.span()[1]:].strip()
        before = " ".join(before[:len(before) - num_words])
        if event:
            before = before + " " + event
        utterance = " ".join([before, replacement, after])
        error_tags.append(f"{replacement}={words_to_replace}")

        match = re.search("(\[[\*|\?]\]\s)*\[=[?|!]? actually says [^\]]+\](\s\[\*\])*", utterance)

    return utterance, error_tags


def remove_superfluous_annotations(utterance):
    """Remove all superfluous annotation information."""
    # remove postcodes
    utterance = re.sub(r"\[\+[^]]*]", "", utterance)
    # remove precodes
    utterance = re.sub(r"\[-[^]]*]", "", utterance)
    # remove comments
    utterance = re.sub(r"\[%[^]]*]", "", utterance)
    # remove explanations:
    if not "[= actually says" in utterance:
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
    utterance = re.sub(r"\[\^ \S*]", "", utterance)
    # Remove arrows:
    utterance = re.sub(r"↓", "", utterance)
    utterance = re.sub(r"→", "", utterance)
    utterance = re.sub(r"↑", "", utterance)
    utterance = re.sub(r"↗", "", utterance)
    # Remove inhalations
    utterance = re.sub(r"∙", "", utterance)
    # Remove quotation marks
    utterance = utterance.replace("“", "")
    utterance = utterance.replace("”", "")
    utterance = utterance.replace("„", "")
    # Remove pitch annotations
    utterance = utterance.replace("▔", "")
    utterance = utterance.replace("▁", "")
    utterance = utterance.replace("⁎", "")
    # Remove speed annotations
    utterance = utterance.replace("∇", "")
    utterance = utterance.replace("∆", "")
    # Remove repetition annotations
    utterance = utterance.replace("↫", "")

    utterance = utterance.replace("⁎", "")
    utterance = utterance.replace("∆", "")

    # Remove smileys
    utterance = utterance.replace("☺", "")

    # Replace pauses
    utterance = re.sub(r"\(\.*\)", "...", utterance)

    # Remove omitted words
    utterance = re.sub(r"\(\S*\)", "", utterance)

    words = split_into_words(utterance, split_on_apostrophe=False, remove_commas=False, remove_trailing_punctuation=False)
    cleaned_utterance = []
    for word in words:
        if not word == "" and not is_excluded_code(word):
            # remove other codes:
            word = re.sub(r"@z:\S*", "", word)
            # child invented forms, family forms, neologisms
            word = re.sub(r"@c", "", word)
            word = re.sub(r"@f", "", word)
            word = re.sub(r"@n", "", word)
            # general special forms
            word = re.sub(r"@g", "", word)
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


def remove_punctuation(utterance, return_removed_trailing_punct=False, remove_commas=False):
    try:
        cleaned_utterance = re.sub(r"[\"„”]", "", utterance)
        cleaned_utterance = re.sub(r"''", "", cleaned_utterance)
        cleaned_utterance = re.sub(r"  ", " ", cleaned_utterance)
        if remove_commas:
            cleaned_utterance = re.sub(",", "", cleaned_utterance)
    except TypeError as e:
        print(utterance)
        raise e

    if return_removed_trailing_punct:
        cleaned_utterance = cleaned_utterance.strip()
        removed_punct = None
        while cleaned_utterance[-1] in [".", "!", "?"]:
            removed_punct = cleaned_utterance[-1]
            cleaned_utterance = cleaned_utterance[:-1]
            cleaned_utterance = cleaned_utterance.strip()

        return cleaned_utterance.strip(), removed_punct
    else:
        cleaned_utterance = re.sub(r"[\.!\?]+\s*$", "", cleaned_utterance)
        return cleaned_utterance.strip()


def split_into_words(utterance, split_on_apostrophe=True, remove_commas=False, remove_trailing_punctuation=False):
    # Copy in order not to modify the original utterance
    utt = utterance
    if remove_trailing_punctuation:
        utt = utterance[:-1]
    regex = '\s'
    if split_on_apostrophe:
        regex += '|\''
    if remove_commas:
        regex += '|,'

    words = re.split(regex, utt)

    # Filter out empty words:
    words = [word for word in words if len(word) > 0]
    return words


def get_num_words(clean_utts, remove_punctuation=True):
    return clean_utts.apply(lambda x: len(split_into_words(x, split_on_apostrophe=True, remove_commas=remove_punctuation, remove_trailing_punctuation=remove_punctuation)))


def get_num_unique_words(clean_utts, remove_punctuation=True):
    return clean_utts.apply(lambda x: len(set(split_into_words(x, split_on_apostrophe=False, remove_commas=remove_punctuation, remove_trailing_punctuation=remove_punctuation))))


# Unintelligible words with an unclear phonetic shape should be transcribed as
CODE_UNINTELLIGIBLE = "xxx"

# Use the symbol yyy when you plan to code all material phonologically on a %pho line.
# (usually used when utterance cannot be matched to particular words)
CODE_PHONETIC = "yyy"

CODE_BABBLING = "@b"
CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION = "@u"
CODE_INTERJECTION = "@i"
CODE_PHONOLOGICAL_CONSISTENT_FORM = "@p"
CODE_PHONOLOGICAL_FRAGMENT = "&"

OTHER_BABBLING = [
    "da",
    "ba",
    "baba",
    "baa",
    "babaa",
    "ababa",
    "bada",
    "gagaa",
    "gaga",
    "ow",
    "ay",
    "pss",
    "ugh",
    "bum",
    "brrr",
    "oop",
    "er",
]
OTHER_NONSPEECH = [
    "ouch",
    "yack",
    "ugh",
    "woah",
    "oy",
    "ee",
    "hee",
    "whoo",
    "oo",
    "hoo",
    "ew",
    "oof",
    "baaee",
    "ewok",
    "ewoks",
    "urgh",
    "ow",
    "heh",
]

VOCAB_CUSTOM = set(
    pd.read_csv("data/childes_custom_vocab.csv", header=None, names=["word"]).word
)


def is_word(word):
    DICT_ENCHANT = enchant.Dict("en_US")
    word = word.lower()
    if word in VOCAB_CUSTOM:
        return True
    if DICT_ENCHANT.check(word):
        return True
    return False


def is_babbling(word, vocab_check=True):
    word = word.replace(",", "")
    # Catching simple events (&=) first, because otherwise they could be interpreted as phonological fragment (&)
    if is_simple_event(word):
        return not paralinguistic_event_is_intelligible(word)
    if (
        word.endswith(CODE_BABBLING)
        or word.endswith(CODE_INTERJECTION)
        or word.startswith(CODE_PHONOLOGICAL_FRAGMENT)
        or word == CODE_UNINTELLIGIBLE
        or word == CODE_PHONETIC
        or word.lower() in OTHER_BABBLING
        or vocab_check and (
            (
            word.endswith(CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION)
            and not is_word(word.replace(CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION, ""))
            )
            or (
                word.endswith(CODE_PHONOLOGICAL_CONSISTENT_FORM)
                and not is_word(word.replace(CODE_PHONOLOGICAL_CONSISTENT_FORM, ""))
            )
        )
        or not vocab_check and (
            word.endswith(CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION)
            or word.endswith(CODE_PHONOLOGICAL_CONSISTENT_FORM)
        )
    ):
        return True
    return False


def remove_events_and_non_parseable_words(utterance):
    event = get_paralinguistic_event(utterance)
    if event:
        if not utterance.startswith(event) or utterance.endswith(event):
            utterance = utterance.replace(event, ",")
            utterance = utterance.replace(" ,", ",")
        else:
            utterance = utterance.replace(event, "")

    if utterance_is_laughter(utterance):
        return ""

    words = split_into_words(utterance, split_on_apostrophe=False, remove_commas=False, remove_trailing_punctuation=False)
    cleaned_utterance = [
        word
        for word in words
        if word_is_parseable_speech(word, vocab_check=False)
    ]
    cleaned_utterance = " ".join(cleaned_utterance)
    return cleaned_utterance.strip()


# We're not fixing words such as "wanna", as it can be both "want to" and "want a"
# Also: "she's": can be either "she has" or "she is"
SLANG_WORDS = {
    "hasta": "has to",
    "hafta": "have to",
    "hadta": "had to",
    "needta": "need to",
    "dat's": "that is",
    "dat": "that",
    "dis": "this",
    "dere": "there",
    "de": "the",
    "gonna": "going to",
    "anoder": "another",
    "dunno": "don't know",
    "'cause": "because",
}


def replace_slang_forms(utterance):
    words = split_into_words(utterance, split_on_apostrophe=False, remove_commas=False, remove_trailing_punctuation=False)
    cleaned_utterance = [
        word if word.replace(",", "") not in SLANG_WORDS.keys() else SLANG_WORDS[word.replace(",", "")]
        for word in words
    ]
    cleaned_utterance = " ".join(cleaned_utterance)
    return cleaned_utterance.strip()


def find_repeated_sequence(utterance):
    words = split_into_words(utterance, split_on_apostrophe=False, remove_commas=True, remove_trailing_punctuation=False)

    for rep_len in range(len(words)//2, 0, -1):
        candidates = ["__".join(words[i:i+rep_len]) for i in range(len(words)) if len(words[i:i+rep_len]) == rep_len]
        repetitions = [candidates[i] for i in range(len(candidates) - rep_len) if candidates[i+rep_len] == candidates[i]]
        repetitions = [rep.replace("__", " ") for rep in repetitions]
        repetitions = [rep for rep in repetitions if rep in utterance]
        if len(repetitions) > 0:
            return repetitions[0]

    return None


DISFLUENCIES = ["uhm", "um", "uh", "erh", "err", "aw", "ehm", "hm"]


def clean_disfluencies(utterance):
    words = utterance.split(" ")
    words = [word for word in words if not word.replace(",", "") in DISFLUENCIES]
    utterance = " ".join(words)

    duplicate = find_repeated_sequence(remove_punctuation(utterance, remove_commas=True))
    while duplicate:
        if (len(remove_punctuation(utterance.replace(duplicate, ""), remove_commas=True).strip()) == 0) and (len(duplicate.split(" ")) == 1):
            # Cases like "bye bye", "no no no!"
            return utterance
        utterance = re.sub(f"(^|\s){duplicate}(\s|$|\.|\,)", " ", utterance, count=1)
        utterance = utterance.strip().replace("  ", " ")
        duplicate = find_repeated_sequence(utterance)

    return utterance


def remove_babbling(utterance):
    # Remove any paralinguistic events
    event = get_paralinguistic_event(utterance)
    keep_event = None
    if event:
        utterance = utterance.replace(event, "")
        if paralinguistic_event_is_intelligible(event):
            keep_event = event
        else:
            if utterance == "":
                return ""
            # For cases like "mm [=! babbling]":
            utterance = utterance.strip()
            words = split_into_words(utterance, split_on_apostrophe=False, remove_commas=True, remove_trailing_punctuation=False)
            if len(words) == 1 and not is_word(words[0]):
                return ""

    utterance = utterance.strip()
    words = split_into_words(utterance, split_on_apostrophe=False, remove_commas=False, remove_trailing_punctuation=False)
    filtered_utterance = [
        word
        for word in words
        if not (is_babbling(word) or is_excluded_code(word))
    ]

    if keep_event:
        filtered_utterance.append(event)

    filtered_utterance = " ".join(filtered_utterance)
    return filtered_utterance.strip()


def filter_transcripts_based_on_num_child_utts(
    conversations, min_child_utts_per_transcript
):
    child_utts_per_transcript = conversations.groupby("transcript_file").size()
    transcripts_enough_utts = child_utts_per_transcript[
        child_utts_per_transcript > min_child_utts_per_transcript
    ]

    return conversations[
        conversations.transcript_file.isin(transcripts_enough_utts.index)
    ]


def add_prev_utts_for_transcript(utterances_transcript):
    utts_speech_related = utterances_transcript[utterances_transcript.is_speech_related.isin([pd.NA, True])]

    def add_prev_utt(utterance):
        if utterance.name in utts_speech_related.index:
            row_number = np.where(utts_speech_related.index.values == utterance.name)[0][0]
            if row_number > 0:
                prev_utt = utts_speech_related.loc[utts_speech_related.index[:row_number][-1]]
                return prev_utt.transcript_clean

        return pd.NA

    def add_prev_utt_speaker_code(utterance):
        if utterance.name in utts_speech_related.index:
            row_number = np.where(utts_speech_related.index.values == utterance.name)[0][0]
            if row_number > 0:
                prev_utt = utts_speech_related.loc[utts_speech_related.index[:row_number][-1]]
                return prev_utt.speaker_code

        return pd.NA

    utterances_transcript["prev_transcript_clean"] = utterances_transcript.apply(
        add_prev_utt,
        axis=1
    )
    utterances_transcript["prev_speaker_code"] = utterances_transcript.apply(
        add_prev_utt_speaker_code,
        axis=1
    )

    return utterances_transcript


def add_prev_utts(utterances):
    # Single-process version for debugging:
    # results = [add_prev_utts_for_transcript(utts_transcript)
    #     for utts_transcript in tqdm([group for _, group in utterances.groupby("transcript_file")])]
    utterances_grouped = [[group] for _, group in utterances.groupby("transcript_file")]
    with Pool(processes=8) as pool:
        results = pool.starmap(
            add_prev_utts_for_transcript,
            tqdm(utterances_grouped, total=len(utterances_grouped)),
        )

    utterances = pd.concat(results, verify_integrity=True)

    return utterances
