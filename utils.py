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

UTTERANCES_WITH_PREV_UTTS_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_with_prev_utts.csv"
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
IS_PAUSE = lambda word: word == "..."
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

    if IS_PAUSE(word):
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


ERR_SUBJECT = "subject"
ERR_VERB = "verb"
ERR_OBJECT = "object"
ERR_POSSESSIVE = "possessive"
ERR_PLURAL = "plural"
ERR_SV_AGREEMENT = "sv_agreement"
ERR_PAST = "past"
ERR_DETERMINER = "determiner"
ERR_PREPOSITION = "preposition"
ERR_AUXILIARY = "auxiliary"
ERR_PRESENT_PROGRESSIVE = "present_progressive"
ERR_OTHER = "other"
ERR_UNKNOWN = "unk"

ERRORS_PAST_TENSE = {('catch', 'caught'), ('draw', 'drew'), ("i'm", "i was"), ('drink', 'drank'), ("go splash", "splashed"), ("winned", "won"), ("gotted", "got"), ("buyeded", "bought"), ("give", "gave"), ("sticked", "stuck"), ("weared", "wore"), ("holded", "held"), ("stinged", "stung"), ("drinked", "drunk"), ("quitted", "quit"), ("safeded", "saved"), ("safed", "saved"), ("aten", "ate"), ("stayded", "stayed"), ("closded", "closed"), ("wearded", "wore"), ("writed", "written"), ("brokened", "broken"), ("brokened", "broke"), ("takened", "took"), ("falled", "fallen"), ("goded", "went"), ("waked", "woken"), ("stucked", "stuck"), ("led", "let"), ("heared", "heard"), ("leaved", "left"), ("meeted", "met"), ('wrote', 'written'), ('writed', 'written'), ('cry', 'cried'), ('calls', 'called'), ('brung', 'brought'), ('bite', 'bit'), ('knowed', 'knew'), ('letted', 'let'), ('go', 'gone'), ('told', 'telled'), ('shutted', 'shut'), ('lend', 'lent'), ('said', 'goed'), ('meaned', 'meant'), ('growed', 'grew'), ('singed', 'sang'), ('broken', 'broke'), ('sang', 'singed'), ("jumpded", "jumped"), ('went', 'goed'), ('stuck', 'stucked'), ("don't", "didn't"), ('goed', 'said'), ('stick', 'stuck'), ('flieded', 'flied'), ('wear', 'wore'), ('took', 'taked'), ('can', 'could'), ('doed', 'did'), ('stuck', 'stick'), ('said', 'sayed'), ('wore', 'wear'), ('did', 'do'), ('blewed', 'blew'), ('slept', 'sleept'), ('lost', 'lose'), ('breaked', 'broke'), ('ate', 'eaten'), ('getted', 'got'), ('bought', 'buyed'), ('sitted', 'sat'), ('writed', 'wrote'), ('eat', 'ate'), ('leaned', 'leant'), ('eated', 'ate'), ('leaped', 'leapt'), ('stung', 'stunged'), ('told', 'tolded'), ('dranked', 'drank'), ('could', 'can'), ('came', 'come'), ('go', 'went'), ('came', 'comed'), ("didn't", "don't"), ('comed', 'came'), ('spat', 'spit'), ('bent', 'bented'), ('readed', 'read'), ('fixeded', 'fixed'), ('ride', 'ridden'), ('done', 'did'), ('had', 'haved'), ('written', 'wrote'), ('read', 'readed'), ('choose', 'chose'), ('hid', 'hided'), ('breaked', 'broken'), ('bented', 'bent'), ('broke', 'broked'), ('runned', 'ran'), ('hided', 'hid'), ('dugged', 'dug'), ('built', 'builded'), ('slept', 'sleeped'), ('sayed', 'said'), ('broken', 'broked'), ('sawed', 'saw'), ('sawed', 'sawded'), ('come', 'came'), ('flew', 'flyed'), ('gone', 'goned'), ('left', 'leave'), ('said', 'say'), ('falled', 'fell'), ('sat', 'sit'), ('chose', 'choose'), ('burned', 'burnt'), ('lost', 'losed'), ('ated', 'ate'), ('goed', 'went'), ('ate', 'ated'), ('lose', 'lost'), ('drawed', 'drew'), ('drew', 'draw'), ('builded', 'built'), ('caught', 'catched'), ('blew', 'blewed'), ('throwed', 'threw'), ('threw', 'throwed'), ('brokened', 'broken'), ('fell', 'fallen'), ('drinked', 'drank'), ('ate', 'et'), ('leapt', 'leaped'), ('ate', 'eat'), ('made', 'maded'), ('forgat', 'forgot'), ('broked', 'broke'), ('sleept', 'slept'), ('dug', 'dugged'), ('fall', 'fell'), ('got', 'get'), ('losed', 'lost'), ('see', 'seen'), ('broke', 'breaked'), ('do', 'did'), ('founded', 'found'), ('goed', 'went like'), ('take', 'took'), ('seed', 'saw'), ('flied', 'flieded'), ('bringed', 'brought'), ('bringeded', 'brought'), ("leaved", "left"), ("cutted", "cut"), ("showded", "showed"), ("buyded", "bought"), ("haveded", "had"), ("pickted", "picked"), ('buyed', 'bought'), ('broken', 'breaked'), ('sent', 'sended'), ('burnt', 'burned'), ('slid', 'slided'), ('took', 'taken'), ('fell', 'fall'), ('eaten', 'ate'), ('et', 'ate'), ('see', 'saw'), ('hold', 'held'), ('leant', 'leaned'), ('stucked', 'stuck'), ('swam', 'swimmed'), ('broked', 'broken'), ('haved', 'had'), ('putted', 'put'), ('brought', 'bringed'), ('hit', 'hitted'), ('say', 'said'), ('made', 'make'), ('felled', 'fell'), ('camed', 'came'), ('ridden', 'ride'), ('sang', 'sanged'), ('drink', 'drunk'), ('spit', 'spat'), ('drew', 'drawed'), ('fell', 'felled'), ('waked', 'woke'), ('bent', 'bended'), ('have', 'had'), ('fixed', 'fixeded'), ('sleeped', 'slept'), ('babysitted', 'babysat'), ('spitted', 'spat'), ('catched', 'caught'), ('wrote', 'writed'), ('buy', 'bought'), ('maked', 'made'), ('maded', 'made'), ('goed', 'got'), ('held', 'hold'), ('got', 'gat'), ('flied', 'flew'), ('stunged', 'stung'), ('made', 'maked'), ('woke', 'waked'), ('bited', 'bit'), ('grew', 'growed'), ('broken', 'brokened'), ('flew', 'flied'), ('shut', 'shutted'), ('telled', 'told'), ('swimmed', 'swam'), ('cwied', 'cried'), ('spat', 'spitted'), ('flewed', 'flew'), ('went like', 'goed'), ('make', 'made'), ('went', 'go'), ('ran', 'runned'), ('swam', 'swammed'), ('flyed', 'flew'), ('leave', 'left'), ('fallen', 'fell'), ('got', 'getted'), ('hitted', 'hit'), ('bit', 'bited'), ('broke', 'broken'), ('cried', 'cwied'), ('gat', 'got'), ('taken', 'took'), ('ate', 'eated'), ('took', 'take'), ('get', 'got'), ('saw', 'seed'), ('goned', 'gone'), ('tolded', 'told'), ('swammed', 'swam'), ('had', 'have'), ('put', 'putted'), ('babysat', 'babysitted'), ('did', 'doed'), ('flew', 'flewed'), ('sawded', 'sawed'), ('taked', 'took'), ('drank', 'dranked'), ('saw', 'sawed'), ('sat', 'sitted'), ('fell', 'falled'), ('forgot', 'forgat'), ('sit', 'sat'), ('bought', 'buy'), ('drunk', 'drink'), ('came', 'camed'), ('saw', 'see'), ('meant', 'meaned'), ('seen', 'see'), ('did', 'done'), ('slided', 'slid'), ('sanged', 'sang'), ('drank', 'drinked'), ('found', 'founded'), ('sended', 'sent'), ('got', 'goed'), ('bended', 'bent')}
ERRORS_PLURAL = [('mans', 'men'), ('mans@c', 'men'), ('baby', 'babies'), ("that's", "they're"), ("name", "names"), ("voice", "voices"), ('this', 'these'), ("wristes", "wrists"), ("half", "halves"), ("bodies", "body"), ("one", "ones"), ("ghostes", "ghosts"), ('mens', 'men'), ('womans', 'women'), ('policemens', 'policemen'), ('firemens', 'firemen'), ('firemans', 'firemen'), ("snowmans", "snowman"),  ('foremans', 'foremen'), ('mouses', 'mice'), ('sheeps', 'sheep'), ('foots', 'feet'), ('foots', 'foot'), ('knifes', 'knives'), ('‹barefeeted', 'barefoot')]
ERRORS_SV_AGREEMENT = [("be", "are"), ("sneez", "sneezes"), ("want", "wants"), ("i's", "i am"), ("coughes", "coughs"), ("mean", "means"), ("be", "am"), ("be", "is"), ("like", "likes"), ("babysit", "babysits"), ("light", "lights"), ("are", "is"), ("are", "am"), ("does", "do"), ("do", "does"), ("weren't", "wasn't"), ("have", "has"), ("there's", "there're"), ("where's", "where're"), ("was", "were"), ("don't", "doesn't"), ('is', 'are'), ('gots', 'got'), ("live", "lives"), ("present", "presents"), ("here's", 'here are'), ("there's", "there are")]
ERRORS_PREPOSITION = [("for", "to"), ("by", "with"), ("a", "to"), ("to", "into"), ("to", "to"), ("with", "to"), ("your", "to"), ("in", "at"), ("the", "as"), ("at", "in"), ("up", "out"), ("a", "for"), ("on", "to"), ('a', 'of'), ('at', 'to'), ('want', 'want to'), ('to', 'at'), ('in', 'on')]
ERRORS_AUXILIARY = [("has", "does"), ("can", "have"), ("is", "has"), ("is", "have"), ("do", "have"), ("do", "won't"), ("do", "will"), ("haven't", "isn't"), ('not', "don't"), ("i'm", "i've"), ("not", "doesn't"), ("is", "has"), ("we", "we'd"), ("aren't", "hasn't"), ("go", "will"), ("i'm", "i've")]
ERRORS_DIALECT = [("them", "those"), ("ain't", "isn't"), ("ma", "my"), ("ya", "you"), ('gonna', 'going to'), ('ya', 'you'), ('yep', 'yes'), ('lemme', 'let me'), ('kinda', 'kind of'), ('yup', 'yes'), ('dyou', 'do you'), ('lil', 'little'), ('bout', 'about'), ('dunno', "don't know"), ("d'you", 'do you'), ('yer', 'your'), ('cause', 'because'), ("where'd", 'where did'), ('nope', 'no'), ("let's", 'let us'), ('em', 'them'), ('wouldja', 'would you'), ('da', 'the'), ('whada', 'what do'), ("'cause", 'because'), ("what're", 'what are'), ('dya', 'do you'), ('gon(na)', 'going to'), ('wanna', 'want to'), ('djou', 'do you'), ('didjou', 'did you'), ('cmon', 'come on'), ("c'mon", 'come on'), ('whatcha', 'what are you'), ('gimme', 'give me'), ('ta', 'to'), ("'em", 'them'), ('wanna', 'want a'), ("c'mere", 'come here'), ('yea:h', 'yes'), ('yeah', 'yes'), ('mkay', 'okay'), ('outta', 'out of'), ('wouldjou', 'would you'), ('lotta', 'lot of'), ('cmere', 'come here'), ('kay', 'okay'), ('gotcha', 'got you'), ('havta', 'have to'), ('whaddya', 'what do you'), ("dat's", 'that (i)s'), ('comere', 'come here'), ('yeah:', 'yes'), ('mhm', 'yes'), ('lookit', 'look'), ("s'more", 'some more'), ('gotta', 'got to'), ('cuz', 'because'), ("you're", 'you are'), ('wan(na)', 'wanna'), ('dontcha', "don't you"), ('howbout', 'how about'), ('whadyou', 'what do you'), ("what's", 'what does'), ('comon', 'come on'), ('camere', 'come here'), ('<gonna', 'going to'), ("com'ere", 'come here'), ('whad', 'what'), ('<dis', 'this'), ('wha', 'what'), ("d'ya", 'do you'), ('wannoo', 'wanna'), ('ya', 'your'), ('getcha', 'get you'), ('wiff', 'with'), ("don't", 'do not'), ('scuse', 'excuse'), ("y'know", 'you know'), ('ya', 'yes'), ('didja', 'did you'), ('whadya', 'what do you'), ('betcha', 'bet you'), ("how'd", 'how did'), ("jumpin'", 'jumping'), ('mmkay', 'okay'), ("how's", 'how does'), ("'bout", 'about'), ('whatcha', 'what do you'), ('cha', 'you'), ('sorta', 'sort of'), ('whaddaya', 'what do you'), ('<de', 'the'), ('putcher', 'put your'), ('k', 'okay'), ('mm+kay', 'okay'), ('wight', 'right'), ("you'd", 'you would'), ('til', 'until'), ("where're", 'where are'), ("n'", 'and'), ('whatcha', 'what you'), ('en', 'and'), ("doin'", 'doing'), ("y'wanna", 'you want to'), ('doncha', "don't you"), ("that'd", 'that would'), ('ye:ah', 'yes'), ('yea', 'yes'), ('sa', "what's that"), ('gotta', 'have got to'), ('udder', 'other'), ('peam', 'cream'), ("where's", 'where are'), ('nah', 'no'), ("doesn't", 'does not'), ("i'm", 'i am'), ('whadda', 'what do'), ('gotchu', 'got you'), ("dere's", 'there (i)s'), ("g'head", 'go ahead'), ('getcher', 'get your'), ('whadyou', 'what are you'), ('woujou', 'would you'), ('outta', 'out_of'), ("come'ere", 'come here'), ("who're", 'who are'), ('till', 'until'), ('dijou', 'did you'), ("we're", 'we are'), ('yessie', 'yes'), ("i've", 'i have'), ("they're", 'they are'), ('donna', 'going to'), ('cos', 'because'), ("mommy's", 'mommy has'), ('peejays', 'pajamas'), ("you've", 'you have'), ('gotta', 'got a'), ('lotsa', 'lots_of'), ('gatta', 'gotta'), ('wan', 'want'), ("<dat's", 'that (i)s'), ('likkle', 'little'), ('gahead', 'go ahead'), ('yeh', 'yes'), ("wha's", 'what does'), ('‹ya', 'yes'), ('aminal', 'animal'), ("why're", 'why are'), ('whadja', 'what did you'), ("where's", 'where does'), ('longit', 'longitudinal'), ("dere's", 'there is'), ("wha's", 'what is'), ('n', 'and'), ('inta', 'into'), ('ywanna', 'you want to'), ('hasta', 'has to'), ('donchou', "don't you"), ("an'", 'and'), ('der', 'there')]


def categorize_error(word_error, word_corrected, row=None):
    if (word_error, word_corrected) in [("me", "i've"), ("me", "i'll"), ("my", "i'm"), ("this", "he's"), ("they're", "he's")]:
        errors = [ERR_SUBJECT, ERR_VERB]
        return errors
    elif (word_error, word_corrected) in [("do", "went"), ("coming", "getting"), ("lay", "lie"), ("where", "what"),
                                          ("people", "person"), ("what", "how"), ("winning", "beating"),
                                          ("wet", "rinse"), ("hear", "listen"), ("get", "put"), ("go", "say"),
                                          ("did", "said"), ("push", "put"), ("lie", "lay"), ("dying", "killing"),
                                          ("round", "down"), ("taking", "teaching"), ("take", "pick"), ("got", "been"),
                                          ("student", "stands"), ("drop", "hit"), ("sheep", "sweep")]:
        # Semantic error
        return []
    elif "$lex" in word_corrected:
        return []   # Lexical error
    elif "$pho" in word_corrected:
        return []   # Phonological error
    elif word_corrected in ["?", "[?]"]:
        return [ERR_UNKNOWN]
    word_corrected = word_corrected.replace("[?]", "").strip()
    word_error = word_error.replace("[?]", "").strip()
    if (word_error, word_corrected) in ERRORS_DIALECT or word_error == word_corrected.replace("th", "d") or "@d" in word_error or word_corrected == word_error.replace("'ll", " will") or word_corrected == word_error.replace("'s", " is") or word_corrected == word_error.replace("'d", " did"):
        # Dialect
        return []
    elif word_corrected in [word_error + suffix for suffix in ["'ve", "'ll"]]:
        err = ERR_VERB
    elif (word_error, word_corrected) in [("i", "i'm"), ("is", "is"), ("read", "read"), ("threw", "threw"), ("in", "is"), ("got", "has"), ("got", "have"), ("it", "is")]:
        err = ERR_VERB
    elif (word_error, word_corrected) in ERRORS_AUXILIARY:
        err = ERR_AUXILIARY
    elif (word_error, word_corrected) in ERRORS_SV_AGREEMENT:
        err = ERR_SV_AGREEMENT
    elif (word_error, word_corrected) in ERRORS_PREPOSITION:
        err = ERR_PREPOSITION
    elif (word_error, word_corrected) in [("a", "an"), ("an", "a"), ("a", "a"), ("the", "the"), ("a", "the"), ("some", "a"), ("my", "the"), ("it", "this"), ("", "a")]:
        err = ERR_DETERMINER
    elif word_corrected == word_error + "'s" or word_error == word_corrected + "'s":
        err = ERR_POSSESSIVE
    elif (word_error, word_corrected) in [("me", "my"), ("your", "my"), ("yours", "your"), ("it's", "her"), ("I", "my"), ("him", "his")]:
        err = ERR_POSSESSIVE
    elif (word_error, word_corrected) in ERRORS_PLURAL:
        err = ERR_PLURAL
    elif word_corrected in [word_error + suffix for suffix in ["ed", "d", "ped"]] or word_error in [word_corrected + suffix for suffix in ["ed", "d", "ped"]]:
        err = ERR_PAST
    elif (word_error, word_corrected) in ERRORS_PAST_TENSE:
        err = ERR_PAST
    elif word_corrected in [word_error + suffix for suffix in ["ing", "ting", "ning"]] or word_error in [word_corrected + suffix for suffix in ["ing", "ting", "ning"]] or word_corrected in [word_error[:-1] + "ing"] or word_error in [word_corrected[:-1] + "ing"] or word_corrected in [word_error[:-2] + "ing"]:
        err = ERR_PRESENT_PROGRESSIVE
    elif (word_error, word_corrected) in [("broking", "breaking")]:
        err = ERR_PRESENT_PROGRESSIVE
    elif (word_error, word_corrected) in [("me", "i"), ("i", "we"), ("you", "he"), ("it", "they"), ("i", "i"), ("him", "he"), ("them", "they"), ("my", "i"), ("who", "what"), ("he's", "it's"), ("they", "he"), ("what's", "that's")]:
        err = ERR_SUBJECT
    elif (word_error, word_corrected) in [("it", "her"), ("the", "them"), ("what", "that")]:
        err = ERR_OBJECT
    elif (word_error, word_corrected) in [("himself", "herself"), ("herself", "himself"), ('theirselves', 'themselves'), ('hisself', 'himself'), ('themself', 'themselves'), ("too", "either"), ("badder", "worse"), ("gooder", "better"), ("taller", "tallest"), ("loud", "louder"), ("much", "many"), ("furry", "fur"), ("really", "real"), ("poison", "poisonous"), ("no", "any"), ("littler", "smaller"), ("anything", "something")]:
        err = ERR_OTHER
    elif word_corrected in [word_error + suffix for suffix in ["s", "es"]] or word_error in [word_corrected + suffix for suffix in ["s", "es"]]:
        # find pos in tokens
        if row is not None and (word_error in row["tokens"] or word_corrected in row["tokens"]):
            if word_error in row["tokens"]:
                pos = row["pos"][row["tokens"].index(word_error)]
            else:
                pos = row["pos"][row["tokens"].index(word_corrected)]
            pos = pos.split(":")[0]
            if pos in ["v", "co"]:
                err = ERR_SV_AGREEMENT
            elif pos == "n":
                err = ERR_PLURAL
            else:
                err = ERR_UNKNOWN
        else:
            err = ERR_UNKNOWN
    else:
        words_error = word_error.split(" ")
        words_corrected = word_corrected.split(" ")
        if len(words_error) > 0 and set(words_corrected) == set(words_error) and words_error != words_corrected:
            err = ERR_OTHER  # word order error
        else:
            err = ERR_UNKNOWN

    return [err]


def replace_actually_said_words(utterance):
    # Remove incomplete annotations
    utterance = utterance.replace("[= actually says]", "")

    # Recursively replace words
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

        errors = categorize_error(replacement, words_to_replace)
        if len(errors) == 0:
            # No grammatical error, most probably semantic
            utterance = " ".join([before, words_to_replace, after])
        else:
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
    utterance = re.sub(r"\.\.\.(\s\.\.\.)*", "...", utterance)

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


DISFLUENCIES = ["uhm", "um", "uh", "erh", "err", "aw", "ehm", "hm"]


def clean_disfluencies(utterance):
    words = utterance.split(" ")
    words = [word for word in words if not word.replace(",", "") in DISFLUENCIES]
    utterance = " ".join(words)
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


def add_prev_utts_for_transcript(utterances_transcript, num_utts=1, add_speaker_codes=True):
    utts_speech_related = utterances_transcript[utterances_transcript.is_speech_related.isin([pd.NA, True])]

    def add_prev_utt(utterance):
        if utterance.name in utts_speech_related.index:
            row_number = np.where(utts_speech_related.index.values == utterance.name)[0][0]
            if row_number > 0:
                prev_utts = utts_speech_related.loc[utts_speech_related.index[:row_number][-num_utts:]]
                return " ".join(prev_utts.transcript_clean)

        return pd.NA

    def add_prev_utt_speaker_codes(utterance):
        if utterance.name in utts_speech_related.index:
            row_number = np.where(utts_speech_related.index.values == utterance.name)[0][0]
            if row_number > 0:
                prev_utts = utts_speech_related.loc[utts_speech_related.index[:row_number][-num_utts:]]
                return " ".join(prev_utts.speaker_code)

        return pd.NA

    utterances_transcript["prev_transcript_clean"] = utterances_transcript.apply(
        add_prev_utt,
        axis=1
    )

    if add_speaker_codes:
        utterances_transcript["prev_speaker_code"] = utterances_transcript.apply(
            add_prev_utt_speaker_codes,
            axis=1
        )

    return utterances_transcript


def add_prev_utts(utterances, num_utts=1):
    # Single-process version for debugging:
    # results = [add_prev_utts_for_transcript(utts_transcript, num_utts)
    #     for utts_transcript in tqdm([group for _, group in utterances.groupby("transcript_file")])]
    utterances_grouped = [[group, num_utts] for _, group in utterances.groupby("transcript_file")]
    with Pool(processes=8) as pool:
        results = pool.starmap(
            add_prev_utts_for_transcript,
            tqdm(utterances_grouped, total=len(utterances_grouped)),
        )

    utterances = pd.concat(results)

    return utterances
