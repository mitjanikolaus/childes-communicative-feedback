import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import (
    remove_punctuation,
    str2bool,
    remove_babbling,
    ANNOTATED_UTTERANCES_FILE,
    UTTERANCES_WITH_SPEECH_ACTS_FILE, remove_events_and_non_parseable_words, replace_slang_forms, clean_disfluencies,
)
from utils import (
    remove_nonspeech_events,
    CODE_UNINTELLIGIBLE,
)

DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED = True

DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE = False

DEFAULT_MODEL_GRAMMATICALITY_ANNOTATION = "cointegrated/roberta-large-cola-krishna2020"
MODELS_ACCEPTABILITY_JUDGMENTS_INVERTED = ["cointegrated/roberta-large-cola-krishna2020"]
BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

# Speech acts that relate to nonverbal/external events
SPEECH_ACTS_NONVERBAL_EVENTS = [
    "CR",  # Criticize or point out error in nonverbal act.
    "PM",  # Praise for motor acts i.e for nonverbal behavior.
    "WD",  # Warn of danger.
    "DS",  # Disapprove scold protest disruptive behavior.
    "AB",  # Approve of appropriate behavior.
    "TO",  # Mark transfer of object to hearer
    "ET",  # Express enthusiasm for hearer's performance.
    "ED",  # Exclaim in disapproval.
]


def is_speech_related(
        utterance,
        label_partially_speech_related=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
        label_unintelligible=pd.NA,
):
    """Label utterances as speech or non-speech."""
    utterance_without_punctuation = remove_punctuation(utterance)
    utt_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)

    utt_without_nonspeech = utt_without_nonspeech.strip()
    if utt_without_nonspeech == "":
        return False

    # We exclude completely unintelligible utterances (we don't know whether it's speech-related or not)
    is_completely_unintelligible = True
    for word in utt_without_nonspeech.split(" "):
        if word != CODE_UNINTELLIGIBLE and word != "":
            is_completely_unintelligible = False
            break
    if is_completely_unintelligible:
        # By returning None, we can filter out these cases later
        return label_unintelligible

    is_partly_speech_related = len(utt_without_nonspeech) != len(
        utterance_without_punctuation
    )
    if is_partly_speech_related:
        return label_partially_speech_related

    return True


def is_intelligible(
        utterance,
        label_partially_intelligible=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
        label_empty_utterance=False,
):
    utterance_without_punctuation = remove_punctuation(utterance)
    utterance_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)
    utterance_without_nonspeech = utterance_without_nonspeech.strip()
    if utterance_without_nonspeech == "":
        return label_empty_utterance

    utt_without_babbling = remove_babbling(utterance_without_nonspeech)

    utt_without_babbling = utt_without_babbling.strip()
    if utt_without_babbling == "":
        return False

    is_partly_intelligible = len(utt_without_babbling) != len(
        utterance_without_nonspeech
    )
    if is_partly_intelligible:
        return label_partially_intelligible

    return True


def get_num_words(utt_gra_tags):
    return len([tag for tag in utt_gra_tags if tag is not None and tag["rel"] != "PUNCT"])


def annotate_grammaticality(clean_utterances, model_name, label_empty_utterance=pd.NA,
                            label_one_word_utterance=pd.NA):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    grammaticalities = np.zeros_like(clean_utterances, dtype=bool).astype(object)  # cast to object to allow for NA
    num_words = torch.tensor([len(re.split('\s|\'', utt)) for utt in clean_utterances])
    grammaticalities[(num_words == 0)] = label_empty_utterance
    grammaticalities[(num_words == 1)] = label_one_word_utterance

    utts_to_annotate = clean_utterances[(num_words > 1)]

    batches = [utts_to_annotate[x:x + BATCH_SIZE] for x in range(0, len(utts_to_annotate), BATCH_SIZE)]

    annotated_grammaticalities = []
    for batch in tqdm(batches):
        tokenized = tokenizer(list(batch), padding=True, return_tensors="pt").to(device)

        predicted_class_ids = model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask).logits.argmax(dim=-1)
        batch_grammaticalities = predicted_class_ids.bool()
        if model_name in MODELS_ACCEPTABILITY_JUDGMENTS_INVERTED:
            batch_grammaticalities = ~batch_grammaticalities
        batch_grammaticalities = batch_grammaticalities.cpu().numpy().astype(object)

        annotated_grammaticalities.extend(batch_grammaticalities.tolist())

    grammaticalities[(num_words > 1)] = annotated_grammaticalities

    return grammaticalities


def clean_preprocessed_utterance(utterance):
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

# TODO:
TEST_UTTS = [2849293, 2251309, 1512119, 1994904, 2167233, 3132184, 2788176, 3056567, 1326340, 1824540, 818333, 2714401, 610898, 2433219, 776345, 1367149, 1871279, 2718571, 297060, 1549140, 665106, 1405135, 1774546, 333982, 1839413, 867910, 376425, 310579, 1884178, 1043358, 1036396, 1416962, 1814659, 2116754, 191715, 1504385, 1347194, 1884848, 76332, 3062382, 1877540, 2740027, 1855156, 555552, 308249, 3008607, 3129315, 2417274, 1936612, 936644, 949924, 2882707, 1490513, 1143718, 2467628, 1067895, 2325737, 1979553, 1397794, 281308, 381005, 2771086, 813996, 615022, 2808463, 3136239, 1363815, 2978897, 1341806, 919576, 1856043, 1358128, 1537058, 1392641, 933772, 2395036, 2944445, 1861299, 905583, 2246053, 1114749, 526012, 2208130, 1659376, 2073162, 3124650, 235577, 375398, 557554, 277279, 1297232, 1854673, 2388858, 1789142, 2434955, 234108, 1056032, 2532608, 2107200, 2206252, 813486, 1355356, 1005499, 237077, 2220164, 187079, 1089197, 1403925, 3017063, 2528075, 647341, 1531750, 1199727, 2151845, 778877, 1720813, 596131, 1124291, 2192516, 2516742, 1113990, 204023, 1394591, 1518290, 951922, 435908, 1550019, 77014, 2287163, 2523147, 964293, 224022, 2266887, 1508453, 2020090, 2974351, 2882765, 3086425, 1962116, 2142143, 3008604, 2191525, 253716, 1479810, 2699007, 2276386, 806112, 325561, 1995670, 1036226, 1892384, 2120844, 1497046, 819645, 1516047, 708564, 3125130, 542621, 1047634, 894206, 2042328, 2435105, 1531593, 2735481, 334958, 2940054, 1393751, 1523682, 107972, 1364230, 1277722, 398692, 1752515, 2039828, 1143225, 2321833, 419433, 1211435, 1924477, 2893678, 2302777, 1737092, 381099, 1051282, 1004830, 197651, 1093579, 2807915, 867395, 1495327, 2279574, 98228, 684006, 445412, 1776479, 2810724, 701805, 376389, 2200479, 2270112, 260480, 1402737, 1140266, 399788, 737532, 951398, 411474, 108082, 1002416, 2318284, 143391, 1769566, 2279411, 700665, 2398000, 3014236, 2862651, 2882752, 1208584, 847995, 1452810, 394807, 1000503, 809542, 2727395, 1977080, 2126067, 1019159, 2970401, 2427001, 985266, 1459866, 2943022, 1331323, 837561, 2922825, 1550520, 295028, 1936216, 1125390, 1531842, 987334, 879499, 430346, 74605, 191865, 1040271, 995273, 1722474, 336402, 2173723, 2423540, 2035574, 1467912, 1367364, 198263, 2394540, 1049303, 582518, 207942, 761816, 1471827, 791953, 1719504, 2440379, 668354, 1792234, 2956252, 107799, 1938628, 607341, 718445, 605764, 2507517, 377599, 2518930, 486400, 1066895, 478346, 1475078, 2109049, 690450, 2523819, 777937, 321054, 2969902, 1113269, 2092224, 728432, 492644, 1513014, 1499792, 655981, 1022068, 666200, 838870, 597532, 616828, 2078778, 2813789, 1098440, 1481899, 2382292, 40911, 2971534, 898467, 941494, 1065224, 1487575, 475598, 1722648, 368807, 2528158, 246302, 1043548, 1130885, 1979680, 1186334, 1766340, 234755, 1355977, 2181951, 1009930, 438597, 1063388, 2479577, 2245982, 130090, 961783, 3019427, 2269663, 2088404, 310434, 381422, 835315, 1332923, 259751, 1945057, 1501548, 528750, 1827298, 1111591, 2889768, 1067172, 2522336, 990821, 618190, 1120110, 820330, 1012121, 1312824, 374353, 2498710, 2416560, 277340, 559115, 2954511, 316879, 2112300, 1150119, 1242908, 1394477, 1484430, 929295, 1918792, 539164, 1725662, 1179623, 539780, 2096241, 1994960, 2058865, 1934723, 2033865, 3027112, 260350, 1964862, 2401847, 2937667, 1863359, 2443196, 761016, 811607, 713048, 2222473, 786554, 1911021, 1927529, 1288455, 777084, 162254, 2062511, 1008518, 2198068, 231992, 498640, 989503, 2518404, 1075592, 127663, 2793683, 2151944, 2518078, 646858, 1729559, 1904401, 751092, 1066722, 2187000, 574542, 701315, 1091299, 516734, 1082096, 787748, 89791, 535911, 402076, 2107131, 2940752, 1215520, 339745, 924903, 467621, 2392146, 949479, 2250403, 757188, 2168299, 2069061, 702106, 2394301, 341234, 1834090, 901471, 2753545, 1133676, 1415265, 1797920, 2429839, 1478057, 1922396, 3141024, 2717088, 2434661, 1260088, 225315, 2220070, 1009601, 611939, 274246, 94301, 2532572, 2212032, 1399745, 441491, 2851090, 1757900, 1153746, 1677513, 466817, 1017335, 1117037, 916657, 2501018, 2201126, 2021828, 1771964, 1102049, 1235179, 307729, 354611, 1310503, 2069640, 1829946, 275200, 775339, 2477742, 2924526, 2057481, 2849825, 621788, 2750769, 1399737, 1095273, 670252, 2773824, 2294038, 688514, 1825321, 1484238, 3123153, 680087, 2747232, 1941928, 1467556, 472463, 437222, 436751, 1082645, 441310, 641401, 1289423, 3066608, 2505486, 1046000, 1140534, 1032207, 1314671, 1795322, 1912125, 317238, 1395023, 911769, 317438, 402365, 97356, 858831, 136721, 1061418, 1521426, 1995968, 2528979, 682198, 3117947, 1690178, 141013, 2087299, 1866031, 2881644, 591432, 2944905, 2926422, 32885, 2405246, 619609, 1552185, 3113594, 1185502, 1123276, 1998461, 2042474, 1950604, 1307630, 2061639, 2172168, 1946196, 2206111, 889758, 2884088, 1043261, 3076557, 506112, 2752764, 277609, 2337228, 527828, 2026646, 695394, 1107816, 2055846, 1939059, 704946, 1050468, 531384, 1977535, 2191229, 1471699, 2476071, 1816546, 135797, 1771906, 336704, 1351099, 2997501, 1340635, 442315, 507546, 1556258, 3080057, 1043271, 2260776, 1866604, 2401242, 919091, 1357636, 2776916, 3028529, 1472164, 1773819, 2517050, 933255, 787221, 1321150, 598697, 136293, 1081698, 98927, 2532850, 2053472, 666198, 1528621, 1141155, 2918683, 1249211, 614722, 2897366, 724010, 2407476, 668811, 1191562, 569389, 1467157, 2526334, 3135611, 557158, 932724, 2365746, 354780, 2156650, 878534, 779325, 2170554, 1796564, 2442940, 1055114, 911639, 1409111, 2481463, 1333009, 1081218, 1348849, 1239341, 1765099, 350520, 2218650, 1241568, 538887, 1454566, 1876149, 180247, 1727457, 389700, 957670, 1824704, 1023944, 2262690, 886292, 947204, 2680294, 812003, 2220571, 1523697, 835320, 1299283, 2700086, 1075813, 2503737, 1480645, 2524814, 677734, 2363932, 2737840, 2273281, 1301062, 1946570, 2309918, 2984437, 991303, 2515863, 1021426, 962815, 1211399, 597577, 762461, 2528532, 854625, 258461, 728065, 2911832, 426598, 2532355, 1135002, 1309915, 911323, 578374, 409625, 2206872, 2232139, 1303323, 2239614, 2097909, 733965, 1012722, 974238, 2495998, 1322033, 440245, 585531, 2023752, 1787253, 1332509, 206050, 1045746, 944115, 1182232, 1128105, 2106004, 1411527, 1219604, 361808, 2092625, 912909, 2476335, 2255858, 2086748, 1050601, 1133860, 1204929, 818326, 1511319, 1483873, 1113657, 666008, 1046126, 1368160, 29974, 753992, 939221, 1207010, 1751770, 1192095, 2515956, 2996507, 976979, 771038, 2911315, 1334343, 2520353, 3116013, 2432012, 2006704, 2139386, 945319, 2403129, 46281, 364203, 3025155, 2037849, 1260188, 2109771, 509053, 315232, 3135220, 3030264, 950066, 100482, 847607, 827247, 1910848, 1523527, 1194936, 1921507, 1416314, 3135474, 1129226, 2319987, 862397, 1376639, 2520014, 1271456, 1031775, 36867, 1052763, 943410, 1003481, 2681454, 714630, 2418466, 2996905, 313844, 421890, 2100055, 1539974, 1694596, 2100963, 3053208, 493696, 2206737, 1360725, 132961, 1545407, 3101476, 738651, 907983, 2323652, 1014925, 2007721, 1085433, 2393564, 2200155, 1842211, 2784490, 2427380, 1018072, 836096, 2720802, 639945, 740823, 1024287, 579825, 3132808, 2162950, 134636, 386942, 660699, 199652, 2718589, 574097, 1710124, 957901, 1198989, 847525, 1486218, 1104258, 1019821, 761256, 1992386, 154034, 1786113, 906488, 1393877, 498914, 894073, 1549333, 630502, 1982670, 930152, 2403007, 1455515, 1856423, 139322, 1801899, 3058732, 2947279, 378323, 491451, 262725, 551737, 1334144, 796210, 1529779, 2420953, 1542219, 257233, 52309, 963134, 751919, 157157, 3071150, 1454217, 2712696, 2067674, 1213859, 2034833, 1338075, 463045, 429712, 1036388, 537884, 74595, 2013370, 1021989, 66576, 2107338, 1908490, 1197475]


def annotate(args):
    utterances = pd.read_pickle(UTTERANCES_WITH_SPEECH_ACTS_FILE)

    # TODO remove:
    utterances = utterances[utterances.speaker_code == "CHI"]
    # utterances = utterances.loc[TEST_UTTS]


    print("Annotating speech-relatedness..")
    utterances = utterances.assign(
        is_speech_related=utterances.transcript_raw.apply(
            is_speech_related,
            label_partially_speech_related=args.label_partially_speech_related,
        )
    )
    utterances.is_speech_related = utterances.is_speech_related.astype("boolean")

    print("Annotating intelligibility..")
    utterances = utterances.assign(
        is_intelligible=utterances.transcript_raw.apply(
            is_intelligible,
            label_partially_intelligible=args.label_partially_intelligible,
        )
    )
    #
    print("Cleaning utterances..")
    utterances = utterances.assign(
        utt_clean=utterances.transcript_raw.apply(
            clean_preprocessed_utterance
        )
    )

    # num_words = np.array([len(re.split('\s|\'', utt)) for utt in utterances.utt_clean.values])
    # utts_to_annotate = utterances[(num_words > 1)]
    # utts_to_annotate = utts_to_annotate[utts_to_annotate.is_speech_related & utts_to_annotate.is_intelligible]
    # utterances = utts_to_annotate.sample(1000, random_state=1)

    # print("Annotating grammaticality..")
    # utterances["is_grammatical"] = annotate_grammaticality(utterances.utt_clean.values,
    #                                                        args.grammaticality_annotation_model)
    # utterances.is_grammatical = utterances.is_grammatical.astype("boolean")


    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--label-partially-speech-related",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
        help="Label for partially speech-related utterances: Set to True to count as speech-related, False to count as "
             "not speech-related or None to exclude these utterances from the analysis",
    )
    argparser.add_argument(
        "--label-partially-intelligible",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
        help="Label for partially intelligible utterances: Set to True to count as intelligible, False to count as unintelligible or None to exclude these utterances from the analysis",
    )
    argparser.add_argument(
        "--grammaticality-annotation-model",
        type=str,
        default=DEFAULT_MODEL_GRAMMATICALITY_ANNOTATION,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    annotated_utts = annotate(args)

    file_path_with_model = f"{ANNOTATED_UTTERANCES_FILE.split('.p')[0]}_{args.grammaticality_annotation_model.replace('/', '_')}.p"
    os.makedirs(os.path.dirname(ANNOTATED_UTTERANCES_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(file_path_with_model), exist_ok=True)

    annotated_utts.to_pickle(ANNOTATED_UTTERANCES_FILE)
    annotated_utts.to_pickle(file_path_with_model)
    annotated_utts.to_csv(file_path_with_model.replace(".p", ".csv"))
