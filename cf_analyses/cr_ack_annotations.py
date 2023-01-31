import os
import pickle

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk import SnowballStemmer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support
from utils import split_into_words

from utils import PROJECT_ROOT_DIR


RESULTS_DIR = PROJECT_ROOT_DIR+"/results/cr_ack/"

# "right" is only considered an acknowledgement if alone, otherwise it can be confounded with
# expressions such as "right there" or "right here"
RESPONSES_ACKNOWLEDGEMENT_IF_ALONE = {"right"}

# We include words like "yes" which can be used as responses, as we check that the previous utterance was not a question
RESPONSES_ACKNOWLEDGEMENT_CERTAIN = {"uhhuh", "uhuh", "uhhum", "mhm", "mm", "hm", "huh", "ummhm", "sure", "okay", "ok",
                                     "'kay", "kay", "alright", "yes", "yep", "yeah", "yeh", "yup", "ahhah", "ah"}
RESPONSES_ACKNOWLEDGEMENT_MULTI_WORD = {"I see", "I see", "I know", "all right", "go ahead", "go on then", "go on",
                                        "good job", "very good", "it is", "it's alright", "that's right", "that's true"}
RESPONSES_ACKNOWLEDGEMENT_EVALUATIVE = {"oh", "ooh", "wow", "uhoh", "actually", "excellent", "exactly", "good", "great",
                                        "hey", "hooray", "oops", "whoops", "woo", "woohoo", "yay"}
ALL_ACKS_SINGLE_WORD = RESPONSES_ACKNOWLEDGEMENT_CERTAIN | RESPONSES_ACKNOWLEDGEMENT_EVALUATIVE | RESPONSES_ACKNOWLEDGEMENT_IF_ALONE

SPEECH_ACTS_CLARIFICATION_REQUEST = [
    "EQ",  # Eliciting question (e.g. hmm?).
    "RR",  # Request to repeat utterance.
]

# List of stopwords to be ignored for repetition calculation
STOPWORDS = {'my', 'doing', 'than', 'doesn', 'do', 'him', 's', 'her', 'won', 'myself', 'his', 'were', 'during', 'few', 'yourself', 'mightn', 'into', 'we', 'above', 'below', 'you', 'what', 'has', 'under', 'each', 'before', 'am', 'after', 'me', 'once', 'out', 'y', 'have', 'ain', 'of', 'will', 'weren', 'with', 'no', 'm', 'whom', 'only', 'ours', 'nor', 'mustn', 'himself', 're', 'was', 'o', 'having', 'for', 'ourselves', 'theirs', 'ma', 'off', 'too', 'i', 'further', 'hadn', 'wasn', 'their', 'more', 'or', 'them', 'again', 't', 'against', 'own', 'those', 'hers', 'does', 've', 'its', 'herself', 'over', 'not', 'should', 'aren', 'that', 'our', 'as', 'been', 'who', 'while', 'to', 'hasn', 'through', 'about', 'haven', 'how', 'can', 'and', 'they', 'in', 'until', 'had', 'an', 'between', 'then', 'both', 'shouldn', 'this', 'down', 'don', 'now', 'yourselves', 'he', 'couldn', 'a', 'where', 'themselves', 'other', 'these', 'wouldn', 'the', 'because', 'but', 'your', 'why', 'up', 'by', 'if', 'most', 'she', 'be', 'is', 'just', 'any', 'such', 'very', 'all', 'are', 'on', 'didn', 'itself', 'll', 'so', 'yours', 'same', 'needn', 'd', 'which', 'isn', 'some', 'here', 'it', 'when', 'at', 'from', 'did', 'being', 'there', 'oh', 'ooh', 'huh', 'ah', 'mhm', 'mm', 'shan'}
STOPWORDS = STOPWORDS | RESPONSES_ACKNOWLEDGEMENT_CERTAIN

stemmer = SnowballStemmer("english")


CAREGIVER_NAMES = [
    "dad",
    "daddy",
    "dada",
    "mom",
    "mum",
    "mommy",
    "mummy",
    "mama",
    "mamma",
    "nin",
]

ACK_CLASSIFIER_FILE = PROJECT_ROOT_DIR+"/data/feedback_classifiers/ack_classifier_weights.p"
CF_CLASSIFIER_FILE = PROJECT_ROOT_DIR+"/data/feedback_classifiers/cf_classifier_weights.p"


def annotate_crs_and_acks(conversations):
    annotate_repetition_ratios(conversations)

    conversations["response_is_clarification_request_speech_act"] = conversations.apply(
        is_clarification_request_speech_act, axis=1)
    conversations["response_is_repetition_clarification_request"] = conversations.apply(
        is_repetition_clarification_request, axis=1)

    conversations["response_is_keyword_acknowledgement"] = conversations.apply(is_keyword_acknowledgement, axis=1)
    conversations["response_is_repetition_acknowledgement"] = conversations.apply(is_repetition_acknowledgement, axis=1)

    conversations["response_is_clarification_request"] = conversations.apply(response_is_clarification_request, axis=1)
    conversations["response_is_acknowledgement"] = conversations.apply(response_is_acknowledgement, axis=1)


def annotate_repetition_ratios(conversations):
    repetition_ratios = conversations.apply(get_repetition_ratios, stem_words=True, axis=1)
    conversations["rep_utt_stemmed"] = repetition_ratios.apply(lambda ratios: ratios[0])
    conversations["rep_response_stemmed"] = repetition_ratios.apply(lambda ratios: ratios[1])

    repetition_ratios = conversations.apply(get_repetition_ratios, stem_words=False, axis=1)
    conversations["rep_utt"] = repetition_ratios.apply(lambda ratios: ratios[0])
    conversations["rep_response"] = repetition_ratios.apply(lambda ratios: ratios[1])


def is_keyword_acknowledgement(micro_conv):
    if micro_conv["utt_transcript_clean"][-1] != "?":
        if micro_conv["response_transcript_clean"][-1] != "?":
            response = micro_conv["response_transcript_clean"].lower()
            words = [word.lower() for word in split_into_words(response, split_on_apostrophe=False, remove_commas=True,
                                                               remove_trailing_punctuation=True)]

            if len(words) > 0:
                response_without_bcs = [word for word in words if word not in ALL_ACKS_SINGLE_WORD]
                if len(response_without_bcs) == 0:
                    return True
                elif words[0] in RESPONSES_ACKNOWLEDGEMENT_CERTAIN:
                    return True
                elif " ".join(words) in RESPONSES_ACKNOWLEDGEMENT_MULTI_WORD:
                    return True
                elif " ".join(response_without_bcs) in RESPONSES_ACKNOWLEDGEMENT_MULTI_WORD:
                    return True
    return False


def get_repetition_ratios(micro_conv, stem_words):
    if pd.isna(micro_conv["response_transcript_clean"]):
        return [0, 0]

    utt = micro_conv["utt_transcript_clean"].lower()
    utt_split = split_into_words(utt, split_on_apostrophe=True, remove_commas=True, remove_trailing_punctuation=True)
    words_utt = set(utt_split)
    if stem_words:
        words_utt = {stemmer.stem(w) for w in words_utt}
    words_utt_no_stopwords = {word for word in words_utt if word not in STOPWORDS}

    response = micro_conv["response_transcript_clean"].lower()
    response_split = split_into_words(response, split_on_apostrophe=True, remove_commas=True, remove_trailing_punctuation=True)
    words_response = set(response_split)
    if stem_words:
        words_response = {stemmer.stem(w) for w in words_response}
    words_response_no_stopwords = {word for word in words_response if word not in STOPWORDS}

    overlap = words_utt_no_stopwords & words_response_no_stopwords

    len_utt = len(words_utt_no_stopwords)
    len_response = len(words_response_no_stopwords)
    if len_utt == 0 or len_response == 0:
        overlap = words_utt & words_response
        len_utt = len(words_utt)
        len_response = len(words_response)

    if len_utt == 0:
        utt_rep_ratio = 0
    else:
        utt_rep_ratio = len(overlap) / len_utt

    if len_response == 0:
        resp_rep_ratio = 0
    else:
        resp_rep_ratio = len(overlap) / len_response

    return [utt_rep_ratio, resp_rep_ratio]


classifier_ack = pickle.load(open(ACK_CLASSIFIER_FILE, "rb"))


def is_repetition_acknowledgement(micro_conv):
    if micro_conv["utt_transcript_clean"][-1] != "?":
        if micro_conv["response_transcript_clean"][-1] != "?":
            return bool(classifier_ack.predict([micro_conv[["rep_utt", "rep_response"]]])[0])
    return False


def response_is_acknowledgement(micro_conv):
    ack_keyword = micro_conv["response_is_keyword_acknowledgement"]
    ack_repetition = micro_conv["response_is_repetition_acknowledgement"]

    return ack_keyword or ack_repetition


classifier_cf = pickle.load(open(CF_CLASSIFIER_FILE, "rb"))


def is_repetition_clarification_request(micro_conv):
    if micro_conv["response_transcript_clean"][-1] == "?":
        return bool(classifier_cf.predict([micro_conv[["rep_utt_stemmed", "rep_response_stemmed"]]])[0])
    return False


def is_clarification_request_speech_act(micro_conv):
    if micro_conv["response_speech_act"] in SPEECH_ACTS_CLARIFICATION_REQUEST:
        utt = micro_conv["utt_transcript_clean"]
        unique_words = set(
            split_into_words(utt, split_on_apostrophe=True, remove_commas=True, remove_trailing_punctuation=True))
        if len(unique_words) == 1 and unique_words.pop().lower() in CAREGIVER_NAMES:
            # If the initial utterance is just a call for attention, the response is not a clarification request.
            return False
        else:
            return True
    return False


def response_is_clarification_request(micro_conv):
    cf_speech_act = micro_conv["response_is_clarification_request_speech_act"]
    cf_repetition = micro_conv["response_is_repetition_clarification_request"]
    return cf_speech_act or cf_repetition


def train_classifier(train_annotations_file, test_annotations_file, target_column):
    train_data = pd.read_csv(train_annotations_file, index_col=0)
    test_data = pd.read_csv(test_annotations_file, index_col=0)

    if target_column == "is_ack":
        train_data = annotate_and_filter_keyword_acks(train_data)
        test_data = annotate_and_filter_keyword_acks(test_data)

    annotate_repetition_ratios(train_data)

    reg = LogisticRegression(penalty="l2", class_weight="balanced")
    reg.fit(train_data[["rep_utt", "rep_response"]].values, train_data[target_column].values)

    precision, recall, f_score, _ = precision_recall_fscore_support(train_data[target_column], reg.predict(train_data[["rep_utt", "rep_response"]].values), average="binary")
    print(f"Train Precision: {precision:.2f}, recall: {recall:.2f}, f-score: {f_score:.2f}")

    plt.figure(figsize=(5.5, 4))
    counts = train_data.groupby(['rep_utt', 'rep_response', target_column]).size().reset_index(name='number')
    ax = sns.scatterplot(data=counts, x="rep_utt", y="rep_response", hue=target_column, size="number", sizes=(30, 1000),
                         alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if target_column == "is_ack":
        handles = [handles[2], handles[1], handles[4], handles[6]] + handles[8:]
        labels = ["Acknowledge-\nment", "Other\nresponse"] + labels[4:6] + labels[8:]
    else:
        handles = [handles[2], handles[1], handles[4], handles[6]] + handles[8:]
        labels = ["Clarification\nRequest", "Other\nresponse"] + labels[4:6] + labels[8:]
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handleheight=0.7,
               handlelength=3, labelspacing=2, fancybox=False, framealpha=0.0)
    plt.tight_layout()
    plt.subplots_adjust(right=0.68)
    x1 = np.arange(0, 1.2, 0.1)
    x2 = (- reg.intercept_[0] - reg.coef_[0, 0] * x1) / reg.coef_[0, 1]
    plt.plot(x1, x2, color="black", lw=1, ls='--')
    ymin, ymax = 0, 1.5
    plt.fill_between(x1, x2, ymin, color=sns.color_palette()[0], alpha=0.1)
    plt.fill_between(x1, x2, ymax, color=sns.color_palette()[1], alpha=0.1)
    plt.xlim((0, 1.1))
    plt.ylim((0, 1.1))

    annotate_repetition_ratios(test_data)

    if target_column == "is_ack":
        plt.savefig(os.path.join(RESULTS_DIR, "repetition_ack.png"), dpi=300)
        pickle.dump(reg, open(ACK_CLASSIFIER_FILE, "wb"))
        test_data["predicted"] = test_data.apply(is_repetition_acknowledgement, axis=1).astype(int)
    else:
        plt.savefig(os.path.join(RESULTS_DIR, "repetition_cr.png"), dpi=300)
        pickle.dump(reg, open(CF_CLASSIFIER_FILE, "wb"))
        test_data["predicted"] = test_data.apply(is_repetition_clarification_request, axis=1).astype(int)

    precision, recall, f_score, _ = precision_recall_fscore_support(test_data[target_column], test_data["predicted"], average="binary")

    # print("Misclassified: ")
    # print(test_data[test_data[target_column] != test_data["predicted"]].to_string(index=False))
    print(f"Test Precision: {precision:.2f}, recall: {recall:.2f}, f-score: {f_score:.2f}")


def annotate_and_filter_keyword_acks(data):
    repetition_ratios = data.apply(get_repetition_ratios, stem_words=False, axis=1)
    data["rep_utt"] = repetition_ratios.apply(lambda ratios: ratios[0])
    data["rep_response"] = repetition_ratios.apply(lambda ratios: ratios[1])
    data["response_is_keyword_acknowledgement"] = data.apply(is_keyword_acknowledgement, axis=1)

    data = data[~data.utt_transcript_clean.str.endswith("?")]

    return data[~data.response_is_keyword_acknowledgement].copy()


def annotate_and_discard_speech_act_crs(data):
    repetition_ratios = data.apply(get_repetition_ratios, stem_words=True, axis=1)
    data["rep_utt"] = repetition_ratios.apply(lambda ratios: ratios[0])
    data["rep_response"] = repetition_ratios.apply(lambda ratios: ratios[1])
    data["response_is_clarification_request_speech_act"] = data.apply(is_clarification_request_speech_act, axis=1)

    return data[~data.response_is_clarification_request_speech_act].copy()


def generate_data_for_annotation(for_clarification_requests: bool):
    conversations = pd.read_csv("results/grammaticality/conversations.csv", index_col=0)
    convs = conversations[
        (conversations.rep_utt > 0.0) & (conversations.rep_response > 0.0) & (
                conversations.response_transcript_clean.str.endswith("?") == for_clarification_requests)].copy()
    if for_clarification_requests:
        convs = annotate_and_discard_speech_act_crs(convs)
    else:
        convs = annotate_and_filter_keyword_acks(convs)

    convs = convs.sample(100, random_state=5)[
        ["utt_transcript_clean", "response_transcript_clean"]]

    if for_clarification_requests:
        convs.to_csv("CR_manual_annotations.csv")
    else:
        convs.to_csv("ACK_manual_annotations.csv")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # generate_data_for_annotation(True)

    print("\nClassifier CR:")
    train_classifier(PROJECT_ROOT_DIR + "/data/feedback_classifiers/CR_manual_annotations.csv",
                     PROJECT_ROOT_DIR + "/data/feedback_classifiers/CR_manual_annotations_test.csv",
                     target_column="is_cr")


    print("\nClassifier ACK:")
    train_classifier(PROJECT_ROOT_DIR + "/data/feedback_classifiers/ACK_manual_annotations.csv",
                     PROJECT_ROOT_DIR + "/data/feedback_classifiers/ACK_manual_annotations_test.csv",
                     target_column="is_ack")

    plt.show()
