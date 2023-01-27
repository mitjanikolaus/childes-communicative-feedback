import os
import pickle

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support
from utils import CF_CLASSIFIER_FILE, ACK_CLASSIFIER_FILE

from cf_analyses.analysis_intelligibility import is_repetition_clarification_request, get_repetition_ratios, \
    is_keyword_acknowledgement, is_repetition_acknowledgement, \
    is_clarification_request_speech_act
from utils import PROJECT_ROOT_DIR


RESULTS_DIR = PROJECT_ROOT_DIR+"/results/cr_ack/"


def train_classifier(train_annotations_file, test_annotations_file, target_column):
    train_data = pd.read_csv(train_annotations_file, index_col=0)
    test_data = pd.read_csv(test_annotations_file, index_col=0)

    if target_column == "is_ack":
        train_data = annotate_and_filter_keyword_acks(train_data)
        test_data = annotate_and_filter_keyword_acks(test_data)

    repetition_ratios = train_data.apply(get_repetition_ratios, axis=1)
    train_data["rep_utt"] = repetition_ratios.apply(lambda ratios: ratios[0])
    train_data["rep_response"] = repetition_ratios.apply(lambda ratios: ratios[1])

    reg = LogisticRegression(penalty="l2", class_weight="balanced")
    reg.fit(train_data[["rep_utt", "rep_response"]].values, train_data[target_column].values)

    prf = precision_recall_fscore_support(train_data[target_column], reg.predict(train_data[["rep_utt", "rep_response"]].values), average="binary")
    print("Train Precision, recall, f-score: ", prf)

    plt.figure(figsize=(5, 4))
    counts = train_data.groupby(['rep_utt', 'rep_response', target_column]).size().reset_index(name='number')
    ax = sns.scatterplot(data=counts, x="rep_utt", y="rep_response", hue=target_column, size="number", sizes=(30, 1000), alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[3:]
    labels = labels[3:]
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handleheight=4, handlelength=3)
    plt.tight_layout()
    plt.subplots_adjust(right=0.72)

    x1 = np.arange(0, 1.2, 0.1)
    x2 = (- reg.intercept_[0] - reg.coef_[0, 0] * x1) / reg.coef_[0, 1]
    plt.plot(x1, x2, color="black", lw=1, ls='--')

    ymin, ymax = 0, 1.5
    plt.fill_between(x1, x2, ymin, color=sns.color_palette()[0], alpha=0.1)
    plt.fill_between(x1, x2, ymax, color=sns.color_palette()[1], alpha=0.1)

    plt.xlim((0, 1.1))
    plt.ylim((0, 1.1))

    repetition_ratios_test = test_data.apply(get_repetition_ratios, axis=1)
    test_data["rep_utt"] = repetition_ratios_test.apply(lambda ratios: ratios[0])
    test_data["rep_response"] = repetition_ratios_test.apply(lambda ratios: ratios[1])

    if target_column == "is_ack":
        plt.savefig(os.path.join(RESULTS_DIR, "repetition_ack.png"), dpi=300)
        pickle.dump(reg, open(ACK_CLASSIFIER_FILE, "wb"))
        test_data["predicted"] = test_data.apply(is_repetition_acknowledgement, axis=1).astype(int)
    else:
        plt.savefig(os.path.join(RESULTS_DIR, "repetition_cr.png"), dpi=300)
        pickle.dump(reg, open(CF_CLASSIFIER_FILE, "wb"))
        test_data["predicted"] = test_data.apply(is_repetition_clarification_request, axis=1).astype(int)

    prf = precision_recall_fscore_support(test_data[target_column], test_data["predicted"], average="binary")

    # print("Misclassified: ")
    # print(test_data[test_data[target_column] != test_data["predicted"]].to_string(index=False))

    print("Test Precision, recall, f-score: ", prf)


def annotate_and_filter_keyword_acks(data):
    repetition_ratios = data.apply(get_repetition_ratios, axis=1)
    data["rep_utt"] = repetition_ratios.apply(lambda ratios: ratios[0])
    data["rep_response"] = repetition_ratios.apply(lambda ratios: ratios[1])
    data["response_is_keyword_acknowledgement"] = data.apply(is_keyword_acknowledgement, axis=1)

    data = data[~data.utt_transcript_clean.str.endswith("?")]

    return data[~data.response_is_keyword_acknowledgement].copy()


def annotate_and_discard_speech_act_crs(data):
    repetition_ratios = data.apply(get_repetition_ratios, axis=1)
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
