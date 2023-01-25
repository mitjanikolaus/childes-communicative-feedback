import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support

from cf_analyses.analysis_intelligibility import is_repetition_clarification_request, get_repetition_ratios, response_is_acknowledgement
from utils import PROJECT_ROOT_DIR


RESULTS_DIR = PROJECT_ROOT_DIR+"/results/cr_ack/"


def eval_crs(annotations_file):
    annotated = pd.read_csv(annotations_file, index_col=0)

    repetition_ratios = annotated.apply(get_repetition_ratios, axis=1)
    annotated["rep_utt"] = repetition_ratios.apply(lambda ratios: ratios[0])
    annotated["rep_response"] = repetition_ratios.apply(lambda ratios: ratios[1])

    plt.figure(figsize=(5, 4))
    counts = annotated.groupby(['rep_utt', 'rep_response', "is_cr"]).size().reset_index(name='number')
    ax = sns.scatterplot(data=counts, x="rep_utt", y="rep_response", hue="is_cr", size="number", sizes=(30, 1000), alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[3:]
    labels = labels[3:]
    plt.axhline(y=0.49, linestyle='--', linewidth=.7, color='black')
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handleheight=4, handlelength=3)
    plt.tight_layout()
    plt.subplots_adjust(right=0.72)
    plt.xlim((0, 1.1))
    plt.ylim((0, 1.1))
    plt.savefig(os.path.join(RESULTS_DIR, "repetition_cr.png"), dpi=300)

    annotated["is_cr_auto_rep"] = annotated.apply(is_repetition_clarification_request, axis=1).astype(int)
    prf = precision_recall_fscore_support(annotated["is_cr"], annotated["is_cr_auto_rep"], average="binary")
    print("\n Precision, recall, f-score: ", prf)
    print("Misclassified: ")
    print(annotated[annotated.is_cr != annotated.is_cr_auto_rep].to_string(index=False))


def eval_acks(annoations_file):
    annotated = pd.read_csv(annoations_file, index_col=0)

    repetition_ratios = annotated.apply(get_repetition_ratios, axis=1)
    annotated["rep_utt"] = repetition_ratios.apply(lambda ratios: ratios[0])
    annotated["rep_response"] = repetition_ratios.apply(lambda ratios: ratios[1])

    annotated["has_response"] = 1
    counts = annotated.groupby(['rep_utt', 'rep_response', "is_ack"]).size().reset_index(name='number')
    plt.figure(figsize=(5, 4))
    ax = sns.scatterplot(data=counts, x="rep_utt", y="rep_response", hue="is_ack", size="number", sizes=(30, 1000), alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[3:6] + [handles[8]]
    labels = labels[3:6] + [labels[8]]
    plt.axhline(y=0.49, linestyle='--', color='black', linewidth=.7)
    plt.axvline(x=0.9, linestyle='--', color='black', linewidth=.7)
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handleheight=4, handlelength=3)
    plt.tight_layout()
    plt.subplots_adjust(right=0.72)
    plt.xlim((0, 1.1))
    plt.ylim((0, 1.1))
    plt.savefig(os.path.join(RESULTS_DIR, "repetition_ack.png"), dpi=300)

    annotated["is_ack_auto_rep"] = annotated.apply(response_is_acknowledgement, axis=1).astype(int)
    prf = precision_recall_fscore_support(annotated["is_ack"], annotated["is_ack_auto_rep"], average="binary")
    print("\n Precision, recall, f-score: ", prf)

    print("Misclassified: ")
    print(annotated[annotated.is_ack != annotated.is_ack_auto_rep].to_string(index=False))


def generate_data_for_annotation(for_clarification_requests: bool):
    conversations = pd.read_csv("results/grammaticality/conversations.csv", index_col=0)
    convs = conversations[
        (conversations.rep_utt > 0.0) & (conversations.rep_response > 0.0) & (
                conversations.response_transcript_clean.str.endswith("?") == for_clarification_requests)].sample(100, random_state=1)[
        ["utt_transcript_clean", "response_transcript_clean"]]
    convs.to_csv("CR_manual_annotations.csv")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    eval_crs(PROJECT_ROOT_DIR+"/data/CR_manual_annotations.csv")
    eval_acks(PROJECT_ROOT_DIR+"/data/ACK_manual_annotations.csv")

    eval_crs(PROJECT_ROOT_DIR+"/data/CR_manual_annotations_test.csv")
    eval_acks(PROJECT_ROOT_DIR+"/data/ACK_manual_annotations_test.csv")

    plt.show()
