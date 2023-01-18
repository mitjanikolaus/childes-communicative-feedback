import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support

from cf_analyses.analysis_intelligibility import is_repetition_clarification_request, get_repetition_ratios, response_is_acknowledgement
from utils import PROJECT_ROOT_DIR


def eval_crs():
    annotated = pd.read_csv(PROJECT_ROOT_DIR+"/data/CR_manual_annotations.csv", index_col=0)

    repetition_ratios = annotated.apply(get_repetition_ratios, axis=1)
    annotated["utt_repetition_ratio"] = repetition_ratios.apply(lambda ratios: ratios[0])
    annotated["resp_repetition_ratio"] = repetition_ratios.apply(lambda ratios: ratios[1])

    annotated["is_cr_auto_rep"] = annotated.apply(is_repetition_clarification_request, axis=1).astype(int)
    counts = annotated.groupby(['utt_repetition_ratio', 'resp_repetition_ratio', "is_clarification_request"]).size().reset_index(name='Count')
    sns.scatterplot(data=counts, x="utt_repetition_ratio", y="resp_repetition_ratio", hue="is_clarification_request", size="Count", alpha=0.7)

    prf = precision_recall_fscore_support(annotated["is_clarification_request"], annotated["is_cr_auto_rep"], average="binary")
    print("Precision, recall, f-score: ", prf)
    print("\n Misclassified: ")
    print(annotated[annotated.is_clarification_request != annotated.is_cr_auto_rep].to_string(index=False))


def eval_acks():
    annotated = pd.read_csv(PROJECT_ROOT_DIR+"/data/ACK_manual_annotations.csv", index_col=0)

    repetition_ratios = annotated.apply(get_repetition_ratios, axis=1)
    annotated["utt_repetition_ratio"] = repetition_ratios.apply(lambda ratios: ratios[0])
    annotated["resp_repetition_ratio"] = repetition_ratios.apply(lambda ratios: ratios[1])

    annotated["has_response"] = 1
    annotated["is_ack_auto_rep"] = annotated.apply(response_is_acknowledgement, axis=1).astype(int)
    counts = annotated.groupby(['utt_repetition_ratio', 'resp_repetition_ratio', "is_ack"]).size().reset_index(name='Count')
    sns.scatterplot(data=counts, x="utt_repetition_ratio", y="resp_repetition_ratio", hue="is_ack", size="Count", alpha=0.7)

    prf = precision_recall_fscore_support(annotated["is_ack"], annotated["is_ack_auto_rep"], average="binary")
    print("Precision, recall, f-score: ", prf)

    print("\n Misclassified: ")
    print(annotated[annotated.is_ack != annotated.is_ack_auto_rep].to_string(index=False))


if __name__ == "__main__":
    # conversations = pd.read_csv("results/grammaticality/conversations.csv", index_col=0)

    # convs = conversations[
    #     (conversations.utt_repetition_ratio > 0.0) & (conversations.resp_repetition_ratio > 0.0) & (
    #             conversations.response_transcript_clean.str.endswith("?") == False)].sample(100, random_state=1)[
    #     ["utt_transcript_clean", "response_transcript_clean", "utt_repetition_ratio", "resp_repetition_ratio"]]
    # convs.to_csv("ACK_manual_annotations.csv")

    # eval_crs()
    eval_acks()

    plt.show()
