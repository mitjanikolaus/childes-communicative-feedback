import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, precision_recall_fscore_support

from grammaticality_manual_annotation.prepare_for_hand_annotation import CORPORA_INCLUDED
from utils import PROJECT_ROOT_DIR, UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE


def eval():
    base_path = PROJECT_ROOT_DIR+"/data/manual_annotation/transcripts"

    utterances = pd.read_csv(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, index_col=0, dtype={"error": object})

    table_data = []
    for corpus in CORPORA_INCLUDED:
        data = []

        for file in os.listdir(base_path):
            if file.startswith(corpus) and file.endswith("_annotated.csv"):
                utts_annotated = pd.read_csv(os.path.join(base_path, file), index_col=0)
                utts_annotated.dropna(subset=["is_grammatical"], inplace=True)
                utts_annotated["is_error"] = ~utts_annotated.is_grammatical.astype(bool)

                # print(len(utts_annotated))
                # assert len(utts_annotated) == 100

                data_childes = utterances[utterances.index.isin(utts_annotated.index)].copy()
                data_childes.dropna(subset=["is_grammatical"], inplace=True)
                data_childes = data_childes[data_childes.is_intelligible]

                data_childes["is_error"] = ~data_childes.is_grammatical.astype(bool)

                utts_annotated = utts_annotated.merge(data_childes[["is_error", "labels"]], how="inner", suffixes=("_manual", "_childes"), left_index=True, right_index=True)
                data.append(utts_annotated)

        data = pd.concat(data, ignore_index=True)
        print(f"\n{corpus}")
        # print(f"Agreement over {len(data)} utterances")

        disagreements = data[data.is_error_manual != data.is_error_childes]
        # TODO nan handling?
        kappa = cohen_kappa_score(data.is_error_manual, data.is_error_childes)
        print(f"Cohen's Kappa: {kappa:.2f}")

        precision, recall, f_score, support = precision_recall_fscore_support(data.is_error_manual, data.is_error_childes, average="binary", zero_division=0)
        print(f"Precision: {precision:.2f}, recall: {recall:.2f}, f-score: {f_score:.2f}")

        table_data.append({"corpus": corpus, "kappa": kappa, "precision": precision, "recall": recall, "f_score": f_score})
        # mcc = matthews_corrcoef(utts_annotated.is_error_manual, utts_annotated.is_error_childes)
        # print(f"MCC: {mcc:.2f}")

    table_data = pd.DataFrame(table_data)
    table_data.set_index("corpus", inplace=True)
    print(table_data.to_latex(float_format="%.2f"))



if __name__ == "__main__":
    eval()