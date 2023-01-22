import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from utils import PROJECT_ROOT_DIR, UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE


def eval():
    base_path = PROJECT_ROOT_DIR+"/data/manual_annotation/transcripts"

    utterances = pd.read_csv(UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE, index_col=0, dtype={"error": object})
    for file in os.listdir(base_path):
        if not file.endswith("_annotated.csv"):
            continue
        print(file)
        utts_annotated = pd.read_csv(os.path.join(base_path, file), index_col=0)
        utts_annotated.dropna(subset=["is_grammatical"], inplace=True)
        utts_annotated["is_grammatical"] = utts_annotated.is_grammatical.astype(bool)

        assert len(utts_annotated) == 100

        data_childes = utterances[utterances.index.isin(utts_annotated.index)].copy()
        data_childes.dropna(subset=["is_grammatical"], inplace=True)
        data_childes["is_grammatical"] = data_childes.is_grammatical.astype(bool)

        print(f"Agreement over {len(data_childes)} utterances")
        utts_annotated = utts_annotated[utts_annotated.index.isin(data_childes.index)].copy()

        # TODO nan handling?
        kappa = cohen_kappa_score(utts_annotated.is_grammatical, data_childes.is_grammatical)
        print(f"Kappa {file}: {kappa:.2f}")

        mcc = matthews_corrcoef(utts_annotated.is_grammatical, data_childes.is_grammatical)
        print(f"MCC {file}: {mcc:.2f}")


if __name__ == "__main__":
    eval()