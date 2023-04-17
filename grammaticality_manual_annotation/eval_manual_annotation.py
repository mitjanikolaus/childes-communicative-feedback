import itertools
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from krippendorff import krippendorff
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from utils import PROJECT_ROOT_DIR

BASE_PATH = PROJECT_ROOT_DIR + "/data/manual_annotation/selection/"
ANNOTATORS = ["mit", "ale"]


def eval():
    files = [f for f in os.listdir(BASE_PATH) if os.path.isfile(os.path.join(BASE_PATH, f))]

    files_ann = {}
    for ann in ANNOTATORS:
        files_ann[ann] = sorted([f for f in files if f[-7:-4] == ann])

    base_ann = ANNOTATORS[1]
    data = []
    for file in files_ann[base_ann]:
        data.append(pd.read_csv(os.path.join(BASE_PATH, file), index_col=0))

    data = pd.concat(data, ignore_index=True)

    columns = ["has_error"]
    for ann in ANNOTATORS:
        if ann == base_ann:
            continue

        data_ann = []
        for file in files_ann[ann]:
            data_ann.append(pd.read_csv(os.path.join(BASE_PATH, file), index_col=0))
        data_ann = pd.concat(data_ann, ignore_index=True)

        assert len(data_ann.dropna(subset=["has_error"])) == len(data.dropna(subset=["has_error"]))
        column_name = f"has_error_{ann}"
        data[column_name] = data_ann["has_error"]
        columns.append(column_name)

    data["disagreement"] = (~data.has_error.isna() & (data.has_error != data.has_error_mit))
    data["disagreement"] = data.disagreement.replace({True: 1, False: ""})
    data.to_csv(os.path.join(BASE_PATH, "agreement.csv"))

    data.dropna(subset=["has_error"], inplace=True)

    kappa_scores = []
    mcc_scores = []
    for ann_1, ann_2 in itertools.combinations(columns, 2):
        kappa = cohen_kappa_score(data[ann_1], data[ann_2])
        kappa_scores.append(kappa)
        print(f"Kappa {(ann_1, ann_2)}: {kappa:.2f}")

        mcc = matthews_corrcoef(data[ann_1], data[ann_2])
        mcc_scores.append(mcc)
    print(f"Kappa: {np.mean(kappa_scores):.2f}")

    print(f"MCC: {np.mean(mcc_scores):.2f}")

    rel_data = [data[name] for name in columns]
    alpha = krippendorff.alpha(reliability_data=rel_data)
    print(f"Alpha: {alpha:.2f}")


def eval_disagreement():
    data = pd.read_csv(os.path.join(BASE_PATH, "disagreement_annotated.csv"))
    data[data.disagreement == 1].groupby("disagreement_reason").size().sort_values(ascending=False).plot(kind="barh")
    plt.subplots_adjust(left=0.5)
    plt.show()


if __name__ == "__main__":
    eval()
    # eval_disagreement()