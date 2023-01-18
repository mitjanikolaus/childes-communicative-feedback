import itertools
import os
import numpy as np
import pandas as pd
from krippendorff import krippendorff
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from utils import PROJECT_ROOT_DIR


def eval():
    base_path = PROJECT_ROOT_DIR+"/data/manual_annotation"
    data = pd.read_csv(os.path.join(base_path, "grammaticality_manual_annotation_400_600_mit.csv"), index_col=0)

    columns = ["is_grammatical"]
    for suffix in ["abd", "abh", "dhi", "lau"]: # abd", "abh", "dhi",
        ann_file = os.path.join(base_path, f"grammaticality_manual_annotation_400_600_{suffix}.csv")
        utts = pd.read_csv(ann_file, index_col=0)
        column_name = f"is_grammatical_{suffix}"
        data[column_name] = utts["is_grammatical"]
        columns.append(column_name)

        # data[f"labels_{suffix}"] = utts["labels"]
        # data[f"note_{suffix}"] = utts["note"]


    all_agree = data[(data.is_grammatical == data.is_grammatical_abh) & (data.is_grammatical == data.is_grammatical_abd) & (data.is_grammatical == data.is_grammatical_lau) & (data.is_grammatical == data.is_grammatical_dhi)]
    somebody_disagrees = data[~data.index.isin(all_agree.index)]
    print("len (all_agree): ", len(all_agree))

    # disagree_strong = data[(data.is_grammatical == 1) & (data.is_grammatical_lau == -1) |
    #                 (data.is_grammatical == -1) & (data.is_grammatical_lau == 1)]
    # #
    # disagree = data[(data.is_grammatical != data.is_grammatical_lau)]

    kappa_scores = []
    mcc_scores = []
    for ann_1, ann_2 in itertools.combinations(columns, 2):
        kappa = cohen_kappa_score(data[ann_1], data[ann_2])
        kappa_scores.append(kappa)
        print(f"Kappa {(ann_1, ann_2)}: {kappa}")

        mcc = matthews_corrcoef(data[ann_1], data[ann_2])
        mcc_scores.append(mcc)
    print("Kappa:", np.mean(kappa_scores))

    print("MCC:")
    print(mcc_scores)
    print(np.mean(mcc_scores))

    print("Alpha:")
    rel_data = [data[name] for name in columns]
    alpha = krippendorff.alpha(reliability_data=rel_data)
    print(alpha)


if __name__ == "__main__":
    eval()