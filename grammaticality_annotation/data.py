import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from grammaticality_annotation.pretrain_lstm import TOKEN_EOS
from grammaticality_data_preprocessing.prepare_hiller_fernandez_data import HILLER_FERNANDEZ_DATA_OUT_PATH
from utils import FILE_FINE_TUNING_CHILDES_ERRORS, FILE_GRAMMATICALITY_ANNOTATIONS

DATA_PATH_ZORRO = "zorro/sentences/babyberta"

DATA_SPLIT_RANDOM_STATE = 8

TEXT_FIELDS = ["prev_transcript_clean", "transcript_clean"]


def prepare_csv(file_path, include_extra_columns=False, val_split_proportion=None):
    data = pd.read_csv(file_path, index_col=0)
    data.dropna(subset=["is_grammatical", "transcript_clean", "prev_transcript_clean"], inplace=True)

    data["is_grammatical"] = data.is_grammatical.astype(int)

    data.rename(columns={"labels": "categories"}, inplace=True)
    data.rename(columns={"is_grammatical": "labels"}, inplace=True)
    if include_extra_columns:
        data = data[TEXT_FIELDS + ["labels", "categories", "note"]]
    else:
        data = data[TEXT_FIELDS + ["labels"]]

    if val_split_proportion:
        train_data_size = int(len(data) * (1 - val_split_proportion))
        data_train = data.sample(train_data_size, random_state=DATA_SPLIT_RANDOM_STATE).copy()
        data_val = data[~data.index.isin(data_train.index)].copy()

        assert (len(set(data_train.index) & set(data_val.index)) == 0)

        return data_train, data_val
    else:
        return data


def prepare_zorro_data():
    path = DATA_PATH_ZORRO
    data_zorro = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    for i, line in enumerate(f.readlines()):
                        data_zorro.append({
                            "transcript_clean": line.replace("\n", ""),
                            "prev_transcript_clean": ".",
                            "labels": 0 if i % 2 == 0 else 1
                        })
    data_zorro = pd.DataFrame(data_zorro)
    return data_zorro


def prepare_manual_annotation_data(val_split_proportion, include_extra_columns=False):
    data_manual_annotations_train, data_manual_annotations_val = prepare_csv(FILE_GRAMMATICALITY_ANNOTATIONS, include_extra_columns, val_split_proportion)
    # Replace unknown grammaticality values
    data_manual_annotations_train = data_manual_annotations_train[data_manual_annotations_train.labels != 0]
    data_manual_annotations_train.labels.replace({-1: 0}, inplace=True)

    data_manual_annotations_val = data_manual_annotations_val[data_manual_annotations_val.labels != 0]
    data_manual_annotations_val.labels.replace({-1: 0}, inplace=True)

    return data_manual_annotations_train, data_manual_annotations_val


def prepare_blimp_data():
    mor = load_dataset("metaeval/blimp_classification", "morphology")["train"].to_pandas()
    syntax = load_dataset("metaeval/blimp_classification", "syntax")["train"].to_pandas()
    data_blimp = pd.concat([mor, syntax], ignore_index=True)
    data_blimp.rename(columns={"text": "transcript_clean"}, inplace=True)
    data_blimp["prev_transcript_clean"] = "."
    data_blimp.set_index("idx", inplace=True)
    return data_blimp


def prepare_cola_data():
    dataset = load_dataset("glue", "cola")
    ds = dataset["train"]
    ds = ds.rename_column("sentence", "transcript_clean")
    ds = ds.rename_column("label", "labels")
    ds = ds.add_column("prev_transcript_clean", ["." for _ in range(len(ds))])
    ds = ds.to_pandas()
    ds.set_index("idx", inplace=True)
    return ds


LOADER_COLUMNS = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]


def create_dataset_dict(train_datasets, additional_val_datasets, val_split_proportion):
    data_manual_annotations_train, data_manual_annotations_val = prepare_manual_annotation_data(val_split_proportion)
    if "childes" in train_datasets + additional_val_datasets:
        data_childes_train, data_childes_val = prepare_csv(FILE_FINE_TUNING_CHILDES_ERRORS, val_split_proportion=val_split_proportion)

    def get_dataset_with_name(ds_name, val=False):
        if ds_name == "manual_annotations":
            if val:
                return data_manual_annotations_val
            else:
                return data_manual_annotations_train
        elif ds_name == "hiller_fernandez":
            return prepare_csv(HILLER_FERNANDEZ_DATA_OUT_PATH)
        elif ds_name == "cola":
            return prepare_cola_data()
        elif ds_name == "blimp":
            return prepare_blimp_data()
        elif ds_name == "childes":
            if val:
                return data_childes_val
            else:
                return data_childes_train
        elif ds_name == "zorro":
            return prepare_zorro_data()
        else:
            raise RuntimeError("Unknown dataset: ", ds_name)

    dataset_dict = DatasetDict()

    data_train = []
    for ds_name in train_datasets:
        data_train.append(get_dataset_with_name(ds_name, val=False))

    data_train = pd.concat(data_train, ignore_index=True)
    ds_train = Dataset.from_pandas(data_train)
    dataset_dict['train'] = ds_train

    ds_val = Dataset.from_pandas(data_manual_annotations_val)
    dataset_dict["validation"] = ds_val

    for ds_name in additional_val_datasets:
        data = get_dataset_with_name(ds_name, val=True)
        ds_val = Dataset.from_pandas(data)
        dataset_dict[f"validation_{ds_name}"] = ds_val

    return dataset_dict


class CHILDESGrammarDataModule(LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str,
            train_batch_size: int,
            eval_batch_size: int,
            train_datasets: list,
            additional_val_datasets: list,
            tokenizer,
            max_seq_length: int = 128,
            val_split_proportion: float = 0.5,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.val_split_proportion = val_split_proportion
        self.train_datasets = train_datasets
        self.additional_val_datasets = additional_val_datasets

        self.num_labels = 2
        self.tokenizer = tokenizer

    def setup(self, stage: str):
        self.dataset = create_dataset_dict(self.train_datasets, self.additional_val_datasets, self.val_split_proportion)
        for split in self.dataset.keys():
            columns = [c for c in self.dataset[split].column_names if c in LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=columns + TEXT_FIELDS)
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, collate_fn=self.tokenize_batch)

    def val_dataloader(self):
        return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch) for x in self.eval_splits]

    def tokenize_batch(self, batch):
        return tokenize(batch, self.tokenizer, self.max_seq_length, add_labels=True)


def tokenize(batch, tokenizer, max_seq_length, add_labels=False):
    if len(TEXT_FIELDS) > 1:
        texts = [tokenizer.sep_token.join([b[TEXT_FIELDS[0]], b[TEXT_FIELDS[1]]]) for b in batch]
        if TOKEN_EOS in tokenizer.all_special_tokens:
            texts = [t + TOKEN_EOS for t in texts]
    else:
        raise NotImplementedError()

    features = tokenizer.batch_encode_plus(
        texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt"
    )
    if add_labels:
        features.data["labels"] = torch.tensor([b["labels"] for b in batch])

    return features


def calc_class_weights(labels):
    class_weight_pos = np.sum(labels).item() / len(labels)
    class_weights = [class_weight_pos, 1 - class_weight_pos]
    return class_weights
