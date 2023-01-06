import argparse
import os
from typing import Optional

import evaluate
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from grammaticatily_data_preprocessing.prepare_hiller_fernandez_data import HILLER_FERNANDEZ_DATA_OUT_PATH
from utils import FILE_FINE_TUNING_CHILDES_ERRORS

FILE_GRAMMATICALITY_ANNOTATIONS = "data/manual_annotation/grammaticality_manually_annotated.csv"

DATA_PATH_ZORRO = "zorro/sentences/babyberta"

DATA_SPLIT_RANDOM_STATE = 7
FINE_TUNE_RANDOM_STATE = 1

MAX_EPOCHS = 15

DEFAULT_BATCH_SIZE = 16

MODELS = [
    "yevheniimaslov/deberta-v3-large-cola",
    "phueb/BabyBERTa-3",
    "cointegrated/roberta-large-cola-krishna2020",  # Inverted labels!!
    "textattack/bert-base-uncased",
    "textattack/bert-base-uncased-CoLA",
]


def prepare_csv(file_path, text_fields):
    data = pd.read_csv(file_path, index_col=0)
    data.dropna(subset=["is_grammatical", "transcript_clean", "prev_transcript_clean"], inplace=True)

    data["is_grammatical"] = data.is_grammatical.astype(int)

    data.rename(columns={"is_grammatical": "label"}, inplace=True)
    data = data[text_fields + ["label"]]
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
                            "label": 0 if i % 2 == 0 else 1
                        })
    data_zorro = pd.DataFrame(data_zorro)
    return data_zorro


def prepare_manual_annotation_data(text_fields, val_split_proportion):
    data_manual_annotations = prepare_csv(FILE_GRAMMATICALITY_ANNOTATIONS, text_fields)
    # Replace unknown grammaticality values
    data_manual_annotations = data_manual_annotations[data_manual_annotations.label != 0]
    data_manual_annotations.label.replace({-1: 0}, inplace=True)

    train_data_size = int(len(data_manual_annotations) * (1 - val_split_proportion))
    data_manual_annotations_train = data_manual_annotations.sample(train_data_size,
                                                                   random_state=DATA_SPLIT_RANDOM_STATE)
    data_manual_annotations_val = data_manual_annotations[
        ~data_manual_annotations.index.isin(data_manual_annotations_train.index)]

    assert (len(set(data_manual_annotations_train.index) & set(data_manual_annotations_val.index)) == 0)

    return data_manual_annotations_train, data_manual_annotations_val


def prepare_blimp_data():
    mor = load_dataset("metaeval/blimp_classification", "morphology")["train"].to_pandas()
    syntax = load_dataset("metaeval/blimp_classification", "syntax")["train"].to_pandas()
    data_blimp = pd.concat([mor, syntax], ignore_index=True)
    data_blimp.rename(columns={"text": "transcript_clean"}, inplace=True)
    data_blimp["prev_transcript_clean"] = "."
    data_blimp.set_index("idx", inplace=True)
    return data_blimp


class CHILDESGrammarDataModule(LightningDataModule):
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            train_batch_size: int,
            eval_batch_size: int,
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

        self.text_fields = ["transcript_clean", "prev_transcript_clean"]
        self.num_labels = 2
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        data_manual_annotations_train, data_manual_annotations_val = prepare_manual_annotation_data(self.text_fields, self.val_split_proportion)

        def get_dataset_with_name(ds_name, text_fields, val=False):
            if ds_name == "manual_annotations":
                if val:
                    return data_manual_annotations_val
                else:
                    return data_manual_annotations_train
            elif ds_name == "hiller_fernandez":
                data_hiller_fernandez = prepare_csv(HILLER_FERNANDEZ_DATA_OUT_PATH, text_fields)
                return data_hiller_fernandez
            elif ds_name == "blimp":
                data_blimp = prepare_blimp_data()
                return data_blimp
            elif ds_name == "childes":
                data_childes = prepare_csv(FILE_FINE_TUNING_CHILDES_ERRORS, text_fields)
                return data_childes
            elif ds_name == "zorro":
                data_zorro = prepare_zorro_data()
                return data_zorro
            else:
                raise RuntimeError("Unknown dataset: ", ds_name)

        self.dataset = DatasetDict()

        data_train = []
        for ds_name in args.train_datasets:
            data_train.append(get_dataset_with_name(ds_name, self.text_fields, val=False))

        data_train = pd.concat(data_train, ignore_index=True)
        ds_train = Dataset.from_pandas(data_train)
        self.dataset['train'] = ds_train

        datasets_val = []
        for ds_name in args.val_datasets:
            data = get_dataset_with_name(ds_name, self.text_fields, val=True)
            ds_val = Dataset.from_pandas(data)
            datasets_val.append(ds_val)
            self.dataset[f"validation_{ds_name}"] = ds_val

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, batch):
        if len(self.text_fields) > 1:
            texts = list(zip(batch[self.text_fields[0]], batch[self.text_fields[1]]))
        else:
            texts = batch[self.text_fields[0]]

        features = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_seq_length, padding='max_length', truncation=True
        )

        features["labels"] = batch["label"]

        return features


class CHILDESGrammarTransformer(LightningModule):
    def __init__(
            self,
            class_weights,
            model_name_or_path: str,
            num_labels: int,
            train_datasets: list,
            val_datasets: list,
            train_batch_size: int,
            eval_batch_size: int,
            learning_rate: float = 1e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            eval_splits: Optional[list] = None,
            val_split_proportion: float = 0.5,
            **kwargs,
    ):
        super().__init__()

        print(f"Model loss class weights: {class_weights}")
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric_mcc = evaluate.load("matthews_correlation")
        self.metric_acc = evaluate.load("accuracy")
        self.metrics = [self.metric_mcc, self.metric_acc]

        weight = torch.tensor(class_weights)
        self.loss_fct = CrossEntropyLoss(weight=weight)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"]
        )
        logits = output["logits"]
        labels = batch["labels"]
        loss = self.loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))

        preds = torch.argmax(logits, axis=1)

        return {"loss": loss, "preds": preds, "labels": labels}

    def training_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()

        acc = self.metric_acc.compute(predictions=preds, references=labels)
        acc = {"train_" + key: value for key, value in acc.items()}
        self.log_dict(acc, prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"]
        )
        logits = output["logits"]
        labels = batch["labels"]
        val_loss = self.loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))

        preds = torch.argmax(logits, axis=1)

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if len(self.hparams.eval_splits) == 1:
            outputs = [outputs]

        for out, split in zip(outputs, self.hparams.eval_splits):
            split = split.replace("validation_", "")
            preds = torch.cat([x["preds"] for x in out]).detach().cpu().numpy()
            labels = torch.cat([x["labels"] for x in out]).detach().cpu().numpy()
            loss = torch.stack([x["loss"] for x in out]).mean()
            self.log(f"{split}_val_loss", loss, prog_bar=True)

            for metric in self.metrics:
                metric_results = metric.compute(predictions=preds, references=labels)
                metric_results = {f"{split}_{key}": value for key, value in metric_results.items()}
                self.log_dict(metric_results, prog_bar=True)

            acc_pos = self.metric_acc.compute(predictions=preds[labels == 1], references=labels[labels == 1])
            acc_neg = self.metric_acc.compute(predictions=preds[labels == 0], references=labels[labels == 0])
            self.log(f"{split}_accuracy_pos", acc_pos["accuracy"])
            self.log(f"{split}_accuracy_neg", acc_neg["accuracy"])

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


def calc_class_weights(dm):
    class_weight_pos = dm.dataset["train"]["labels"].sum().item() / dm.dataset["train"]["labels"].shape[0]
    class_weights = [class_weight_pos, 1 - class_weight_pos]
    return class_weights


def main(args):
    seed_everything(FINE_TUNE_RANDOM_STATE)

    dm = CHILDESGrammarDataModule(val_split_proportion=args.val_split_proportion,
                                  model_name_or_path=args.model,
                                  eval_batch_size=args.batch_size,
                                  train_batch_size=args.batch_size)
    dm.setup("fit")
    model = CHILDESGrammarTransformer(
        class_weights=calc_class_weights(dm),
        train_datasets=args.train_datasets,
        val_datasets=args.val_datasets,
        eval_batch_size=args.batch_size,
        train_batch_size=args.batch_size,
        model_name_or_path=args.model,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        val_split_proportion=args.val_split_proportion,
    )

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    )

    print("\n\n\nInitial validation:")
    trainer.validate(model, dm)

    print("\n\n\nTraining:")
    trainer.fit(model, datamodule=dm)

    print("\n\n\nFinal validation:")
    trainer.validate(model, dm)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    argparser.add_argument(
        "--train-datasets",
        type=str,
        nargs="+",
        default=[],
    )
    argparser.add_argument(
        "--val-datasets",
        type=str,
        nargs="+",
        default=[],
    )
    argparser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    argparser.add_argument(
        "--val-split-proportion",
        type=float,
        default=0.5,
        help="Val split proportion (only for manually annotated data)"
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
