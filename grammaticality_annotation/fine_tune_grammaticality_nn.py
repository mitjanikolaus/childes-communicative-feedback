import argparse
import os
from typing import Optional

import evaluate
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup, PreTrainedTokenizerFast,
)

from grammaticality_annotation.pretrain_lstm import TOKENIZER_PATH, TOKEN_PAD, TOKEN_EOS, TOKEN_UNK, TOKEN_SEP, LSTMSequenceClassification
from grammaticality_data_preprocessing.prepare_hiller_fernandez_data import HILLER_FERNANDEZ_DATA_OUT_PATH
from utils import FILE_FINE_TUNING_CHILDES_ERRORS, FILE_GRAMMATICALITY_ANNOTATIONS

DATA_PATH_ZORRO = "zorro/sentences/babyberta"

DATA_SPLIT_RANDOM_STATE = 7
FINE_TUNE_RANDOM_STATE = 1

DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-5

MODELS = [
    "yevheniimaslov/deberta-v3-large-cola",
    "phueb/BabyBERTa-3",
    "cointegrated/roberta-large-cola-krishna2020",  # Inverted labels!!
    "bert-base-uncased",
    "textattack/bert-base-uncased-CoLA",
]


TEXT_FIELDS = ["transcript_clean", "prev_transcript_clean"]


def prepare_csv(file_path, include_extra_columns=False):
    data = pd.read_csv(file_path, index_col=0)
    data.dropna(subset=["is_grammatical", "transcript_clean", "prev_transcript_clean"], inplace=True)

    data["is_grammatical"] = data.is_grammatical.astype(int)

    data.rename(columns={"labels": "categories"}, inplace=True)
    data.rename(columns={"is_grammatical": "labels"}, inplace=True)
    if include_extra_columns:
        data = data[TEXT_FIELDS + ["labels", "categories", "note"]]
    else:
        data = data[TEXT_FIELDS + ["labels"]]
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
    data_manual_annotations = prepare_csv(FILE_GRAMMATICALITY_ANNOTATIONS, include_extra_columns)
    # Replace unknown grammaticality values
    data_manual_annotations = data_manual_annotations[data_manual_annotations.labels != 0]
    data_manual_annotations.labels.replace({-1: 0}, inplace=True)

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


def prepare_cola_data():
    dataset = load_dataset("glue", "cola")
    ds = dataset["train"]
    ds = ds.rename_column("sentence", "transcript_clean")
    ds = ds.rename_column("label", "labels")
    ds = ds.add_column("prev_transcript_clean", ["." for _ in range(len(ds))])
    ds = ds.to_pandas()
    ds.set_index("idx", inplace=True)
    return ds


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
        data_manual_annotations_train, data_manual_annotations_val = prepare_manual_annotation_data(self.val_split_proportion)

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
                return prepare_csv(FILE_FINE_TUNING_CHILDES_ERRORS)
            elif ds_name == "zorro":
                return prepare_zorro_data()
            else:
                raise RuntimeError("Unknown dataset: ", ds_name)

        self.dataset = DatasetDict()

        data_train = []
        for ds_name in self.train_datasets:
            data_train.append(get_dataset_with_name(ds_name, val=False))

        data_train = pd.concat(data_train, ignore_index=True)
        ds_train = Dataset.from_pandas(data_train)
        self.dataset['train'] = ds_train

        ds_val = Dataset.from_pandas(data_manual_annotations_val)
        self.dataset[f"validation"] = ds_val

        for ds_name in self.additional_val_datasets:
            data = get_dataset_with_name(ds_name, val=True)
            ds_val = Dataset.from_pandas(data)
            self.dataset[f"validation_{ds_name}"] = ds_val

        for split in self.dataset.keys():
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns+TEXT_FIELDS)

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
        if len(TEXT_FIELDS) > 1:
            texts = [self.tokenizer.sep_token.join([b[TEXT_FIELDS[0]], b[TEXT_FIELDS[1]]]) for b in batch]
            if TOKEN_EOS in self.tokenizer.all_special_tokens:
                texts = [t + TOKEN_EOS for t in texts]
        else:
            raise NotImplementedError()

        features = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_seq_length, padding=True, truncation=True, return_tensors="pt"
        )
        features.data["labels"] = torch.tensor([b["labels"] for b in batch])

        return features


class CHILDESGrammarModel(LightningModule):
    def __init__(
            self,
            class_weights,
            model_name_or_path: str,
            num_labels: int,
            train_datasets: list,
            additional_val_datasets: list,
            train_batch_size: int,
            eval_batch_size: int,
            learning_rate: float,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            eval_splits: Optional[list] = None,
            val_split_proportion: float = 0.5,
            **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate

        print(f"Model loss class weights: {class_weights}")
        self.save_hyperparameters(ignore=["tokenizer"])

        if os.path.isfile(args.model):
            self.model = LSTMSequenceClassification.load_from_checkpoint(args.model, num_labels=num_labels)
        else:
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

        self.metric_mcc = evaluate.load("matthews_correlation")
        self.metric_acc = evaluate.load("accuracy")
        self.metrics = [self.metric_mcc, self.metric_acc]

        weight = torch.tensor(class_weights)
        self.loss_fct = CrossEntropyLoss(weight=weight)

        self.val_error_analysis = False

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
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
            attention_mask=batch["attention_mask"],
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
            preds = torch.cat([x["preds"] for x in out]).detach().cpu().numpy()
            labels = torch.cat([x["labels"] for x in out]).detach().cpu().numpy()
            loss = torch.stack([x["loss"] for x in out]).mean()

            if split == "validation":
                self.log(f"val_loss", loss, prog_bar=True)
                for metric in self.metrics:
                    metric_results = metric.compute(predictions=preds, references=labels)
                    self.log_dict(metric_results, prog_bar=True)

                acc_pos = self.metric_acc.compute(predictions=preds[labels == 1], references=labels[labels == 1])
                acc_neg = self.metric_acc.compute(predictions=preds[labels == 0], references=labels[labels == 0])
                self.log(f"accuracy_pos", acc_pos["accuracy"])
                self.log(f"accuracy_neg", acc_neg["accuracy"])

                if self.val_error_analysis:
                    _, data_manual_annotations_val = prepare_manual_annotation_data(self.hparams.val_split_proportion,
                                                                                    include_extra_columns=True)
                    data_manual_annotations_val["pred"] = preds
                    errors = data_manual_annotations_val[
                        data_manual_annotations_val.pred != data_manual_annotations_val.label]
                    correct = data_manual_annotations_val[
                        data_manual_annotations_val.pred == data_manual_annotations_val.label]

                    errors.to_csv(os.path.join(self.logger.log_dir, "manual_annotations_errors.csv"))
                    correct.to_csv(os.path.join(self.logger.log_dir, "manual_annotations_correct.csv"))

            else:
                split = split.replace("validation_", "")
                self.log(f"{split}_val_loss", loss)
                for metric in self.metrics:
                    metric_results = metric.compute(predictions=preds, references=labels)
                    metric_results = {f"{split}_{key}": value for key, value in metric_results.items()}
                    self.log_dict(metric_results)

                acc_pos = self.metric_acc.compute(predictions=preds[labels == 1], references=labels[labels == 1])
                acc_neg = self.metric_acc.compute(predictions=preds[labels == 0], references=labels[labels == 0])
                self.log(f"{split}_accuracy_pos", acc_pos["accuracy"])
                self.log(f"{split}_accuracy_neg", acc_neg["accuracy"])
                self.log(f"{split}_val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        if isinstance(self.model, LSTMSequenceClassification):
            optimizer = Adam(self.parameters(), lr=self.learning_rate, eps=self.hparams.adam_epsilon)
            return [optimizer]
        else:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.hparams.adam_epsilon)

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

    if os.path.isfile(args.model):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
        tokenizer.add_special_tokens(
            {'pad_token': TOKEN_PAD, 'eos_token': TOKEN_EOS, 'unk_token': TOKEN_UNK, 'sep_token': TOKEN_SEP})
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    dm = CHILDESGrammarDataModule(val_split_proportion=args.val_split_proportion,
                                  model_name_or_path=args.model,
                                  eval_batch_size=args.batch_size,
                                  train_batch_size=args.batch_size,
                                  train_datasets=args.train_datasets,
                                  additional_val_datasets=args.additional_val_datasets,
                                  tokenizer=tokenizer)
    dm.setup("fit")
    model = CHILDESGrammarModel(
        class_weights=calc_class_weights(dm),
        train_datasets=args.train_datasets,
        additional_val_datasets=args.additional_val_datasets,
        eval_batch_size=args.batch_size,
        train_batch_size=args.batch_size,
        model_name_or_path=args.model,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        val_split_proportion=args.val_split_proportion,
        learning_rate=args.learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(monitor="matthews_correlation", mode="max", save_last=True,
                                            filename="{epoch:02d}-{matthews_correlation:.2f}")
    early_stop_callback = EarlyStopping(monitor="matthews_correlation", patience=10, verbose=True, mode="max",
                                        min_delta=0.01, stopping_threshold=0.99)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    print("\n\n\nInitial validation:")
    trainer.validate(model, dm)

    print("\n\n\nTraining:")
    trainer.fit(model, datamodule=dm)

    print(f"\n\n\nFinal validation (using {checkpoint_callback.best_model_path}:")
    best_model = CHILDESGrammarModel.load_from_checkpoint(checkpoint_callback.best_model_path)

    model.val_error_analysis = True
    trainer.validate(best_model, dm)


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
        "--additional-val-datasets",
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
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    argparser.add_argument(
        "--val-split-proportion",
        type=float,
        default=0.5,
        help="Val split proportion (only for manually annotated data)"
    )
    argparser = Trainer.add_argparse_args(argparser)

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
