import argparse
import os
from typing import Optional

import evaluate
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup, PreTrainedTokenizerFast,
)

from grammaticality_annotation.data import prepare_manual_annotation_data, CHILDESGrammarDataModule, calc_class_weights
from grammaticality_annotation.pretrain_lstm import TOKENIZER_PATH, TOKEN_PAD, TOKEN_EOS, TOKEN_UNK, TOKEN_SEP, LSTMSequenceClassification

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


class CHILDESGrammarModel(LightningModule):
    def __init__(
            self,
            class_weights,
            model_name_or_path: str,
            num_labels: int,
            train_datasets: list,
            val_datasets: list,
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

        if os.path.isfile(model_name_or_path):
            self.model = LSTMSequenceClassification.load_from_checkpoint(model_name_or_path, num_labels=num_labels)
        else:
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

        self.metric_mcc = evaluate.load("matthews_correlation")
        self.metric_acc = evaluate.load("accuracy")
        self.metrics = [self.metric_mcc, self.metric_acc]

        weight = torch.tensor(class_weights)
        self.loss_fct = CrossEntropyLoss(weight=weight)

        self.val_error_analysis = False

        self.reference_val = self.hparams.eval_splits[0]

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

            if split == self.reference_val:
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
                                  val_datasets=args.val_datasets,
                                  tokenizer=tokenizer)
    dm.setup("fit")
    class_weights = calc_class_weights(dm.dataset["train"]["labels"].numpy())

    model = CHILDESGrammarModel(
        class_weights=class_weights,
        train_datasets=args.train_datasets,
        val_datasets=args.val_datasets,
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
    trainer.validate(model, datamodule=dm)

    print("\n\n\nTraining:")
    trainer.fit(model, datamodule=dm)

    print(f"\n\n\nFinal validation (using {checkpoint_callback.best_model_path}:")
    best_model = CHILDESGrammarModel.load_from_checkpoint(checkpoint_callback.best_model_path)

    model.val_error_analysis = True
    trainer.validate(best_model, datamodule=dm)


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
