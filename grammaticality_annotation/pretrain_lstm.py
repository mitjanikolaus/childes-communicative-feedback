import argparse
import math
import os.path

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch import nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from utils import UTTERANCES_WITH_PREV_UTTS_FILE, PROJECT_ROOT_DIR

LM_DATA = os.path.expanduser("~/data/communicative_feedback/sentences.txt")
TOKENIZER_PATH = PROJECT_ROOT_DIR+"/data/tokenizer-childes.json"

BATCH_SIZE = 32

TRUNCATION_LENGTH = 40

MAX_EPOCHS = 10

TOKEN_PAD = "[PAD]"
TOKEN_EOS = "[EOS]"
TOKEN_UNK = "[UNK]"
TOKEN_SEP = "[SEP]"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_tokenizer():
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=[TOKEN_PAD, TOKEN_UNK, TOKEN_EOS, TOKEN_SEP], show_progress=True, vocab_size=10000)
    tokenizer.train(files=[LM_DATA], trainer=trainer)

    tokenizer.save(TOKENIZER_PATH)


def prepare_data():
    data = pd.read_csv(UTTERANCES_WITH_PREV_UTTS_FILE, index_col=0)
    data = data[data.is_speech_related & data.is_intelligible]
    data.dropna(subset=["transcript_clean", "prev_transcript_clean"], inplace=True)

    sentences = data.apply(lambda row: TOKEN_SEP.join([row.prev_transcript_clean, row.transcript_clean + TOKEN_EOS]), axis=1).values
    with open(LM_DATA, 'w') as f:
        f.write("\n".join(sentences))


class CHILDESDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, tokenizer):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        data = load_dataset("text", data_files={"train": LM_DATA})
        self.data = data["train"].train_test_split(test_size=0.001)

    def tokenize_batch(self, batch):
        text = [t["text"] for t in batch]
        encodings = self.tokenizer.batch_encode_plus(text, padding=True, max_length=TRUNCATION_LENGTH, truncation=True,
                                                     return_tensors="pt")
        encodings.data["labels"] = encodings.data["input_ids"][:, 1:]

        return encodings

    def train_dataloader(self):
        return DataLoader(self.data["train"], batch_size=self.batch_size, collate_fn=self.tokenize_batch)

    def val_dataloader(self):
        return DataLoader(self.data["test"], batch_size=self.batch_size, collate_fn=self.tokenize_batch)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if num_layers > 1:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                                dropout=dropout_rate, batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.fc_classification = nn.Linear(hidden_dim, 2)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, input_ids, hidden=None, attention_mask=None, token_type_ids=None):
        if not hidden:
            hidden = self.init_hidden(input_ids.shape[0])
        embedding = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).cpu().numpy()
        packed_input = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_input, hidden)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        logits = self.fc(output)

        return {"logits": logits, "hidden": hidden}

    def forward_classification(self, input_ids, hidden=None, attention_mask=None, token_type_ids=None):
        if not hidden:
            hidden = self.init_hidden(input_ids.shape[0])
        embedding = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).cpu().numpy()
        packed_input = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_input, hidden)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)

        # Zero all output that is padded
        output = output * attention_mask.unsqueeze(2)
        # Max pool over time dimension
        output = self.max_pool(output.permute(0, 2, 1)).squeeze()

        logits = self.fc_classification(output)

        return {"logits": logits, "hidden": hidden}

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell


class CHILDESGrammarLSTM(LightningModule):
    def __init__(
            self,
            pad_token_id,
            vocab_size,
            tokenizer=None,
            embedding_dim: int = 256,
            hidden_dim: int = 256,
            num_layers: int = 1,
            dropout_rate: float = 0.1,
            learning_rate: float = 0.003,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])

        self.tokenizer = tokenizer

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        self.vocab_size = vocab_size
        self.model = LSTM(self.vocab_size, embedding_dim,  hidden_dim, num_layers, dropout_rate)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        )
        labels = batch["labels"]
        logits = output["logits"]
        logits = logits[:, :-1, :]
        loss = self.loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log("train_loss", loss.mean().item(), prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        )
        labels = batch["labels"]
        logits = output["logits"]
        logits = logits[:, :-1, :]
        val_loss = self.loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))

        preds = torch.argmax(logits, axis=1)

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        print("\n\n")
        print(self.generate("you", max_seq_len=20, temperature=0.3))
        print(self.generate("you", max_seq_len=20, temperature=0.5))
        print(self.generate("you", max_seq_len=20, temperature=0.7))

        self.log(f"val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return [optimizer]

    def generate(self, prompt, max_seq_len, temperature, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        encoding = self.tokenizer.encode_plus(prompt, return_tensors="pt").to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        with torch.no_grad():
            hidden = None
            for i in range(max_seq_len):
                output = self.model(input_ids, attention_mask=attention_mask, hidden=hidden)
                logits = output["logits"]
                hidden = output["hidden"]
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                prediction = torch.multinomial(probs, num_samples=1)

                if prediction.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat([input_ids, prediction], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.tensor(1, device=input_ids.device).reshape(1, 1)], dim=1)

        decoded = self.tokenizer.decode(input_ids[0].cpu().numpy())
        self.train()
        return decoded


class LSTMSequenceClassification(CHILDESGrammarLSTM):
    def __init__(
            self,
            pad_token_id: int,
            vocab_size: int,
            tokenizer = None,
            embedding_dim: int = 256,
            hidden_dim: int = 256,
            num_layers: int = 1,
            dropout_rate: float = 0.1,
            learning_rate: float = 0.001,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            num_labels: int = 2,
            **kwargs,
    ):
        super().__init__(
                         tokenizer=tokenizer,
                         embedding_dim=embedding_dim,
                         hidden_dim=hidden_dim,
                         num_layers=num_layers,
                         dropout_rate=dropout_rate,
                         learning_rate=learning_rate,
                         adam_epsilon=adam_epsilon,
                         warmup_steps=warmup_steps,
                         weight_decay=weight_decay,
                         pad_token_id=pad_token_id,
                         vocab_size=vocab_size,
                         )

        self.num_labels = num_labels

        # self.freeze_base_weights()

    def freeze_base_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc_classification.weight.requires_grad = True
        self.model.fc_classification.bias.requires_grad = True

    def forward(self, **inputs):
        return self.model.forward_classification(**inputs)


def train(args):
    if not os.path.isfile(LM_DATA):
        print("Preparing data...")
        prepare_data()
    if not os.path.isfile(TOKENIZER_PATH):
        print("Training tokenizer...")
        train_tokenizer()

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    tokenizer.add_special_tokens(
        {'pad_token': TOKEN_PAD, 'eos_token': TOKEN_EOS, 'unk_token': TOKEN_UNK, 'sep_token': TOKEN_SEP})

    data_module = CHILDESDataModule(BATCH_SIZE, tokenizer)

    model = CHILDESGrammarLSTM(tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id, vocab_size=tokenizer.vocab_size)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True,
                                            filename="{epoch:02d}-{val_loss:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min",
                                        min_delta=0.01, stopping_threshold=0.0)

    tb_logger = TensorBoardLogger(name="logs_pretrain_lstm", save_dir=os.path.curdir)
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1 if torch.cuda.is_available() else None,
        accelerator="gpu" if torch.cuda.is_available() else None,
        val_check_interval=1000,
        auto_lr_find=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
    )

    # trainer.tune(model, datamodule=data_module)
    #Learning rate set to 0.003311311214825908

    print("\n\n\nInitial validation:")
    initial_eval = trainer.validate(model, data_module)
    print(f"Perplexity: {math.exp(initial_eval[0]['val_loss']):.2f}")

    trainer.fit(model, datamodule=data_module)

    final_eval = trainer.validate(model, data_module)
    print(f"Perplexity: {math.exp(final_eval[0]['val_loss']):.2f}")


def parse_args():
    argparser = argparse.ArgumentParser()

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
