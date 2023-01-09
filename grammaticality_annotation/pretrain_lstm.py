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

from utils import UTTERANCES_WITH_PREV_UTTS_FILE

LM_DATA = os.path.expanduser("~/data/communicative_feedback/sentences.txt")
TOKENIZER_PATH = "data/tokenizer-childes.json"

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

    sentences = data.apply(lambda row: TOKEN_SEP.join([row.prev_transcript_clean, row.transcript_clean]), axis=1).values
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
        text = [t["text"] + TOKEN_EOS for t in batch]
        encodings = self.tokenizer.batch_encode_plus(text, padding=True, max_length=TRUNCATION_LENGTH, truncation=True)
        return encodings.data

    def train_dataloader(self):
        return DataLoader(self.data["train"], batch_size=self.batch_size, collate_fn=self.tokenize_batch)

    def val_dataloader(self):
        return DataLoader(self.data["test"], batch_size=self.batch_size, collate_fn=self.tokenize_batch)


class CHILDESLSTM(LightningModule):
    def __init__(
            self,
            train_batch_size: int,
            eval_batch_size: int,
            tokenizer,
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

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

        self.init_weights()

    def forward(self, batch):
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        seq_lengths = [att.index(0) if 0 in att else len(att) for att in batch["attention_mask"]]
        hidden = self.init_hidden(len(input_ids))
        logits, _ = self.forward_step(input_ids, seq_lengths, hidden)
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        return logits, labels

    def forward_step(self, input_ids, seq_lengths, hidden):
        embedding = self.embedding(input_ids)
        packed_input = pack_padded_sequence(embedding, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_input, hidden)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

    def init_weights(self):
        #TODO check
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
                                                            self.hidden_dim).uniform_(-init_range_other,
                                                                                      init_range_other).to(device)
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_dim,
                                                            self.hidden_dim).uniform_(-init_range_other,
                                                                                      init_range_other).to(device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell

    def training_step(self, batch, batch_idx):
        logits, labels = self(batch)
        loss = self.loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log("loss", loss.mean().item(), prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits, labels = self(batch)
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
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return [optimizer]

    def generate(self, prompt, max_seq_len, temperature, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        input_ids = self.tokenizer.encode(prompt)
        hidden = self.init_hidden(batch_size=1)
        with torch.no_grad():
            for i in range(max_seq_len):
                src = torch.LongTensor([input_ids]).to(device)
                prediction, hidden = self.forward_step(src, [src.shape[1]], hidden)
                probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
                prediction = torch.multinomial(probs, num_samples=1).item()

                if prediction == self.tokenizer.eos_token_id:
                    break

                input_ids.append(prediction)

        decoded = self.tokenizer.decode(input_ids)
        self.train()
        return decoded


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

    model = CHILDESLSTM(BATCH_SIZE, BATCH_SIZE, tokenizer=tokenizer)

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
