import argparse
import torch
from pytorch_lightning import Trainer

from grammaticality_annotation.fine_tune_grammaticality_nn import CHILDESGrammarDataModule, \
    CHILDESGrammarTransformer


def main(args):
    model = CHILDESGrammarTransformer.load_from_checkpoint(args.model_checkpoint)
    hparams = model.hparams
    dm = CHILDESGrammarDataModule(val_split_proportion=hparams.val_split_proportion,
                                  model_name_or_path=hparams.model_name_or_path,
                                  eval_batch_size=hparams.eval_batch_size,
                                  train_batch_size=hparams.train_batch_size,
                                  train_datasets=hparams.train_datasets,
                                  val_datasets=hparams.val_datasets,)
    dm.setup("fit")

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    )

    model.val_error_analysis = True

    print("\n\n\nValidation:")
    trainer.validate(model, dm)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
