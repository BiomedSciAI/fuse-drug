"""
(C) Copyright 2021 IBM Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from fusedrug_examples.design.amp.datasets import PeptidesDatasets

from typing import Any, Optional, List, Tuple
import hydra
from omegaconf import DictConfig
from functools import partial

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader

from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.models.heads import Head1D
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC
from fuse.dl.losses import LossDefault
from fuse.utils import NDict
from fuse.data import DatasetDefault

from fusedrug_examples.design.amp.model import (
    Embed,
    WordDropout,
    Tokenizer,
    GRUEncoder,
    TransformerEncoder,
    RandomOverride,
    RandomMix,
    RandomAdjacentSwap,
    RandomShift,
)
from fusedrug_examples.design.amp.losses import LossRecon


def data(
    peptides_datasets: dict,
    target_key: str,
    infill: bool,
    batch_size: int,
    num_batches: int,
    data_loader: dict,
) -> Tuple[DatasetDefault, DataLoader, DataLoader]:
    """
    Data preparation
    :param peptides_datasets: kw_args for PeptidesDatasets.dataset()
    :param target_key: used to to balance the trainset
    :param infill: if True will keep the unlabeled data. Otherwise it will filter it.
    :param batch_size: batch size - used for both train dataloader and validation dataloader.
    :param num_batches: number of batches
    :param data_loader: arguments for pytorch dataloader constructor
    :return: train dataset and dataloaders for both train-set and validation-set
    """
    ds_train, ds_valid, _ = PeptidesDatasets.dataset(**peptides_datasets)

    # filter unknown labels
    if not infill:
        labels_train = ds_train.get_multi(None, keys=[target_key, "source"], desc="filter train set")
        indices_train = [i for i, v in enumerate(labels_train) if v[target_key] >= 0]
        ds_train.subset(indices_train)

    labels_valid = ds_valid.get_multi(None, keys=[target_key, "source"], desc="filter validation set")
    indices_valid = [i for i, v in enumerate(labels_valid) if v[target_key] >= 0]
    ds_valid.subset(indices_valid)

    dl_train = DataLoader(
        ds_train,
        collate_fn=CollateDefault(keep_keys=["sequence", target_key, "data.sample_id"]),
        batch_sampler=BatchSamplerDefault(
            ds_train,
            balanced_class_name=target_key,
            batch_size=batch_size,
            mode="approx",
            num_batches=num_batches,
        ),
        **data_loader,
    )
    dl_valid = DataLoader(
        ds_valid,
        collate_fn=CollateDefault(keep_keys=["sequence", target_key, "data.sample_id"]),
        **data_loader,
    )

    return ds_train, dl_train, dl_valid


def model(
    seqs: List[str],
    encoder_type: str,
    transformer_encoder: dict,
    gru_encoder: dict,
    embed: dict,
    word_dropout: dict,
    random_override: dict,
    random_mix: dict,
    random_adjacent_swap: dict,
    random_shift: dict,
    classifier_head: dict,
    classifier_recon_head: dict,
    z_dim: int,
    max_seq_len: int,
) -> Tuple[torch.nn.Module, Tokenizer]:
    """
    :param seqs: list of all train sequences - used to create a tokenizer
    :param encoder_type: either "transformer" or "gru"
    :param transformer_encoder: argument for TransformerEncoder constructor - used only with encoder_type=="transformer"
    :param gru_encoder: arguments for GRUEncoder constructor - used only with encoder_type=="gru"
    :param embed: arguments for Embed constructor
    :param word_dropout: arguments for WordDropout constructor
    :param random_override: arguments for RandomOverride constructor
    :param random_mix: arguments for RandomMix constructor
    :param random_adjacent_swap: arguments for RandomAdjacentSwap constructor
    :param random_shift: arguments for RandomShift constructor
    :param classifier_head: arguments for Head1D constructor that used as a classifier head for target classification
    :param classifier_recon_head: arguments for Head1D constructor that used to reconstruct the sequence - used if infill set to True
    :param z_dim: size of latent space
    :param max_seq_len: maximum number of tokens in a sequence
    """

    tokenizer = Tokenizer(seqs, "sequence", "model.tokens", max_seq_len)
    if encoder_type == "transformer":
        encoder_model = TransformerEncoder(**transformer_encoder)
    elif encoder_type == "gru":
        encoder_model = GRUEncoder(**gru_encoder)
    else:
        raise Exception(f"Error: unsupported encoder {encoder_type}")

    embed = Embed(
        key_in="model.tokens_encoder",
        key_out="model.embedding",
        n_vocab=len(tokenizer._tokenizer.vocab),
        **embed,
    )
    model = torch.nn.Sequential(
        tokenizer,
        RandomAdjacentSwap(
            key_in="model.tokens",
            key_out="model.tokens_encoder",
            **random_adjacent_swap,
        ),
        WordDropout(
            key_in="model.tokens_encoder",
            key_out="model.tokens_encoder",
            **word_dropout,
        ),
        RandomOverride(
            key_in="model.tokens_encoder",
            key_out="model.tokens_encoder",
            values=list(range(len(tokenizer._tokenizer.vocab))),
            **random_override,
        ),
        RandomShift(
            key_in="model.tokens_encoder",
            key_out="model.tokens_encoder",
            **random_shift,
        ),
        embed,
        RandomMix(
            key_in="model.embedding",
            key_out="model.embedding",
            values=list(range(len(tokenizer._tokenizer.vocab))),
            embed=embed,
            **random_mix,
        ),
        ModelWrapSeqToDict(
            model=encoder_model,
            model_inputs=["model.embedding"],
            model_outputs=["model.z", "model.z_recon"],
        ),
        Head1D(head_name="cls", conv_inputs=[("model.z", z_dim)], **classifier_head),
        Head1D(
            head_name="recon",
            conv_inputs=[("model.z_recon", z_dim)],
            num_outputs=len(tokenizer._tokenizer.vocab),
            **classifier_recon_head,
        ),
    )

    return model, tokenizer


def filter_label_unknown(batch_dict: NDict, label_key: str, out_key: str) -> NDict:
    """ignore samples with label -1"""
    # filter out samples

    keep_indices = batch_dict[label_key] != -1
    # keep_indices = keep_indices.cpu().numpy()
    return {label_key: batch_dict[label_key][keep_indices], out_key: batch_dict[out_key][keep_indices]}


def train(
    model: torch.nn.Module,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    model_dir: str,
    opt: callable,
    trainer_kwargs: dict,
    target_key: str,
    tokenizer: Any,
    infill: bool,
    lr_scheduler: callable = None,
    track_clearml: Optional[dict] = None,
) -> None:
    """
    train code for the task
    :param model: the model to train
    :param dl_train: train-set dataloader
    :param dl_valid: validation dataloader
    :param model_dir: path to store the training outputs
    :param opt: callable that given model parameters returns optimizer
    :param trainer_kwargs: arguments for pl.Trainer constructor
    :param target_key: key for the labels
    :param tokenizer: with the following interface to the special token for padding -  tokenizer._tokenizer.vocab.stoi["<pad>"].
                     TODO: get the only the padding token to simplify experiments with other tokenizers
    :param infill: if True will keep the unlabeled data. Otherwise it will filter it.
    :param lr_scheduler: callable to return the learning rate scheduler given an optimizer.
    :param track_clearml: optional. Arguments for start_clearml_logger() method. If not None will track the experiments with clearml.
    """

    if track_clearml is not None:
        from fuse.dl.lightning.pl_funcs import start_clearml_logger

        start_clearml_logger(**track_clearml)

    #  Loss
    losses = {
        "ce": LossDefault(
            pred="model.logits.cls",
            target=target_key,
            callable=partial(F.cross_entropy, ignore_index=-1),
        ),
    }
    if infill:
        losses["recon"] = LossRecon(
            "model.tokens",
            "model.logits.recon",
            pad_index=tokenizer._tokenizer.vocab.stoi["<pad>"],
        )

    # Metrics
    filter_func = partial(filter_label_unknown, label_key=target_key, out_key="model.output.cls")
    train_metrics = {
        "auc": MetricAUCROC(pred="model.output.cls", target=target_key, batch_pre_collect_process_func=filter_func),
    }

    validation_metrics = {
        "auc": MetricAUCROC(pred="model.output.cls", target=target_key),
    }

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.metrics.auc",
        mode="max",
    )

    # optimizer and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizer = opt(params=model.parameters())
    optimizers_and_lr_schs = dict(optimizer=optimizer)

    if lr_scheduler is not None:
        optimizers_and_lr_schs["lr_scheduler"] = lr_scheduler(optimizer)
        if isinstance(
            optimizers_and_lr_schs["lr_scheduler"],
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            optimizers_and_lr_schs["monitor"] = "validation.losses.total_loss"

    #  Train
    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

    # create lightning trainer.
    pl_trainer = pl.Trainer(**trainer_kwargs)

    # train
    pl_trainer.fit(pl_module, dl_train, dl_valid, ckpt_path=None)


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)
    print(NDict(cfg).print_tree(True))

    # data
    ds_train, dl_train, dl_valid = data(**cfg.data)

    # model
    seqs = [sample["sequence"] for sample in ds_train.get_multi(None, keys=["sequence"], desc="tokenizer")]
    nn_model, tokenizer = model(seqs=seqs, **cfg.model)

    # train
    train(
        model=nn_model,
        dl_train=dl_train,
        dl_valid=dl_valid,
        tokenizer=tokenizer,
        **cfg.train,
    )


if __name__ == "__main__":
    main()
