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
from fuse.data import DatasetDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.losses.loss_wrap_to_dict import LossWrapToDict
from fuse.dl.models.heads import Head1D
from fuse.eval.metrics.metrics_common import Filter
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC
from fuse.dl.losses import LossDefault
from fuse.utils import NDict

from fusedrug_examples.design.amp.losses import kl_gaussian_sharedmu, LossRecon, LossWAE
from fusedrug_examples.design.amp.model import (
    Embed,
    WordDropout,
    Sample,
    Tokenizer,
    GRUEncoder,
    GRUDecoder,
    LogitsToSeq,
    TransformerEncoder,
    TransformerDecoder,
    RandomOverride,
    RandomMix,
    RandomAdjacentSwap,
    RandomShift,
)
from fusedrug_examples.design.amp.metrics import (
    MetricSeqAccuracy,
    MetricPrintRandomSubsequence,
)

def filter_label_unknown(batch_dict: NDict, label_key: str, out_key: str) -> NDict:
    """ignore samples with label -1"""
    # filter out samples

    keep_indices = batch_dict[label_key] != -1
    # keep_indices = keep_indices.cpu().numpy()
    return {label_key: batch_dict[label_key][keep_indices], out_key: batch_dict[out_key][keep_indices]}

def data(
    peptides_datasets: dict, batch_size: int, data_loader: dict
) -> Tuple[DatasetDefault, DataLoader, DataLoader]:
    """
    Data preparation
    :param peptides_datasets: kw_args for PeptidesDatasets.dataset()
    :param batch_size: batch size - used for both train dataloader and validation dataloader.
    :param data_loader: arguments for pytorch dataloader constructor
    :return: train dataset and dataloaders for both train-set and validation-set
    """

    ds_train, ds_valid, _ = PeptidesDatasets.dataset(**peptides_datasets)

    dl_train = DataLoader(
        ds_train,
        collate_fn=CollateDefault(
            keep_keys=["sequence", "amp.label", "toxicity.label", "data.sample_id"]
        ),
        batch_sampler=BatchSamplerDefault(
            ds_train,
            balanced_class_name="amp.label",
            balanced_class_weights={0: 0.25, 1: 0.25, -1: 0.5},
            batch_size=batch_size,
            mode="approx",
        ),
        **data_loader,
    )
    dl_valid = DataLoader(
        ds_valid,
        collate_fn=CollateDefault(
            keep_keys=["sequence", "amp.label", "toxicity.label", "data.sample_id"]
        ),
        batch_sampler=BatchSamplerDefault(
            ds_valid,
            balanced_class_name="amp.label",
            balanced_class_weights={0: 0.5, 1: 0.5, -1: 0.0},
            batch_size=batch_size,
            mode="approx",
        ),
        **data_loader,
    )

    return ds_train, dl_train, dl_valid


def model(
    seqs: List[str],
    encoder_type: str,
    transformer_encoder: dict,
    gru_encoder: dict,
    decoder_type: str,
    transformer_decoder: dict,
    gru_decoder: dict,
    embed: dict,
    word_dropout: dict,
    random_override: dict,
    random_mix: dict,
    random_adjacent_swap: dict,
    random_shift: dict,
    classification_amp: dict,
    classification_toxicity: dict,
    z_dim: int,
    cls_detached: bool,
    max_seq_len: int,
):
    """
    :param seqs: list of all train sequences - used to create a tokenizer
    :param encoder_type: either "transformer" or "gru"
    :param transformer_encoder: argument for TransformerEncoder constructor - used only with encoder_type=="transformer"
    :param gru_encoder: arguments for GRUEncoder constructor - used only with encoder_type=="gru"
    :param decoder_type: either "transformer" or "gru"
    :param transformer_decoder: arguments for TransformerDecoder constructor - used only with decoder_type=="transformer"
    :param gru_decoder: argument for GRUDecoder constructor - used only with decoder_type=="gru"
    :param embed: arguments for Embed constructor
    :param word_dropout: arguments for WordDropout constructor
    :param random_override: arguments for RandomOverride constructor
    :param random_mix: arguments for RandomMix constructor
    :param random_adjacent_swap: arguments for RandomAdjacentSwap constructor
    :param random_shift: arguments for RandomShift constructor
    :param classification_amp: arguments for Head1D constructor that used as a classifier head for amp classification
    :param classification_toxicity: arguments for Head1D constructor that used as a classifier head for toxicity classification
    :param z_dim: size of latent space
    :param cls_detached: if set to False the classification gradients will not flow into the encoder
    :param max_seq_len: maximum number of tokens in a sequence
    """

    tokenizer = Tokenizer(seqs, "sequence", "model.tokens", max_seq_len)
    if encoder_type == "transformer":
        encoder_model = TransformerEncoder(**transformer_encoder)
    elif encoder_type == "gru":
        encoder_model = GRUEncoder(**gru_encoder)
    else:
        raise Exception(f"Error: unsupported encoder {encoder_type}")

    if decoder_type == "transformer":
        decoder_model = TransformerDecoder(
            output_dim=len(tokenizer._tokenizer.vocab), **transformer_decoder
        )
    elif decoder_type == "gru":
        decoder_model = GRUDecoder(
            output_dim=len(tokenizer._tokenizer.vocab), **gru_decoder
        )
    else:
        raise Exception(f"Error: unsupported decoder {decoder_type}")
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
            model_outputs=["model.mu", "model.logvar"],
        ),
        Sample("model.mu", "model.logvar", key_out="model.z"),
        ModelWrapSeqToDict(
            model=decoder_model,
            model_inputs=["model.z"],
            model_outputs=["model.logits.gen"],
        ),
        LogitsToSeq(
            key_in="model.logits.gen",
            key_out="model.out",
            itos=tokenizer._tokenizer.vocab.itos,
        ),
        ModelWrapSeqToDict(
            model=torch.detach,
            model_inputs=["model.z"],
            model_outputs=["model.z_detached"],
        ),
        Head1D(
            head_name="amp",
            conv_inputs=[("model.z_detached" if cls_detached else "model.z", z_dim)],
            **classification_amp,
        ),
        Head1D(
            head_name="toxicity",
            conv_inputs=[("model.z_detached" if cls_detached else "model.z", z_dim)],
            **classification_toxicity,
        ),
    )

    return model, tokenizer


def train(
    model: torch.nn.Module,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    tokenizer: Any,
    model_dir: str,
    num_iter: int,
    opt: callable,
    trainer_kwargs: dict,
    losses: dict,
    track_clearml: Optional[dict] = None,
):
    """
    train code for the task
    :param model: the model to train
    :param dl_train: train-set dataloader
    :param dl_valid: validation dataloader
    :param tokenizer: with the following interface to the special token for padding -  tokenizer._tokenizer.vocab.stoi["<pad>"].
                     TODO: get the only the padding token id to simplify experiments with other tokenizers
    :param model_dir: path to store the training outputs
    :param num_iter: total number of iterations in training process
    :param opt: callable that given model parameters returns optimizer
    :param trainer_kwargs: arguments for pl.Trainer constructor
    :param losses: extra arguments for each loss
    :param track_clearml: optional. Arguments for start_clearml_logger() method. If not None will track the experiments with clearml.
    """
    if track_clearml is not None:
        from fuse.dl.lightning.pl_funcs import start_clearml_logger

        start_clearml_logger(**track_clearml)

    #  Loss

    losses = {
        "wae": LossWAE("model.z", num_iter),
        "recon": LossRecon(
            "model.tokens",
            "model.logits.gen",
            pad_index=tokenizer._tokenizer.vocab.stoi["<pad>"],
        ),
        "kl_shared_mu": LossWrapToDict(
            loss_module=kl_gaussian_sharedmu,
            loss_arg_to_batch_key={"logvar": "model.logvar"},
            **losses.kl_shared_mu,
        ),
        "amp_ce": LossDefault(
            pred="model.logits.amp",
            target="amp.label",
            callable=partial(F.cross_entropy, ignore_index=-1),
            **losses.amp_ce,
        ),
        "toxicity_ce": LossDefault(
            pred="model.logits.toxicity",
            target="toxicity.label",
            callable=partial(F.cross_entropy, ignore_index=-1),
            **losses.toxicity_ce,
        ),
    }

    # Metrics
    train_metrics = {}

    filter_func_amp = partial(filter_label_unknown, label_key="amp.label", out_key="model.output.amp")
    filter_func_toxicity = partial(filter_label_unknown, label_key="toxicity.label", out_key="model.output.toxicity")
    
    validation_metrics = {
        "acc": MetricSeqAccuracy(pred="model.out", target="sequence"),
        "auc_amp": 
            MetricAUCROC(pred="model.output.amp", target="amp.label", batch_pre_collect_process_func=filter_func_amp),
        "auc_toxicity": MetricAUCROC(pred="model.output.toxicity", target="toxicity.label", batch_pre_collect_process_func=filter_func_toxicity),

        "dump": MetricPrintRandomSubsequence(
            pred="model.out", target="sequence", num_sample_to_print=1
        ),
    }

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.losses.total_loss",
        mode="min",
    )

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizer = opt(params=model.parameters())
    optimizers_and_lr_schs = dict(optimizer=optimizer)

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
def main(cfg: DictConfig):
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)

    # data
    ds_train, dl_train, dl_valid = data(**cfg.data)

    # model
    seqs = [
        sample["sequence"]
        for sample in ds_train.get_multi(None, keys=["sequence"], desc="tokenizer")
    ]
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
