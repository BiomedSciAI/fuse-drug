from fuse.utils.ndict import NDict
import torch
import pytorch_lightning as pl

# The Contrastive_PLM_DTI submodule is the repository found at https://github.com/samsledje/Contrastive_PLM_DTI
# and described in the paper "Adapting protein language models for rapid DTI prediction": https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf
import fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.models as models
import metrics

# and described in the paper "Adapting protein language models for rapid DTI prediction": https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.Contrastive_PLM_DTI.src import (
    architectures as architectures,
)
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI import (
    metrics,
)

from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.losses.loss_default import LossDefault
import fuse.dl.lightning.pl_funcs as fuse_pl
import losses
from functools import partial
import os
import itertools
import pandas as pd


class PLM_DTI_Module(pl.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        print("Initializing model")
        self.cfg = cfg
        if cfg.model.model_architecture.lower() == "deepcoembedding":
            model_types = models
        else:
            model_types = architectures
        model = getattr(model_types, cfg.model.model_architecture)(
            cfg.model.drug_shape,
            cfg.model.target_shape,
            latent_dimension=cfg.model.latent_dimension,
            latent_distance=cfg.model.latent_distance,
            classify=cfg.model.classify,
        )

        # wrap model with fuse:
        self.model = ModelWrapSeqToDict(
            model=model,
            model_inputs=["data.drug", "data.target"],
            model_outputs=["model.output"],
        )

        if cfg.trainer.loss is None:
            if cfg.experiment.task == "dti_dg":
                loss_fct = torch.nn.MSELoss()
            else:
                loss_fct = torch.nn.BCELoss()
        else:
            if cfg.trainer.loss.lower() in (
                "bce",
                "cross_entropy",
                "ce",
                "cross entropy",
            ):
                loss_fct = torch.nn.BCELoss()
            elif cfg.trainer.loss.lower() in ("focal", "focal_loss", "focal loss"):
                loss_fct = partial(
                    losses.focal_loss,
                    is_input_logits=False,
                    alpha=-1,
                    gamma=2,
                    reduction="mean",
                )
        # wrap loss with fuse:
        # losses
        self.losses = {
            "cls_loss": LossDefault(
                pred="model.output", target="data.label", callable=loss_fct, weight=1.0
            ),
        }
        self.train_metrics, self.val_metrics, self.test_metrics = metrics.get_metrics(
            cfg.experiment.task
        )

        self.output_dir = cfg.experiment.dir
        if (
            "save_preds_for_benchmark_eval" in cfg.test
            and cfg.test.save_preds_for_benchmark_eval
        ):
            self.do_save_preds_for_benchmark_eval = True
        else:
            self.do_save_preds_for_benchmark_eval = False
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch_dict):
        return self.model(batch_dict)

    def training_step(self, batch_dict: NDict) -> torch.Tensor:
        batch_dict = self.model(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss and save losses values in batch_dict["losses"]
        loss = fuse_pl.step_losses(self.losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        fuse_pl.step_metrics(self.train_metrics, batch_dict)
        self.log("training_loss", loss)
        self.training_step_outputs.append({"losses": batch_dict["losses"]})
        # return the total_loss, the losses and drop everything else
        return {"loss": loss, "losses": batch_dict["losses"]}

    def validation_step(self, batch_dict: NDict, batch_idx: int) -> None:
        batch_dict = self.model(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss (ignored) and save losses values in batch_dict["losses"]
        loss = fuse_pl.step_losses(self.losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        fuse_pl.step_metrics(self.val_metrics, batch_dict)
        self.log("validation_loss", loss)
        self.validation_step_outputs.append({"losses": batch_dict["losses"]})
        return {"losses": batch_dict["losses"]}

    def on_train_epoch_end(self) -> None:
        step_outputs = self.training_step_outputs
        # calc average epoch loss and log it
        fuse_pl.epoch_end_compute_and_log_losses(
            self, "train", [e["losses"] for e in step_outputs]
        )
        # evaluate  and log it
        fuse_pl.epoch_end_compute_and_log_metrics(self, "train", self.train_metrics)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        step_outputs = self.validation_step_outputs
        # calc average epoch loss and log it
        fuse_pl.epoch_end_compute_and_log_losses(
            self, "validation", [e["losses"] for e in step_outputs]
        )
        # evaluate  and log it
        fuse_pl.epoch_end_compute_and_log_metrics(self, "validation", self.val_metrics)
        self.validation_step_outputs.clear()

    def test_step(self, batch_dict: NDict, batch_idx: int) -> None:
        batch_dict = self.model(batch_dict)
        loss = fuse_pl.step_losses(self.losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        fuse_pl.step_metrics(self.test_metrics, batch_dict)
        self.log("test_loss", loss)
        sample_ids = (
            batch_dict["data.sample_id"]
            if self.do_save_preds_for_benchmark_eval
            else None
        )
        self.test_step_outputs.append({"losses": batch_dict["losses"]})
        return {
            "losses": batch_dict["losses"],
            "preds": batch_dict["model.output"],
            "ids": sample_ids,
        }

    def on_test_epoch_end(self) -> None:
        step_outputs = self.test_step_outputs
        ### add saving of predictions and labels

        # calc average epoch loss and log it
        fuse_pl.epoch_end_compute_and_log_losses(
            self, "test", [e["losses"] for e in step_outputs]
        )
        # evaluate  and log it
        fuse_pl.epoch_end_compute_and_log_metrics(self, "test", self.test_metrics)

        # save predictions and labels (for later evaluation)
        if self.do_save_preds_for_benchmark_eval:
            self.save_preds_for_benchmark_eval(step_outputs)

        self.test_step_outputs.clear()

    def save_preds_for_benchmark_eval(self, test_step_outputs):
        output_filepath = os.path.join(self.output_dir, "test_results.tsv")
        preds = (
            torch.cat([item["preds"] for item in test_step_outputs], 0).cpu().numpy()
        )
        # labels = torch.cat([item['labels'] for item in test_step_outputs], 0).cpu().numpy()
        ids_lists = [item["ids"] for item in test_step_outputs]
        ids = list(itertools.chain.from_iterable(ids_lists))
        source_dataset_versioned_names = [item[0] for item in ids]
        source_dataset_activity_ids = [item[1] for item in ids]
        df = pd.DataFrame(
            columns=[
                "source_dataset_versioned_name",
                "source_dataset_activity_id",
                "pred",
            ]
        )
        df["source_dataset_versioned_name"] = source_dataset_versioned_names
        df["source_dataset_activity_id"] = source_dataset_activity_ids
        df["pred"] = preds
        df.to_csv(output_filepath, index=False, sep="\t")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.trainer.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.cfg.trainer.lr_t0
        )
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}
