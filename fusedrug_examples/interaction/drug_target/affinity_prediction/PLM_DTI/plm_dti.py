from typing import List
from fuse.utils.ndict import NDict
import torch
import pytorch_lightning as pl

# The Contrastive_PLM_DTI submodule is the repository found at https://github.com/samsledje/Contrastive_PLM_DTI
# and described in the paper "Adapting protein language models for rapid DTI prediction": https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.Contrastive_PLM_DTI.src import (
    architectures as model_types,
)
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI import metrics
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.losses.loss_default import LossDefault
import fuse.dl.lightning.pl_funcs as fuse_pl


class PLM_DTI_Module(pl.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        print("Initializing model")
        self.cfg = cfg
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

        if cfg.experiment.task == "dti_dg":
            loss_fct = torch.nn.MSELoss()
        else:
            loss_fct = torch.nn.BCELoss()

        # wrap loss with fuse:
        # losses
        self.losses = {
            "cls_loss": LossDefault(pred="model.output", target="data.label", callable=loss_fct, weight=1.0),
        }
        self.val_metrics, self.test_metrics = metrics.get_metrics(cfg.experiment.task)
        self.val_metric_dict = metrics.get_metrics_instances(self.val_metrics)
        self.test_metric_dict = metrics.get_metrics_instances(self.test_metrics)
        # TBD - handle checkpoint loading

    def forward(self, batch_dict: NDict) -> NDict:
        return self.model(batch_dict)

    def training_step(self, batch_dict: NDict) -> torch.Tensor:
        batch_dict = self.model(batch_dict)
        loss = fuse_pl.step_losses(self.losses, batch_dict)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch_dict: NDict, batch_idx: int) -> None:
        batch_dict = self.model(batch_dict)
        loss = fuse_pl.step_losses(self.losses, batch_dict)
        for _, met_instance in self.val_metric_dict.items():
            met_instance(batch_dict["model.output"].type(torch.float32), batch_dict["data.label"])
        self.log("validation_loss", loss)

    def validation_epoch_end(self, validation_step_outputs: List) -> dict:
        results = {}
        for (k, met_instance) in self.val_metric_dict.items():
            res = met_instance.compute()
            results[k] = res
            self.log(k, results[k])
        return results

    def test_step(self, batch_dict: NDict, batch_idx: int) -> None:
        batch_dict = self.model(batch_dict)
        loss = fuse_pl.step_losses(self.losses, batch_dict)
        for _, met_instance in self.test_metric_dict.items():
            met_instance(batch_dict["model.output"].type(torch.float32), batch_dict["data.label"])
        self.log("test_loss", loss)

    def test_epoch_end(self, test_step_outputs: List) -> dict:
        results = {}
        for (k, met_instance) in self.test_metric_dict.items():
            res = met_instance.compute()
            results[k] = res
            self.log(k, results[k])
        return results

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.trainer.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=self.cfg.trainer.lr_t0)
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}
