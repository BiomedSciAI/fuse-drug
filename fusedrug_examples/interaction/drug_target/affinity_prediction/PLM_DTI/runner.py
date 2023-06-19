"""
PLM-DTI affinity predictor (see https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf)
"""
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI import (
    data,
    plm_dti,
)
import os
from omegaconf import DictConfig, OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from clearml import Task

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
SELECTED_CONFIG = "train_config.yaml"


@hydra.main(config_path=CONFIGS_DIR, config_name=SELECTED_CONFIG)
def main(cfg: DictConfig) -> None:

    if len(cfg) == 0:
        raise Exception("You should provide --config-dir and --config-name.")

    print("Hydra config:")
    print(OmegaConf.to_yaml(cfg))
    print("End of Hydra config.")

    OmegaConf.resolve(
        cfg
    )  # to make sure that all "interpolated" values are resolved ( https://omegaconf.readthedocs.io/_/downloads/en/latest/pdf/ )
    # cfg_raw = OmegaConf.to_object(cfg)

    # Set random state
    pl.seed_everything(cfg.experiment.seed)

    # Load DataLoaders (wrapped by FuseMedML)
    train_dataloader, valid_dataloader, test_dataloader, cfg = data.get_dataloaders(cfg)

    model = plm_dti.PLM_DTI_Module(cfg)

    # Initialize clearml
    if cfg.experiment.clearml:
        _ = Task.init(
            project_name=cfg.experiment.project_name,
            task_name=cfg.experiment.experiment_name,
        )

    # Train model
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.trainer.watch_metric, save_top_k=1, mode="max"
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.experiment.dir,
        check_val_every_n_epoch=cfg.trainer.every_n_val,
        max_epochs=cfg.trainer.epochs,
        benchmark=True,
    )
    ckpt_path = (
        None if "checkpoint" not in cfg.experiment else cfg.experiment.checkpoint
    )  # start from checkpoint if exists
    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=ckpt_path)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
