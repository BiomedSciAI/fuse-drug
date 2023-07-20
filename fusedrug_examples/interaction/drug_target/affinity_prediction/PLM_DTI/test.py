"""
PLM-DTI affinity predictor (see https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf)
Test-only script using an existing trained model 
To run from the CLI you need to append the test.checkpoint argument if it doesn't exist in the config, i.e:
python test.py +test.checkpoint=/path/to/model/checkpoint.ckpt
(If it does exist, note that it will be used in train.py as the initial checkpoint for training)
"""

import os
from omegaconf import DictConfig, OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from clearml import Task
import data
import plm_dti

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
SELECTED_CONFIG = "config.yaml"


@hydra.main(config_path=CONFIGS_DIR, config_name=SELECTED_CONFIG)
def main(cfg: DictConfig) -> None:

    if len(cfg) == 0:
        raise Exception(f"You should provide --config-dir and --config-name.")

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
    _, _, test_dataloader, cfg = data.get_dataloaders(cfg, test_mode=True)

    model = plm_dti.PLM_DTI_Module(cfg)

    trainer = pl.Trainer(
        default_root_dir=cfg.experiment.dir,
        devices=1,
        accelerator="gpu",
        benchmark=True,
    )

    _ = trainer.test(model, test_dataloader, ckpt_path=cfg.test.checkpoint)


if __name__ == "__main__":
    main()
