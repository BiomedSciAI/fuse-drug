"""
BimodalMCA affinity predictor (see https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c00889)
"""

import os
from omegaconf import DictConfig, OmegaConf
import hydra
import socket
import model
import data
import utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), 'configs')
SELECTED_CONFIG = 'train_config.yaml'

OmegaConf.register_new_resolver("pytoda_ligand_tokenizer_path", utils.pytoda_ligand_tokenizer_path)
@hydra.main(config_path=CONFIGS_DIR, config_name=SELECTED_CONFIG)
def run_train_and_val(cfg : DictConfig) -> None:

    if len(cfg)==0:
        raise Exception(f'You should provide --config-dir and --config-name  . Note - config-name should be WITHOUT the .yaml postfix')

    SESSION_FULL_PATH = os.path.realpath(os.getcwd())
    
    print('Hydra config:')
    print(OmegaConf.to_yaml(cfg))
    print('End of Hydra config.')

    
    print(f'Running on hostname={socket.gethostname()}')
    STOP_FILENAME = os.path.join(SESSION_FULL_PATH, 'STOP')
    print(f'Will monitor the presence of a stop file to enable stopping a session gracefully: {STOP_FILENAME}')
    exit_on_stopfile_callback = utils.ExitOnStopFileCallback(STOP_FILENAME)

    OmegaConf.resolve(cfg) #to make sure that all "interpolated" values are resolved ( https://omegaconf.readthedocs.io/_/downloads/en/latest/pdf/ )
    cfg_raw = OmegaConf.to_object(cfg)

    lightning_data = data.AffinityDataModule(**cfg_raw['data']['lightning_data_module'])
    lightning_model = model.AffinityPredictorModule(**cfg_raw['model'])

    val_loss_callback = ModelCheckpoint(
        dirpath=SESSION_FULL_PATH,
        filename="val_loss",
        mode="min",
        monitor="val_loss",
        verbose=True,
        every_n_epochs=1,
        save_last=True,
        save_top_k=1,
    )
    val_rmse_callback = ModelCheckpoint(
        dirpath=SESSION_FULL_PATH,
        filename="val_rmse",
        mode="min",
        monitor="val_rmse",
        verbose=True,
        every_n_epochs=1,
        save_top_k=1,
    )
    val_pearson_callback = ModelCheckpoint(
        dirpath=SESSION_FULL_PATH,
        filename="val_pearson",
        mode="max",
        monitor="val_pearson",
        verbose=True,
        every_n_epochs=1,
        save_top_k=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[val_rmse_callback, val_loss_callback, val_pearson_callback, exit_on_stopfile_callback],
    )

    trainer.fit(lightning_model, lightning_data)


if __name__ == '__main__':
    run_train_and_val()









