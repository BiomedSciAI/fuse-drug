"""
training BimodalMCA affinity predictor (see https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520)
with focus on multiprocessing to improve GPU utilization
"""

import os

# import tensorflow as tf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import hydra
from PPI_utils import (
    hydra_resolvers,
)  # examples.fuse_examples.interaction.drug_target.affinity_prediction.bimodal_mca_PPI.

from omegaconf import DictConfig, OmegaConf
import sys
import socket
from fuse.utils.file_io import read_simple_int_file
from clearml import Task
from pl_model import AffinityPredictorModule
from pl_data import AffinityDataModule
import colorama

from colorama import Fore
from typing import Dict

colorama.init(autoreset=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


print(
    "CUDA_VISIBLE_DEVICES=",
    os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else None,
)


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
SELECTED_CONFIG = "train_bmmca_full_PPI.yaml"

for name, func in hydra_resolvers.items():
    OmegaConf.register_new_resolver(name, func)


# logging.info(f'CONFIGS_DIR={CONFIGS_DIR}')
# logging.info(f'SELECTED_CONFIG={SELECTED_CONFIG}')


def session_manager_and_logger_setup(cfg: dict) -> None:
    cfg_raw = OmegaConf.to_object(cfg)
    if "LSB_JOBID" in os.environ:
        print(f'LSB_JOBID={os.environ["LSB_JOBID"]}')
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    SESSION_FULL_PATH = os.path.realpath(os.getcwd())
    HYDRA_SESSION_NUM = os.path.basename(os.getcwd())

    OmegaConf.save(config=cfg, f=os.path.join(SESSION_FULL_PATH, "resolved_config.yaml"))

    session_manager_num = None
    session_num_file = os.path.realpath(os.path.join(SESSION_FULL_PATH, "../session_created"))
    if os.path.isfile(session_num_file):
        session_manager_num = read_simple_int_file(session_num_file)
        print(f"found session num {session_manager_num} in {session_num_file}")
    else:
        print(
            "WARNING: this run was initiated without the session manager. This is ok for debugging, but it will pollute your code directory with all sorts of files/dirs.\n"
            "Few tips:\n"
            "To able to run on CCC make sure you use scripts/session_runner_ccc.py"
            "not only it would let you run it on a cluster, it would copy the config and scripts into a new directory, allowing to rerun it,\n"
            "which is very useful for many things - like continuous training accross sessions, distributed training, and more\n"
            "without worrying that the config or the code (within this project) changes\n"
        )

        print(
            f'Note: the debug session directory, in which artifacts like logs and checkpoints will be saved, is: {cfg_raw["paths"]["debug_session_dir"]}'
        )
        print("to choose a different location modify paths.debug_session_dir in configs/train_config.yaml")
        print("You may need to manually delete this directory.")

        SESSION_FULL_PATH = cfg_raw["paths"]["debug_session_dir"]
        os.makedirs(SESSION_FULL_PATH, exist_ok=True)

    default_checkpoints_dir = os.path.realpath(os.path.join(SESSION_FULL_PATH, "../rank_0"))
    print(f"checkpoints will be written into {default_checkpoints_dir}")
    os.makedirs(default_checkpoints_dir, exist_ok=True)

    # clearml
    project_name = cfg_raw["clearml"]["project_name"]
    print("CLEARML - using project_name=", project_name)
    task_name = cfg_raw["clearml"]["task_name"]

    if session_manager_num is None:
        task_name = f"{task_name}@{HYDRA_SESSION_NUM}"
    else:
        # Note: not using hydra num here to allow continued training, see description here: https://github.com/allegroai/clearml/issues/160 (description in the official doc is too cryptic)
        task_name = f"session_{session_manager_num}@{task_name}"
        # task_name=f"session_{session_manager_num}@{task_name}@{HYDRA_SESSION_NUM}"

    task_name += f'@{cfg_raw["_free_text_description"]}' + cfg_raw["model"]["base_model"]
    print("CLEARML - using task_name=", task_name)

    if cfg_raw["clearml"]["active"]:
        if (cfg_raw["load_from_checkpoint"] is not None) and cfg_raw["clearml"][
            "load_from_checkpoint_continues_within_session"
        ]:
            print(
                Fore.GREEN
                + f"for clearml: continuing existing task - project_name={project_name}, task_name={task_name}, continue_last_task=True"
            )
            Task.init(
                project_name=project_name,
                task_name=task_name,
                continue_last_task=True,
            )
        else:
            print(
                Fore.GREEN
                + f"for clearml: starting a new task - project_name={project_name}, task_name={task_name}, reuse_last_task_id=False"
            )
            Task.init(
                project_name=project_name,
                task_name=task_name,
                reuse_last_task_id=False,
            )
    return SESSION_FULL_PATH


# @hydra.main(config_path=None) #use this if you intend to pass --config-dir and --config-name from CLI
@hydra.main(config_path=CONFIGS_DIR, config_name=SELECTED_CONFIG)
def main(cfg: DictConfig) -> None:
    if 0 == len(cfg):
        raise Exception(
            "You should provide --config-dir and --config-name  . Note - config-name should be WITHOUT the .yaml postfix"
        )

    OmegaConf.resolve(
        cfg
    )  # to make sure that all "interpolated" values are resolved ( https://omegaconf.readthedocs.io/_/downloads/en/latest/pdf/ )
    cfg_raw = OmegaConf.to_object(cfg)

    SESSION_FULL_PATH = session_manager_and_logger_setup(cfg=cfg)

    if ("ONLY_GET_EXPECTED_WORKING_DIR" in os.environ) or cfg.only_get_expected_working_dir:
        QUERY_SEP_TOKEN = "<@@@QUERY_SEP_TOKEN@@@>"
        print(
            f"SESSION_FULL_PATH_QUERY_ANSWER={QUERY_SEP_TOKEN}{SESSION_FULL_PATH}{QUERY_SEP_TOKEN}"
        )  # this is used by "run_detached_fuse_based.sh" to know the working dir, to be able to create it and redirect stdour/err to it
        return 0
    print("Hydra config:")
    print(OmegaConf.to_yaml(cfg))
    print("/hydra config")

    # os.makedirs(SESSION_FULL_PATH, exist_ok=True)

    print(f"Running on hostname={socket.gethostname()}")
    STOP_FILENAME = os.path.join(SESSION_FULL_PATH, "STOP")
    print(f"created a stop filename - create it to stop a session gracefully. [{STOP_FILENAME}]")

    def _check_stopfile(stop_filename: str) -> None:
        if os.path.isfile(stop_filename):
            print(f"detected request stop file: [{STOP_FILENAME}]. Exiting from process.")
            sys.stdout.flush()
            sys.exit()

    class ExitOnStopFileCallback(Callback):
        def __init__(self, stop_filename: str = None) -> None:
            super().__init__()
            if not isinstance(stop_filename, str):
                raise Exception("stop_filename must be str")
            self.stop_filename = stop_filename
            print(
                f"ExitOnStopFileCallback: To stop the session (even if it is detached) create a file named: {self.stop_filename}"
            )

        # def on_predict_batch_start(self, trainer=trainer, pl_module, batch, batch_idx, dataloader_idx):
        #     _check_stopfile(self.stop_filename)
        def on_predict_batch_start(self, **kwargs: Dict) -> None:
            _check_stopfile(self.stop_filename)

        def on_train_batch_start(self, **kwargs: Dict) -> None:
            _check_stopfile(self.stop_filename)

        def on_test_batch_start(self, **kwarg: Dict) -> None:
            _check_stopfile(self.stop_filename)

    print("cfg_raw = ")
    print(cfg_raw)
    print("done printing cfg_raw")

    lightning_data = AffinityDataModule(
        **cfg_raw["data"]["lightning_data_module"],
    )
    lightning_model_module = AffinityPredictorModule(
        **cfg_raw["model"],
    )

    exit_on_stopfile_callback = ExitOnStopFileCallback(STOP_FILENAME)

    val_loss_callback = pl.callbacks.ModelCheckpoint(
        dirpath=SESSION_FULL_PATH,
        filename="val_loss",
        mode="min",
        monitor="val_loss",
        verbose=True,
        every_n_epochs=1,
        save_last=True,
        save_top_k=1,
    )
    val_rmse_callback = pl.callbacks.ModelCheckpoint(
        dirpath=SESSION_FULL_PATH,
        filename="val_rmse",
        mode="min",
        monitor="val_rmse",
        verbose=True,
        every_n_epochs=1,
        save_top_k=1,
    )
    val_pearson_callback = pl.callbacks.ModelCheckpoint(
        dirpath=SESSION_FULL_PATH,
        filename="val_pearson",
        mode="max",
        monitor="val_pearson",
        verbose=True,
        every_n_epochs=1,
        save_top_k=1,
    )

    #

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            val_rmse_callback,
            val_loss_callback,
            val_pearson_callback,
            exit_on_stopfile_callback,
        ],
    )

    trainer.fit(lightning_model_module, lightning_data)


if __name__ == "__main__":
    print("GOT HERE!!! start of script")

    main()
