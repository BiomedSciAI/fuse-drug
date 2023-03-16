from pytorch_lightning.callbacks import Callback
import os, sys
from pathlib import Path
from typing import Union, List
import pandas as pd
from fusedrug.data.interaction.drug_target.datasets.dti_binding_dataset import (
    DTIBindingDataset,
)
import numpy as np
from time import time
import torch
import random
import pytz
from datetime import datetime
from collections import OrderedDict
import getpass
import re


def get_cwd():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    return curr_dir


def _check_stopfile(stop_filename):
    if os.path.isfile(stop_filename):
        print(f"detected request stop file: [{stop_filename}]. Exiting from process.")
        sys.stdout.flush()
        sys.exit()


class ExitOnStopFileCallback(Callback):
    def __init__(self, stop_filename=None):
        super().__init__()
        if not isinstance(stop_filename, str):
            raise Exception("stop_filename must be str")
        self.stop_filename = stop_filename
        print(
            f"ExitOnStopFileCallback: To stop the session (even if it is detached) create a file named: {self.stop_filename}"
        )

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        _check_stopfile(self.stop_filename)

    def on_batch_start(self, trainer, pl_module):
        _check_stopfile(self.stop_filename)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        _check_stopfile(self.stop_filename)


def listify(x):
    if not isinstance(x, list):
        return [x]
    return x


def get_local_timestamp(timezone: str):
    """
    #to get a list of all available timezones use `pytz.all_timezones`
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    date_str = now.strftime("%Y-%m-%d@%H-%M-%S@%f")
    # print('local time:', date_str)
    return date_str


def worker_init_fn(worker_id):
    # to ensure different seed on each worker
    worker_seed = (torch.initial_seed() + os.getpid()) % (2**32)
    print(f"worker_init_fn - seed={worker_seed}")
    np.random.seed(worker_seed)
    random.seed(worker_seed)


hydra_resolvers = OrderedDict(
    current_username=lambda: getpass.getuser(),
    local_time=get_local_timestamp,
    cwd=get_cwd,
)


def get_valid_filename(s):
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "@", s)


def listify(x):
    if not isinstance(x, list):
        x = [x]
    return x
