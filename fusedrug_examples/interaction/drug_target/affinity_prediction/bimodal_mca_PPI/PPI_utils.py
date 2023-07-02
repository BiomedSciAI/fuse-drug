import os
import numpy as np
import torch
import random
import pytz
from datetime import datetime
from collections import OrderedDict
import getpass
import re
from typing import Any, List


def get_cwd() -> str:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    return curr_dir


def listify(x: Any) -> List:
    if not isinstance(x, list):
        return [x]
    return x


def get_local_timestamp(timezone: str) -> str:
    """
    #to get a list of all available timezones use `pytz.all_timezones`
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    date_str = now.strftime("%Y-%m-%d@%H-%M-%S@%f")
    # print('local time:', date_str)
    return date_str


def worker_init_fn(worker_id: Any) -> None:
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


def get_valid_filename(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "@", s)
