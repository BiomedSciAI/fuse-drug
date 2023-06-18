import torch
import os
import numpy as np
import random
from typing import Any, List


def listify(x: Any) -> List:
    if not isinstance(x, list):
        return [x]
    return x


def worker_init_fn(worker_id: int) -> None:
    # to ensure different seed on each worker
    worker_seed = (torch.initial_seed() + os.getpid() + worker_id) % (2 ** 32)
    print(f"worker_init_fn - seed={worker_seed}")
    set_rng_seed(worker_seed)
    # add torch random seed selection as well?


def set_rng_seed(seed: int) -> None:
    print(f"set_rng_seed: setting python random, numpy and torch rng seeds to {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
