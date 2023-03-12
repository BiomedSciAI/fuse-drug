import torch
import os
import numpy as np
import random

def listify(x):
    if not isinstance(x, list):
        return [x]
    return x

def worker_init_fn(worker_id):
    #to ensure different seed on each worker
    worker_seed = (torch.initial_seed()+os.getpid()+worker_id) % (2**32)
    print(f'worker_init_fn - seed={worker_seed}')
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    