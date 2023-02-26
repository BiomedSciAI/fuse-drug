#TODO: move to fuse, possibly not needed and already supported
import torch
from fuse.utils import NDict
from fuse.data import OpBase
import numpy as np

class OpConvertAllNumpyToTorch(OpBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, device='cpu', keys=None):
        '''
        :param keys: if None will recursively look for numpy arrays and move them to torch
            if it is a list, will only conver the items found in it
        '''
    
        if keys is None:
            keys = sample_dict.flatten().keys()

        for k in keys:
            if isinstance(sample_dict[k], np.ndarray):
                sample_dict[k] = torch.from_numpy(sample_dict[k]).to(device)                                

        return sample_dict