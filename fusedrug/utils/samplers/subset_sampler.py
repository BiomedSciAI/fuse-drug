import torch
import numpy as np
from fuse.data.datasets.dataset_base import DatasetBase
from typing import Optional, Union

class SubsetSampler(torch.utils.data.Sampler):
    """
    A sequential sampler with optional pre-shuffling, but with the option of 
    limiting the number of sampler per epoch to be less than the total dataset size.
    In this way only a subset of the dataset is used in each epoch.
    :param dataset: dataset to sample from
    :param: sample_ids: One of:
        - Int denoting dataset length (needed in case the __len__ method of the dataset is undefined).
          If both __len__ is defined and sample_ids is defined as an Int, sample_ids will be used. 
          If sample_ids is defined as a list, the length of the list will be used.
        - List of sample ids to use as indices
        - None: use a running index: range(len(dataset))
    :param num_samples_per_epoch: number of samples that define an "epoch". if larger than the dataset length, or None, the dataset length will be used.
    :param shuffle: whether to shuffle in the beginning of every epoch
    """
    def __init__(self, dataset: DatasetBase, sample_ids: Optional[Union[int, list]], num_samples_per_epoch:Union[int, None], shuffle:bool=True):
        if sample_ids is None:
            dataset_len = len(dataset)
        elif isinstance(sample_ids, int):
            dataset_len = sample_ids
        elif isinstance(sample_ids, list):
            dataset_len = len(sample_ids)

        if num_samples_per_epoch is None or num_samples_per_epoch>dataset_len:
            num_samples_per_epoch = dataset_len
        self.dataset = dataset
        self.num_samples_per_epoch = num_samples_per_epoch
        self.dataset_len = dataset_len
        if sample_ids is None or isinstance(sample_ids, int):
            self.indices = list(range(dataset_len))
        elif isinstance(sample_ids, list):
            self.indices = sample_ids
            
        self.shuffle = shuffle
        self.epoch = 0 

    def __iter__(self):
        # offset for the current epoch
        offset = self.epoch * self.num_samples_per_epoch
        if offset + self.num_samples_per_epoch >= self.dataset_len:
            self.epoch = 0
            offset = 0

        # shuffle the indices before the first epoch
        if self.shuffle and self.epoch == 0:
            np.random.shuffle(self.indices)

        # yield indices that fall within the specified range:
        for i in range(offset, offset + self.num_samples_per_epoch):
            yield self.indices[i]

        self.epoch += 1

    def __len__(self):
        return self.num_samples_per_epoch
