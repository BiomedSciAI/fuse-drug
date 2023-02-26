from typing import List, Union
from torch.utils.data import Dataset, Sampler 
import numpy as np

# TODO: consider moving to core fuse if needed on top of the existing balanced sampler
class FastBalancedSampler(Sampler):
    def __init__(self, 
        datasets_lengths:List[Union[int, Dataset]], 
        minibatch_pattern:List[int], 
        shuffle:bool=False,
        shuffle_within_minibatch:bool=None,
        yield_minibatch:bool=True,
        epoch_minibatches_count_mode:str = 'see_all',
        verbose=1,
        ):
        '''
        Creates a balanced sampler of indices. Useful when combined with ConcatDataset.
        I created it because WeightedRandomSampler had too many issues, and in addition it was too slow,
        especially when dealing with *massive* datasets, containing billions of samples.

        for example:
            dataset1 = ...
            dataset2 = ...
            dataset3 = ...
            all_datasets = [dataset1, dataset2, dataset3]
            combined_dataset = ConcatDataset()
            balanced_sampler = BalancedSampler(
                datasets_lengths = [len(ds) for ds in all_datasets], #you may also pass the datasets themselves
                minibatch_pattern = [3,2,1], #each minibatch consists of 3 samples from the first dataset, 2 samples from the second dataset, and 1 sample from the third dataset
                shuffle=True,
                epoch_minibatches_count_mode = 'see_all',
            )

        Args:
            datasets_lengths: the lengths of the sub parts (consequitive) that you want to verify equal sampling from.
             For example, if the dataset that you intend to use with this sampler was created by calling ConcatDataset([dataset1, dataset2, dataset3]), 
              then you would pass [len(x) for x in [dataset1, dataset2, dataset3]]
              You may also pass the list of datasets, in the same order that you passed them to ConcatDataset
             
            minibatch_pattern: how many samples to take from each sub part. 
             For example, setting minibatch_pattern=[1,3,2] would mean that each minibatch has 6 samples (1+3+2) and contains:
                1 sample of the first dataset
                3 samples of the second dataset
                2 sampler of the third dataset
            shuffle: should the order be shuffled (without replacement)
            shuffle_within_minibatch: should every minibatch be shuffled within itself 
                this is useful especially when data parallelization is used, to make sure we don't get into a degenerate state in which a certain gpu/node always gets the same classes
                The default behavior (None) is:
                    * if shuffle is True, then shuffle_within_minibatch will be considered as True.
                    * if shuffle is False, then shuffle_within_minibatch will be considered as False.
                Pass False or True to override the default behavior
            yield_minibatch: when set to True will return a minibatch. Set to False to return an individual sample each time.

            epoch_minibatches_count_mode: determines how the len() of the sampler will be determined.
                options:
                    pass epoch_minibatches_count_mode='see_all' to automatically calculate the number of minibatches required in order to see *all* of the samples
                    pass epoch_minibatches_count_mode=('see_all_of_specific_dataset', <dataset index (an integer)>) to automatically calculate the number of minibatches required in order to see all samples of the given dataset index
                        for example - epoch_minibatches_count_mode=('see_all_of_specific_dataset', 0) to define length to be minibatches num that is required to see all samples from dataset 0
                    pass epoch_minibatches_count_mode = integer to define any desired integer number of minibatches that defines len()
                        for example - epoch_minibatches_count_mode=1234
            
            
        '''

        self._datasets_lengths = [len(x) if isinstance(x, Dataset) else x for x in datasets_lengths]
        if verbose>0:
            print('FastBalancedSampler::datasets lengths =',self._datasets_lengths)
        self._datasets_num = len(self._datasets_lengths)
        self._minibatch_pattern = minibatch_pattern
        if verbose>0:
            print('FastBalancedSampler::minibatch_pattern =',self._minibatch_pattern)
        assert isinstance(self._minibatch_pattern, list)
        for x in self._minibatch_pattern:
            assert isinstance(x, int)
        if len(self._datasets_lengths) != len(self._minibatch_pattern):
            raise Exception('datasets_lengths and minibatch_pattern are expected to be lists with the same length')
        if len(self._datasets_lengths)<1:
            raise Exception('got an empty datasets_lengths!')

        self._minibatch_size = sum(self._minibatch_pattern)

        self._epoch_minibatches_count_mode = epoch_minibatches_count_mode        
        self._shuffle = shuffle
        self._shuffle_within_minibatch = shuffle_within_minibatch
        if self._shuffle_within_minibatch is None:
            self._shuffle_within_minibatch = self._shuffle

        self._yield_minibatch = yield_minibatch

        self._per_dataset_indices = [np.arange(x) for x in self._datasets_lengths]
        #points to one after the last valid pointer, so it will trigger shuffle (or go to start if no shuffle is requested)
        self._per_dataset_pointers = [x for x in self._datasets_lengths] 

        # calculate the needed minibatches per epoch (which is defined in len())
        _err_msg = f'Invalid self._epoch_minibatches_count_mode provided = {self._epoch_minibatches_count_mode} - please see the doc for description of valid values'

        if isinstance(self._epoch_minibatches_count_mode, int):
            pass
        elif isinstance(self._epoch_minibatches_count_mode, tuple):
            if not len(self._epoch_minibatches_count_mode) == 2:
                raise Exception(_err_msg)
            if not self._epoch_minibatches_count_mode[0] == 'see_all_of_specific_dataset':
                raise Exception(_err_msg)
            if not self._epoch_minibatches_count_mode[1] >=0 and self._epoch_minibatches_count_mode[1] < self._datasets_num:
                raise Exception(_err_msg)
            _focus_on_dataset_index = self._epoch_minibatches_count_mode[1]
            #calculate needed minibatches to see all samples from this specific dataset

            _required_minibatches_num = self._datasets_lengths[_focus_on_dataset_index]  // self._minibatch_pattern[_focus_on_dataset_index]
            _required_minibatches_num += self._datasets_lengths[_focus_on_dataset_index]  % self._minibatch_pattern[_focus_on_dataset_index]
            self._epoch_minibatches_count_mode = _required_minibatches_num
        else:
            if not self._epoch_minibatches_count_mode == 'see_all':
                raise Exception(_err_msg)
            required_minibatches_to_see_each_dataset = [
                (self._datasets_lengths[idx] // self._minibatch_pattern[idx]) + (self._datasets_lengths[idx] % self._minibatch_pattern[idx]) 
                for idx in range(self._datasets_num)
            ]

            self._epoch_minibatches_count_mode = max(required_minibatches_to_see_each_dataset)

        assert isinstance( self._epoch_minibatches_count_mode, int)


    def __len__(self):
        if self._yield_minibatch:
            return self._epoch_minibatches_count_mode
        return self._epoch_minibatches_count_mode*self._minibatch_size
    
    def __iter__(self):
        for _ in range(self._epoch_minibatches_count_mode):
            curr_mb = self._get_one_minibatch()
            if self._yield_minibatch:
                yield curr_mb
            else:
                for idx in curr_mb:
                    yield idx

    def _get_one_minibatch(self):
        # build our minibatch indices
        mb_indices = []
        for dataset_index, count in enumerate(self._minibatch_pattern):                    
            for _ in range(count):
                curr_idx = self._get_next_sample_idx(dataset_index)
                mb_indices.append(curr_idx)

        if self._shuffle_within_minibatch:
            np.random.shuffle(mb_indices)

        return mb_indices

    def _get_next_sample_idx(self, dataset_index):
        assert dataset_index >= 0 and dataset_index < self._datasets_num
        indices_num = len(self._per_dataset_indices[dataset_index])

        assert self._per_dataset_pointers[dataset_index] >=0
        assert self._per_dataset_pointers[dataset_index] <= indices_num

        if self._per_dataset_pointers[dataset_index] == indices_num:
            #we reached the end, so let's reshuffle (if requested) and set position to the start
            if self._shuffle:
                np.random.shuffle(self._per_dataset_indices[dataset_index])
            self._per_dataset_pointers[dataset_index] = 0

        position = self._per_dataset_pointers[dataset_index]
        local_sample_index = self._per_dataset_indices[dataset_index][position]
        self._per_dataset_pointers[dataset_index] += 1

        #now, convert the local position in the dataset into the global position in the concat dataset
        total_sample_index = sum(self._datasets_lengths[:dataset_index]) + local_sample_index
        return int(total_sample_index)
