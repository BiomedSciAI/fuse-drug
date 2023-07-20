from typing import List, Union
from torch.utils.data import Sampler
import numpy as np
import pandas as pd


class BalancedClassDataFrameSampler(Sampler):
    """
    Balanced sampler for large dataframes.
    (Native fuse BatchSamplerDefault caused memory issues when used with very large datasets).
    :param df: the dataframe which stores the data
    :param label_column_name: name of the label column in the dataframe
    :param classes: list of class names to use
    :param counts: list of number of times each class should be used in a minibatch. the minibatch size will be the sum of counts
    :param shuffle: whether to shuffle  - both each minibatch, and once saw all samples from a class. TODO: consider separating into two params
    :total_minibatches: total minibatches per epoch
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_column_name: str,
        classes: List[str],
        counts: List[int],
        shuffle: bool = True,
        total_minibatches: Union[int, str] = "see_all_smallest_class",
    ):
        self.df = df
        assert len(classes) == len(counts)
        self.shuffle = shuffle
        self.minibatch_size = sum(counts)
        assert self.minibatch_size > 0

        self.classes = classes
        self.counts = counts

        self.indices_per_class = []
        self.pointers = []

        minibatches_needed_to_see_all = []

        for curr_class, curr_count in zip(self.classes, self.counts):
            assert isinstance(curr_count, int)
            indices = np.flatnonzero(df[label_column_name] == curr_class)
            if len(indices) == 0:
                raise Exception(f"found zero cases for class {curr_class} !")
            else:
                print(f"found {len(indices)} cases of class {curr_class}")
            self.indices_per_class.append(indices)
            self.pointers.append(indices.shape[0])

            minibatches_needed_to_see_all.append(
                int(np.ceil(indices.shape[0] / curr_count))
            )

        smallest_class_index = np.argmin([len(x) for x in self.indices_per_class])

        total_minibatches_to_see_all_samples_at_least_once = (
            minibatches_needed_to_see_all[smallest_class_index]
        )
        print(
            f"To see all {self.indices_per_class[smallest_class_index].shape[0]} samples of the most rare class ({self.classes[smallest_class_index]}) it would take {total_minibatches_to_see_all_samples_at_least_once} minibatches."
        )
        print(
            'set total_minibatches to "see_all_smallest_class" to use this number of minibatches.'
        )
        if isinstance(total_minibatches, str):
            assert total_minibatches == "see_all_smallest_class"
            self.total_minibatches = total_minibatches_to_see_all_samples_at_least_once
        else:
            print(
                f"a custom total_minibatches was provided ({total_minibatches}). This is useful since if the number is large then something in pytorch (or pytorch lightning) makes GPU utilization drop drastically."
            )
            self.total_minibatches = total_minibatches

            epochs_to_see_all = (
                total_minibatches_to_see_all_samples_at_least_once / total_minibatches
            )
            print(
                f"Under the provided epoch size of {total_minibatches} minibatches per epoch, it would take {epochs_to_see_all:.2f} epochs to see at least once every sample."
            )

    def __len__(self):
        return self.total_minibatches

    def _get_next_sample(self, class_index):
        if self.pointers[class_index] >= self.indices_per_class[class_index].shape[0]:
            if self.shuffle:
                print(f"shuffling for (class name={self.classes[class_index]})")
                np.random.shuffle(self.indices_per_class[class_index])
            self.pointers[class_index] = 0

        final_raw_index = self.indices_per_class[class_index][
            self.pointers[class_index]
        ]
        ans = self.df.index[final_raw_index]

        self.pointers[class_index] += 1

        return ans

    def __iter__(self):
        for _ in range(self.total_minibatches):
            curr_mb = []
            for class_idx, count in enumerate(self.counts):
                for _ in range(count):
                    curr_mb += [self._get_next_sample(class_idx)]
            if self.shuffle:
                np.random.shuffle(curr_mb)
            yield curr_mb
