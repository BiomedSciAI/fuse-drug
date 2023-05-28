"""
(C) Copyright 2021 IBM Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from typing import List, Optional
from statistics import mean
import random

from fuse.eval.metrics.classification.metrics_classification_common import MetricDefault


class MetricSeqAccuracy(MetricDefault):
    """
    Accuracy of a generated sequence.
    """

    def __init__(self, pred: Optional[str] = None, target: Optional[str] = None, **kwargs: dict):
        super().__init__(metric_func=self.sequence_accuracy, pred=pred, target=target, **kwargs)

    @staticmethod
    def sequence_accuracy(pred: List[str], target: List[str]) -> float:
        accuracy_list = []
        for pred_i, target_i in zip(pred, target):
            correct_count = 0
            for pred_char, target_char in zip(pred_i, target_i):
                if pred_char == target_char:
                    correct_count += 1
            accuracy_list.append(correct_count / len(target_i))

        return mean(accuracy_list)


class MetricPrintRandomSubsequence(MetricDefault):
    """Dump prediction and target of random samples (num_sample_to_print)"""

    def __init__(self, pred: str, target: str, num_sample_to_print: int):
        super().__init__(metric_func=self.dump, pred=pred, target=target)
        self._num_samples_to_print = num_sample_to_print

    def dump(self, pred: list, target: list) -> None:
        indices = random.sample(range(len(pred)), self._num_samples_to_print)
        for index in indices:
            print(f"\n{target[index]}->\n{pred[index]}\n---------")
