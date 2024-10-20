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

Created on June 30, 2021

"""
from typing import List, Dict, Any
from functools import partial
import pandas as pd
from fuse.eval.metrics.metrics_common import MetricPerBatchDefault
from Bio import Align
import difflib


class MetricPairwiseProteinSequenceAlignmentScore(MetricPerBatchDefault):
    def __init__(
        self,
        preds: str,
        target: str,
        substitution_matrix: str = "BLOSUM62",
        **kwargs: dict,
    ) -> None:
        super().__init__(
            preds=preds,
            target=target,
            metric_per_batch_func=partial(
                _pairwise_protein_sequence_alignment_score,
                substitution_matrix=substitution_matrix,
            ),
            result_aggregate_func=_pairwise_protein_sequence_alignment_compute,
            post_keys_to_collect=["pairwise_alignment_score"],
            **kwargs,
        )


def _pairwise_protein_sequence_alignment_score(
    preds: List[str],
    target: List[str],
    substitution_matrix: str,
) -> List[float]:
    """Compute pairwise sequence alignment statistics
    Args:

    Returns:

    """
    assert isinstance(preds, list)
    assert isinstance(target, list)
    assert len(preds) == len(target)

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = Align.substitution_matrices.load(substitution_matrix)

    penalty_score = 1000
    scores = []
    for (curr_pred, curr_gt) in zip(preds, target):
        try:
            score = aligner.align(curr_gt, curr_pred)[0].score
        except Exception as e:
            print(e)
            print(
                f'Error: _pairwise_protein_sequence_alignment_score::could not calculate alignment pairwise score for target {curr_gt} and prediction {curr_pred} a default "penalty" score of {penalty_score} will be returned'
            )
            score = penalty_score
        scores.append(score)

    return scores


def _pairwise_protein_sequence_alignment_compute(
    pairwise_alignment_score: List,
) -> float:
    return float(pairwise_alignment_score / len(pairwise_alignment_score))


################


def _pairwise_aligned_score(preds: List[str], target: List[str]) -> List[float]:
    assert isinstance(preds, (list, pd.Series))
    assert isinstance(target, (list, pd.Series))
    assert len(preds) == len(target)

    penalty_score = 0.0

    scores = []
    for (curr_pred, curr_gt) in zip(preds, target):
        try:
            sample_indels = compare_strings(curr_gt, curr_pred)
            sample_total = sum(sample_indels.values())
            if sample_total == 0:
                score = 0.0
            else:
                score = sample_indels["equal"] / sample_total
        except Exception as e:
            print(e)
            print(
                f'Error: _pairwise_aligned_score::could not calculate alignment pairwise score for target {curr_gt} and prediction {curr_pred} a default "penalty" score of {penalty_score} will be returned'
            )
            score = penalty_score
        scores.append(score)

    return scores


def compare_strings(from_text: Any, to_text: Any, return_score: bool = False) -> Dict:
    matcher = difflib.SequenceMatcher(None, from_text, to_text)
    counts = dict(
        insert=0,
        delete=0,
        replace=0,
        equal=0,
    )
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            counts[opcode] += a1 - a0
        elif opcode == "replace":
            b_len = b1 - b0
            a_len = a1 - a0
            if a_len == b_len:
                counts["replace"] += a_len
            else:
                counts["replace"] += min(a_len, b_len)
                if a_len > b_len:
                    counts["delete"] += a_len - b_len
                elif b_len > a_len:
                    counts["insert"] += b_len - a_len
                else:
                    assert False, "should not reach here"
        elif opcode == "insert":
            counts[opcode] += b1 - b0
        elif opcode == "delete":
            counts[opcode] += a1 - a0
        else:
            assert False

    if return_score:
        total = sum([val for (_, val) in counts.items()])
        ans = counts["equal"] / total
        return ans

    return counts


def naive_aar_no_indels(preds: List[str], target: List[str]) -> List[float]:
    """
    evlauates the edit distance, normalized by the input length, between 2 strings of the same lengths (i.e., no insertions and deletion)
    incases where len(preds) != len(target) raises an exception.
    """
    scores = []
    for s1, s2 in zip(preds, target):

        if len(s1) != len(s2):
            raise Exception(
                "Encountered different sequence lengths in prediction and target - can not eval naive AAR which assumes equal length"
            )

        score = 0.0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                score += 1
        score = score / len(s1)
        scores.append(score)

    return scores
