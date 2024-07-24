from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
from typing import Optional, List, Tuple, Dict
from tokenizers import Encoding
import torch
import re
from fuse.utils import NDict


class InjectorTokenizer(ModularTokenizer):
    """
    InjectorTokenizer builds on top of ModularTokenizer.

    Its purpose is to extend beyond "standard" input tokens as integers as input for a model.
    Instead, it provides control on *vectors* that are to be used as input for a model.

    Example use cases:
    1. Providing scalars (floating point) as inputs
    2. Providing vectors of embeddings - for example of a protein embedding

    Each input "token" becomes a tensor of a defined size, and is built of:
    1. Header
        made of 4 floats
        [
            0.0 or 1.0 #is this a sentinel/mask or not
            0.0 or 1.0 #is this a standard vocabulary token
            0.0 or 1.0 #is this a scalar
            0.0 or 1.0 #is this a full injected vector (e.g. an embedding)
        ]
    2. Content
        the rest of each input vector is made of input_dim-4 float elements.


    Note - in the "standard vocabulary token" - we support providing an external embeding layer (like in vanilla T5),
        as it's part of the trained weights.

    """

    @staticmethod
    def build_placeholder_meta_tokenization(
        *,
        sequence: str,
        sample_dict: Optional[NDict] = None,
    ) -> Tuple[str, List[str]]:
        """
        In order to avoid modifying and rewriting the logic in modular tokenizer, especially regarding padding, limitation of max length of certain sub-parts,
         we put placeholders to make sure that the total size is known/fixed and respects the meta instructions to the modular tokenizer

         Returns: a tuple with 2 elements
         (
            a single string with the full query containing placeholder tokens for FLOAT and VECTOR meta tokenizer parts,
            a list of [meta-tokenizer name, data, meta-tokenizer name, data, meta-tokenizer name, data,  ...]
         )
        """
        hints_and_subseq = re.split("<@TOKENIZER-TYPE=([^>]*)>", sequence)[
            1:
        ]  # the first element is blank - removing it
        assert (
            len(hints_and_subseq) > 0 and len(hints_and_subseq) % 2 == 0
        ), f"Error: expecting leading modular tokenizer hints followed by a sequence to tokenize, got {sequence}"

        with_placeholders = []

        for tokenizer_type, subseq in zip(
            hints_and_subseq[::2], hints_and_subseq[1::2]
        ):
            if tokenizer_type.startswith("SCALARS_"):
                with_placeholders.append(
                    "<@TOKENIZER-TYPE=AA>"
                )  # won't use AA tokens, just an arbitrary one to be able to use a token like <SCALAR>

                if (
                    tokenizer_type == "SCALARS_LITERALS"
                ):  # note: masking is only supported in literals (not in "from dict")
                    values = subseq.split(",")
                    # seq = "<SCALAR>" * len(values)
                    seq = "".join(
                        [
                            "<MASKED_SCALAR>" if x == "<MASK>" else "<SCALAR>"
                            for x in values
                        ]
                    )
                elif tokenizer_type == "SCALARS_FROM_DICT":
                    if sample_dict is None:
                        raise Exception(
                            "SCALARS_FROM_DICT used but the provided sample_dict is None"
                        )
                    values = sample_dict[subseq]
                    assert len(values.shape) == 1
                    seq = "<SCALAR>" * len(values)
                else:
                    raise Exception(f"tokenizer_type={tokenizer_type} is not supported")

                # elif tokenizer_type == "SCALARS_MASKED":
                #     values = subseq.split(",")
                #     assert all([x=='<MASK>' for x in values]) #only <MASK> is currently supported
                #     seq = "<MASKED_SCALAR>" * len(values)

                with_placeholders.append(seq)

            elif tokenizer_type.startswith("VECTORS_"):
                raise Exception("VECTOR_* are not supported yet")
            else:
                with_placeholders.append("<@TOKENIZER-TYPE=" + tokenizer_type + ">")
                with_placeholders.append(subseq)

        return "".join(with_placeholders), hints_and_subseq

    @staticmethod
    def prepare_info_for_model_step(
        *,
        per_meta_tokenizer_data: List[str],
        per_meta_encoding_including_placeholders: List[Encoding],
        sample_dict: Optional[NDict] = None,
    ) -> Dict:
        """
        since we:
        1. Need to use the model embedding layer (allowing gradients flow if needed)
        2. We prefer not to use the model during the data pipeline

        In this function we prepare everything so that during the train/val/test_step we'll be able to do what's needed before doing the forward pass

        Args:
            per_meta_tokenizer_data: a list of [meta-tokenizer name, data, meta-tokenizer name, data, meta-tokenizer name, data,  ...]
            per_meta_encoding_including_placeholders: a list of Encoding elements. This is used to extract per tokenizer final tokens num (after all of the padding and cropping logic was already done)
            sample_dict: a fuse sample_dict - optional.
                needed only if the meta tokenizer instruction uses a syntax of lookup from the dictionary


        """
        scalars_indices = []
        scalars_values = []
        scalars_masked_indices = []
        prev_index_end = -1

        for tokenizer_name, curr_str_data, curr_placeholder_encoding in zip(
            per_meta_tokenizer_data[::2],
            per_meta_tokenizer_data[1::2],
            per_meta_encoding_including_placeholders,
        ):
            if tokenizer_name.startswith("SCALARS_"):
                if "SCALARS_LITERALS" == tokenizer_name:
                    curr_str_data = curr_str_data.strip().split(",")
                    if len(curr_str_data) != len(curr_placeholder_encoding.ids):
                        raise Exception(
                            f"should match expected length. Found length {len(curr_str_data)} but placeholders length was {len(curr_placeholder_encoding.ids)}"
                        )

                    curr_indices = []
                    curr_data = []

                    for i, val in enumerate(curr_str_data):
                        if val != "<MASK>":
                            curr_indices.append(i + prev_index_end + 1)
                            curr_data.append(float(val))
                        else:
                            scalars_masked_indices.append(i + prev_index_end + 1)

                    if len(curr_indices) > 0:
                        curr_indices = torch.tensor(curr_data, dtype=torch.int64)
                        curr_data = torch.tensor(curr_data, dtype=torch.float32)

                        scalars_indices.append(curr_indices)
                        scalars_values.append(curr_data)

                        assert len(curr_data.shape) == 1
                elif "SCALARS_FROM_DICT" == tokenizer_name:
                    if sample_dict is None:
                        raise Exception(
                            "SCALARS_FROM_DICT used but the provided sample_dict is None"
                        )
                    curr_data = sample_dict[curr_str_data]
                    assert len(curr_data.shape) == 1
                    curr_indices = torch.arange(
                        prev_index_end + 1, prev_index_end + 1 + curr_data.shape[0]
                    )

                    scalars_indices.append(curr_indices)
                    scalars_values.append(curr_data)

                    prev_index_end += curr_data.shape[0]

                else:
                    raise Exception(
                        "Only supported SCALARS_* tokenizers are SCALARS_LITERALS and SCALARS_FROM_DICT"
                    )

            elif tokenizer_name.startswith("VECTORS_"):
                raise NotImplementedError
            else:
                prev_index_end += len(curr_placeholder_encoding.ids)

        if len(scalars_indices) > 0:
            scalars_indices = torch.concat(scalars_indices)
            scalars_values = torch.concat(scalars_values)

        if len(scalars_masked_indices) > 0:
            scalars_masked_indices = torch.tensor(
                scalars_masked_indices, dtype=torch.int64
            )
        else:
            scalars_masked_indices = None

        return {
            "scalars_indices": scalars_indices,
            "scalars_values": scalars_values,
            "scalars_masked_indices": scalars_masked_indices,
        }
