from fuse.utils import NDict

# from fuse.data import OpBase, get_sample_id
from fusedrug.data.tokenizer.injectortokenizer.injector_tokenizer import (
    InjectorTokenizer,
)

# from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
from fusedrug.data.tokenizer.ops import FastModularTokenizer

# from warnings import warn
# from collections import defaultdict
from typing import Optional, Union, Any

# import os
# import re
# import torch


class InjectorTokenizerOp(FastModularTokenizer):
    """
    applies a injector tokenizer

    injector tokenizer builds on top of modular tokenizer.
    its purpose is to build inputs_emb for the model (instead of input_ids)
        this allows to support more advanced inputs beyond token ids, like:
        * scalars inputs
        * embeddings vector within a single input
    """

    def __init__(
        self,
        input_dim: int,
        tokenizer_path: str,
        max_size: Union[int, None] = None,
        pad_token: Union[str, None] = None,
        pad_type_id: Union[int, None] = None,
        validate_ends_with_eos: Optional[bool] = True,
        eos: Optional[str] = "<EOS>",
        verbose: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            tokenizer_path: full path to a directory that the tokenizer will be loaded from
            max_size: sequences below this size will be padded, and above this size will be truncated
            pad: a string of the pad token
            pad_type_id: see tokenizers.Tokenizer.enable_padding() docstring
            validate_ends_with_eos: during encoder request (a _call_ to the op) will make sure that it ends with the provided eos token, and raise exception otherwise.
                having an eos (end of sentence) token in the end is useful for multiple scenarios, for example in a generative transformer (like T5 encoder-decoder)
            verbose:
        """
        if verbose:
            print(
                f"DEBUG:InjectorTokenizerOp __init__ called for path {tokenizer_path}"
            )

        super().__init__(
            tokenizer_path=tokenizer_path,
            max_size=max_size,
            pad_token=pad_token,
            pad_type_id=pad_type_id,
            validate_ends_with_eos=validate_ends_with_eos,
            eos=eos,
            verbose=verbose,
            **kwargs,
        )

        self._input_dim = input_dim

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str,
        key_out_tokenized_object: Optional[str] = None,
        key_out_tokens_ids: Optional[str] = None,
        key_out_attention_mask: Optional[str] = None,
        convert_attention_mask_to_bool: Optional[bool] = True,
        max_seq_len: Optional[int] = None,
        on_unknown: Optional[str] = "warn",
        verbose: Optional[int] = 1,
        validate_ends_with_eos: Optional[bool] = None,
        key_out_scalars_indices: Optional[str] = None,
        key_out_scalars_values: Optional[str] = None,
        key_out_masked_scalars_indices: Optional[str] = None,
    ) -> NDict:
        """_summary_

        Args:
            sample_dict (NDict): _description_
            key_in (str): key to either a:
                (1) string that contains, in addition to the text that is to be tokenized, special delimiters signifying the type
                of input within each span of text (e.g. <@TOKENIZER-TYPE=AA> sequence, <@TOKENIZER-TYPE=SMILES>, etc.).
                (2) list of modular_tokenizer.TypedInput specifying the tokenizer type and the subsequence to tokenize
            key_out_tokenized_object (Optional[str], optional): _description_. Defaults to None.
            key_out_tokens_ids (Optional[str], optional): _description_. Defaults to None.
            key_out_attention_mask (Optional[str], optional): _description_. Defaults to None.
            convert_attention_mask_to_bool (Optional[bool], optional): _description_. Defaults to True.
            max_seq_len (Optional[int], optional): set maximum sequence len dynamically, used for both padding and truncation.. Defaults to None.
            on_unknown (Optional[str], optional): What happens if unknown tokens (i.e. ones mapped to <UNK>) are encountered: 'raise' or 'warn'. Defaults to "warn".
            verbose (Optional[int], optional): verbosity level. 0: no notification, 1: warning notification, 2: warning with partial data, 3: warning
                with full data. Defaults to 1.
            validate_ends_with_eos (Optional[bool], optional): if not None, overrides self._validate_ends_with_eos
            key_out_scalars_inputs_indices:str optional
                if provided, will write to sample_dict in this key a 1D torch tensor with indices of all inputs scalar elements.
            key_out_scalars_inputs_values:str optional
                if provided, will write to sample_dict in this key a 1D torch tensor with indices of all inputs scalar values.



        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            NDict: _description_
        """

        (
            with_placeholders_str,
            per_meta_orig,
        ) = InjectorTokenizer.build_placeholder_meta_tokenization(
            sequence=sample_dict[key_in], sample_dict=sample_dict
        )
        sample_dict[key_in + ".with_placeholders"] = with_placeholders_str

        super().__call__(
            sample_dict=sample_dict,
            key_in=key_in + ".with_placeholders",
            key_out_tokenized_object=key_out_tokenized_object,
            key_out_tokens_ids=key_out_tokens_ids,
            key_out_attention_mask=key_out_attention_mask,
            convert_attention_mask_to_bool=convert_attention_mask_to_bool,
            max_seq_len=max_seq_len,
            on_unknown=on_unknown,
            verbose=verbose,
            validate_ends_with_eos=validate_ends_with_eos,
            key_out_encoding_per_meta=key_in
            + ".per_meta_part_encoding",  # using the key_in as base for the name because key_out_* are optional
        )

        prepared_data = InjectorTokenizer.prepare_info_for_model_step(
            per_meta_tokenizer_data=per_meta_orig,
            per_meta_encoding_including_placeholders=sample_dict[
                key_in + ".per_meta_part_encoding"
            ],
            sample_dict=sample_dict,
        )

        if key_out_scalars_indices is not None:
            sample_dict[key_out_scalars_indices] = prepared_data["scalars_indices"]
        else:
            if prepared_data["scalars_indices"] is not None:
                raise Exception(
                    "non None scalars_indices found but no key_out_scalars_indices found"
                )

        if key_out_scalars_values is not None:
            sample_dict[key_out_scalars_values] = prepared_data["scalars_values"]
        else:
            if prepared_data["scalars_values"] is not None:
                raise Exception(
                    "non None scalars_value found but no key_out_scalars_values found"
                )

        if key_out_masked_scalars_indices is not None:
            sample_dict[key_out_masked_scalars_indices] = prepared_data[
                "scalars_masked_indices"
            ]
        else:
            if prepared_data["scalars_masked_indices"] is not None:
                raise Exception(
                    "non None scalars_masked_indices found but no key_out_masked_scalars_indices found"
                )

        return sample_dict
