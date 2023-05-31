from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
from fusedrug.data.tokenizer.modulartokenizer.multi_tokenizer import (
    ModularTokenizer as Tokenizer,
)
from warnings import warn
from collections import defaultdict
from typing import Tuple, Optional, Union, Any
import os
import re


class FastModularTokenizer(OpBase):
    """
    applies a modular tokenizer
    """

    def __init__(
        self,
        tokenizer_path: str,
        max_size: Union[int, None] = None,
        pad_token: Union[str, None] = None,
        pad_type_id: Union[int, None] = None,
        validate_ends_with_eos: Optional[str] = "<EOS>",
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
        super().__init__(**kwargs)

        if verbose:
            print(f"DEBUG:FastModularTokenizer __init__ called for path {tokenizer_path}")

        self._tokenizer_path = tokenizer_path
        self._tokenizer = Tokenizer.from_file(self._tokenizer_path)
        pad_id = self._tokenizer.token_to_id(pad_token)

        if pad_token is None:
            raise Exception(f"Could not find pad token = {pad_token} in {tokenizer_path}")

        self._validate_ends_with_eos = validate_ends_with_eos

        if self._validate_ends_with_eos is not None:
            eos_id = self._tokenizer.token_to_id(self._validate_ends_with_eos)
            if eos_id is None:
                raise Exception(
                    f"Could not find eos token = {validate_ends_with_eos} in {tokenizer_path}. You can disable the validation by setting validate_ends_with_eos=None"
                )

        self._pad_id = pad_id
        self._verbose = verbose

        if max_size is not None:
            assert isinstance(max_size, int)
            assert max_size > 0
            assert isinstance(pad_id, int)

            padding_kwargs = dict(length=max_size, pad_id=pad_id)
            if pad_type_id is not None:
                assert isinstance(pad_type_id, int)
                padding_kwargs["pad_type_id"] = pad_type_id

            self._tokenizer.enable_padding(direction="right", **padding_kwargs)

            self._tokenizer.enable_truncation(
                max_length=max_size,
                direction="right",
            )

        self._max_size = max_size

        if self._verbose:
            self._debug_max_tokenized_len_encountered = defaultdict(int)

    # note: use normalizer.Sequence to chain multiple normalizers
    def set_normalizer(self, normalizer: Any) -> None:
        # self._tokenizer.normalizer = normalizer
        raise Exception("Not implemented")

    # note: use pre_toknizers.Sequence to chain multiple pre_toknizers
    def set_pre_tokenizer(self, pre_tokenizer: Any) -> None:
        # self._tokenizer.pre_tokenizer = pre_tokenizer
        raise Exception("Not implemented")

    def set_post_processor(self, post_processor: Any) -> None:
        # self._tokenizer.post_processor = post_processor
        raise Exception("Not implemented")

    def get_vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def get_max_token_id(self) -> Tuple[str, int]:
        """
        scans the vocab for the max observed token id and returns a tuple for it
            [its token string (str), the token id (int)]
        """
        max_token_id = self._tokenizer.get_max_id()
        token_str_of_max_token_id = self._tokenizer.id_to_token(max_token_id)

        if max_token_id is None:
            raise Exception("Could not find max_token_id! possibly an empty vocab.")
        if token_str_of_max_token_id is None:
            warn("max_token_id does not correspond to a token. It may be an upper bound placeholder.")
            return "placeholder upper-bound token", max_token_id
        return token_str_of_max_token_id, max_token_id

    def get_min_max_sentinels(
        self,
        sentinel_prefix: Optional[str] = "<SENTINEL_ID",
        integer_find_regex: Optional[str] = "\d{1,}",
    ) -> Tuple[int, int]:
        """
        returns a Tuple [min encountered sentinel name, max encountered sentinel name]

        For example, if the vocab contains:

        SENTINEL_ID_101: 1000,
        SENTINEL_ID_102: 1001,
        SENTINEL_ID_103: 1002,

        will return [101,103]
        """
        min_token = None
        max_token = None

        for k, _ in self._tokenizer.get_added_vocab().items():
            if sentinel_prefix in k:
                val = re.findall(integer_find_regex, k)
                if len(val) != 1:
                    raise Exception(f"expected exactly one integer number in {k} but found {val}")
                val = val[0]
                val = int(val)

                if (min_token is None) or (val < min_token):
                    min_token = val

                if (max_token is None) or (val > max_token):
                    max_token = val

        if (min_token is None) or (max_token is None):
            raise Exception(f'Could not find any sentinels with the prefix "{sentinel_prefix}"')

        return (min_token, max_token)

    def get_token_id(self, token_str: str) -> int:
        ans = self._tokenizer.token_to_id(token_str)
        assert ans is not None, f"could not find token id for token:{token_str}!"
        return ans

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str,
        key_out_tokenized_object: Optional[str] = None,
        key_out_tokens_ids: Optional[str] = None,
        key_out_attention_mask: Optional[str] = None,
        convert_attention_mask_to_bool: Optional[bool] = True,
    ) -> NDict:
        # if self._verbose:
        #     print(
        #         f'PID:{os.getpid()} FastModularTokenizer op sample_id {sample_dict["data.sample_id"]} key_in={key_in} pdb={sample_dict["pdb"]} HeavyChain: {sample_dict["Hchain"]} LightChain: {sample_dict["Lchain"]}'
        #     )

        data_lst = sample_dict[key_in]
        if not isinstance(data_lst, list):
            # data_lst is a list of named tuples of type collections.namedtuple("TypedInput", ["input_type", "input_string", "max_len"])
            raise Exception(
                f"Expected key_in={key_in} to point to a list of inputs, and instead got a {type(data_lst)}. value={data_lst}"
            )

        if self._validate_ends_with_eos is not None:
            if not data_lst[-1].input_string.rstrip().endswith(self._validate_ends_with_eos):
                raise Exception(
                    f"self._validate_ends_with_eos was set to {self._validate_ends_with_eos}, but about to encode a string that does not end with it. The str end was: {data_lst[-1].input_string}"
                )

        encoded = self._tokenizer.encode_list(data_lst)

        if self._max_size is not None:  # we tightly couple padding length and max size.
            assert self._max_size == len(encoded.ids)

        if self._verbose:
            if self._pad_id in encoded.ids:
                _encoded_len_unpadded = encoded.ids.index(self._pad_id)
            else:
                # no padding, therefore it was fully used (either exactly the size, or most likely it was clipped)
                _encoded_len_unpadded = len(encoded.ids)

            if _encoded_len_unpadded > self._debug_max_tokenized_len_encountered[self._tokenizer_path]:
                print(
                    "DEBUG: FastModularTokenizer: encountered new max encoded size:",
                    _encoded_len_unpadded,
                    " for tokenizer: ",
                    self._tokenizer_path,
                )
                self._debug_max_tokenized_len_encountered[self._tokenizer_path] = _encoded_len_unpadded

        # KEEP THIS AS DOC FOR NOW
        # encoded has attributes [ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing]
        # ids are the encoded tokens,
        # type_ids are for things like "which sentence is this from"
        # tokens are the actual tokens (for example - ['c1ccc(', '/C(', '=N/N', 'c2nc3ccccc3', 's2)', 'c2cccc', 'n2)cc1', '[PAD]', '[PAD]', '[PAD]'])
        # offsets describe the starting point and length of each original token
        # attention_mask - by default puts 1 for everything that isn't padding, and 0 for those that are padding
        # special_tokens_mask - 1 for anything that is a special token (e.g. padding, separator, etc.) 0 for the rest
        # overflowing - it's encoded original content that get clipped out due to max length definition

        if (
            len(encoded.overflowing) > 0
        ):  # note, encoded.overflowing may have multiple items, and each item can contain multiple items
            overall_char_len = sum([len(x.input_string) for x in data_lst])
            print(
                f"Warning: FastModularTokenizer (pid={os.getpid()}) had to truncate sequence. Original Sequence Length = {overall_char_len} max supported = {self._max_size} for tokenizer: {self._tokenizer_path} for sample_id {get_sample_id(sample_dict)}"
            )

        if key_out_tokenized_object is not None:
            # if requested, store the entire tokenizer.Encoding object (which provides access to attributes such as  [ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
            sample_dict[key_out_tokenized_object] = encoded

        if key_out_tokens_ids is not None:
            sample_dict[key_out_tokens_ids] = encoded.ids

        if key_out_attention_mask is not None:
            sample_dict[key_out_attention_mask] = encoded.attention_mask
            if convert_attention_mask_to_bool:
                sample_dict[key_out_attention_mask] = [bool(x) for x in sample_dict[key_out_attention_mask]]

        if (key_out_tokens_ids is None) and (key_out_tokenized_object is None):
            warn(
                "FastModularTokenizer Op got key_out_tokens_ids=None and key_out_tokenized_object=None, which means it will not modify anything in the sample. Is this intended?"
            )

        return sample_dict