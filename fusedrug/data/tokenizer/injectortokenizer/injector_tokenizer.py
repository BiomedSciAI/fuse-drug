from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
from typing import Dict
from typing import Optional, List, Union, Tuple, Any
from tokenizers import Encoding
import omegaconf
import torch
from collections.abc import Iterable
import re


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

    def __init__(
        self,
        input_dim: int,
        embedding_layer: torch.nn.Module,
        tokenizers_info: Union[List, omegaconf.listconfig.ListConfig],
        load_adjusted_jsons: Optional[bool] = False,
        special_tokens_dict: Optional[Dict] = None,
        additional_tokens_list: Optional[List] = None,
        max_possible_token_id: Optional[int] = None,
        max_special_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        input_dim: the size of a vector of each input. The total output will be [sequence length, input_dim]


        """
        self._input_dim = input_dim
        self._embedding_layer = embedding_layer
        self._modular_tokenizer = ModularTokenizer(
            tokenizers_info=tokenizers_info,
            load_adjusted_jsons=load_adjusted_jsons,
            special_tokens_dict=special_tokens_dict,
            additional_tokens_list=additional_tokens_list,
            max_possible_token_id=max_possible_token_id,
            max_special_token_id=max_special_token_id,
            **kwargs,
        )

        print("")

    def encode_list(
        self,
        typed_input_list: List,
        max_len: Optional[int] = None,
        padding_token_id: Optional[int] = None,
        padding_token: Optional[str] = "<PAD>",
        pad_type_id: Optional[int] = None,
        return_overflow_info: Optional[bool] = False,
        on_unknown: Optional[str] = "warn",
        verbose: int = 1,
    ) -> Union[Encoding, Tuple[Encoding, str]]:
        """_summary_

        Args:
            typed_input_list (List): list of collections.namedtuple("input_type", ["input_string", "max_len"]), with
                input type: the name of input type,
                input_string: the string to be encoded
                max_len: maximal length of the encoding (in tokens). Only relevant for truncation, as we do not need to
                pad individual sub-tokenizer encodings - we only pad the final encoding of the ModularTokenizer.
                The smallest value between config-defined and tuple-defined is used. If None, the max_len
                that was defined for the sub-tokenizer in the config is used.
            max_len (Optional[int], optional): _description_. Defaults to None.
            padding_token_id (Optional[str], optional): _description_. Defaults to 0. TODO: default to None and infer it
            padding_token (Optional[str], optional): _description_. Defaults to "<PAD>".
            pad_type_id (Optional[int], optional): _description_. Defaults to 0. (TODO: raise exception)
            return_overflow_info (Optional[bool], optional): If True return an additional string with overflow information. Defaults to False.
            on_unknown: (Optional[str], optional): What happens if unknown tokens (i.e. ones mapped to <UNK>) are encountered: 'raise' or 'warn'
            verbose (Optional[int], optional): verbosity level. 0: no notification, 1: warning notification, 2: warning with partial data, 3: warning
                with full data. Defaults to 1.
        Returns:
            Encoding: _description_
        """

        raise NotImplementedError

    def encode(
        self,
        sequence: str,
        max_len: Optional[int] = None,
        padding_token_id: Optional[int] = 0,
        padding_token: Optional[str] = "<PAD>",
        pad_type_id: Optional[int] = 0,
        return_overflow_info: Optional[bool] = False,
        on_unknown: Optional[str] = "warn",
        verbose: Optional[int] = 1,
    ) -> Encoding:
        # (self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True)
        """Receives a user-supplied string that contains, in addition to the text that is to be tokenized, special delimiters signifying the type
        of input within each span of text (e.g. <@TOKENIZER-TYPE=AA> sequence, <@TOKENIZER-TYPE=SMILES>, etc.). These determine the type of tokenizer to use on each span,
        and are not encoded.
        Optionaly, you may also describe maximum length per section, for example:
            "<@TOKENIZER-TYPE=AA><BLAH><BLAH2>QKPGQAPRLLIYG<@TOKENIZER-TYPE=AA@MAX-LEN=122><BLAH3>SGSDFSDFSFD"
            would not have a local limitation of the first AA section, but will have a local maximum length of 122 on the second section.
            local in this context means that the maximum length will be imposed on the individual section prior to applying any global "entire sequence" maximum size limitations (if any).

        Args:
            input_string (str): _description_
            max_len (Optional[int], optional): _description_. Defaults to None.
            padding_token_id (Optional[str], optional): _description_. Defaults to 0.
            padding_token (Optional[str], optional): _description_. Defaults to "<PAD>".
            pad_type_id (Optional[int], optional): _description_. Defaults to 0.
            return_overflow_info (Optional[bool], optional): _description_. If True return an additional string with overflow information. Defaults to False.
            on_unknown: (Optional[str], optional): What happens if unknown tokens (i.e. ones mapped to <UNK>) are encountered: 'raise' or 'warn'
            verbose (int, optional): verbosity level. 0: no notification, 1: warning notification, 2: warning with partial data, 3: warning
                with full data. Defaults to 1.
        Returns:
            Encoding: _description_
            str: _description_ information on overflow, if return_overflow_info=True
        """

        raise NotImplementedError

    def decode(self, ids: Iterable, skip_special_tokens: Optional[bool] = False) -> str:
        """Receives a list of IDs and returns a string of tokens
            TODO: possibly output also the type of token (AA, SMILES, etc)
        Args:
            ids (Iterable): _description_
            skip_special_tokens (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            str: _description_
        """

        raise NotImplementedError

    @staticmethod
    def build_placeholder_meta_tokenization(sequence: str) -> Tuple[str, List[str]]:
        """
        In order to avoid modifying and rewriting the logic in modular tokenizer, especially regarding padding, limitation of max length of certain sub-parts,
         we put placeholders to make sure that the total size is known/fixed and respects the meta instructions to the modular tokenizer
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
            if tokenizer_type == "FLOAT":
                with_placeholders.append(
                    "<@TOKENIZER-TYPE=AA>"
                )  # won't use AA tokens, just an arbitrary one to be able to use a token like <1>
                values = subseq.split(",")
                seq = "<1>" * len(values)
                with_placeholders.append(seq)
            elif tokenizer_type == "VECTOR":
                with_placeholders.append(
                    "<@TOKENIZER-TYPE=AA>"
                )  # won't use AA tokens, just an arbitrary one to be able to use a token like <1>
                values = subseq.split("@")
                seq = "<1>" * len(values)
                with_placeholders.append(seq)
            else:
                with_placeholders.append(tokenizer_type)
                with_placeholders.append(subseq)

        return "".join(with_placeholders), with_placeholders
