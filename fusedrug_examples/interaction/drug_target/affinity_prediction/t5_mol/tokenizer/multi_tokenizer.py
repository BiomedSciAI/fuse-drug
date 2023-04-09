from typing import Dict
from collections.abc import Iterable
from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
from tokenizers import Tokenizer, Encoding
from warnings import warn
from typing import Optional, List, Set, Union, Tuple
import json


class ModularTokenizer:
    def __init__(
        self,
        tokenizers_info: Dict,
        load_adjusted_jsons: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            tokenizers_info (Dict): _description_
            modular_tokenizers_out_path (Optional[str], optional): _description_. Defaults to None.
            load_adjusted_jsons (Optional[bool], optional): Whether to load json files created by ModularTokenizer, or to adjust the indices of given jsons. Defaults to False.
        """

        # TODO: if there is only one tokenizer, leave it as is, without changing its id mappings. Not needed - if there's only one - we can just load its json
        self.tokenizers_info = tokenizers_info

        if not load_adjusted_jsons:
            # store special tokens in a list to preserve their order:
            all_special_tokens = []

            # collect all special tokens (without keeping indices):
            for t_type in self.tokenizers_info:
                t_info = self.tokenizers_info[t_type]
                t_json = json.load(open(t_info["json_path"]))
                self.tokenizers_info[t_type]["json_instance"] = t_json
                model_type = t_json["model"]["type"]
                assert (
                    model_type == "BPE"
                ), f"Only tokenizer models of type BPE are supported (got {model_type}). Other types were not tested. Comment this assertion at your own risk"

                part_special_tokens = ModularTokenizer.get_special_tokens(t_json, force_special=False)
                part_special_tokens = [t for t in part_special_tokens if not t in all_special_tokens]
                all_special_tokens = all_special_tokens + part_special_tokens

            all_special_token_structs = ModularTokenizer.build_special_token_list(all_special_tokens)
            # rearrange regular token indices
            next_index = max([t["id"] for t in all_special_token_structs]) + 1
        else:
            for t_type in self.tokenizers_info:
                t_info = self.tokenizers_info[t_type]
                t_json = json.load(open(t_info["modular_json_path"]))
                self.tokenizers_info[t_type]["json_instance"] = t_json

        for t_type in self.tokenizers_info:
            t_info = self.tokenizers_info[t_type]
            t_json = self.tokenizers_info[t_type]["json_instance"]
            # operations on the tokenizer json
            if not load_adjusted_jsons:
                t_json["added_tokens"] = all_special_token_structs
                (t_json["model"]["vocab"], next_index,) = ModularTokenizer.remap_vocab(
                    vocab=t_json["model"]["vocab"],
                    special_token_structs=all_special_token_structs,
                    starting_index=next_index,
                )
            # end operations on json
            json_str = json.dumps(t_json)
            tokenizer_inst = Tokenizer.from_str(json_str)
            max_size = t_info["max_len"]
            tokenizer_inst.enable_truncation(
                max_length=max_size,
                direction="right",
            )
            json_str = tokenizer_inst.to_str()
            t_json = json.loads(json_str)
            self.tokenizers_info[t_type]["tokenizer_inst"] = tokenizer_inst
            self.tokenizers_info[t_type]["json_instance"] = t_json

        test_res, test_res_detail = self.diagnose()
        assert False not in test_res.values(), "resulting tokenizer is not consistent"
        self.build_inner_decoder()

    @staticmethod
    def remap_vocab(
        vocab: Dict,
        special_token_structs: Optional[List] = None,
        starting_index: Optional[int] = None,
    ) -> Tuple[Dict, int]:
        """Receives a vocabulary, a list of special token structures and a starting index. Returns a new vocabulary that
        a. contains all the special tokens with their IDs, as were given in special_token_structs.
        b. contains all the tokens in vocab (except special ones), numbered consecutively starting with starting_index.
        c. the order of the regular tokens remains unchanged (they are usually ordered by appearance frequency - we do not want to change that)

        Args:
            vocab (Dict): _description_
            special_token_structs (Optional[List]): a list of special token structures to be added to the tokenizer. If None or empty, no special tokens are added. Defaults to None
            starting_index (Optional[int], optional): Starting id of regular tokens. If None - inferred from special_tokens. Defaults to None.

        Returns:
            Tuple[Dict,int]: Returns the updated vocabulary and the next starting index (its max ID + 1)
        """
        if special_token_structs != None and len(special_token_structs) > 0:
            init_vocab = {t["content"]: t["id"] for t in special_token_structs}
            special_tokens = set(init_vocab.keys())
            special_inds = list(init_vocab.values())
            if starting_index is None:
                starting_index = max(special_inds) + 1
        else:
            special_tokens = set()
            init_vocab = []
            if starting_index is None:
                starting_index = 0

        # At this point our vocab contains all the special tokens with their IDs, and we know from which ID to start the regular token mappings

        regular_tokens = set(vocab.keys()) - special_tokens
        regular_vocab = {
            r_t: vocab[r_t] for r_t in regular_tokens
        }  # These are only regular tokens with their original indices
        regular_sorted = sorted(
            regular_vocab.items(), key=lambda x: x[1], reverse=False
        )  # regular tokens sorted by their ID in ascending order.

        regular_vocab = {t[0]: i + starting_index for i, t in enumerate(regular_sorted)}
        init_vocab.update(regular_vocab)
        starting_index = max(regular_vocab.values()) + 1
        return init_vocab, starting_index

    @staticmethod
    def build_special_token_list(special_tokens: Union[List, Set]) -> List:
        """Creates a list of special token structures with consecutive indices, according to the following template
            special token template:
            {
                'id': 0,
                'content': '<UNK>',
                'single_word': False,
                'lstrip': False,
                'rstrip': False,
                'normalized': False,
                'special': True
            }


        Args:
            special_tokens (Union[List, Set]): _description_

        Returns:
            List: _description_
        """
        special_tokens = [
            {
                "id": i,
                "content": v,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i, v in enumerate(special_tokens)
        ]
        return special_tokens

    @staticmethod
    def get_special_tokens(tokenizer_json_inst: Dict, force_special: Optional[bool] = False) -> Set:
        """returns the special tokens from tokenizer defined by json_inst.
            Note: An alternative would be to call tokenizer_inst.get_vocab(with_added_tokens), using with_added_tokens False and True, which
            should've given us just regular and regular+special tokens, but for some reason both these options return the same output,
            so we must resort to json parsing.


        Args:
            json_inst (Dict): _description_
            force_special (Optional[bool], optional): If False, treats all added tokens as special. If True, considers only those that have "special": True as special. Defaults to False.

        Returns:
            Set: _description_
        """
        special_token_structs = tokenizer_json_inst["added_tokens"]
        if force_special:
            special_tokens = [t["content"] for t in special_token_structs if t["special"]]
        else:
            special_tokens = [t["content"] for t in special_token_structs]

        return special_tokens

    @staticmethod
    def get_regular_tokens(tokenizer_json_inst: Dict, force_special: Optional[bool] = False) -> Set:
        """returns the regular tokens from tokenizer defined by json_inst.
            Note: An alternative would be to call tokenizer_inst.get_vocab(with_added_tokens), using with_added_tokens False and True, which
            should've given us just regular and regular+special tokens, but for some reason both these options return the same output,
            so we must resort to json parsing.


        Args:
            json_inst (Dict): _description_
            force_special (Optional[bool], optional): If False, treats all added tokens as special when deciding which token is regular and which is special.
                If True, considers only those that have "special": True as special. Defaults to False.

        Returns:
            Set: _description_
        """
        special_tokens = ModularTokenizer.get_special_tokens(
            tokenizer_json_inst=tokenizer_json_inst, force_special=force_special
        )
        all_tokens = set(tokenizer_json_inst["model"]["vocab"].keys())
        return all_tokens - set(special_tokens)

    @staticmethod
    def get_vocab(tokenizer_json_inst: Dict, token_list: Optional[List] = None) -> Dict:
        """Returns a dictionary of {token:id} of tokenizer tokenizer_json_inst for all tokens in token_list

        Args:
            tokenizer_json_inst (Dict): json instance representation of a tokenizer
            token_list (Optional[List], optional): list of tokens. If None - all tokens are used. Defaults to None.

        Returns:
            Dict: _description_
        """
        all_vocab = tokenizer_json_inst["model"]["vocab"]
        if token_list is None:
            return all_vocab
        output = {t: all_vocab[t] for t in token_list if t in all_vocab}
        return output

    @staticmethod
    def load_from_jsons(tokenizers_info: Dict) -> object:
        """Reads a list of json paths (from tokenizer_info dictionary, as defined in the config), that were created by ModularTokenizer.save_jsons, and creates a modular tokenizer, keeping the ID mappings
        of the jsons.
        TODO: Check the resulting ModularTokenizer for consistency

        Args:
            tokenizer_info (Dict): _description_

        Returns:
            object: _description_
        """
        return ModularTokenizer(tokenizers_info=tokenizers_info, load_adjusted_jsons=True)

    @staticmethod
    def update_id2token_mapping(id2token: Dict, add_vocab: Dict, is_special: Optional[bool] = False) -> Dict:
        """Updates id2token mapping with tokens from add_vocab. Returns the updated id2token

        Args:
            id2token (Dict):
            add_vocab (Dict): vocabulary as returned
            is_special (Optional[bool], optional): whether or not add_vocab holds special tokens. Defaults to False.

        Returns:
            Dict: _description_
        """

        for token in add_vocab:
            if add_vocab[token] in id2token:
                print("Warning: ID collision during update_id2token_mapping for token {token}, id {add_vocab[token]}")
            else:
                tmp_dict = {"token": token, "is_special": is_special}
                id2token[add_vocab[token]] = tmp_dict
        return id2token

    def build_inner_decoder(self) -> None:
        """Goes over all the inner tokenizers and builds an id-to-token mapping with the following structure:
        self.decoder_dict = {id: {
                                    token:token_id,     #token corresponding to the id
                                    is_special:bool,    #whether the token is special or not
                                    }
                                }
        There are two ways to implement this:
        - automatic understanding of relevant tokenizer for each subsequence (subsequences can be recognized from sequence_ids and mask), and using tokenizer.decode
            Pros:
            -   Decoding takes less time by using efficient implementation from tokenizers
            Cons:
            -   Inference may be difficult/inefficient (need to divide into sequences of regular tokens)
        - maintaining a single decoder dictionary, and using it. Currently implemented this option.
            Pros:
            -   straightforward implementation
            Cons:
            -   not as efficient as built-in tokenizer decode.

        """
        self.decoder_dict = {}
        for t_type in self.tokenizers_info:
            t_info = self.tokenizers_info[t_type]
            assert (
                "json_instance" in t_info
            ), f"tokenizer of type {t_type} hasn't been instantiated yet. Call init first."
            if len(self.decoder_dict) == 0:  # Add
                sp_tokens = ModularTokenizer.get_special_tokens(t_info["json_instance"])
                sp_vocab = ModularTokenizer.get_vocab(tokenizer_json_inst=t_info["json_instance"], token_list=sp_tokens)
                self.decoder_dict = ModularTokenizer.update_id2token_mapping(
                    id2token=self.decoder_dict, add_vocab=sp_vocab, is_special=True
                )
            reg_tokens = ModularTokenizer.get_regular_tokens(t_info["json_instance"])
            reg_vocab = ModularTokenizer.get_vocab(tokenizer_json_inst=t_info["json_instance"], token_list=reg_tokens)
            self.decoder_dict = ModularTokenizer.update_id2token_mapping(
                id2token=self.decoder_dict, add_vocab=reg_vocab, is_special=False
            )

    def diagnose(self) -> Tuple[Dict, Dict]:
        """_summary_

        Returns:
            Tuple[Dict, Dict]: brief (pass/fail for each test) and detailed (which tokenizers failed) description of failed tests
        """
        tests = [
            "special token consistency",
            "ID duplicates in vocab",
            "ID collisions across vocabs",
        ]
        result = {t_name: True for t_name in tests}
        result_details = {t_name: [] for t_name in tests}
        tokenizer_types = list(self.tokenizers_info.keys())
        all_inds_set = set()
        all_inds_len = 0
        if len(tokenizer_types) > 1:
            special_tokens = list(
                ModularTokenizer.get_special_tokens(self.tokenizers_info[tokenizer_types[0]]["json_instance"])
            )
            special_tokens_vocab = ModularTokenizer.get_vocab(
                tokenizer_json_inst=self.tokenizers_info[tokenizer_types[0]]["json_instance"],
                token_list=special_tokens,
            )

            # check if all special tokens are the same across all tokenizers
            for t_type in tokenizer_types:
                special_tokens_t = list(
                    ModularTokenizer.get_special_tokens(self.tokenizers_info[t_type]["json_instance"])
                )
                special_tokens_vocab_t = ModularTokenizer.get_vocab(
                    tokenizer_json_inst=self.tokenizers_info[t_type]["json_instance"],
                    token_list=special_tokens_t,
                )

                if special_tokens_vocab != special_tokens_vocab_t:
                    result["special token consistency"] = False
                    result_details["special token consistency"].append(t_type)

            # check if there are no ID collisions within/between vocabs
            for t_type in tokenizer_types:
                regular_tokens = list(
                    ModularTokenizer.get_regular_tokens(self.tokenizers_info[t_type]["json_instance"])
                )
                regular_tokens_vocab = ModularTokenizer.get_vocab(
                    tokenizer_json_inst=self.tokenizers_info[t_type]["json_instance"],
                    token_list=regular_tokens,
                )
                regular_tokens_IDs = regular_tokens_vocab.values()
                regular_tokens_ID_set = set(regular_tokens_IDs)
                if len(regular_tokens_IDs) != len(regular_tokens_ID_set):
                    result["ID duplicates in vocab"] = False
                    result_details["ID duplicates in vocab"].append(t_type)

                all_inds_set = all_inds_set.union(regular_tokens_ID_set)
                if len(all_inds_set) != all_inds_len + len(regular_tokens_ID_set):
                    result["ID collisions across vocabs"] = False
                    result_details["ID collisions across vocabs"].append(t_type)
                all_inds_len = len(all_inds_set)

            special_tokens_ID_set = set(special_tokens_vocab.values())
            if len(special_tokens_vocab.values()) != len(special_tokens_ID_set):
                result["ID duplicates in vocab"] = False
                result_details["ID duplicates in vocab"].append("special")

            all_inds_set = all_inds_set.union(special_tokens_ID_set)
            if len(all_inds_set) != all_inds_len + len(set(special_tokens_ID_set)):
                result["ID collisions across vocabs"] = False
                result_details["ID collisions across vocabs"].append("special")
            all_inds_len = len(all_inds_set)

        return result, result_details

    def is_consistent(self) -> bool:
        """Returns True if the modular tokenizer is consistent, i.e.:
        a. Special tokens are the same (and map to the same indices) across all the tokenizers
        b. Regular token ID mappings of any given tokenizer do not collide with special token mappings, nor with ID mappings of other tokenizers

        Raises:
            Exception: _description_

        Returns:
            bool: _description_
        """
        raise Exception("Not implemented")

    def save_jsons(self, tokenizers_info: Optional[Dict] = None) -> None:
        """_summary_

        Args:
            tokenizers_info (Optional[Dict], optional): Dictionary containing the following:
            {
                tokenizer_type: {
                        "modular_json_path":out_path for tokenizer_type
                    }
            }
            In case of None, paths stored in self.tokenizers_info (modular_json_path for each tokenizer) are used.
            In case of partial tokenizer_types, only those tokenizers will be saved
            Defaults to None.
            TODO: also save the config yaml there
        """
        if tokenizers_info == None:
            for t_type in self.tokenizers_info:
                tokenizer_inst = self.tokenizers_info[t_type]["tokenizer_inst"]
                out_path = self.tokenizers_info[t_type]["modular_json_path"]
                tokenizer_inst.save(out_path)
        else:
            for t_type in tokenizers_info:
                tokenizer_inst = self.tokenizers_info[t_type]["tokenizer_inst"]
                out_path = tokenizers_info[t_type]["modular_json_path"]
                tokenizer_inst.save(out_path)

    def _add_single_tokenizer(
        self,
        tokenizer_info: Dict,
    ) -> None:
        raise Exception("Not implemented")

    def add_special_tokens(
        self,
        special_token_list: List,
    ) -> None:
        raise Exception("Not implemented")

    def add_tokenizers(
        self,
    ) -> None:
        raise Exception("Not implemented")

    def _encode_single_type(self, data_str: str, input_type: str, sequence_id: Optional[int] = None) -> Encoding:
        assert isinstance(data_str, str)
        assert isinstance(input_type, str)

        if not input_type in self.tokenizers_info:
            raise Exception(f"Input type {input_type} not found")

        encoded = self.tokenizers_info[input_type]["tokenizer_inst"].encode(data_str)

        if len(encoded.overflowing) > 0:
            print(
                f"Warning: FastTokenizer had to truncate sequence. Original Sequence Length = {len(data_str)}, max tokens supported = {self.tokenizers_info[input_type]['max_len']}, exceeded by {len(encoded.overflowing[0].ids)} tokens, for tokenizer: {input_type}"
            )

        if sequence_id is None:
            sequence_id = int(self.tokenizers_info[input_type]["tokenizer_id"])  # this does not work.
            # Instead of changing the sequence IDS, it does nothing (probably due to nonunique seq. ids)
        encoded.set_sequence_id(sequence_id)
        # encoded.sequence_ids = [sequence_id] * len(encoded.sequence_ids)

        return encoded

    def encode(
        self,
        typed_input_list: List,
        max_len: Optional[int] = None,
        padding_token_id: Optional[str] = 0,
        padding_token: Optional[str] = "<PAD>",
        pad_type_id: Optional[int] = 0,
    ) -> Encoding:
        """_summary_

        Args:
            typed_input_list (List): _description_
            max_len (Optional[int], optional): _description_. Defaults to None.
            padding_token_id (Optional[str], optional): _description_. Defaults to 0.
            padding_token (Optional[str], optional): _description_. Defaults to "<PAD>".
            pad_type_id (Optional[int], optional): _description_. Defaults to 0.

        Returns:
            Encoding: _description_
        """
        encoded_list = []
        curr_sequence_id = 0
        for input_type, data_str in typed_input_list:
            encoded_list.append(
                self._encode_single_type(
                    data_str=data_str,
                    input_type=input_type,
                    sequence_id=curr_sequence_id,
                )
            )
            curr_sequence_id += 1
            # KEEP THIS AS DOC FOR NOW
            # encoded has attributes [ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing]
            # ids are the encoded tokens,
            # type_ids are for things like "which sentence is this from". There seem to be several limitations on those:
            #   - they must be unique, i.e. several different sentences cannot receive the same type_id from outside (for some reason they can be mapped
            #     to the same value if wrong ids were supplied by the user
            #   - it seems like the ids must be consecutive, starting with 0, in order to work as expected. If they do not start with 0,
            #     it is forced on the last sequence.
            # tokens are the actual tokens (for example - ['c1ccc(', '/C(', '=N/N', 'c2nc3ccccc3', 's2)', 'c2cccc', 'n2)cc1', '[PAD]', '[PAD]', '[PAD]'])
            # offsets describe the starting point and length of each original token
            # attention_mask - by default puts 1 for everything that isn't padding, and 0 for those that are padding
            # special_tokens_mask - 1 for anything that is a special token (e.g. padding, separator, etc.) 0 for the rest
            # overflowing - It's a list of Encoding structures of original content that got clipped out due to max length definition.
            #               In my experience, only the zeroth index contains anything. Don't know when there's more then one member in the list.

        merged_encoding = Encoding.merge(encoded_list)

        if max_len != None:
            merged_encoding.truncate(max_length=max_len)
        if padding_token_id != None:
            # TODO: find the actual padding token ID
            rnd_type = list(self.tokenizers_info.keys())[0]
            rnd_inst = self.tokenizers_info[rnd_type]["tokenizer_inst"]
            padding_token_id = rnd_inst.token_to_id(padding_token)
            merged_encoding.pad(
                length=max_len,
                direction="right",
                pad_id=padding_token_id,
                pad_token=padding_token,
                pad_type_id=pad_type_id,
            )

        return merged_encoding

    def decode(self, id_list: Iterable, skip_special_tokens: Optional[bool] = False) -> str:
        """Receives a list of IDs and returns a string of tokens

        Args:
            id_list (Iterable): _description_
            skip_special_tokens (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            str: _description_
        """

        if skip_special_tokens:
            ret_val = [self.decoder_dict[id]["token"] for id in id_list if not self.decoder_dict[id]["is_special"]]
        else:
            ret_val = [self.decoder_dict[id]["token"] for id in id_list]
        return "".join(ret_val)

    def get_token_id(self, token: str) -> int:
        raise Exception("not implemented yet")

    def encode_string(
        self,
        input_string: str,
        max_len: Optional[int] = None,
        padding_token_id: Optional[str] = 0,
        padding_token: Optional[str] = "<PAD>",
        pad_type_id: Optional[int] = 0,
    ) -> Encoding:
        """Receives a user-supplied string that contains, in addition to the text that is to be tokenized, special delimiters signifying the type
        of input within each span of text (e.g. AA sequence, SMILES, task definition, etc.). These determine the type of tokenizer to use on each span,
        and are not encoded.

        Args:
            input_string (str): _description_
            max_len (Optional[int], optional): _description_. Defaults to None.
            padding_token_id (Optional[str], optional): _description_. Defaults to 0.
            padding_token (Optional[str], optional): _description_. Defaults to "<PAD>".
            pad_type_id (Optional[int], optional): _description_. Defaults to 0.

        Returns:
            Encoding: _description_
        """
        raise Exception("Not implemented yet")

    def get_tokenizer_types(self) -> List:
        return list(self.tokenizers_info.keys())

    # TODO: add get_id method


class ModularMultiTokenizerOp(OpBase):
    """
    Tokenizes multiple types of molecule representations (e.g. AA sequences, SMILES, SELFIES, etc.), by applying a corresponding tokenizer for each type of input
    applies a tokenizers (https://github.com/huggingface/tokenizers) based tokenizer
    """

    def __init__(
        self,
        tokenizer_gen_inst: ModularTokenizer,
        verbose=0,
        **kwargs,
    ):
        """

        Args:
            tokenizer_gen_inst (ModularTokenizer): an instance of ModularTokenizer.
            The tokenizers must map to a single ID space, i.e.
            1. different tokens from each tokenizer must map to different IDs
            2. All tokenizers have the same special tokens, with the same IDs
            3. Same tokens from different tokenizers may map to different IDs
            Every value is a dict containing the following:
                                "tokenizer_inst": tokenizer instance
                                "max_size": max number of output tokens

            pad_type_id (_type_, optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 0.
        """
        super().__init__(**kwargs)

        if verbose > 0:
            print(
                f"DEBUG:ModularMultiTokenizer __init__ called for input types {list(tokenizer_gen_inst.get_tokenizer_types())}"
            )

        self.mtokenizer = tokenizer_gen_inst
        self._verbose = verbose

    def get_token_id(self, token_str: str) -> int:
        """returns the id the token maps to

        Args:
            token_str (str): _description_

        Raises:
            Exception: if could not find the token in any of the internal tokenizers

        Returns:
            int: ID of the input token if it exists
        """
        return self.mtokenizer.get_token_id(token_str)
        # return None

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str,
        key_out_tokenized_object: str = None,
        key_out_tokens_ids: str = None,
        key_out_attention_mask: str = None,
        convert_attention_mask_to_bool: bool = True,
    ) -> NDict:
        """_summary_

        Args:
            sample_dict (NDict): _description_
            key_in (str): a key to a list of tuples (type, input_str) that is the ModularTokenizer.encode input
            key_out_tokenized_object (str, optional): _description_. Defaults to None.
            key_out_tokens_ids (str, optional): _description_. Defaults to None.
            key_out_attention_mask (str, optional): _description_. Defaults to None.
            convert_attention_mask_to_bool (bool, optional): _description_. Defaults to True.

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            NDict: _description_
        """
        typed_input_list = sample_dict[key_in]
        if not isinstance(typed_input_list, list):
            raise Exception(
                f"Expected key_in={key_in} to point to a string, and instead got a {type(typed_input_list)}. value={typed_input_list}"
            )
        encoded = self.mtokenizer.encode(typed_input_list)

        if len(encoded.overflowing) > 0:
            print(
                f"Warning: FastTokenizer had to truncate sequence. Truncated {encoded.overflowing[0].tokens} for sample_id {get_sample_id(sample_dict)}"
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
                "FastTokenizer Op got key_out_tokens_ids=None and key_out_tokenized_object=None, which means it will not modify anything in the sample. Is this intended?"
            )

        return sample_dict


class ModularStringTokenizer(OpBase):
    """
    Tokenizes a raw input string containing multiple types of molecule representations (e.g. AA sequences, SMILES, SELFIES, etc.),
    by inferring which parts of the string correspond to which tokenizer, and applying the corresponding tokenizers.
    applies huggingface(https://github.com/huggingface/tokenizers)-based tokenizers
    """

    def __init__(
        self,
        tokenizer_dict: Dict,  # a dictionary of tokenizer instances whose keys are the input types to be transferred to _call_
        # Every value is a dict containing the following:
        # "tokenizer_inst": tokenizer instance
        # "max_size": max number of output tokens
        max_size=None,  # determines both padding length and max size.
        pad_id=None,
        pad_type_id=None,
        verbose=0,
        **kwargs,
    ):
        """

        Args:
            tokenizer_dict (Dict): a dictionary of tokenizers, keyed by input type (i.e. to which input to apply each tokenizer).
            The tokenizers must map to a single ID space, i.e.
            1. different tokens from each tokenizer must map to different IDs
            2. All tokenizers have the same special tokens, with the same IDs
            3. Same tokens from different tokenizers may map to different IDs
            Every value is a dict containing the following:
                "tokenizer_inst": tokenizer instance
                "max_size": max number of output tokens
                "start_delimiter": starting delimiter of an input sequence that is to be tokenized by the given tokenizer
                "end_delimiter": ending delimiter of an input sequence that is to be tokenized by the given tokenizer
            pad_type_id (_type_, optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 0.
        """
        super().__init__(**kwargs)

        raise Exception("Not implemented")

    def get_token_id(self, token_str: str) -> int:
        """returns the id the token maps to

        Args:
            token_str (str): _description_

        Raises:
            Exception: if could not find the token in any of the internal tokenizers

        Returns:
            int: ID of the input token if it exists
        """
        raise Exception("Not implemented")

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str,
        key_out_tokenized_object: str = None,
        key_out_tokens_ids: str = None,
        key_out_attention_mask: str = None,
        convert_attention_mask_to_bool: bool = True,
        input_type: Optional[str] = None,
    ) -> NDict:
        raise Exception("Not implemented")


class ModularMultiDecoder(OpBase):
    """
    Decodes IDs encoded by ModularMultiTokenizer
    """

    def __init__(
        self,
        tokenizer_dict: Dict,  # a dictionary of tokenizer instances whose keys are the input types to be transferred to _call_
        max_size=None,  # determines both padding length and max size.
        pad_id=None,
        pad_type_id=None,
        verbose=0,
        **kwargs,
    ):
        """

        Args:
            tokenizer_dict (Dict): a dictionary of tokenizers, keyed by input type (i.e. to which input to apply each tokenizer).
            The tokenizers must map to a single ID space, i.e.
            1. different tokens from each tokenizer must map to different IDs
            2. All tokenizers have the same special tokens, with the same IDs
            3. Same tokens from different tokenizers may map to different IDs
            pad_type_id (_type_, optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 0.
        """
        raise Exception("Not implemented")
        super().__init__(**kwargs)

    def __call__(
        self,
        sample_dict: NDict,
        input_type: str,
        key_in: str,
        key_out_tokenized_object: str = None,
        key_out_tokens_ids: str = None,
        key_out_attention_mask: str = None,
        convert_attention_mask_to_bool: bool = True,
    ) -> NDict:
        raise Exception("Not implemented")

    """<TASK_START><BINDING><TASK_END><START_AA>ABCD<END_AA><START_SMILES>CCCCC<END_SMILES>
    """
